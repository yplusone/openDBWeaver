/*
query_template: SELECT SearchEngineID, ClientIP, COUNT(*) AS c, SUM(IsRefresh) AS sum_isrefresh, AVG(ResolutionWidth) AS avg_resolutionwidth
                FROM hits
                WHERE SearchPhrase <> ''
                GROUP BY SearchEngineID, ClientIP
                ORDER BY c DESC
                LIMIT 10;

split_template: select * from dbweaver((SELECT SearchEngineID, ClientIP, IsRefresh, ResolutionWidth FROM hits WHERE (SearchPhrase!='')));

query_example: SELECT SearchEngineID, ClientIP, COUNT(*) AS c, SUM(IsRefresh) AS sum_isrefresh, AVG(ResolutionWidth) AS avg_resolutionwidth
               FROM hits
               WHERE SearchPhrase <> ''
               GROUP BY SearchEngineID, ClientIP
               ORDER BY c DESC
               LIMIT 10;

split_query: select * from dbweaver((SELECT SearchEngineID, ClientIP, IsRefresh, ResolutionWidth FROM hits WHERE (SearchPhrase!='')));
*/

#define DUCKDB_EXTENSION_MAIN

#include "dbweaver_extension.hpp"
#include "duckdb.hpp"
#include "duckdb/common/types/vector.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/common/vector_size.hpp"

#include <absl/container/flat_hash_map.h>

#include <atomic>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <utility>
#include <vector>
#include <algorithm>

namespace duckdb {

// ============================================================
//  Key: SearchEngineID + ClientIP + precomputed hash
// ============================================================
struct HashedKey {
	int32_t search_engine_id;
	int64_t client_ip;

	bool operator==(const HashedKey &o) const noexcept {
		return search_engine_id == o.search_engine_id && client_ip == o.client_ip;
	}
};

struct HashedKeyHash {
	size_t operator()(const HashedKey &k) const noexcept {
		uint64_t h = (uint64_t)k.client_ip;
		h ^= (uint64_t)k.search_engine_id + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
		return (size_t)h;
	}
};


// ============================================================
//  Agg payload
// ============================================================

struct AggPayload {
	int64_t c = 0;
	int64_t sum_isrefresh = 0;
	int64_t sum_resolutionwidth = 0;
};

// ============================================================
//  Bind data
// ============================================================

struct FnBindData : public FunctionData {
	unique_ptr<FunctionData> Copy() const override { return make_uniq<FnBindData>(); }
	bool Equals(const FunctionData &) const override { return true; }
};

// ============================================================
//  Global state
// ============================================================

struct FnGlobalState : public GlobalTableFunctionState {
	std::mutex lock;

	absl::flat_hash_map<HashedKey, AggPayload, HashedKeyHash> map;

	std::atomic<idx_t> active_local_states{0};
	std::atomic<idx_t> merged_local_states{0};

	// 0:not started, 1:adopting, 2:adopt done
	std::atomic<uint8_t> adopt_stage{0};

	idx_t MaxThreads() const override { return std::numeric_limits<idx_t>::max(); }
};

// ============================================================
//  Local state
// ============================================================
struct FnLocalState : public LocalTableFunctionState {
	bool merged = false;

	absl::flat_hash_map<HashedKey, AggPayload, HashedKeyHash> map;

	inline AggPayload &FindOrInsertPayload(int32_t engine_id, int64_t client_ip) {
		HashedKey tmp{engine_id, client_ip};
		return map[tmp];
	}
};


// ============================================================
//  Type-Specialized Dispatching Helpers
// ============================================================

template <typename T>
static void FillColumnBuffer(int64_t *dest, const UnifiedVectorFormat &uvf, idx_t size) {
	const T *data = (const T *)uvf.data;
	const auto sel = uvf.sel;
	for (idx_t i = 0; i < size; ++i) {
		dest[i] = (int64_t)data[sel->get_index(i)];
	}
}

static void DispatchFillBuffer(int64_t *dest, const UnifiedVectorFormat &uvf, LogicalTypeId tid, idx_t size) {
	switch (tid) {
	case LogicalTypeId::BOOLEAN:
		FillColumnBuffer<bool>(dest, uvf, size);
		break;
	case LogicalTypeId::TINYINT:
		FillColumnBuffer<int8_t>(dest, uvf, size);
		break;
	case LogicalTypeId::SMALLINT:
		FillColumnBuffer<int16_t>(dest, uvf, size);
		break;
	case LogicalTypeId::INTEGER:
		FillColumnBuffer<int32_t>(dest, uvf, size);
		break;
	case LogicalTypeId::BIGINT:
		FillColumnBuffer<int64_t>(dest, uvf, size);
		break;
	case LogicalTypeId::UTINYINT:
		FillColumnBuffer<uint8_t>(dest, uvf, size);
		break;
	case LogicalTypeId::USMALLINT:
		FillColumnBuffer<uint16_t>(dest, uvf, size);
		break;
	case LogicalTypeId::UINTEGER:
		FillColumnBuffer<uint32_t>(dest, uvf, size);
		break;
	case LogicalTypeId::UBIGINT:
		FillColumnBuffer<uint64_t>(dest, uvf, size);
		break;
	default:
		std::memset(dest, 0, size * sizeof(int64_t));
		break;
	}
}

static inline bool IsIntegral(LogicalTypeId id) {
	return id == LogicalTypeId::TINYINT || id == LogicalTypeId::SMALLINT || id == LogicalTypeId::INTEGER ||
	       id == LogicalTypeId::BIGINT || id == LogicalTypeId::UTINYINT || id == LogicalTypeId::USMALLINT ||
	       id == LogicalTypeId::UINTEGER || id == LogicalTypeId::UBIGINT || id == LogicalTypeId::BOOLEAN;
}

// ============================================================
//  Execute
// ============================================================

static OperatorResultType FnExecute(ExecutionContext &, TableFunctionInput &in, DataChunk &input, DataChunk &) {
	if (input.size() == 0) return OperatorResultType::NEED_MORE_INPUT;

	auto &l = in.local_state->Cast<FnLocalState>();

	if (input.ColumnCount() < 4) {
		throw InvalidInputException("dbweaver expects four columns: SearchEngineID, ClientIP, IsRefresh, ResolutionWidth");
	}

	auto &v_engine = input.data[0];
	auto &v_ip = input.data[1];
	auto &v_refresh = input.data[2];
	auto &v_width = input.data[3];

	const auto t_engine = v_engine.GetType().id();
	const auto t_ip = v_ip.GetType().id();
	const auto t_refresh = v_refresh.GetType().id();
	const auto t_width = v_width.GetType().id();

	if (!IsIntegral(t_engine) || !IsIntegral(t_ip) || !IsIntegral(t_refresh) || !IsIntegral(t_width)) {
		throw InvalidInputException("dbweaver expects integral types for SearchEngineID, ClientIP, IsRefresh, ResolutionWidth");
	}

	UnifiedVectorFormat u_engine, u_ip, u_refresh, u_width;
	v_engine.ToUnifiedFormat(input.size(), u_engine);
	v_ip.ToUnifiedFormat(input.size(), u_ip);
	v_refresh.ToUnifiedFormat(input.size(), u_refresh);
	v_width.ToUnifiedFormat(input.size(), u_width);

	const idx_t size = input.size();
	const bool all_valid = u_engine.validity.AllValid() && u_ip.validity.AllValid() &&
	                       u_refresh.validity.AllValid() && u_width.validity.AllValid();

	alignas(16) int64_t b_engine[STANDARD_VECTOR_SIZE];
	alignas(16) int64_t b_ip[STANDARD_VECTOR_SIZE];
	alignas(16) int64_t b_refresh[STANDARD_VECTOR_SIZE];
	alignas(16) int64_t b_width[STANDARD_VECTOR_SIZE];

	DispatchFillBuffer(b_engine, u_engine, t_engine, size);
	DispatchFillBuffer(b_ip, u_ip, t_ip, size);
	DispatchFillBuffer(b_refresh, u_refresh, t_refresh, size);
	DispatchFillBuffer(b_width, u_width, t_width, size);
	if (all_valid) {
		for (idx_t i = 0; i < size; ++i) {
			int32_t eng = (int32_t)b_engine[i];
			int64_t ip = b_ip[i];
			AggPayload &p = l.FindOrInsertPayload(eng, ip);
			p.c += 1;
			p.sum_isrefresh += b_refresh[i];
			p.sum_resolutionwidth += b_width[i];
		}
	} else {
		for (idx_t i = 0; i < size; ++i) {
			if (!u_engine.validity.RowIsValid(u_engine.sel->get_index(i)) ||
			    !u_ip.validity.RowIsValid(u_ip.sel->get_index(i)) ||
			    !u_refresh.validity.RowIsValid(u_refresh.sel->get_index(i)) ||
			    !u_width.validity.RowIsValid(u_width.sel->get_index(i))) {
				continue;
			}
			int32_t eng = (int32_t)b_engine[i];
			int64_t ip = b_ip[i];
			AggPayload &p = l.FindOrInsertPayload(eng, ip);
			p.c += 1;
			p.sum_isrefresh += b_refresh[i];
			p.sum_resolutionwidth += b_width[i];
		}
	}


	return OperatorResultType::NEED_MORE_INPUT;
}

// ============================================================
//  Merge / Finalize
// ============================================================

static inline void MergeLocalIntoGlobal(FnLocalState &local, FnGlobalState &g) {
	for (auto const &kv : local.map) {
		const HashedKey &k = kv.first;
		const AggPayload &lp = kv.second;
		if (lp.c == 0) continue;

		AggPayload &gp = g.map[k];
		gp.c += lp.c;
		gp.sum_isrefresh += lp.sum_isrefresh;
		gp.sum_resolutionwidth += lp.sum_resolutionwidth;
	}
}

struct TopRow {
	int64_t c;
	int32_t search_engine_id;
	int64_t client_ip;
	int64_t sum_isrefresh;
	int64_t sum_resolutionwidth;
};

struct TopRowMinCmp {
	bool operator()(const TopRow &a, const TopRow &b) const { return a.c > b.c; }
};

static void EmitTop10(absl::flat_hash_map<HashedKey, AggPayload, HashedKeyHash> &map_ref, DataChunk &out) {
	std::priority_queue<TopRow, std::vector<TopRow>, TopRowMinCmp> pq;

	for (auto const &kv : map_ref) {
		const HashedKey &k = kv.first;
		const AggPayload &p = kv.second;
		if (p.c <= 0) continue;

		TopRow row;
		row.c = p.c;
		row.search_engine_id = k.search_engine_id;
		row.client_ip = k.client_ip;
		row.sum_isrefresh = p.sum_isrefresh;
		row.sum_resolutionwidth = p.sum_resolutionwidth;

		if (pq.size() < 10) {
			pq.push(row);
		} else if (row.c > pq.top().c) {
			pq.pop();
			pq.push(row);
		}
	}

	std::vector<TopRow> top;
	top.reserve(pq.size());
	while (!pq.empty()) {
		top.push_back(pq.top());
		pq.pop();
	}

	std::sort(top.begin(), top.end(), [](const TopRow &a, const TopRow &b) {
		if (a.c != b.c) return a.c > b.c;
		if (a.search_engine_id != b.search_engine_id) return a.search_engine_id < b.search_engine_id;
		return a.client_ip < b.client_ip;
	});

	int32_t *out_eng = FlatVector::GetData<int32_t>(out.data[0]);
	int64_t *out_ip = FlatVector::GetData<int64_t>(out.data[1]);
	int64_t *out_c = FlatVector::GetData<int64_t>(out.data[2]);
	int64_t *out_sumref = FlatVector::GetData<int64_t>(out.data[3]);
	double *out_avgw = FlatVector::GetData<double>(out.data[4]);

	idx_t out_idx = 0;
	for (const auto &r : top) {
		out_eng[out_idx] = r.search_engine_id;
		out_ip[out_idx] = r.client_ip;
		out_c[out_idx] = r.c;
		out_sumref[out_idx] = r.sum_isrefresh;
		out_avgw[out_idx] = (r.c > 0) ? (double)r.sum_resolutionwidth / (double)r.c : 0.0;
		++out_idx;
	}
	out.SetCardinality(out_idx);
}

static OperatorFinalizeResultType FnFinalize(ExecutionContext &, TableFunctionInput &in, DataChunk &out) {
	auto &g = in.global_state->Cast<FnGlobalState>();
	auto &l = in.local_state->Cast<FnLocalState>();

	const auto active = g.active_local_states.load(std::memory_order_relaxed);
	if (active == 1) {
		EmitTop10(l.map, out);
		return OperatorFinalizeResultType::FINISHED;
	}

	if (!l.merged) {
		uint8_t expected = 0;
		if (g.adopt_stage.compare_exchange_strong(expected, 1, std::memory_order_acq_rel)) {
			std::lock_guard<std::mutex> guard(g.lock);
			if (g.map.empty()) g.map = std::move(l.map);
			else MergeLocalIntoGlobal(l, g);
			g.adopt_stage.store(2, std::memory_order_release);
			l.merged = true;
			g.merged_local_states.fetch_add(1, std::memory_order_relaxed);
		} else {
			while (g.adopt_stage.load(std::memory_order_acquire) == 1) std::this_thread::yield();
			std::lock_guard<std::mutex> guard(g.lock);
			MergeLocalIntoGlobal(l, g);
			l.merged = true;
			g.merged_local_states.fetch_add(1, std::memory_order_relaxed);
		}
	}

	if (g.merged_local_states.load(std::memory_order_relaxed) == active) {
		EmitTop10(g.map, out);
		return OperatorFinalizeResultType::FINISHED;
	}

	out.SetCardinality(0);
	return OperatorFinalizeResultType::FINISHED;
}

// ============================================================
//  Bind / Init
// ============================================================

static unique_ptr<GlobalTableFunctionState> FnInit(ClientContext &, TableFunctionInitInput &) {
	return unique_ptr<GlobalTableFunctionState>(make_uniq<FnGlobalState>());
}

static unique_ptr<LocalTableFunctionState> FnInitLocal(ExecutionContext &, TableFunctionInitInput &,
                                                      GlobalTableFunctionState *global_state) {
	auto &g = global_state->Cast<FnGlobalState>();
	g.active_local_states.fetch_add(1, std::memory_order_relaxed);
	return unique_ptr<LocalTableFunctionState>(make_uniq<FnLocalState>());
}

static unique_ptr<FunctionData> FnBind(ClientContext &, TableFunctionBindInput &,
                                       vector<LogicalType> &return_types,
                                       vector<string> &names) {
	return_types.push_back(LogicalType::INTEGER); // SearchEngineID
	return_types.push_back(LogicalType::BIGINT);  // ClientIP
	return_types.push_back(LogicalType::BIGINT);  // c
	return_types.push_back(LogicalType::BIGINT);  // sum_isrefresh
	return_types.push_back(LogicalType::DOUBLE);  // avg_resolutionwidth

	names.push_back("SearchEngineID");
	names.push_back("ClientIP");
	names.push_back("c");
	names.push_back("sum_isrefresh");
	names.push_back("avg_resolutionwidth");

	return make_uniq<FnBindData>();
}

static void LoadInternal(ExtensionLoader &loader) {
	TableFunction f("dbweaver", {LogicalType::TABLE}, nullptr, FnBind, FnInit, FnInitLocal);
	f.in_out_function = FnExecute;
	f.in_out_function_final = FnFinalize;
	loader.RegisterFunction(f);
}

void DbweaverExtension::Load(ExtensionLoader &loader) { LoadInternal(loader); }
std::string DbweaverExtension::Name() { return "dbweaver"; }
std::string DbweaverExtension::Version() const { return DuckDB::LibraryVersion(); }

} // namespace duckdb

extern "C" {
DUCKDB_CPP_EXTENSION_ENTRY(dbweaver, loader) {
	duckdb::LoadInternal(loader);
}
}
