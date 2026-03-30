/*
query_template: SELECT WatchID, ClientIP, COUNT(*) AS c, SUM(IsRefresh) AS sum_isrefresh, AVG(ResolutionWidth) AS avg_resolutionwidth
                FROM hits
                GROUP BY WatchID, ClientIP
                ORDER BY c DESC
                LIMIT 10;

split_template: select * from dbweaver((SELECT WatchID, ClientIP, IsRefresh, ResolutionWidth FROM hits));

query_example: SELECT WatchID, ClientIP, COUNT(*) AS c, SUM(IsRefresh) AS sum_isrefresh, AVG(ResolutionWidth) AS avg_resolutionwidth
               FROM hits
               GROUP BY WatchID, ClientIP
               ORDER BY c DESC
               LIMIT 10;

split_query: select * from dbweaver((SELECT WatchID, ClientIP, IsRefresh, ResolutionWidth FROM hits));
*/

#define DUCKDB_EXTENSION_MAIN

#include "dbweaver_extension.hpp"
#include "duckdb.hpp"
#include "duckdb/common/types/vector.hpp"
#include "duckdb/function/table_function.hpp"

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
//  Key: WatchID + ClientIP + precomputed hash
// ============================================================

struct HashedKey {
	int64_t watch_id;
	int64_t client_ip;
	size_t hash;

	bool operator==(const HashedKey &o) const noexcept {
		return hash == o.hash && watch_id == o.watch_id && client_ip == o.client_ip;
	}
};

struct HashedKeyHash {
	size_t operator()(const HashedKey &k) const noexcept { return k.hash; }
};

static inline size_t HashCompositeKey(int64_t watch_id, int64_t client_ip) {
	return CombineHash(duckdb::Hash(watch_id), duckdb::Hash(client_ip));
}

// ============================================================
//  Agg payload - stored inline in the map
// ============================================================
struct AggPayload {
	int64_t c;
	int64_t sum_isrefresh;
	int64_t sum_resolutionwidth;
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

	std::atomic<idx_t> active_local_states {0};
	std::atomic<idx_t> merged_local_states {0};

	// finalize coordination
	std::atomic<uint8_t> adopt_stage {0};      // 0:not started, 1:adopting, 2:adopt done
	std::atomic<bool> output_emitted {false};  // ensure exactly one emitter

	idx_t MaxThreads() const override { return std::numeric_limits<idx_t>::max(); }
};

// ============================================================
//  Local state
// ============================================================
struct FnLocalState : public LocalTableFunctionState {
	bool merged = false;

	absl::flat_hash_map<HashedKey, AggPayload, HashedKeyHash> map;

	inline AggPayload *FindOrInsertPayload(int64_t watch_id, int64_t client_ip, size_t hash) {
		HashedKey key;
		key.watch_id = watch_id;
		key.client_ip = client_ip;
		key.hash = hash;

		auto it = map.find(key);
		if (it != map.end()) {
			return &it->second;
		}

		AggPayload payload;
		payload.c = 0;
		payload.sum_isrefresh = 0;
		payload.sum_resolutionwidth = 0;

		return &map.emplace(key, payload).first->second;
	}
};


// ============================================================
//  Init / Bind
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
                                       vector<LogicalType> &return_types, vector<string> &names) {
	return_types.push_back(LogicalType::BIGINT); // ClientIP
	return_types.push_back(LogicalType::BIGINT); // WatchID
	return_types.push_back(LogicalType::DOUBLE); // avg_resolutionwidth
	return_types.push_back(LogicalType::BIGINT); // c
	return_types.push_back(LogicalType::BIGINT); // sum_isrefresh

	names.push_back("ClientIP");
	names.push_back("WatchID");
	names.push_back("avg_resolutionwidth");
	names.push_back("c");
	names.push_back("sum_isrefresh");

	return make_uniq<FnBindData>();
}

static inline bool IsIntegral(LogicalTypeId id) {
	return id == LogicalTypeId::TINYINT || id == LogicalTypeId::SMALLINT || id == LogicalTypeId::INTEGER ||
	       id == LogicalTypeId::BIGINT || id == LogicalTypeId::UTINYINT || id == LogicalTypeId::USMALLINT ||
	       id == LogicalTypeId::UINTEGER || id == LogicalTypeId::UBIGINT || id == LogicalTypeId::BOOLEAN;
}

static inline int64_t ReadInt64At(const UnifiedVectorFormat &uvf, idx_t ridx, LogicalTypeId tid) {
	const idx_t idx = uvf.sel->get_index(ridx);
	switch (tid) {
	case LogicalTypeId::BOOLEAN:
		return (int64_t)((bool *)uvf.data)[idx];
	case LogicalTypeId::TINYINT:
		return (int64_t)((int8_t *)uvf.data)[idx];
	case LogicalTypeId::SMALLINT:
		return (int64_t)((int16_t *)uvf.data)[idx];
	case LogicalTypeId::INTEGER:
		return (int64_t)((int32_t *)uvf.data)[idx];
	case LogicalTypeId::BIGINT:
		return (int64_t)((int64_t *)uvf.data)[idx];
	case LogicalTypeId::UTINYINT:
		return (int64_t)((uint8_t *)uvf.data)[idx];
	case LogicalTypeId::USMALLINT:
		return (int64_t)((uint16_t *)uvf.data)[idx];
	case LogicalTypeId::UINTEGER:
		return (int64_t)((uint32_t *)uvf.data)[idx];
	case LogicalTypeId::UBIGINT:
		return (int64_t)((uint64_t *)uvf.data)[idx];
	default:
		return 0;
	}
}

// ============================================================
//  Execute
// ============================================================

static OperatorResultType FnExecute(ExecutionContext &, TableFunctionInput &in, DataChunk &input, DataChunk &) {
	if (input.size() == 0) return OperatorResultType::NEED_MORE_INPUT;

	auto &l = in.local_state->Cast<FnLocalState>();

	if (input.ColumnCount() < 4) {
		throw InvalidInputException("dbweaver expects four columns: WatchID, ClientIP, IsRefresh, ResolutionWidth");
	}

	auto &v_watch = input.data[0];
	auto &v_ip = input.data[1];
	auto &v_refresh = input.data[2];
	auto &v_width = input.data[3];

	const auto t_watch = v_watch.GetType().id();
	const auto t_ip = v_ip.GetType().id();
	const auto t_refresh = v_refresh.GetType().id();
	const auto t_width = v_width.GetType().id();

	if (!IsIntegral(t_watch) || !IsIntegral(t_ip) || !IsIntegral(t_refresh) || !IsIntegral(t_width)) {
		throw InvalidInputException("dbweaver expects integral types for WatchID, ClientIP, IsRefresh, ResolutionWidth");
	}

	UnifiedVectorFormat u_watch, u_ip, u_refresh, u_width;
	v_watch.ToUnifiedFormat(input.size(), u_watch);
	v_ip.ToUnifiedFormat(input.size(), u_ip);
	v_refresh.ToUnifiedFormat(input.size(), u_refresh);
	v_width.ToUnifiedFormat(input.size(), u_width);

	const bool all_valid = u_watch.validity.AllValid() && u_ip.validity.AllValid() &&
	                       u_refresh.validity.AllValid() && u_width.validity.AllValid();

	if (all_valid) {
		const bool all_flat = (v_watch.GetVectorType() == VectorType::FLAT_VECTOR &&
		                       v_ip.GetVectorType() == VectorType::FLAT_VECTOR &&
		                       v_refresh.GetVectorType() == VectorType::FLAT_VECTOR &&
		                       v_width.GetVectorType() == VectorType::FLAT_VECTOR);

		const bool standard_types = (t_watch == LogicalTypeId::BIGINT && t_ip == LogicalTypeId::BIGINT &&
		                             t_refresh == LogicalTypeId::BOOLEAN && t_width == LogicalTypeId::SMALLINT);

		if (all_flat && standard_types) {
			const int64_t *__restrict p_watch = (const int64_t *)u_watch.data;
			const int64_t *__restrict p_ip = (const int64_t *)u_ip.data;
			const bool *__restrict p_refresh = (const bool *)u_refresh.data;
			const int16_t *__restrict p_width = (const int16_t *)u_width.data;

			for (idx_t ridx = 0; ridx < input.size(); ++ridx) {
				const int64_t watch_id = p_watch[ridx];
				const int64_t ip = p_ip[ridx];
				const int64_t refresh = (int64_t)p_refresh[ridx];
				const int64_t width = (int64_t)p_width[ridx];
				const size_t hash = HashCompositeKey(watch_id, ip);

				AggPayload *p = l.FindOrInsertPayload(watch_id, ip, hash);
				p->c += 1;
				p->sum_isrefresh += refresh;
				p->sum_resolutionwidth += width;
			}
		} else {
			for (idx_t ridx = 0; ridx < input.size(); ++ridx) {
				const int64_t watch_id = ReadInt64At(u_watch, ridx, t_watch);
				const int64_t ip = ReadInt64At(u_ip, ridx, t_ip);
				const int64_t refresh = ReadInt64At(u_refresh, ridx, t_refresh);
				const int64_t width = ReadInt64At(u_width, ridx, t_width);
				const size_t hash = HashCompositeKey(watch_id, ip);

				AggPayload *p = l.FindOrInsertPayload(watch_id, ip, hash);
				p->c += 1;
				p->sum_isrefresh += refresh;
				p->sum_resolutionwidth += width;
			}
		}
	} else {
		for (idx_t ridx = 0; ridx < input.size(); ++ridx) {
			if (!u_watch.validity.RowIsValid(u_watch.sel->get_index(ridx)) ||
			    !u_ip.validity.RowIsValid(u_ip.sel->get_index(ridx)) ||
			    !u_refresh.validity.RowIsValid(u_refresh.sel->get_index(ridx)) ||
			    !u_width.validity.RowIsValid(u_width.sel->get_index(ridx))) {
				continue;
			}

			const int64_t watch_id = ReadInt64At(u_watch, ridx, t_watch);
			const int64_t ip = ReadInt64At(u_ip, ridx, t_ip);
			const int64_t refresh = ReadInt64At(u_refresh, ridx, t_refresh);
			const int64_t width = ReadInt64At(u_width, ridx, t_width);

			AggPayload *p = l.FindOrInsertPayload(watch_id, ip, HashCompositeKey(watch_id, ip));
			p->c += 1;
			p->sum_isrefresh += refresh;
			p->sum_resolutionwidth += width;
		}
	}

	return OperatorResultType::NEED_MORE_INPUT;
}

// ============================================================
//  Merge helper
// ============================================================

static inline void MergeLocalIntoGlobal(FnLocalState &local, FnGlobalState &g) {
	for (auto &kv : local.map) {
		const HashedKey &k = kv.first;
		const AggPayload &lp = kv.second;
		if (lp.c == 0) continue;

		auto it = g.map.find(k);
		if (it != g.map.end()) {
			AggPayload &gp = it->second;
			gp.c += lp.c;
			gp.sum_isrefresh += lp.sum_isrefresh;
			gp.sum_resolutionwidth += lp.sum_resolutionwidth;
		} else {

			g.map.emplace(k, lp);
		}
	}
}

// ============================================================
//  Finalize: adopt-first finisher + merge rest + Top10 by c
// ============================================================
struct TopRow {
	int64_t c;
	int64_t watch_id;
	int64_t client_ip;
	int64_t sum_isrefresh;
	int64_t sum_resolutionwidth;
};

struct TopRowCmp {
	bool operator()(const TopRow &a, const TopRow &b) const {
		if (a.c != b.c) return a.c > b.c;
		if (a.client_ip != b.client_ip) return a.client_ip < b.client_ip;
		return a.watch_id < b.watch_id;
	}
};

static void EmitTop10(DataChunk &out, absl::flat_hash_map<HashedKey, AggPayload, HashedKeyHash> &map_ref) {
	std::priority_queue<TopRow, std::vector<TopRow>, TopRowCmp> pq;
	TopRowCmp cmp;

	for (auto &kv : map_ref) {
		const HashedKey &k = kv.first;
		const AggPayload &p = kv.second;
		if (p.c <= 0) continue;

		TopRow row;
		row.c = p.c;
		row.watch_id = k.watch_id;
		row.client_ip = k.client_ip;
		row.sum_isrefresh = p.sum_isrefresh;
		row.sum_resolutionwidth = p.sum_resolutionwidth;

		if (pq.size() < 10) {
			pq.push(row);
		} else if (cmp(row, pq.top())) {
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
		if (a.client_ip != b.client_ip) return a.client_ip < b.client_ip;
		return a.watch_id < b.watch_id;
	});


	auto *out_ip = FlatVector::GetData<int64_t>(out.data[0]);
	auto *out_watch = FlatVector::GetData<int64_t>(out.data[1]);
	auto *out_avgw = FlatVector::GetData<double>(out.data[2]);
	auto *out_c = FlatVector::GetData<int64_t>(out.data[3]);
	auto *out_sumref = FlatVector::GetData<int64_t>(out.data[4]);

	idx_t out_idx = 0;
	for (auto &r : top) {
		out_ip[out_idx] = r.client_ip;
		out_watch[out_idx] = r.watch_id;
		out_avgw[out_idx] = (r.c > 0) ? (double)r.sum_resolutionwidth / (double)r.c : 0.0;
		out_c[out_idx] = r.c;
		out_sumref[out_idx] = r.sum_isrefresh;
		++out_idx;
	}
	out.SetCardinality(out_idx);
}

static OperatorFinalizeResultType FnFinalize(ExecutionContext &, TableFunctionInput &in, DataChunk &out) {
	auto &g = in.global_state->Cast<FnGlobalState>();
	auto &l = in.local_state->Cast<FnLocalState>();

	const auto active = g.active_local_states.load(std::memory_order_relaxed);

	if (active == 1) {
		bool expected = false;
		if (!g.output_emitted.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
			out.SetCardinality(0);
			return OperatorFinalizeResultType::FINISHED;
		}
		EmitTop10(out, l.map);
		return OperatorFinalizeResultType::FINISHED;
	}

	if (!l.merged) {
		uint8_t expected = 0;
		if (g.adopt_stage.compare_exchange_strong(expected, (uint8_t)1, std::memory_order_acq_rel)) {
			std::lock_guard<std::mutex> guard(g.lock);
			if (g.map.empty()) {
				g.map = std::move(l.map);
			} else {
				MergeLocalIntoGlobal(l, g);
			}
			g.adopt_stage.store(2, std::memory_order_release);
		} else {
			while (g.adopt_stage.load(std::memory_order_acquire) == 1) {
				std::this_thread::yield();
			}
			std::lock_guard<std::mutex> guard(g.lock);
			MergeLocalIntoGlobal(l, g);
		}
		l.merged = true;
		g.merged_local_states.fetch_add(1, std::memory_order_relaxed);
	}

	const auto merged = g.merged_local_states.load(std::memory_order_relaxed);

	if (!(active > 0 && merged == active)) {
		out.SetCardinality(0);
		return OperatorFinalizeResultType::FINISHED;
	}
	bool expected = false;
	if (!g.output_emitted.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
		out.SetCardinality(0);
		return OperatorFinalizeResultType::FINISHED;
	}

	EmitTop10(out, g.map);
	return OperatorFinalizeResultType::FINISHED;
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
