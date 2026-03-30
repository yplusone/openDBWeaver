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
	size_t hash;

	bool operator==(const HashedKey &o) const noexcept {
		return hash == o.hash && search_engine_id == o.search_engine_id && client_ip == o.client_ip;
	}
};

struct HashedKeyHash {
	size_t operator()(const HashedKey &k) const noexcept { return k.hash; }
};

static inline size_t HashCompositeKey(int32_t search_engine_id, int64_t client_ip) {
	return CombineHash(duckdb::Hash(search_engine_id), duckdb::Hash(client_ip));
}

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
//  Global state (Partitioned)
// ============================================================

struct Partition {
	std::mutex lock;
	absl::flat_hash_map<HashedKey, AggPayload, HashedKeyHash> map;
};

struct FnGlobalState : public GlobalTableFunctionState {
	static constexpr idx_t N_PARTITIONS = 64;
	std::unique_ptr<Partition[]> partitions;

	FnGlobalState() {
		partitions = std::unique_ptr<Partition[]>(new Partition[N_PARTITIONS]);
	}

	std::atomic<idx_t> active_local_states{0};
	std::atomic<idx_t> merged_local_states{0};

	idx_t MaxThreads() const override { return std::numeric_limits<idx_t>::max(); }
};

// ============================================================
//  Local state
// ============================================================

struct FnLocalState : public LocalTableFunctionState {
	bool merged = false;

	absl::flat_hash_map<HashedKey, AggPayload, HashedKeyHash> map;

	inline AggPayload *FindOrInsertPayload(int32_t engine_id, int64_t client_ip, size_t hash) {
		HashedKey tmp{engine_id, client_ip, hash};
		return &map[tmp];
	}
};

// ============================================================
//  Init / Bind
// ============================================================

static unique_ptr<GlobalTableFunctionState> FnInit(ClientContext &, TableFunctionInitInput &) {
	return make_uniq<FnGlobalState>();
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

// ============================================================
//  Small helpers for "accept any integral type" reads
// ============================================================

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

	auto &val_engine = u_engine.validity;
	auto &val_ip = u_ip.validity;
	auto &val_refresh = u_refresh.validity;
	auto &val_width = u_width.validity;

	const bool all_valid =
	    val_engine.AllValid() && val_ip.AllValid() && val_refresh.AllValid() && val_width.AllValid();
	constexpr idx_t BATCH = 32;

	if (all_valid) {
		for (idx_t rr = 0; rr < input.size(); rr += BATCH) {
			const idx_t n = std::min(BATCH, input.size() - rr);

			int32_t engines[BATCH];
			int64_t ips[BATCH];
			int64_t refreshes[BATCH];
			int64_t widths[BATCH];
			size_t hashes[BATCH];

			for (idx_t k = 0; k < n; ++k) {
				const idx_t ridx = rr + k;
				engines[k] = (int32_t)ReadInt64At(u_engine, ridx, t_engine);
				ips[k] = ReadInt64At(u_ip, ridx, t_ip);
				refreshes[k] = ReadInt64At(u_refresh, ridx, t_refresh);
				widths[k] = ReadInt64At(u_width, ridx, t_width);
				hashes[k] = HashCompositeKey(engines[k], ips[k]);
			}
#if defined(__GNUC__) || defined(__clang__)
			for (idx_t k = 0; k < n; ++k) {
				l.map.prefetch(HashedKey {engines[k], ips[k], hashes[k]});
			}
#endif


			for (idx_t k = 0; k < n; ++k) {
				AggPayload *p = l.FindOrInsertPayload(engines[k], ips[k], hashes[k]);
				p->c += 1;
				p->sum_isrefresh += refreshes[k];
				p->sum_resolutionwidth += widths[k];
			}
		}
	} else {
		for (idx_t rr = 0; rr < input.size(); rr += BATCH) {
			const idx_t n = std::min(BATCH, input.size() - rr);

			int32_t engines[BATCH];
			int64_t ips[BATCH];
			int64_t refreshes[BATCH];
			int64_t widths[BATCH];
			size_t hashes[BATCH];
			bool valid[BATCH];

			for (idx_t k = 0; k < n; ++k) {
				const idx_t ridx = rr + k;
				valid[k] = val_engine.RowIsValid(u_engine.sel->get_index(ridx)) &&
				           val_ip.RowIsValid(u_ip.sel->get_index(ridx)) &&
				           val_refresh.RowIsValid(u_refresh.sel->get_index(ridx)) &&
				           val_width.RowIsValid(u_width.sel->get_index(ridx));

				if (valid[k]) {
					engines[k] = (int32_t)ReadInt64At(u_engine, ridx, t_engine);
					ips[k] = ReadInt64At(u_ip, ridx, t_ip);
					refreshes[k] = ReadInt64At(u_refresh, ridx, t_refresh);
					widths[k] = ReadInt64At(u_width, ridx, t_width);
					hashes[k] = HashCompositeKey(engines[k], ips[k]);
				}
			}
#if defined(__GNUC__) || defined(__clang__)
			for (idx_t k = 0; k < n; ++k) {
				if (valid[k]) {
					l.map.prefetch(HashedKey {engines[k], ips[k], hashes[k]});
				}
			}
#endif


			for (idx_t k = 0; k < n; ++k) {
				if (valid[k]) {
					AggPayload *p = l.FindOrInsertPayload(engines[k], ips[k], hashes[k]);
					p->c += 1;
					p->sum_isrefresh += refreshes[k];
					p->sum_resolutionwidth += widths[k];
				}
			}
		}
	}


	return OperatorResultType::NEED_MORE_INPUT;
}

// ============================================================
//  Merge helpers
// ============================================================
static inline void MergeLocalIntoGlobalSharded(FnLocalState &local, FnGlobalState &g) {
	if (local.map.empty()) {
		return;
	}

	// Buffer local entries into partition-specific buckets to minimize locking overhead.
	// Instead of O(Groups) lock/unlock operations, we reduce it to O(N_PARTITIONS).
	std::vector<std::pair<HashedKey, AggPayload>> buffers[FnGlobalState::N_PARTITIONS];

	// Heuristic reserve to minimize reallocations
	size_t reserve_size = (local.map.size() / FnGlobalState::N_PARTITIONS) + 128;
	for (idx_t i = 0; i < FnGlobalState::N_PARTITIONS; ++i) {
		buffers[i].reserve(reserve_size);
	}

	for (auto const &kv : local.map) {
		const HashedKey &k = kv.first;
		const AggPayload &lp = kv.second;
		if (lp.c == 0) {
			continue;
		}
		size_t part_idx = (k.hash >> (sizeof(size_t) * 8 - 6)) & (FnGlobalState::N_PARTITIONS - 1);
		buffers[part_idx].emplace_back(k, lp);
	}

	for (idx_t i = 0; i < FnGlobalState::N_PARTITIONS; ++i) {
		if (buffers[i].empty()) {
			continue;
		}
		auto &part = g.partitions[i];
		std::lock_guard<std::mutex> guard(part.lock);
		for (auto const &entry : buffers[i]) {
			auto &gp = part.map[entry.first];
			gp.c += entry.second.c;
			gp.sum_isrefresh += entry.second.sum_isrefresh;
			gp.sum_resolutionwidth += entry.second.sum_resolutionwidth;
		}
	}
}


// ============================================================
//  Finalize: Partitioned Merging + Top10
// ============================================================

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

static void FinalizeAndEmit(std::priority_queue<TopRow, std::vector<TopRow>, TopRowMinCmp> &pq, DataChunk &out) {
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

static void EmitTop10FromLocal(FnLocalState &l, DataChunk &out) {
	std::priority_queue<TopRow, std::vector<TopRow>, TopRowMinCmp> pq;
	for (auto &kv : l.map) {
		const HashedKey &k = kv.first;
		const AggPayload &p = kv.second;
		if (p.c <= 0) continue;
		TopRow row{p.c, k.search_engine_id, k.client_ip, p.sum_isrefresh, p.sum_resolutionwidth};
		if (pq.size() < 10) pq.push(row);
		else if (row.c > pq.top().c) { pq.pop(); pq.push(row); }
	}
	FinalizeAndEmit(pq, out);
}

static void EmitTop10Sharded(FnGlobalState &g, DataChunk &out) {
	std::priority_queue<TopRow, std::vector<TopRow>, TopRowMinCmp> pq;
	for (idx_t i = 0; i < FnGlobalState::N_PARTITIONS; ++i) {
		auto &part = g.partitions[i];
		for (auto &kv : part.map) {
			const HashedKey &k = kv.first;
			const AggPayload &p = kv.second;
			if (p.c <= 0) continue;
			TopRow row{p.c, k.search_engine_id, k.client_ip, p.sum_isrefresh, p.sum_resolutionwidth};
			if (pq.size() < 10) pq.push(row);
			else if (row.c > pq.top().c) { pq.pop(); pq.push(row); }
		}
	}
	FinalizeAndEmit(pq, out);
}

static OperatorFinalizeResultType FnFinalize(ExecutionContext &, TableFunctionInput &in, DataChunk &out) {
	auto &g = in.global_state->Cast<FnGlobalState>();
	auto &l = in.local_state->Cast<FnLocalState>();

	const auto active = g.active_local_states.load(std::memory_order_relaxed);

	if (active == 1) {
		EmitTop10FromLocal(l, out);
		return OperatorFinalizeResultType::FINISHED;
	}

	if (!l.merged) {
		MergeLocalIntoGlobalSharded(l, g);
		l.merged = true;
	}

	if (g.merged_local_states.fetch_add(1, std::memory_order_acq_rel) + 1 == active) {
		EmitTop10Sharded(g, out);
	} else {
		out.SetCardinality(0);
	}

	return OperatorFinalizeResultType::FINISHED;
}

// ============================================================
//  Extension load
// ============================================================

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
