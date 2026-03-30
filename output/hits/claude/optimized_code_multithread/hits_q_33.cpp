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
//  Key: WatchID + ClientIP (Compact 16 bytes)
// ============================================================

struct HashedKey {
	int64_t watch_id;
	int64_t client_ip;

	bool operator==(const HashedKey &o) const noexcept {
		return watch_id == o.watch_id && client_ip == o.client_ip;
	}
};

struct HashedKeyHash {
	size_t operator()(const HashedKey &k) const noexcept {
		const uint64_t kMul = 0x9ddfea08eb382d69ULL;
		uint64_t a = (static_cast<uint64_t>(k.watch_id) ^ static_cast<uint64_t>(k.client_ip)) * kMul;
		a ^= (a >> 47);
		uint64_t b = (static_cast<uint64_t>(k.client_ip) ^ a) * kMul;
		b ^= (b >> 47);
		b *= kMul;
		return static_cast<size_t>(b);
	}
};

// ============================================================
//  Agg payload
// ============================================================

struct AggPayload {
	int64_t c = 0;
	int64_t sum_isrefresh = 0;
	int64_t sum_resolutionwidth = 0;
	uint64_t first_seen_seq = 0;
};

static constexpr idx_t NUM_PARTITIONS = 128;

// ============================================================
//  Local / Global state Forward decls
// ============================================================

struct FnLocalState;

struct FnGlobalState : public GlobalTableFunctionState {
	struct Partition {
		absl::flat_hash_map<HashedKey, AggPayload, HashedKeyHash> map;
	};

	Partition partitions[NUM_PARTITIONS];
	std::vector<FnLocalState*> local_states;
	std::mutex local_states_lock;

	std::atomic<idx_t> active_local_states {0};
	std::atomic<idx_t> threads_entered_finalize {0};
	std::atomic<idx_t> threads_finished_redistribute {0};
	std::atomic<idx_t> next_partition_to_merge {0};
	std::atomic<idx_t> threads_finished_merge {0};
	std::atomic<bool> output_emitted {false};

	idx_t MaxThreads() const override { return std::numeric_limits<idx_t>::max(); }
};

struct FnLocalState : public LocalTableFunctionState {
	bool redistributed = false;
	uint64_t local_seq = 0;
	absl::flat_hash_map<HashedKey, AggPayload, HashedKeyHash> map;
	std::vector<std::pair<HashedKey, AggPayload>> buckets[NUM_PARTITIONS];

	inline AggPayload &FindOrInsertPayload(int64_t watch_id, int64_t client_ip) {
		HashedKey tmp {watch_id, client_ip};
		auto res = map.try_emplace(tmp);
		if (res.second) {
			res.first->second.first_seen_seq = local_seq++;
		}
		return res.first->second;
	}
};

// ============================================================
//  Bind data
// ============================================================

struct FnBindData : public FunctionData {
	unique_ptr<FunctionData> Copy() const override { return make_uniq<FnBindData>(); }
	bool Equals(const FunctionData &) const override { return true; }
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
	auto ls = make_uniq<FnLocalState>();
	{
		std::lock_guard<std::mutex> lock(g.local_states_lock);
		g.local_states.push_back(ls.get());
	}
	return unique_ptr<LocalTableFunctionState>(std::move(ls));
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

static inline int64_t ReadInt64At(const UnifiedVectorFormat &uvf, idx_t ridx, LogicalTypeId tid) {
	const idx_t idx = uvf.sel->get_index(ridx);
	switch (tid) {
	case LogicalTypeId::BOOLEAN: return (int64_t)((bool *)uvf.data)[idx];
	case LogicalTypeId::TINYINT: return (int64_t)((int8_t *)uvf.data)[idx];
	case LogicalTypeId::SMALLINT: return (int64_t)((int16_t *)uvf.data)[idx];
	case LogicalTypeId::INTEGER: return (int64_t)((int32_t *)uvf.data)[idx];
	case LogicalTypeId::BIGINT: return (int64_t)((int64_t *)uvf.data)[idx];
	case LogicalTypeId::UTINYINT: return (int64_t)((uint8_t *)uvf.data)[idx];
	case LogicalTypeId::USMALLINT: return (int64_t)((uint16_t *)uvf.data)[idx];
	case LogicalTypeId::UINTEGER: return (int64_t)((uint32_t *)uvf.data)[idx];
	case LogicalTypeId::UBIGINT: return (int64_t)((uint64_t *)uvf.data)[idx];
	default: return 0;
	}
}

// ============================================================
//  Execute
// ============================================================

static OperatorResultType FnExecute(ExecutionContext &, TableFunctionInput &in, DataChunk &input, DataChunk &) {
	if (input.size() == 0) return OperatorResultType::NEED_MORE_INPUT;
	auto &l = in.local_state->Cast<FnLocalState>();

	UnifiedVectorFormat u_watch, u_ip, u_refresh, u_width;
	input.data[0].ToUnifiedFormat(input.size(), u_watch);
	input.data[1].ToUnifiedFormat(input.size(), u_ip);
	input.data[2].ToUnifiedFormat(input.size(), u_refresh);
	input.data[3].ToUnifiedFormat(input.size(), u_width);

	const auto t_watch = input.data[0].GetType().id();
	const auto t_ip = input.data[1].GetType().id();
	const auto t_refresh = input.data[2].GetType().id();
	const auto t_width = input.data[3].GetType().id();

	const bool all_valid = u_watch.validity.AllValid() && u_ip.validity.AllValid() &&
	                       u_refresh.validity.AllValid() && u_width.validity.AllValid();

	if (all_valid && t_watch == LogicalTypeId::BIGINT && t_ip == LogicalTypeId::INTEGER &&
	    t_refresh == LogicalTypeId::SMALLINT && t_width == LogicalTypeId::SMALLINT) {
		const int64_t *d_watch = (const int64_t *)u_watch.data;
		const int32_t *d_ip = (const int32_t *)u_ip.data;
		const int16_t *d_refresh = (const int16_t *)u_refresh.data;
		const int16_t *d_width = (const int16_t *)u_width.data;

		for (idx_t i = 0; i < input.size(); ++i) {
			int64_t watch_id = d_watch[u_watch.sel->get_index(i)];
			int64_t ip = (int64_t)d_ip[u_ip.sel->get_index(i)];
			int64_t ref = (int64_t)d_refresh[u_refresh.sel->get_index(i)];
			int64_t w = (int64_t)d_width[u_width.sel->get_index(i)];
			AggPayload &p = l.FindOrInsertPayload(watch_id, ip);
			p.c++;
			p.sum_isrefresh += ref;
			p.sum_resolutionwidth += w;
		}
	} else {
		for (idx_t i = 0; i < input.size(); ++i) {
			if (!u_watch.validity.RowIsValid(u_watch.sel->get_index(i)) ||
			    !u_ip.validity.RowIsValid(u_ip.sel->get_index(i)) ||
			    !u_refresh.validity.RowIsValid(u_refresh.sel->get_index(i)) ||
			    !u_width.validity.RowIsValid(u_width.sel->get_index(i))) continue;
			int64_t watch_id = ReadInt64At(u_watch, i, t_watch);
			int64_t ip = ReadInt64At(u_ip, i, t_ip);
			int64_t ref = ReadInt64At(u_refresh, i, t_refresh);
			int64_t w = ReadInt64At(u_width, i, t_width);
			AggPayload &p = l.FindOrInsertPayload(watch_id, ip);
			p.c++;
			p.sum_isrefresh += ref;
			p.sum_resolutionwidth += w;
		}
	}
	return OperatorResultType::NEED_MORE_INPUT;
}

// ============================================================
//  Finalize: Lock-free Partitioned Merge
// ============================================================

struct TopRow {
	int64_t c;
	int64_t watch_id;
	int64_t client_ip;
	int64_t sum_isrefresh;
	int64_t sum_resolutionwidth;
	uint64_t first_seen_seq;
};

struct TopRowMinCmp {
	bool operator()(const TopRow &a, const TopRow &b) const {
		if (a.c != b.c) return a.c > b.c;
		return a.first_seen_seq < b.first_seen_seq;
	}
};

static void ProcessMapEntries(absl::flat_hash_map<HashedKey, AggPayload, HashedKeyHash> &map,
                              std::priority_queue<TopRow, std::vector<TopRow>, TopRowMinCmp> &pq) {
	for (auto &kv : map) {
		const HashedKey &k = kv.first;
		const AggPayload &p = kv.second;
		if (p.c <= 0) continue;
		TopRow row {p.c, k.watch_id, k.client_ip, p.sum_isrefresh, p.sum_resolutionwidth, p.first_seen_seq};
		if (pq.size() < 10) {
			pq.push(row);
		} else if (row.c > pq.top().c || (row.c == pq.top().c && row.first_seen_seq < pq.top().first_seen_seq)) {
			pq.pop();
			pq.push(row);
		}
	}
}

static void EmitFinalData(DataChunk &out, std::priority_queue<TopRow, std::vector<TopRow>, TopRowMinCmp> &pq) {
	std::vector<TopRow> top;
	while (!pq.empty()) {
		top.push_back(pq.top());
		pq.pop();
	}
	std::sort(top.begin(), top.end(), [](const TopRow &a, const TopRow &b) {
		if (a.c != b.c) return a.c > b.c;
		if (a.first_seen_seq != b.first_seen_seq) return a.first_seen_seq < b.first_seen_seq;
		if (a.client_ip != b.client_ip) return a.client_ip < b.client_ip;
		return a.watch_id < b.watch_id;
	});

	idx_t out_idx = 0;
	for (auto &r : top) {
		FlatVector::GetData<int64_t>(out.data[0])[out_idx] = r.client_ip;
		FlatVector::GetData<int64_t>(out.data[1])[out_idx] = r.watch_id;
		FlatVector::GetData<double>(out.data[2])[out_idx] = (double)r.sum_resolutionwidth / (double)r.c;
		FlatVector::GetData<int64_t>(out.data[3])[out_idx] = r.c;
		FlatVector::GetData<int64_t>(out.data[4])[out_idx] = r.sum_isrefresh;
		++out_idx;
	}
	out.SetCardinality(out_idx);
}

static OperatorFinalizeResultType FnFinalize(ExecutionContext &, TableFunctionInput &in, DataChunk &out) {
	auto &g = in.global_state->Cast<FnGlobalState>();
	auto &l = in.local_state->Cast<FnLocalState>();

	const idx_t num_threads = g.active_local_states.load(std::memory_order_acquire);

	// Synchronization Barrier: Wait for all threads to reach Finalize
	g.threads_entered_finalize.fetch_add(1, std::memory_order_release);
	while (g.threads_entered_finalize.load(std::memory_order_acquire) < num_threads) {
		std::this_thread::yield();
	}

	// Pass 0: Optimization for single thread
	if (num_threads == 1) {
		bool expected = false;
		if (g.output_emitted.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
			std::priority_queue<TopRow, std::vector<TopRow>, TopRowMinCmp> pq;
			ProcessMapEntries(l.map, pq);
			EmitFinalData(out, pq);
		} else {
			out.SetCardinality(0);
		}
		return OperatorFinalizeResultType::FINISHED;
	}

	// Pass 1: Local Redistribution into Buckets
	if (!l.redistributed) {
		HashedKeyHash hasher;
		for (auto &kv : l.map) {
			size_t p_idx = hasher(kv.first) % NUM_PARTITIONS;
			l.buckets[p_idx].push_back(kv);
		}
		l.map = {}; // Release map memory
		l.redistributed = true;
		g.threads_finished_redistribute.fetch_add(1, std::memory_order_release);
	}
	while (g.threads_finished_redistribute.load(std::memory_order_acquire) < num_threads) {
		std::this_thread::yield();
	}

	// Pass 2: Partition Ownership Merging
	while (true) {
		idx_t p_idx = g.next_partition_to_merge.fetch_add(1, std::memory_order_relaxed);
		if (p_idx >= NUM_PARTITIONS) break;

		auto &dest_map = g.partitions[p_idx].map;
		for (auto *ls : g.local_states) {
			auto &src_bucket = ls->buckets[p_idx];
			for (auto &kv : src_bucket) {
				auto res = dest_map.try_emplace(kv.first, kv.second);
				if (!res.second) {
					AggPayload &gp = res.first->second;
					const AggPayload &lp = kv.second;
					gp.c += lp.c;
					gp.sum_isrefresh += lp.sum_isrefresh;
					gp.sum_resolutionwidth += lp.sum_resolutionwidth;
					if (lp.first_seen_seq < gp.first_seen_seq) gp.first_seen_seq = lp.first_seen_seq;
				}
			}
			src_bucket = {}; // Release bucket memory
		}
	}

	g.threads_finished_merge.fetch_add(1, std::memory_order_release);
	while (g.threads_finished_merge.load(std::memory_order_acquire) < num_threads) {
		std::this_thread::yield();
	}

	// Pass 3: Collect Top 10 from Global Partitions
	bool expected = false;
	if (g.output_emitted.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
		std::priority_queue<TopRow, std::vector<TopRow>, TopRowMinCmp> pq;
		for (idx_t i = 0; i < NUM_PARTITIONS; ++i) {
			ProcessMapEntries(g.partitions[i].map, pq);
		}
		EmitFinalData(out, pq);
	} else {
		out.SetCardinality(0);
	}

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
