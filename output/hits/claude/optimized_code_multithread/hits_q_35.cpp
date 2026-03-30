/*
query_template: SELECT 1, URL, COUNT(*) AS c
                FROM hits
                GROUP BY 1, URL
                ORDER BY c DESC
                LIMIT 10;

split_template: select * from dbweaver((SELECT URL FROM hits));

query_example: SELECT 1, URL, COUNT(*) AS c
               FROM hits
               GROUP BY 1, URL
               ORDER BY c DESC
               LIMIT 10;

split_query: select * from dbweaver((SELECT URL FROM hits));
*/

#define DUCKDB_EXTENSION_MAIN

#include "dbweaver_extension.hpp"
#include "duckdb.hpp"
#include "duckdb/common/types/string_heap.hpp"
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
#include <immintrin.h>

namespace duckdb {

// ============================================================
//  Key: string_t(URL) + precomputed hash (content-based equality)
// ============================================================

struct HashedStringKey {
	string_t s;
	size_t hash;

	bool operator==(const HashedStringKey &o) const noexcept {
	if (hash != o.hash) return false;
		const auto a_len = s.GetSize();
		const auto b_len = o.s.GetSize();
		if (a_len != b_len) return false;
		if (a_len == 0) return true;
		return std::memcmp(s.GetDataUnsafe(), o.s.GetDataUnsafe(), a_len) == 0;
	}
};

struct HashedStringKeyHash {
	size_t operator()(const HashedStringKey &k) const noexcept { return k.hash; }
};

static inline size_t HashURL(const string_t &s) {
	return duckdb::Hash(s);
}

// ============================================================
//  Bind data
// ============================================================

struct FnBindData : public FunctionData {
	unique_ptr<FunctionData> Copy() const override { return make_uniq<FnBindData>(); }
	bool Equals(const FunctionData &) const override { return true; }
};

// ============================================================
//  Finalize Helpers - TopRow
// ============================================================

struct TopRow {
	int64_t c;
	string_t url;
	size_t hash;
};
struct TopRowMinCmp {
	bool operator()(const TopRow &a, const TopRow &b) const { return a.c > b.c; } // min-heap by c
};

static void UpdatePQ(std::priority_queue<TopRow, std::vector<TopRow>, TopRowMinCmp> &pq, const HashedStringKey &k, int64_t cnt) {
	if (cnt <= 0) return;
	TopRow row{cnt, k.s, k.hash};
	if (pq.size() < 10) {
		pq.push(row);
	} else if (row.c > pq.top().c) {
		pq.pop();
		pq.push(row);
	}
}

static void UpdatePQ(std::priority_queue<TopRow, std::vector<TopRow>, TopRowMinCmp> &pq, const TopRow &row) {
	if (row.c <= 0) return;
	if (pq.size() < 10) {
		pq.push(row);
	} else if (row.c > pq.top().c) {
		pq.pop();
		pq.push(row);
	}
}

// ============================================================
//  Global state (Sharded)
// ============================================================
struct Shard {
	std::mutex lock;
	absl::flat_hash_map<HashedStringKey, int64_t, HashedStringKeyHash> map;
};


struct FnGlobalState : public GlobalTableFunctionState {
	static constexpr idx_t NUM_SHARDS = 256;
	Shard shards[NUM_SHARDS];

	std::atomic<idx_t> active_local_states{0};
	std::atomic<idx_t> merged_local_states{0};

	// Parallel top-K extraction state
	std::atomic<idx_t> shard_idx_counter{0};
	std::mutex global_pq_lock;
	std::priority_queue<TopRow, std::vector<TopRow>, TopRowMinCmp> global_pq;
	std::atomic<idx_t> extraction_done_count{0};
	std::atomic<bool> all_done{false};

	idx_t MaxThreads() const override { return std::numeric_limits<idx_t>::max(); }
};


// ============================================================
//  Local state
// ============================================================

struct FnLocalState : public LocalTableFunctionState {
	bool merged = false;

	StringHeap heap;
	absl::flat_hash_map<HashedStringKey, int64_t, HashedStringKeyHash> map;

	inline void AddOne(const HashedStringKey &probe) {
		auto it = map.find(probe);
		if (it != map.end()) {
			it->second += 1;
			return;
		}
		auto copied = heap.AddString(probe.s);
		map.emplace(HashedStringKey{copied, probe.hash}, 1);
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
                                       vector<LogicalType> &return_types,
                                       vector<string> &names) {
	return_types.push_back(LogicalType::INTEGER); // 1
	return_types.push_back(LogicalType::VARCHAR); // URL
	return_types.push_back(LogicalType::BIGINT);  // c

	names.push_back("1");
	names.push_back("URL");
	names.push_back("c");

	return make_uniq<FnBindData>();
}

// ============================================================
//  Execute
// ============================================================

static OperatorResultType FnExecute(ExecutionContext &, TableFunctionInput &in, DataChunk &input, DataChunk &) {
	if (input.size() == 0) return OperatorResultType::NEED_MORE_INPUT;

	auto &l = in.local_state->Cast<FnLocalState>();

	if (input.ColumnCount() < 1) {
		throw InvalidInputException("dbweaver expects one column: URL");
	}

	auto &v_url = input.data[0];
	if (v_url.GetType().id() != LogicalTypeId::VARCHAR) {
		throw InvalidInputException("dbweaver expects URL as VARCHAR");
	}

	UnifiedVectorFormat u_url;
	v_url.ToUnifiedFormat(input.size(), u_url);

	auto &validity = u_url.validity;
	auto *data = (string_t *)u_url.data;

	constexpr idx_t BATCH = 16;
	HashedStringKey keys[BATCH];
	bool is_valid[BATCH];

	for (idx_t rr = 0; rr < input.size(); rr += BATCH) {
		const idx_t n = std::min(BATCH, input.size() - rr);

		// Pass 1: Compute hashes and prefetch target bucket addresses
		for (idx_t k = 0; k < n; ++k) {
			const idx_t ridx = rr + k;
			const idx_t idx = u_url.sel->get_index(ridx);
			if (validity.RowIsValid(idx)) {
				is_valid[k] = true;
				keys[k].s = data[idx];
				keys[k].hash = HashURL(keys[k].s);

				// Software prefetch for absl::flat_hash_map bucket
				l.map.prefetch(keys[k]);

				// Prefetch string data for memcmp equality checks
				if (keys[k].s.GetSize() > 0) {
					_mm_prefetch(keys[k].s.GetDataUnsafe(), _MM_HINT_T0);
				}
			} else {
				is_valid[k] = false;
			}
		}

		// Pass 2: Perform map lookups and insertions
		for (idx_t k = 0; k < n; ++k) {
			if (is_valid[k]) {
				l.AddOne(keys[k]);
			}
		}
	}

	return OperatorResultType::NEED_MORE_INPUT;
}

// ============================================================
//  Finalize Helpers
// ============================================================

static void FinalSortAndEmit(DataChunk &out, std::priority_queue<TopRow, std::vector<TopRow>, TopRowMinCmp> &pq) {
	std::vector<TopRow> top;
	top.reserve(pq.size());
	while (!pq.empty()) {
		top.push_back(pq.top());
		pq.pop();
	}

	std::sort(top.begin(), top.end(), [](const TopRow &a, const TopRow &b) {
		if (a.c != b.c) return a.c > b.c;
		if (a.hash != b.hash) return a.hash < b.hash;
		const auto al = a.url.GetSize();
		const auto bl = b.url.GetSize();
		const auto ml = std::min(al, bl);
		int cmp = 0;
		if (ml > 0) cmp = std::memcmp(a.url.GetDataUnsafe(), b.url.GetDataUnsafe(), ml);
		if (cmp != 0) return cmp < 0;
		return al < bl;
	});

	auto *out_one = FlatVector::GetData<int32_t>(out.data[0]);
	auto *out_url = FlatVector::GetData<string_t>(out.data[1]);
	auto *out_c = FlatVector::GetData<int64_t>(out.data[2]);
	idx_t out_idx = 0;
	for (size_t i = 0; i < top.size(); ++i) {
		const TopRow &r = top[i];
		out_one[out_idx] = 1;
		out_url[out_idx] = StringVector::AddString(out.data[1], r.url);
		out_c[out_idx] = r.c;
		++out_idx;
	}

	out.SetCardinality(out_idx);
}
static void MergeLocalIntoGlobalSharded(FnLocalState &l, FnGlobalState &g) {
	if (l.map.empty()) {
		return;
	}
	struct EntryPtr {
		const HashedStringKey *k;
		int64_t v;
	};

	const idx_t num_shards = FnGlobalState::NUM_SHARDS;
	idx_t counts[num_shards];
	std::memset(counts, 0, sizeof(counts));

	for (auto const &kv : l.map) {
		counts[kv.first.hash % num_shards]++;
	}

	idx_t offsets[num_shards + 1];
	offsets[0] = 0;
	for (idx_t i = 0; i < num_shards; i++) {
		offsets[i + 1] = offsets[i] + counts[i];
	}

	std::vector<EntryPtr> entries(l.map.size());
	idx_t pos[num_shards];
	std::memcpy(pos, offsets, sizeof(idx_t) * num_shards);

	for (auto const &kv : l.map) {
		const idx_t shard_idx = kv.first.hash % num_shards;
		entries[pos[shard_idx]++] = {&kv.first, kv.second};
	}

	for (idx_t i = 0; i < num_shards; i++) {
		const idx_t start = offsets[i];
		const idx_t end = offsets[i + 1];
		if (start == end) {
			continue;
		}
		auto &shard = g.shards[i];
		std::lock_guard<std::mutex> guard(shard.lock);
		for (idx_t j = start; j < end; j++) {
			auto &entry = entries[j];
			auto it = shard.map.find(*entry.k);
			if (it != shard.map.end()) {
				it->second += entry.v;
			} else {
				shard.map.emplace(*entry.k, entry.v);
			}
		}
	}
}


static OperatorFinalizeResultType FnFinalize(ExecutionContext &, TableFunctionInput &in, DataChunk &out) {
	auto &g = in.global_state->Cast<FnGlobalState>();
	auto &l = in.local_state->Cast<FnLocalState>();

	if (l.merged) {
		out.SetCardinality(0);
		return OperatorFinalizeResultType::FINISHED;
	}

	const auto active = g.active_local_states.load(std::memory_order_acquire);

	// Fast path if only one thread participated
	if (active == 1) {
		std::priority_queue<TopRow, std::vector<TopRow>, TopRowMinCmp> pq;
		for (auto &kv : l.map) {
			UpdatePQ(pq, kv.first, kv.second);
		}
		FinalSortAndEmit(out, pq);
		l.merged = true;
		return OperatorFinalizeResultType::FINISHED;
	}

	// Parallel sharded merge
	MergeLocalIntoGlobalSharded(l, g);
	l.merged = true;

	g.merged_local_states.fetch_add(1, std::memory_order_release);

	// Barrier: Wait for all local threads to finish merging into global shards
	while (g.merged_local_states.load(std::memory_order_acquire) < active) {
		std::this_thread::yield();
	}

	// Parallel Extraction Phase: All threads collaborate to extract Top-K from the 256 shards
	std::priority_queue<TopRow, std::vector<TopRow>, TopRowMinCmp> local_pq;
	idx_t s_idx;
	while ((s_idx = g.shard_idx_counter.fetch_add(1, std::memory_order_relaxed)) < FnGlobalState::NUM_SHARDS) {
		auto &shard = g.shards[s_idx];
		for (auto &kv : shard.map) {
			UpdatePQ(local_pq, kv.first, kv.second);
		}
	}

	// Merge local Top 10 results into the global priority queue
	{
		std::lock_guard<std::mutex> lock(g.global_pq_lock);
		while (!local_pq.empty()) {
			UpdatePQ(g.global_pq, local_pq.top());
			local_pq.pop();
		}
	}
	const auto finished = g.extraction_done_count.fetch_add(1, std::memory_order_acq_rel) + 1;

	// Only the last thread to finish the parallel extraction outputs the final results
	if (finished == active) {
		FinalSortAndEmit(out, g.global_pq);
		g.all_done.store(true, std::memory_order_release);
	} else {
		out.SetCardinality(0);
		while (!g.all_done.load(std::memory_order_acquire)) {
			std::this_thread::yield();
		}
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
