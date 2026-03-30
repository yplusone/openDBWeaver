/*
query_template: SELECT SearchPhrase, COUNT(DISTINCT UserID) AS u FROM hits WHERE SearchPhrase <> '' GROUP BY SearchPhrase ORDER BY u DESC LIMIT 10;

split_template: select * from dbweaver((SELECT SearchPhrase, UserID FROM hits WHERE (SearchPhrase!='')))
query_example: SELECT SearchPhrase, COUNT(DISTINCT UserID) AS u FROM hits WHERE SearchPhrase <> '' GROUP BY SearchPhrase ORDER BY u DESC LIMIT 10;

split_query: select * from dbweaver((SELECT SearchPhrase, UserID FROM hits WHERE (SearchPhrase!='')))
*/

#define DUCKDB_EXTENSION_MAIN

#include "dbweaver_extension.hpp"
#include "duckdb.hpp"
#include "duckdb/common/types/string_heap.hpp"
#include "duckdb/common/types/vector.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/common/types/hash.hpp"

#include <absl/container/flat_hash_map.h>
#include <absl/container/flat_hash_set.h>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>
#include <algorithm>
#include <memory>
#include <condition_variable>

namespace duckdb {

// ============================================================
//  Types and Helpers
// ============================================================

struct HashedStringT {
	string_t str;
	hash_t hash;

	bool operator==(const HashedStringT &o) const noexcept {
		if (hash != o.hash) return false;
		const auto an = str.GetSize();
		const auto bn = o.str.GetSize();
		if (an != bn) return false;
		if (str.GetData() == o.str.GetData()) return true;
		return !an || std::memcmp(str.GetData(), o.str.GetData(), an) == 0;
	}
};

struct HashedStringTHash {
	size_t operator()(const HashedStringT &k) const noexcept { return static_cast<size_t>(k.hash); }
};



using user_id_t = int64_t;

struct TopRow {
	int64_t u;
	string_t key;
};

static bool LessLex(const string_t &a, const string_t &b) {
	const auto an = a.GetSize();
	const auto bn = b.GetSize();
	const auto mn = std::min(an, bn);
	int cmp = mn ? std::memcmp(a.GetData(), b.GetData(), mn) : 0;
	if (cmp != 0) return cmp < 0;
	return an < bn;
}

struct TopRowMinCmp {
	bool operator()(const TopRow &a, const TopRow &b) const {
		if (a.u != b.u) return a.u > b.u;
		return !LessLex(a.key, b.key);
	}
};
struct UpdateEntry {
	string_t phrase;
	hash_t h_phrase;
	uint64_t pair_key;
};

struct ThreadData {
	unique_ptr<StringHeap> heap;
	std::vector<UpdateEntry> buffers[256];

	static constexpr idx_t PARTITION_BUFFER_SIZE = 64;
	UpdateEntry local_buffers[256][PARTITION_BUFFER_SIZE];
	uint32_t local_buffer_counts[256];

	ThreadData() {
		std::memset(local_buffer_counts, 0, sizeof(local_buffer_counts));
	}
};


// ============================================================
//  State structures
// ============================================================

struct FnBindData : public FunctionData {
	unique_ptr<FunctionData> Copy() const override { return make_uniq<FnBindData>(); }
	bool Equals(const FunctionData &) const override { return true; }
};

struct FnGlobalState : public GlobalTableFunctionState {
	static constexpr idx_t NUM_PARTITIONS = 256;

	std::mutex mu;
	std::condition_variable cv;

	// Phase 1 Synchronization
	idx_t total_threads = 0;
	idx_t threads_at_barrier1 = 0;
	std::vector<unique_ptr<ThreadData>> all_thread_data;
	bool phase1_done = false;

	// Phase 2: Parallel Partition Aggregation
	std::atomic<idx_t> next_partition{0};
	std::vector<TopRow> phase2_top_rows;
	idx_t threads_at_barrier2 = 0;
	bool phase2_done = false;

	// Final Result Extraction
	std::vector<TopRow> final_result;
	std::atomic<bool> result_ready{false};
	std::atomic<uint8_t> output_returned{0};

	idx_t MaxThreads() const override { return std::numeric_limits<idx_t>::max(); }
};
struct FnLocalState : public LocalTableFunctionState {
	unique_ptr<ThreadData> data;
	absl::flat_hash_set<uint64_t> local_seen;

	FnLocalState() {

		data = make_uniq<ThreadData>();
		data->heap = make_uniq<StringHeap>();
	}
};

// ============================================================
//  Table Function Implementation
// ============================================================

static unique_ptr<GlobalTableFunctionState> FnInit(ClientContext &, TableFunctionInitInput &) {
	return make_uniq<FnGlobalState>();
}

static unique_ptr<LocalTableFunctionState> FnInitLocal(ExecutionContext &, TableFunctionInitInput &,
                                                      GlobalTableFunctionState *global_state) {
	auto &g = global_state->Cast<FnGlobalState>();
	{
		std::lock_guard<std::mutex> lock(g.mu);
		g.total_threads++;
	}
	return make_uniq<FnLocalState>();
}

static unique_ptr<FunctionData> FnBind(ClientContext &, TableFunctionBindInput &,
                                       vector<LogicalType> &return_types,
                                       vector<string> &names) {
	return_types.push_back(LogicalType::VARCHAR);
	return_types.push_back(LogicalType::BIGINT);
	names.push_back("SearchPhrase");
	names.push_back("u");
	return make_uniq<FnBindData>();
}

static OperatorResultType FnExecute(ExecutionContext &, TableFunctionInput &in, DataChunk &input, DataChunk &) {
	if (input.size() == 0) return OperatorResultType::NEED_MORE_INPUT;

	auto &l = in.local_state->Cast<FnLocalState>();

	UnifiedVectorFormat uvf0, uvf1;
	input.data[0].ToUnifiedFormat(input.size(), uvf0);
	input.data[1].ToUnifiedFormat(input.size(), uvf1);
	auto *ptr0 = (string_t *)uvf0.data;
	auto *ptr1 = (user_id_t *)uvf1.data;

	for (idx_t i = 0; i < input.size(); ++i) {
		idx_t i0 = uvf0.sel->get_index(i);
		idx_t i1 = uvf1.sel->get_index(i);
		if (!uvf0.validity.RowIsValid(i0) || !uvf1.validity.RowIsValid(i1)) continue;

		string_t phrase = ptr0[i0];
		if (phrase.GetSize() == 0) continue;

		user_id_t uid = ptr1[i1];
		hash_t h_phrase = duckdb::Hash(phrase);
		hash_t h_uid = duckdb::Hash(uid);
		// 64-bit key for (phrase, user) deduplication
		uint64_t key = duckdb::CombineHash(h_phrase, h_uid);
		if (l.local_seen.insert(key).second) {

			idx_t part_idx = h_phrase & (FnGlobalState::NUM_PARTITIONS - 1);
			string_t owned = l.data->heap->AddString(phrase);

			auto &cnt = l.data->local_buffer_counts[part_idx];
			l.data->local_buffers[part_idx][cnt++] = {owned, h_phrase, key};
			if (cnt == ThreadData::PARTITION_BUFFER_SIZE) {
				auto &vec = l.data->buffers[part_idx];
				idx_t old_size = vec.size();
				vec.resize(old_size + ThreadData::PARTITION_BUFFER_SIZE);
				std::memcpy(&vec[old_size], l.data->local_buffers[part_idx], sizeof(UpdateEntry) * ThreadData::PARTITION_BUFFER_SIZE);
				cnt = 0;
			}
		}

	}

	return OperatorResultType::NEED_MORE_INPUT;
}

static OperatorFinalizeResultType FnFinalize(ExecutionContext &, TableFunctionInput &in, DataChunk &out) {
	auto &g = in.global_state->Cast<FnGlobalState>();
	auto &l = in.local_state->Cast<FnLocalState>();
	if (g.total_threads == 0) {
		out.SetCardinality(0);
		return OperatorFinalizeResultType::FINISHED;
	}

	// Flush remaining thread-local buffers to global partition vectors
	for (idx_t p = 0; p < 256; ++p) {
		uint32_t cnt = l.data->local_buffer_counts[p];
		if (cnt > 0) {
			auto &vec = l.data->buffers[p];
			idx_t old_size = vec.size();
			vec.resize(old_size + cnt);
			std::memcpy(&vec[old_size], l.data->local_buffers[p], sizeof(UpdateEntry) * cnt);
			l.data->local_buffer_counts[p] = 0;
		}
	}

	// Phase 1 Barrier: Collect all thread-local data

	{
		std::unique_lock<std::mutex> lock(g.mu);
		g.all_thread_data.push_back(std::move(l.data));
		g.threads_at_barrier1++;
		if (g.threads_at_barrier1 == g.total_threads) {
			g.phase1_done = true;
			g.cv.notify_all();
		} else {
			g.cv.wait(lock, [&] { return g.phase1_done; });
		}
	}
	// Phase 2: Aggregate partitions in parallel
	std::priority_queue<TopRow, std::vector<TopRow>, TopRowMinCmp> pq;
	absl::flat_hash_set<uint64_t> seen;
	absl::flat_hash_map<HashedStringT, int64_t, HashedStringTHash> counts;


	while (true) {
		idx_t p = g.next_partition.fetch_add(1);
		if (p >= FnGlobalState::NUM_PARTITIONS) break;

		seen.clear();
		counts.clear();

		for (const auto &td : g.all_thread_data) {
			for (const auto &entry : td->buffers[p]) {
				if (seen.insert(entry.pair_key).second) {
					auto it = counts.find({entry.phrase, entry.h_phrase});
					if (it == counts.end()) {
						counts.emplace(HashedStringT{entry.phrase, entry.h_phrase}, 1);
					} else {
						it->second++;
					}
				}
			}
		}

		for (auto &kv : counts) {
			TopRow tr {kv.second, kv.first.str};
			if (pq.size() < 10) {
				pq.push(tr);
			} else if (tr.u > pq.top().u || (tr.u == pq.top().u && LessLex(tr.key, pq.top().key))) {
				pq.pop();
				pq.push(tr);
			}
		}
	}

	// Phase 2 Barrier: Collect top 10 from each thread and combine
	{
		std::unique_lock<std::mutex> lock(g.mu);
		while (!pq.empty()) {
			g.phase2_top_rows.push_back(pq.top());
			pq.pop();
		}
		g.threads_at_barrier2++;
		if (g.threads_at_barrier2 == g.total_threads) {
			std::sort(g.phase2_top_rows.begin(), g.phase2_top_rows.end(), [&](const TopRow &a, const TopRow &b) {
				if (a.u != b.u) return a.u > b.u;
				return LessLex(a.key, b.key);
			});
			for (auto &tr : g.phase2_top_rows) {
				if (g.final_result.size() >= 10) break;
				g.final_result.push_back(tr);
			}
			g.phase2_done = true;
			g.result_ready.store(true, std::memory_order_release);
			g.cv.notify_all();
		} else {
			g.cv.wait(lock, [&] { return g.phase2_done; });
		}
	}

	while (!g.result_ready.load(std::memory_order_acquire)) {
		std::this_thread::yield();
	}

	uint8_t expected = 0;
	if (g.output_returned.compare_exchange_strong(expected, 1)) {
		auto *out_keys = FlatVector::GetData<string_t>(out.data[0]);
		auto *out_counts = FlatVector::GetData<int64_t>(out.data[1]);
		idx_t out_idx = 0;
		for (auto &r : g.final_result) {
			out_keys[out_idx] = StringVector::AddString(out.data[0], r.key);
			out_counts[out_idx] = r.u;
			++out_idx;
		}
		out.SetCardinality(out_idx);
	} else {
		out.SetCardinality(0);
	}

	return OperatorFinalizeResultType::FINISHED;
}

// ============================================================
//  Extension Registration
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