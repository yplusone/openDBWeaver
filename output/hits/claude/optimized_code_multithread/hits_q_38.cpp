/*
query_template: SELECT Title, COUNT(*) AS PageViews
                FROM hits
                WHERE CounterID = 62
                  AND EventDate >= '2013-07-01' AND EventDate <= '2013-07-31'
                  AND DontCountHits = 0
                  AND IsRefresh = 0
                  AND Title <> ''
                GROUP BY Title
                ORDER BY PageViews DESC
                LIMIT 10;

split_template: select * from dbweaver((SELECT Title
                                       FROM hits
                                       WHERE (CounterID=62)
                                         AND (DontCountHits=0)
                                         AND (IsRefresh=0)
                                         AND (Title!='')));

query_example: SELECT Title, COUNT(*) AS PageViews
               FROM hits
               WHERE CounterID = 62
                 AND EventDate >= '2013-07-01' AND EventDate <= '2013-07-31'
                 AND DontCountHits = 0
                 AND IsRefresh = 0
                 AND Title <> ''
               GROUP BY Title
               ORDER BY PageViews DESC
               LIMIT 10;

split_query: select * from dbweaver((SELECT Title
                                    FROM hits
                                    WHERE (CounterID=62)
                                      AND (DontCountHits=0)
                                      AND (IsRefresh=0)
                                      AND (Title!='')));
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
#include <utility>
#include <vector>
#include <algorithm>

namespace duckdb {

// ============================================================
// Key: string_t(Title) + precomputed hash (content-based equality)
// ============================================================

struct HashedStringKey {
	string_t s;
	size_t hash;

	bool operator==(const HashedStringKey &o) const noexcept {
		if (hash != o.hash) {
			return false;
		}
		const auto a_len = s.GetSize();
		const auto b_len = o.s.GetSize();
		if (a_len != b_len) {
			return false;
		}
		if (a_len == 0) {
			return true;
		}
		return std::memcmp(s.GetDataUnsafe(), o.s.GetDataUnsafe(), a_len) == 0;
	}
};

struct HashedStringKeyHash {
	size_t operator()(const HashedStringKey &k) const noexcept {
		return k.hash;
	}
};

static inline size_t HashTitle(const string_t &s) {
	return duckdb::Hash(s);
}

// ============================================================
// Bind data
// ============================================================

struct FnBindData : public FunctionData {
	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<FnBindData>();
	}
	bool Equals(const FunctionData &) const override {
		return true;
	}
};

// ============================================================
// Global state
// ============================================================
struct FnGlobalShard {
	std::mutex lock;
	absl::flat_hash_map<HashedStringKey, int64_t, HashedStringKeyHash> map;
};

struct FnGlobalState : public GlobalTableFunctionState {
	static constexpr size_t NUM_SHARDS = 64;
	FnGlobalShard shards[NUM_SHARDS];

	std::mutex heap_lock;
	std::vector<unique_ptr<StringHeap>> local_heaps;

	std::atomic<idx_t> active_local_states{0};
	std::atomic<idx_t> merged_local_states{0};
	std::atomic<bool> result_emitted{false};

	idx_t MaxThreads() const override {
		return std::numeric_limits<idx_t>::max();
	}
};

// ============================================================
// Local state
// ============================================================

struct FnLocalState : public LocalTableFunctionState {
	bool merged = false;

	unique_ptr<StringHeap> heap;
	absl::flat_hash_map<HashedStringKey, int64_t, HashedStringKeyHash> map;

	FnLocalState() : heap(make_uniq<StringHeap>()) {}

	inline void Add(const string_t &title, const size_t h, int64_t count = 1) {
		HashedStringKey probe{title, h};
		auto it = map.find(probe);
		if (it != map.end()) {
			it->second += count;
			return;
		}
		if (title.IsInlined()) {
			map.emplace(probe, count);
		} else {
			auto copied = heap->AddString(title);
			map.emplace(HashedStringKey{copied, h}, count);
		}
	}
};

// ============================================================
// Init / Bind
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
	return_types.push_back(LogicalType::VARCHAR); // Title
	return_types.push_back(LogicalType::BIGINT);  // PageViews

	names.push_back("Title");
	names.push_back("PageViews");

	return make_uniq<FnBindData>();
}

// ============================================================
// Execute
// Expects pre-filtered input: only Title column, already constrained
// ============================================================

static OperatorResultType FnExecute(ExecutionContext &, TableFunctionInput &in, DataChunk &input, DataChunk &) {
	if (input.size() == 0) {
		return OperatorResultType::NEED_MORE_INPUT;
	}

	auto &l = in.local_state->Cast<FnLocalState>();

	if (input.ColumnCount() < 1) {
		throw InvalidInputException("dbweaver expects one column: Title");
	}

	auto &v_title = input.data[0];
	if (v_title.GetType().id() != LogicalTypeId::VARCHAR) {
		throw InvalidInputException("dbweaver expects Title as VARCHAR");
	}

	UnifiedVectorFormat u_title;
	v_title.ToUnifiedFormat(input.size(), u_title);

	auto *data = (string_t *)u_title.data;
	auto v_type = v_title.GetVectorType();

	if (v_type == VectorType::DICTIONARY_VECTOR && u_title.sel) {
		auto *sel_ptr = u_title.sel->data();
		absl::flat_hash_map<sel_t, int64_t> counts;
		counts.reserve(input.size());
		if (u_title.validity.AllValid()) {
			for (idx_t i = 0; i < input.size(); i++) {
				counts[sel_ptr[i]]++;
			}
		} else {
			for (idx_t i = 0; i < input.size(); i++) {
				sel_t idx = sel_ptr[i];
				if (u_title.validity.RowIsValid(idx)) {
					counts[idx]++;
				}
			}
		}
		for (auto const &entry : counts) {
			string_t title = data[entry.first];
			l.Add(title, HashTitle(title), entry.second);
		}
	} else if (v_type == VectorType::CONSTANT_VECTOR) {
		if (u_title.validity.RowIsValid(0)) {
			string_t title = data[0];
			l.Add(title, HashTitle(title), (int64_t)input.size());
		}
	} else {
		if (u_title.validity.AllValid()) {
			for (idx_t i = 0; i < input.size(); i++) {
				idx_t idx = u_title.sel ? u_title.sel->get_index(i) : i;
				string_t title = data[idx];
				l.Add(title, HashTitle(title));
			}
		} else {
			for (idx_t i = 0; i < input.size(); i++) {
				idx_t idx = u_title.sel ? u_title.sel->get_index(i) : i;
				if (u_title.validity.RowIsValid(idx)) {
					string_t title = data[idx];
					l.Add(title, HashTitle(title));
				}
			}
		}
	}

	return OperatorResultType::NEED_MORE_INPUT;
}

// ============================================================
// Merge helper
// ============================================================
static void MergeLocalIntoGlobal(FnLocalState &local, FnGlobalState &g) {
	if (local.map.empty()) {
		return;
	}

	// Use a single contiguous buffer and offsets to bucket entries without 64 vector allocations
	using EntryPtr = const std::pair<const HashedStringKey, int64_t> *;
	idx_t counts[FnGlobalState::NUM_SHARDS] = {0};
	for (const auto &kv : local.map) {
		counts[kv.first.hash % FnGlobalState::NUM_SHARDS]++;
	}

	idx_t offsets[FnGlobalState::NUM_SHARDS + 1];
	offsets[0] = 0;
	for (size_t i = 0; i < FnGlobalState::NUM_SHARDS; i++) {
		offsets[i + 1] = offsets[i] + counts[i];
	}

	std::vector<EntryPtr> sorted_entries(local.map.size());
	idx_t current_offsets[FnGlobalState::NUM_SHARDS];
	std::memcpy(current_offsets, offsets, sizeof(idx_t) * FnGlobalState::NUM_SHARDS);

	for (const auto &kv : local.map) {
		size_t shard_idx = kv.first.hash % FnGlobalState::NUM_SHARDS;
		sorted_entries[current_offsets[shard_idx]++] = &kv;
	}

	for (size_t i = 0; i < FnGlobalState::NUM_SHARDS; ++i) {
		idx_t start = offsets[i];
		idx_t end = offsets[i + 1];
		if (start == end) {
			continue;
		}
		auto &shard = g.shards[i];
		std::lock_guard<std::mutex> guard(shard.lock);
		for (idx_t j = start; j < end; j++) {
			EntryPtr ptr = sorted_entries[j];
			auto it = shard.map.find(ptr->first);
			if (it != shard.map.end()) {
				it->second += ptr->second;
			} else {
				shard.map.emplace(ptr->first, ptr->second);
			}
		}
	}


	{
		std::lock_guard<std::mutex> guard(g.heap_lock);
		g.local_heaps.push_back(std::move(local.heap));
	}
}


// ============================================================
// Top10
// ============================================================

struct TopRow {
	int64_t c;
	string_t title;
	size_t hash;
};

struct TopRowMinCmp {
	bool operator()(const TopRow &a, const TopRow &b) const {
		return a.c > b.c; // min-heap by count
	}
};
static void EmitTop10(FnGlobalState &g, DataChunk &out) {
	std::priority_queue<TopRow, std::vector<TopRow>, TopRowMinCmp> pq;

	for (size_t i = 0; i < FnGlobalState::NUM_SHARDS; ++i) {
		auto &shard = g.shards[i];
		for (auto const &kv : shard.map) {
			const HashedStringKey &k = kv.first;
			const int64_t cnt = kv.second;
			if (cnt <= 0) {
				continue;
			}

			TopRow row;
			row.c = cnt;
			row.title = k.s;
			row.hash = k.hash;

			if (pq.size() < 10) {
				pq.push(row);
			} else if (row.c > pq.top().c) {
				pq.pop();
				pq.push(row);
			}
		}
	}

	std::vector<TopRow> top;
	top.reserve(pq.size());
	while (!pq.empty()) {
		top.push_back(pq.top());
		pq.pop();
	}

	std::sort(top.begin(), top.end(), [](const TopRow &a, const TopRow &b) {
		if (a.c != b.c) {
			return a.c > b.c;
		}
		if (a.hash != b.hash) {
			return a.hash < b.hash;
		}
		const idx_t al = a.title.GetSize();
		const idx_t bl = b.title.GetSize();
		const idx_t ml = std::min(al, bl);
		int cmp = 0;
		if (ml > 0) {
			cmp = std::memcmp(a.title.GetDataUnsafe(), b.title.GetDataUnsafe(), ml);
		}
		if (cmp != 0) {
			return cmp < 0;
		}
		return al < bl;
	});

	auto *out_title = FlatVector::GetData<string_t>(out.data[0]);
	auto *out_pv = FlatVector::GetData<int64_t>(out.data[1]);

	idx_t out_idx = 0;
	for (idx_t i = 0; i < top.size(); ++i) {
		out_title[out_idx] = StringVector::AddString(out.data[0], top[i].title);
		out_pv[out_idx] = top[i].c;
		++out_idx;
	}
	out.SetCardinality(out_idx);
}


// ============================================================
// Finalize
// ============================================================
static OperatorFinalizeResultType FnFinalize(ExecutionContext &, TableFunctionInput &in, DataChunk &out) {
	auto &g = in.global_state->Cast<FnGlobalState>();
	auto &l = in.local_state->Cast<FnLocalState>();

	if (!l.merged) {
		MergeLocalIntoGlobal(l, g);
		l.merged = true;
		g.merged_local_states.fetch_add(1, std::memory_order_acq_rel);
	}

	const idx_t merged = g.merged_local_states.load(std::memory_order_acquire);
	const idx_t active = g.active_local_states.load(std::memory_order_acquire);

	if (active > 0 && merged == active) {
		bool expected = false;
		if (g.result_emitted.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
			EmitTop10(g, out);
			return OperatorFinalizeResultType::FINISHED;
		}
	}

	out.SetCardinality(0);
	return OperatorFinalizeResultType::FINISHED;
}

// ============================================================
// Extension load
// ============================================================

static void LoadInternal(ExtensionLoader &loader) {
	TableFunction f("dbweaver", {LogicalType::TABLE}, nullptr, FnBind, FnInit, FnInitLocal);
	f.in_out_function = FnExecute;
	f.in_out_function_final = FnFinalize;
	loader.RegisterFunction(f);
}

void DbweaverExtension::Load(ExtensionLoader &loader) {
	LoadInternal(loader);
}
std::string DbweaverExtension::Name() {
	return "dbweaver";
}
std::string DbweaverExtension::Version() const {
	return DuckDB::LibraryVersion();
}

} // namespace duckdb

extern "C" {

DUCKDB_CPP_EXTENSION_ENTRY(dbweaver, loader) {
	duckdb::LoadInternal(loader);
}

}
