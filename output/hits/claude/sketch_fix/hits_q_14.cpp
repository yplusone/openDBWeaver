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

namespace duckdb {

// ============================================================
//  Key wrapper: string_t + precomputed hash
// ============================================================

struct HashedStringT {
	string_t str;
	size_t hash;

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
	size_t operator()(const HashedStringT &k) const noexcept { return k.hash; }
};

static inline size_t HashString(const char *p, idx_t n) {
	return duckdb::Hash(p, n);
}

// ============================================================
//  Distinct Arena: stable address for sets w/o "big reserve"
//  - We store flat_hash_set<user_id_t> in heap blocks.
//  - Map stores pointers -> stable even if hash table rehashes.
// ============================================================

using user_id_t = int64_t;

struct DistinctArena {
	static constexpr idx_t BLOCK_SIZE = 512;

	std::vector<std::unique_ptr<absl::flat_hash_set<user_id_t>[]>> blocks;
	idx_t offset = BLOCK_SIZE;

	inline absl::flat_hash_set<user_id_t>* AllocateSet() {
		if (offset >= BLOCK_SIZE) {
			blocks.emplace_back(std::unique_ptr<absl::flat_hash_set<user_id_t>[]>(new absl::flat_hash_set<user_id_t>[BLOCK_SIZE]));
			offset = 0;
		}
		return &blocks.back()[offset++];
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
//  Global state
// ============================================================

struct FnGlobalState : public GlobalTableFunctionState {
	std::mutex lock;

	unique_ptr<StringHeap> heap;
	DistinctArena arena;

	absl::flat_hash_map<HashedStringT, absl::flat_hash_set<user_id_t>*, HashedStringTHash> map;

	std::atomic<idx_t> active_local_states{0};
	std::atomic<idx_t> merged_local_states{0};

	// 0:not started, 1:adopting, 2:adopt done
	std::atomic<uint8_t> adopt_stage{0};

	idx_t MaxThreads() const override { return std::numeric_limits<idx_t>::max(); }
};

// ============================================================
//  Local state
// ============================================================

struct AggregateDictionaryState {
	string dictionary_id;
	unique_ptr<Vector> dictionary_addresses;
	unsafe_unique_array<bool> found_entry;
	idx_t capacity = 0;
};

struct FnLocalState : public LocalTableFunctionState {
	bool merged = false;

	unique_ptr<StringHeap> heap;
	DistinctArena arena;

	absl::flat_hash_map<HashedStringT, absl::flat_hash_set<user_id_t>*, HashedStringTHash> map;

	std::vector<int64_t> dict_chunk_counts;
	std::vector<uint8_t> dict_chunk_seen;
	std::vector<idx_t> dict_chunk_touched;

	AggregateDictionaryState dict_state;

	void EnsureDictChunkCapacity(idx_t dict_size) {
		if (dict_size <= dict_chunk_counts.size()) return;
		dict_chunk_counts.resize(dict_size, 0);
		dict_chunk_seen.resize(dict_size, 0);
	}

	inline absl::flat_hash_set<user_id_t>* FindOrInsertSetPtr(const string_t &s, size_t hash) {
		HashedStringT tmp{s, hash};
		auto it = map.find(tmp);
		if (it != map.end()) {
			return it->second;
		}
		const auto owned = heap->AddString(s);
		absl::flat_hash_set<user_id_t>* set_ptr = arena.AllocateSet();
		map.emplace(HashedStringT{owned, hash}, set_ptr);
		return set_ptr;
	}

	inline void AddDistinct(const string_t &s, user_id_t uid) {
		const size_t hash = HashString(s.GetData(), s.GetSize());
		FindOrInsertSetPtr(s, hash)->insert(uid);
	}
};

// ============================================================
//  Init / Bind
// ============================================================

static unique_ptr<GlobalTableFunctionState> FnInit(ClientContext &, TableFunctionInitInput &) {
	auto gs = make_uniq<FnGlobalState>();
	gs->heap = make_uniq<StringHeap>();
	return unique_ptr<GlobalTableFunctionState>(std::move(gs));
}

static unique_ptr<LocalTableFunctionState> FnInitLocal(ExecutionContext &, TableFunctionInitInput &,
                                                      GlobalTableFunctionState *global_state) {
	auto &g = global_state->Cast<FnGlobalState>();
	g.active_local_states.fetch_add(1, std::memory_order_relaxed);

	auto ls = make_uniq<FnLocalState>();
	ls->heap = make_uniq<StringHeap>();
	return unique_ptr<LocalTableFunctionState>(std::move(ls));
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

// ============================================================
//  Execute
// ============================================================

static OperatorResultType FnExecute(ExecutionContext &, TableFunctionInput &in, DataChunk &input, DataChunk &) {
	if (input.size() == 0) return OperatorResultType::NEED_MORE_INPUT;

	auto &l = in.local_state->Cast<FnLocalState>();
	// No dictionary optimization for composite-group DISTINCT

	UnifiedVectorFormat uvf0, uvf1;
	input.data[0].ToUnifiedFormat(input.size(), uvf0);
	input.data[1].ToUnifiedFormat(input.size(), uvf1);
	auto *ptr0 = (string_t *)uvf0.data;
	auto *ptr1 = (user_id_t *)uvf1.data;

	auto &valid0 = uvf0.validity;
	auto &valid1 = uvf1.validity;
	const bool all_valid = valid0.AllValid() && valid1.AllValid();

	constexpr idx_t BATCH = 8;
	if (all_valid) {
		for (idx_t rr = 0; rr < input.size(); rr += BATCH) {
			idx_t n = std::min(BATCH, input.size() - rr);

			string_t batch_strings[BATCH];
			size_t batch_hashes[BATCH];
			user_id_t batch_userids[BATCH];

			for (idx_t k = 0; k < n; ++k) {
				idx_t i0 = uvf0.sel->get_index(rr+k);
				idx_t i1 = uvf1.sel->get_index(rr+k);
				batch_strings[k] = ptr0[i0];
				batch_userids[k] = ptr1[i1];
				if (batch_strings[k].GetSize() == 0) batch_hashes[k] = 0; // skip
				else batch_hashes[k] = HashString(batch_strings[k].GetData(), batch_strings[k].GetSize());
			}
			for (idx_t k = 0; k < n; ++k) {
				if (batch_hashes[k] == 0) continue;
#ifdef __GNUC__
				HashedStringT pf_probe{batch_strings[k], batch_hashes[k]};
				const void *bucket_ptr = (const void *)(&pf_probe);
				__builtin_prefetch(bucket_ptr, 0, 1);
#endif
			}
			for (idx_t k = 0; k < n; ++k) {
				if (batch_hashes[k] == 0) continue;
				absl::flat_hash_set<user_id_t> *set_ptr = l.FindOrInsertSetPtr(batch_strings[k], batch_hashes[k]);
				set_ptr->insert(batch_userids[k]);
			}
		}
	} else {
		for (idx_t rr = 0; rr < input.size(); rr += BATCH) {
			idx_t n = std::min(BATCH, input.size() - rr);

			string_t batch_strings[BATCH];
			size_t batch_hashes[BATCH];
			user_id_t batch_userids[BATCH];

			for (idx_t k = 0; k < n; ++k) {
				idx_t i0 = uvf0.sel->get_index(rr+k);
				idx_t i1 = uvf1.sel->get_index(rr+k);
				if (!valid0.RowIsValid(i0) || !valid1.RowIsValid(i1)) { batch_hashes[k] = 0; continue; }
				batch_strings[k] = ptr0[i0];
				batch_userids[k] = ptr1[i1];
				if (batch_strings[k].GetSize() == 0) batch_hashes[k] = 0;
				else batch_hashes[k] = HashString(batch_strings[k].GetData(), batch_strings[k].GetSize());
			}
			for (idx_t k = 0; k < n; ++k) {
				if (batch_hashes[k] == 0) continue;
#ifdef __GNUC__
				HashedStringT pf_probe{batch_strings[k], batch_hashes[k]};
				const void *bucket_ptr = (const void *)(&pf_probe);
				__builtin_prefetch(bucket_ptr, 0, 1);
#endif
			}
			for (idx_t k = 0; k < n; ++k) {
				if (batch_hashes[k] == 0) continue;
				absl::flat_hash_set<user_id_t> *set_ptr = l.FindOrInsertSetPtr(batch_strings[k], batch_hashes[k]);
				set_ptr->insert(batch_userids[k]);
			}
		}
	}
	return OperatorResultType::NEED_MORE_INPUT;
}

// ============================================================
//  Merge helper
// ============================================================

static inline void MergeLocalIntoGlobal(FnLocalState &local, FnGlobalState &g) {
	// g.lock must be held by caller
	for (auto &kv : local.map) {
		const HashedStringT &k_hs = kv.first;
		const absl::flat_hash_set<user_id_t> &local_set = *kv.second;
		if (local_set.empty()) continue;

		auto it = g.map.find(k_hs);
		if (it != g.map.end()) {
			auto *global_set = it->second;
			// add all local distinct userIDs into global set
			global_set->insert(local_set.begin(), local_set.end());
			continue;
		}
		const auto owned = g.heap->AddString(k_hs.str);
		absl::flat_hash_set<user_id_t>* set_ptr = g.arena.AllocateSet();
		*set_ptr = local_set;
		g.map.emplace(HashedStringT{owned, k_hs.hash}, set_ptr);
	}
}

// ============================================================
//  Finalize: adopt-first finisher + merge rest + Top10
// ============================================================

struct TopRow {
	int64_t u;
	const string_t *key;
};

struct TopRowMinCmp {
	bool operator()(const TopRow &a, const TopRow &b) const { return a.u > b.u; }
};

static OperatorFinalizeResultType FnFinalize(ExecutionContext &, TableFunctionInput &in, DataChunk &out) {
	auto &g = in.global_state->Cast<FnGlobalState>();
	auto &l = in.local_state->Cast<FnLocalState>();

	const auto active = g.active_local_states.load(std::memory_order_relaxed);

	if (active == 1) {
		// Single-threaded: direct output from local state with zero lock/synchronization
		std::priority_queue<TopRow, std::vector<TopRow>, TopRowMinCmp> pq;
		for (auto &kv : l.map) {
			const int64_t u = kv.second->size();
			if (u <= 0) continue;
			if (pq.size() < 10) {
				pq.push(TopRow{u, &kv.first.str});
			} else if (u > pq.top().u) {
				pq.pop();
				pq.push(TopRow{u, &kv.first.str});
			}
		}
		std::vector<TopRow> top;
		top.reserve(pq.size());
		while (!pq.empty()) {
			top.push_back(pq.top());
			pq.pop();
		}
		auto less_lex = [](const string_t &a, const string_t &b) {
			const auto an = a.GetSize();
			const auto bn = b.GetSize();
			const auto mn = std::min(an, bn);
			int cmp = mn ? std::memcmp(a.GetData(), b.GetData(), mn) : 0;
			if (cmp != 0) return cmp < 0;
			return an < bn;
		};
		std::sort(top.begin(), top.end(), [&](const TopRow &a, const TopRow &b) {
			if (a.u != b.u) return a.u > b.u;
			return less_lex(*a.key, *b.key);
		});
		auto *out_keys = FlatVector::GetData<string_t>(out.data[0]);
		auto *out_counts = FlatVector::GetData<int64_t>(out.data[1]);
		idx_t out_idx = 0;
		for (auto &r : top) {
			out_keys[out_idx] = StringVector::AddString(out.data[0], *r.key);
			out_counts[out_idx] = r.u;
			++out_idx;
		}
		out.SetCardinality(out_idx);
		return OperatorFinalizeResultType::FINISHED;
	}

	if (!l.merged) {
		uint8_t expected = 0;
		if (g.adopt_stage.compare_exchange_strong(expected, 1, std::memory_order_acq_rel)) {
			std::lock_guard<std::mutex> guard(g.lock);

			if (g.map.empty()) {
				g.heap = std::move(l.heap);
				g.arena = std::move(l.arena);
				g.map = std::move(l.map);
				if (!g.heap) g.heap = make_uniq<StringHeap>();
			} else {
				MergeLocalIntoGlobal(l, g);
			}

			g.adopt_stage.store(2, std::memory_order_release);
			l.merged = true;
			g.merged_local_states.fetch_add(1, std::memory_order_relaxed);
		} else {
			while (g.adopt_stage.load(std::memory_order_acquire) == 1) {
				std::this_thread::yield();
			}
			std::lock_guard<std::mutex> guard(g.lock);
			MergeLocalIntoGlobal(l, g);
			l.merged = true;
			g.merged_local_states.fetch_add(1, std::memory_order_relaxed);
		}
	}

	const auto merged = g.merged_local_states.load(std::memory_order_relaxed);

	if (!(active > 0 && merged == active)) {
		out.SetCardinality(0);
		return OperatorFinalizeResultType::FINISHED;
	}

	std::priority_queue<TopRow, std::vector<TopRow>, TopRowMinCmp> pq;

	for (auto &kv : g.map) {
		const int64_t u = kv.second->size();
		if (u <= 0) continue;

		if (pq.size() < 10) {
			pq.push(TopRow{u, &kv.first.str});
		} else if (u > pq.top().u) {
			pq.pop();
			pq.push(TopRow{u, &kv.first.str});
		}
	}

	std::vector<TopRow> top;
	top.reserve(pq.size());
	while (!pq.empty()) {
		top.push_back(pq.top());
		pq.pop();
	}

	auto less_lex = [](const string_t &a, const string_t &b) {
		const auto an = a.GetSize();
		const auto bn = b.GetSize();
		const auto mn = std::min(an, bn);
		int cmp = mn ? std::memcmp(a.GetData(), b.GetData(), mn) : 0;
		if (cmp != 0) return cmp < 0;
		return an < bn;
	};

	std::sort(top.begin(), top.end(), [&](const TopRow &a, const TopRow &b) {
		if (a.u != b.u) return a.u > b.u;
		return less_lex(*a.key, *b.key);
	});

	auto *out_keys = FlatVector::GetData<string_t>(out.data[0]);
	auto *out_counts = FlatVector::GetData<int64_t>(out.data[1]);

	idx_t out_idx = 0;
	for (auto &r : top) {
		out_keys[out_idx] = StringVector::AddString(out.data[0], *r.key);
		out_counts[out_idx] = r.u;
		++out_idx;
	}
	out.SetCardinality(out_idx);

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