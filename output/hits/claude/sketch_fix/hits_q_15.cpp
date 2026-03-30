/*
query_template: SELECT SearchEngineID, SearchPhrase, COUNT(*) AS c
                FROM hits
                WHERE SearchPhrase <> ''
                GROUP BY SearchEngineID, SearchPhrase
                ORDER BY c DESC
                LIMIT 10;

split_template: select * from dbweaver((SELECT SearchPhrase, SearchEngineID FROM hits WHERE (SearchPhrase!='')));

query_example: SELECT SearchEngineID, SearchPhrase, COUNT(*) AS c
               FROM hits
               WHERE SearchPhrase <> ''
               GROUP BY SearchEngineID, SearchPhrase
               ORDER BY c DESC
               LIMIT 10;

split_query: select * from dbweaver((SELECT SearchPhrase, SearchEngineID FROM hits WHERE (SearchPhrase!='')));
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

namespace duckdb {

// ============================================================
//  Key wrapper: SearchEngineID + string_t + precomputed hash
// ============================================================

struct HashedCompositeKey {
	string_t phrase;
	int32_t search_engine_id;
	size_t hash;

	bool operator==(const HashedCompositeKey &o) const noexcept {
		if (hash != o.hash) return false;
		if (search_engine_id != o.search_engine_id) return false;
		const auto an = phrase.GetSize();
		const auto bn = o.phrase.GetSize();
		if (an != bn) return false;
		if (phrase.GetData() == o.phrase.GetData()) return true;
		return !an || std::memcmp(phrase.GetData(), o.phrase.GetData(), an) == 0;
	}
};

struct HashedCompositeKeyHash {
	size_t operator()(const HashedCompositeKey &k) const noexcept { return k.hash; }
};

static inline size_t HashCompositeKey(int32_t search_engine_id, const char *p, idx_t n) {
	// Combine SearchEngineID and phrase string hash
	// Note: phrase is filtered to non-empty by SQL, but we still guard.
	if (!p || n == 0) return 0;
	return CombineHash(duckdb::Hash(search_engine_id), duckdb::Hash(p, n));
}

// ============================================================
//  CountArena: stable address for counts w/o "big reserve"
// ============================================================

struct CountArena {
	static constexpr idx_t BLOCK_SIZE = 4096;

	std::vector<std::unique_ptr<int64_t[]>> blocks;
	idx_t offset = BLOCK_SIZE;

	inline int64_t *AllocateZeroed() {
		if (offset >= BLOCK_SIZE) {
			blocks.emplace_back(std::unique_ptr<int64_t[]>(new int64_t[BLOCK_SIZE]));
			offset = 0;
		}
		int64_t *p = &blocks.back()[offset++];
		*p = 0;
		return p;
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
	CountArena counts;

	// key -> pointer-to-count (stable)
	absl::flat_hash_map<HashedCompositeKey, int64_t *, HashedCompositeKeyHash> map;

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
	unsafe_unique_array<bool> found_entry;
	idx_t capacity = 0;
};

struct FnLocalState : public LocalTableFunctionState {
	bool merged = false;

	unique_ptr<StringHeap> heap;
	CountArena counts;

	absl::flat_hash_map<HashedCompositeKey, int64_t *, HashedCompositeKeyHash> map;

	AggregateDictionaryState dict_state;

	inline int64_t *FindOrInsertCountPtr(int32_t search_engine_id, const string_t &s, size_t hash) {
		if (s.GetSize() == 0 || s.GetData() == nullptr) {
			return nullptr;
		}
		HashedCompositeKey tmp{s, search_engine_id, hash};
		auto it = map.find(tmp);
		if (it != map.end()) {
			return it->second;
		}
		const auto owned = heap->AddString(s);
		int64_t *cnt_ptr = counts.AllocateZeroed();
		map.emplace(HashedCompositeKey{owned, search_engine_id, hash}, cnt_ptr);
		return cnt_ptr;
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
	return_types.push_back(LogicalType::INTEGER);
	return_types.push_back(LogicalType::VARCHAR);
	return_types.push_back(LogicalType::BIGINT);
	names.push_back("SearchEngineID");
	names.push_back("SearchPhrase");
	names.push_back("c");
	return make_uniq<FnBindData>();
}

// ============================================================
//  Column order resolver (fixes your bug)
// ============================================================

struct ColOrder {
	idx_t engine_col;
	idx_t phrase_col;
	bool ok;
};

static inline ColOrder ResolveColOrder(DataChunk &input) {
	// We expect exactly two columns from the passed subquery:
	// (SearchPhrase VARCHAR, SearchEngineID INTEGER) OR (SearchEngineID INTEGER, SearchPhrase VARCHAR).
	if (input.ColumnCount() < 2) {
		return {0, 1, false};
	}
	auto &c0 = input.data[0];
	auto &c1 = input.data[1];
	auto t0 = c0.GetType().id();
	auto t1 = c1.GetType().id();

	const bool c0_is_varchar = t0 == LogicalTypeId::VARCHAR;
	const bool c1_is_varchar = t1 == LogicalTypeId::VARCHAR;
	const bool c0_is_int32 = t0 == LogicalTypeId::INTEGER;
	const bool c1_is_int32 = t1 == LogicalTypeId::INTEGER;

	// phrase, engine
	if (c0_is_varchar && c1_is_int32) return {1, 0, true};
	// engine, phrase
	if (c0_is_int32 && c1_is_varchar) return {0, 1, true};

	// Fallback: not the expected schema (could be BIGINT for engine etc.)
	// Try to accept any integral for engine.
	const bool c0_is_integral =
	    t0 == LogicalTypeId::TINYINT || t0 == LogicalTypeId::SMALLINT || t0 == LogicalTypeId::INTEGER ||
	    t0 == LogicalTypeId::BIGINT || t0 == LogicalTypeId::UTINYINT || t0 == LogicalTypeId::USMALLINT ||
	    t0 == LogicalTypeId::UINTEGER || t0 == LogicalTypeId::UBIGINT;
	const bool c1_is_integral =
	    t1 == LogicalTypeId::TINYINT || t1 == LogicalTypeId::SMALLINT || t1 == LogicalTypeId::INTEGER ||
	    t1 == LogicalTypeId::BIGINT || t1 == LogicalTypeId::UTINYINT || t1 == LogicalTypeId::USMALLINT ||
	    t1 == LogicalTypeId::UINTEGER || t1 == LogicalTypeId::UBIGINT;

	if (c0_is_varchar && c1_is_integral) return {1, 0, true};
	if (c1_is_varchar && c0_is_integral) return {0, 1, true};

	return {0, 1, false};
}

// ============================================================
//  Dictionary fast path: phrase column must be dictionary vector
// ============================================================
static inline int32_t ReadEngineId(const UnifiedVectorFormat &uvf, idx_t logical_idx, LogicalTypeId type_id) {
	auto phys_idx = uvf.sel->get_index(logical_idx);
	switch (type_id) {
	case LogicalTypeId::TINYINT:
		return static_cast<int32_t>(((int8_t*)uvf.data)[phys_idx]);
	case LogicalTypeId::SMALLINT:
		return static_cast<int32_t>(((int16_t*)uvf.data)[phys_idx]);
	case LogicalTypeId::INTEGER:
		return static_cast<int32_t>(((int32_t*)uvf.data)[phys_idx]);
	case LogicalTypeId::BIGINT:
		return static_cast<int32_t>(((int64_t*)uvf.data)[phys_idx]);
	case LogicalTypeId::UTINYINT:
		return static_cast<int32_t>(((uint8_t*)uvf.data)[phys_idx]);
	case LogicalTypeId::USMALLINT:
		return static_cast<int32_t>(((uint16_t*)uvf.data)[phys_idx]);
	case LogicalTypeId::UINTEGER:
		return static_cast<int32_t>(((uint32_t*)uvf.data)[phys_idx]);
	case LogicalTypeId::UBIGINT:
		return static_cast<int32_t>(((uint64_t*)uvf.data)[phys_idx]);
	default:
		throw InternalException("dbweaver: unexpected engine id logical type in ReadEngineId");
	}
}

static inline bool TryCountDictionaryChunk(FnLocalState &l, DataChunk &input) {

	auto ord = ResolveColOrder(input);
	if (!ord.ok) return false;
	auto &phrase_vec = input.data[ord.phrase_col];
	auto &engine_vec = input.data[ord.engine_col];
	auto engine_type = engine_vec.GetType().id();

	if (phrase_vec.GetVectorType() != VectorType::DICTIONARY_VECTOR) return false;



	auto opt_dict_size = DictionaryVector::DictionarySize(phrase_vec);
	if (!opt_dict_size.IsValid()) return false;

	const idx_t dict_size = opt_dict_size.GetIndex();
	if (dict_size == 0) return true;

	auto &dict_state = l.dict_state;
	auto &dictionary_id = DictionaryVector::DictionaryId(phrase_vec);

	if (dict_state.dictionary_id.empty() || dict_state.dictionary_id != dictionary_id) {
		if (dict_size > dict_state.capacity) {
			dict_state.found_entry = make_unsafe_uniq_array<bool>(dict_size);
			dict_state.capacity = dict_size;
		}
		memset(dict_state.found_entry.get(), 0, dict_size * sizeof(bool));
		dict_state.dictionary_id = dictionary_id;
	} else if (dict_size > dict_state.capacity) {
		throw InternalException(
		    "dbweaver: cached dictionary id matches but dict_size grew (id %s, dict_size %d, cap %d)",
		    dict_state.dictionary_id, dict_size, dict_state.capacity);
	}

	auto &offsets = DictionaryVector::SelVector(phrase_vec);
	auto &dictionary_vector = DictionaryVector::Child(phrase_vec);

	UnifiedVectorFormat dict_uvf;
	dictionary_vector.ToUnifiedFormat(dict_size, dict_uvf);
	auto *dict_data = (string_t *)dict_uvf.data;
	auto &dict_valid = dict_uvf.validity;
	UnifiedVectorFormat engine_uvf;
	engine_vec.ToUnifiedFormat(input.size(), engine_uvf);
	auto &engine_valid = engine_uvf.validity;



	// Aggregate within chunk by (engine_id, dict_idx) using a tiny open-addressing table would be faster,
	// but this is the minimal-correct fix while keeping your overall structure.
	// We implement a small linear-probe map specialized for this chunk to avoid std::unordered_map overhead.

	struct Entry {
		uint64_t key;   // (engine_id << 32) | dict_idx
		int64_t count;
		bool used;
	};

	auto make_key = [](int32_t engine_id, idx_t dict_idx) -> uint64_t {
		return (uint64_t(uint32_t(engine_id)) << 32) | uint64_t(uint32_t(dict_idx));
	};

	// capacity: next pow2 >= 2*rows
	idx_t cap = 1;
	while (cap < input.size() * 2) cap <<= 1;
	std::vector<Entry> table(cap);
	auto mask = cap - 1;

	auto mix64 = [](uint64_t x) -> uint64_t {
		// splitmix64
		x += 0x9e3779b97f4a7c15ULL;
		x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
		x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
		return x ^ (x >> 31);
	};

	auto insert_inc = [&](uint64_t k, int64_t inc) {
		uint64_t h = mix64(k);
		idx_t pos = idx_t(h) & mask;
		while (true) {
			auto &e = table[pos];
			if (!e.used) {
				e.used = true;
				e.key = k;
				e.count = inc;
				return;
			}
			if (e.key == k) {
				e.count += inc;
				return;
			}
			pos = (pos + 1) & mask;
		}
	};
	for (idx_t r = 0; r < input.size(); ++r) {
		const idx_t dict_idx = offsets.get_index(r);

		const idx_t er = engine_uvf.sel->get_index(r);
		if (!engine_valid.RowIsValid(er)) continue;

		int32_t engine_id = ReadEngineId(engine_uvf, r, engine_type);
		insert_inc(make_key(engine_id, dict_idx), 1);
	}



	// flush chunk table into main hash map
	for (auto &e : table) {
		if (!e.used) continue;
		int32_t engine_id = int32_t(uint32_t(e.key >> 32));
		idx_t dict_idx = idx_t(uint32_t(e.key & 0xffffffffULL));
		int64_t inc = e.count;

		const idx_t di = dict_uvf.sel->get_index(dict_idx);
		if (!dict_valid.RowIsValid(di)) continue;

		const string_t s = dict_data[di];
		if (s.GetSize() == 0 || s.GetData() == nullptr) continue;

		const size_t hash = HashCompositeKey(engine_id, s.GetData(), s.GetSize());
		if (hash == 0) continue;

		int64_t *cnt_ptr = l.FindOrInsertCountPtr(engine_id, s, hash);
		if (!cnt_ptr) continue;
		*cnt_ptr += inc;
	}

	return true;
}

// ============================================================
//  Execute
// ============================================================

static OperatorResultType FnExecute(ExecutionContext &, TableFunctionInput &in, DataChunk &input, DataChunk &) {
	if (input.size() == 0) return OperatorResultType::NEED_MORE_INPUT;

	auto &l = in.local_state->Cast<FnLocalState>();

	auto ord = ResolveColOrder(input);
	if (!ord.ok) {
		throw InvalidInputException(
		    "dbweaver expects two columns (SearchPhrase VARCHAR, SearchEngineID INTEGER) in any order");
	}

	if (TryCountDictionaryChunk(l, input)) return OperatorResultType::NEED_MORE_INPUT;
	UnifiedVectorFormat engine_uvf, phrase_uvf;
	auto &engine_vec = input.data[ord.engine_col];
	auto engine_type = engine_vec.GetType().id();
	engine_vec.ToUnifiedFormat(input.size(), engine_uvf);
	input.data[ord.phrase_col].ToUnifiedFormat(input.size(), phrase_uvf);

	auto &engine_valid = engine_uvf.validity;

	auto *phrase_ptr = (string_t *)phrase_uvf.data;
	auto &phrase_valid = phrase_uvf.validity;



	const bool all_valid_phrase = phrase_valid.AllValid();
	const bool all_valid_engine = engine_valid.AllValid();

	constexpr idx_t BATCH = 8;
	for (idx_t rr = 0; rr < input.size(); rr += BATCH) {
		idx_t n = std::min(BATCH, input.size() - rr);

		string_t batch_strings[BATCH];
		size_t batch_hashes[BATCH];
		int32_t batch_engines[BATCH];
		for (idx_t k = 0; k < n; ++k) {
			idx_t er = engine_uvf.sel->get_index(rr + k);
			idx_t pr = phrase_uvf.sel->get_index(rr + k);

			bool phrase_ok = all_valid_phrase || phrase_valid.RowIsValid(pr);
			bool engine_ok = all_valid_engine || engine_valid.RowIsValid(er);

			batch_hashes[k] = 0;
			if (phrase_ok && engine_ok) {
				batch_engines[k] = ReadEngineId(engine_uvf, rr + k, engine_type);
				batch_strings[k] = phrase_ptr[pr];
				if (batch_strings[k].GetSize() > 0 && batch_strings[k].GetData() != nullptr) {
					batch_hashes[k] = HashCompositeKey(batch_engines[k], batch_strings[k].GetData(),
					                                  batch_strings[k].GetSize());
				}
			}
		}



#ifdef __GNUC__
		for (idx_t k = 0; k < n; ++k) {
			if (batch_hashes[k] == 0) continue;
			HashedCompositeKey pf_probe{batch_strings[k], batch_engines[k], batch_hashes[k]};
			__builtin_prefetch((const void *)(&pf_probe), 0, 1);
		}
#endif

		for (idx_t k = 0; k < n; ++k) {
			if (batch_hashes[k] == 0) continue;
			int64_t *cnt = l.FindOrInsertCountPtr(batch_engines[k], batch_strings[k], batch_hashes[k]);
			if (!cnt) continue;
			++(*cnt);
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
		const HashedCompositeKey &k_hs = kv.first;
		const int64_t v = *kv.second;
		if (v == 0) continue;

		auto it = g.map.find(k_hs);
		if (it != g.map.end()) {
			*it->second += v;
			continue;
		}

		const auto owned = g.heap->AddString(k_hs.phrase);
		int64_t *cnt_ptr = g.counts.AllocateZeroed();
		*cnt_ptr = v;
		g.map.emplace(HashedCompositeKey{owned, k_hs.search_engine_id, k_hs.hash}, cnt_ptr);
	}
}

// ============================================================
//  Finalize: adopt-first finisher + merge rest + Top10
// ============================================================

struct TopRow {
	int64_t c;
	int32_t search_engine_id;
	const string_t *phrase;
};

struct TopRowMinCmp {
	bool operator()(const TopRow &a, const TopRow &b) const { return a.c > b.c; } // min-heap
};

static OperatorFinalizeResultType FnFinalize(ExecutionContext &, TableFunctionInput &in, DataChunk &out) {
	auto &g = in.global_state->Cast<FnGlobalState>();
	auto &l = in.local_state->Cast<FnLocalState>();

	const auto active = g.active_local_states.load(std::memory_order_relaxed);

	if (active == 1) {
		// Single-threaded: direct output from local state
		std::priority_queue<TopRow, std::vector<TopRow>, TopRowMinCmp> pq;
		for (auto &kv : l.map) {
			const int64_t c = *kv.second;
			if (c <= 0) continue;
			if (pq.size() < 10) {
				pq.push(TopRow{c, kv.first.search_engine_id, &kv.first.phrase});
			} else if (c > pq.top().c) {
				pq.pop();
				pq.push(TopRow{c, kv.first.search_engine_id, &kv.first.phrase});
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
			if (a.c != b.c) return a.c > b.c;
			if (a.search_engine_id != b.search_engine_id) return a.search_engine_id < b.search_engine_id;
			return less_lex(*a.phrase, *b.phrase);
		});

		auto *out_engines = FlatVector::GetData<int32_t>(out.data[0]);
		auto *out_phrases = FlatVector::GetData<string_t>(out.data[1]);
		auto *out_counts = FlatVector::GetData<int64_t>(out.data[2]);

		idx_t out_idx = 0;
		for (auto &r : top) {
			out_engines[out_idx] = r.search_engine_id;
			out_phrases[out_idx] = StringVector::AddString(out.data[1], *r.phrase);
			out_counts[out_idx] = r.c;
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
				g.counts = std::move(l.counts);
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

	// only one finalize should output
	if (!(active > 0 && merged == active)) {
		out.SetCardinality(0);
		return OperatorFinalizeResultType::FINISHED;
	}

	std::priority_queue<TopRow, std::vector<TopRow>, TopRowMinCmp> pq;
	for (auto &kv : g.map) {
		const int64_t c = *kv.second;
		if (c <= 0) continue;
		if (pq.size() < 10) {
			pq.push(TopRow{c, kv.first.search_engine_id, &kv.first.phrase});
		} else if (c > pq.top().c) {
			pq.pop();
			pq.push(TopRow{c, kv.first.search_engine_id, &kv.first.phrase});
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
		if (a.c != b.c) return a.c > b.c;
		if (a.search_engine_id != b.search_engine_id) return a.search_engine_id < b.search_engine_id;
		return less_lex(*a.phrase, *b.phrase);
	});

	auto *out_engines = FlatVector::GetData<int32_t>(out.data[0]);
	auto *out_phrases = FlatVector::GetData<string_t>(out.data[1]);
	auto *out_counts = FlatVector::GetData<int64_t>(out.data[2]);

	idx_t out_idx = 0;
	for (auto &r : top) {
		out_engines[out_idx] = r.search_engine_id;
		out_phrases[out_idx] = StringVector::AddString(out.data[1], *r.phrase);
		out_counts[out_idx] = r.c;
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