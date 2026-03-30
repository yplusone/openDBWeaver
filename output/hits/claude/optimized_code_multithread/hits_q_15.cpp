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
#include "duckdb/common/vector_operations/vector_operations.hpp"
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
	hash_t hash;

	bool operator==(const HashedCompositeKey &o) const noexcept {
		if (hash != o.hash) return false;
		if (search_engine_id != o.search_engine_id) return false;
		const auto len = phrase.GetSize();
		if (len != o.phrase.GetSize()) return false;
		// Optimization: Prefix-first comparison using content comparison
		// DuckDB string_t padding ensures 4 bytes are always safe to compare in the prefix region.
		if (std::memcmp(phrase.GetPrefix(), o.phrase.GetPrefix(), 4) != 0) return false;
		if (len <= 4) return true;
		if (phrase.GetData() == o.phrase.GetData()) return true;
		return std::memcmp(phrase.GetData(), o.phrase.GetData(), len) == 0;
	}
};

struct HashedCompositeKeyHash {
	hash_t operator()(const HashedCompositeKey &k) const noexcept { return k.hash; }
};

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

struct FnBindData : public FunctionData {
	unique_ptr<FunctionData> Copy() const override { return make_uniq<FnBindData>(); }
	bool Equals(const FunctionData &) const override { return true; }
};

struct FnGlobalState : public GlobalTableFunctionState {
	struct Partition {
		std::mutex lock;
		unique_ptr<StringHeap> heap;
		CountArena counts;
		absl::flat_hash_map<HashedCompositeKey, int64_t *, HashedCompositeKeyHash> map;
		Partition() : heap(make_uniq<StringHeap>()) {}
	};

	static constexpr idx_t NUM_PARTITIONS = 256;
	Partition partitions[NUM_PARTITIONS];

	std::atomic<idx_t> active_local_states{0};
	std::atomic<idx_t> merged_local_states{0};

	idx_t MaxThreads() const override { return std::numeric_limits<idx_t>::max(); }
};

struct FnLocalState : public LocalTableFunctionState {
	bool merged = false;
	unique_ptr<StringHeap> heap;
	CountArena counts;
	absl::flat_hash_map<HashedCompositeKey, int64_t *, HashedCompositeKeyHash> map;

	inline int64_t *FindOrInsertCountPtr(int32_t search_engine_id, const string_t &s, hash_t hash) {
		if (s.GetSize() == 0 || s.GetData() == nullptr) return nullptr;
		HashedCompositeKey tmp{s, search_engine_id, hash};
		auto it = map.find(tmp);
		if (it != map.end()) return it->second;
		const auto owned = heap->AddString(s);
		int64_t *cnt_ptr = counts.AllocateZeroed();
		map.emplace(HashedCompositeKey{owned, search_engine_id, hash}, cnt_ptr);
		return cnt_ptr;
	}
};

static unique_ptr<GlobalTableFunctionState> FnInit(ClientContext &, TableFunctionInitInput &) {
	return make_uniq<FnGlobalState>();
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

struct ColOrder { idx_t engine_col; idx_t phrase_col; bool ok; };

static inline ColOrder ResolveColOrder(DataChunk &input) {
	if (input.ColumnCount() < 2) return {0, 1, false};
	auto t0 = input.data[0].GetType().id();
	const bool c0_is_varchar = t0 == LogicalTypeId::VARCHAR;
	if (c0_is_varchar) return {1, 0, true};
	return {0, 1, true};
}

template <typename T, bool IS_FLAT>
static void ProcessEngineType(FnLocalState &l, idx_t count, const UnifiedVectorFormat &engine_uvf,
                             const UnifiedVectorFormat &phrase_uvf, const hash_t *hashes_ptr) {
	const T *engine_ptr = (const T *)engine_uvf.data;
	const string_t *phrase_ptr = (const string_t *)phrase_uvf.data;

	if (IS_FLAT) {
		if (engine_uvf.validity.AllValid() && phrase_uvf.validity.AllValid()) {
			for (idx_t i = 0; i < count; ++i) {
				const string_t &s = phrase_ptr[i];
				if (s.GetSize() > 0) {
					int64_t *cnt = l.FindOrInsertCountPtr(static_cast<int32_t>(engine_ptr[i]), s, hashes_ptr[i]);
					if (cnt) ++(*cnt);
				}
			}
		} else {
			for (idx_t i = 0; i < count; ++i) {
				if (engine_uvf.validity.RowIsValid(i) && phrase_uvf.validity.RowIsValid(i)) {
					const string_t &s = phrase_ptr[i];
					if (s.GetSize() > 0) {
						int64_t *cnt = l.FindOrInsertCountPtr(static_cast<int32_t>(engine_ptr[i]), s, hashes_ptr[i]);
						if (cnt) ++(*cnt);
					}
				}
			}
		}
	} else {
		for (idx_t i = 0; i < count; ++i) {
			idx_t er = engine_uvf.sel->get_index(i);
			idx_t pr = phrase_uvf.sel->get_index(i);
			if (engine_uvf.validity.RowIsValid(er) && phrase_uvf.validity.RowIsValid(pr)) {
				const string_t &s = phrase_ptr[pr];
				if (s.GetSize() > 0) {
					int64_t *cnt = l.FindOrInsertCountPtr(static_cast<int32_t>(engine_ptr[er]), s, hashes_ptr[i]);
					if (cnt) ++(*cnt);
				}
			}
		}
	}
}

static OperatorResultType FnExecute(ExecutionContext &, TableFunctionInput &in, DataChunk &input, DataChunk &) {
	if (input.size() == 0) return OperatorResultType::NEED_MORE_INPUT;
	auto &l = in.local_state->Cast<FnLocalState>();
	auto ord = ResolveColOrder(input);
	if (!ord.ok) return OperatorResultType::NEED_MORE_INPUT;

	// Vectorized Hash Pass
	Vector hash_vec(LogicalType(LogicalTypeId::UBIGINT), input.size());
	VectorOperations::Hash(input.data[ord.engine_col], hash_vec, input.size());
	Vector phrase_hash_vec(LogicalType(LogicalTypeId::UBIGINT), input.size());
	VectorOperations::Hash(input.data[ord.phrase_col], phrase_hash_vec, input.size());
	// Use 3-arg CombineHash which updates the first vector
	VectorOperations::CombineHash(hash_vec, phrase_hash_vec, input.size());
	auto *hashes_ptr = FlatVector::GetData<hash_t>(hash_vec);

	UnifiedVectorFormat engine_uvf, phrase_uvf;
	input.data[ord.engine_col].ToUnifiedFormat(input.size(), engine_uvf);
	input.data[ord.phrase_col].ToUnifiedFormat(input.size(), phrase_uvf);
	auto engine_type = input.data[ord.engine_col].GetType().id();

	// Use VectorType::FLAT_VECTOR as a safe proxy for identity selection vectors
	bool is_flat = (input.data[ord.engine_col].GetVectorType() == VectorType::FLAT_VECTOR) &&
	               (input.data[ord.phrase_col].GetVectorType() == VectorType::FLAT_VECTOR);

	if (is_flat) {
		switch (engine_type) {
		case LogicalTypeId::TINYINT: ProcessEngineType<int8_t, true>(l, input.size(), engine_uvf, phrase_uvf, hashes_ptr); break;
		case LogicalTypeId::SMALLINT: ProcessEngineType<int16_t, true>(l, input.size(), engine_uvf, phrase_uvf, hashes_ptr); break;
		case LogicalTypeId::INTEGER: ProcessEngineType<int32_t, true>(l, input.size(), engine_uvf, phrase_uvf, hashes_ptr); break;
		case LogicalTypeId::BIGINT: ProcessEngineType<int64_t, true>(l, input.size(), engine_uvf, phrase_uvf, hashes_ptr); break;
		case LogicalTypeId::UTINYINT: ProcessEngineType<uint8_t, true>(l, input.size(), engine_uvf, phrase_uvf, hashes_ptr); break;
		case LogicalTypeId::USMALLINT: ProcessEngineType<uint16_t, true>(l, input.size(), engine_uvf, phrase_uvf, hashes_ptr); break;
		case LogicalTypeId::UINTEGER: ProcessEngineType<uint32_t, true>(l, input.size(), engine_uvf, phrase_uvf, hashes_ptr); break;
		case LogicalTypeId::UBIGINT: ProcessEngineType<uint64_t, true>(l, input.size(), engine_uvf, phrase_uvf, hashes_ptr); break;
		default: break;
		}
	} else {
		switch (engine_type) {
		case LogicalTypeId::TINYINT: ProcessEngineType<int8_t, false>(l, input.size(), engine_uvf, phrase_uvf, hashes_ptr); break;
		case LogicalTypeId::SMALLINT: ProcessEngineType<int16_t, false>(l, input.size(), engine_uvf, phrase_uvf, hashes_ptr); break;
		case LogicalTypeId::INTEGER: ProcessEngineType<int32_t, false>(l, input.size(), engine_uvf, phrase_uvf, hashes_ptr); break;
		case LogicalTypeId::BIGINT: ProcessEngineType<int64_t, false>(l, input.size(), engine_uvf, phrase_uvf, hashes_ptr); break;
		case LogicalTypeId::UTINYINT: ProcessEngineType<uint8_t, false>(l, input.size(), engine_uvf, phrase_uvf, hashes_ptr); break;
		case LogicalTypeId::USMALLINT: ProcessEngineType<uint16_t, false>(l, input.size(), engine_uvf, phrase_uvf, hashes_ptr); break;
		case LogicalTypeId::UINTEGER: ProcessEngineType<uint32_t, false>(l, input.size(), engine_uvf, phrase_uvf, hashes_ptr); break;
		case LogicalTypeId::UBIGINT: ProcessEngineType<uint64_t, false>(l, input.size(), engine_uvf, phrase_uvf, hashes_ptr); break;
		default: break;
		}
	}

	return OperatorResultType::NEED_MORE_INPUT;
}

static void MergeLocalIntoPartitions(FnLocalState &l, FnGlobalState &g) {
	if (l.map.empty()) return;

	// Radix-style grouping by partition index to minimize lock acquisition overhead
	uint32_t counts[256] = {0};
	for (auto &kv : l.map) {
		if (*kv.second == 0) continue;
		counts[kv.first.hash >> 56]++;
	}

	uint32_t offsets[257];
	offsets[0] = 0;
	for (int i = 0; i < 256; ++i) offsets[i + 1] = offsets[i] + counts[i];

	if (offsets[256] == 0) return;

	struct LocalEntry {
		const HashedCompositeKey *key;
		int64_t count;
	};
	std::vector<LocalEntry> partitioned(offsets[256]);
	uint32_t current_offsets[256];
	memcpy(current_offsets, offsets, 256 * sizeof(uint32_t));

	for (auto &kv : l.map) {
		if (*kv.second == 0) continue;
		uint8_t p_idx = (uint8_t)(kv.first.hash >> 56);
		partitioned[current_offsets[p_idx]++] = {&kv.first, *kv.second};
	}

	for (int i = 0; i < 256; ++i) {
		if (counts[i] == 0) continue;
		auto &p = g.partitions[i];
		std::lock_guard<std::mutex> guard(p.lock);
		for (uint32_t j = offsets[i]; j < offsets[i + 1]; ++j) {
			const auto &le = partitioned[j];
			auto it = p.map.find(*le.key);
			if (it != p.map.end()) {
				*it->second += le.count;
			} else {
				const auto owned = p.heap->AddString(le.key->phrase);
				int64_t *cnt_ptr = p.counts.AllocateZeroed();
				*cnt_ptr = le.count;
				p.map.emplace(HashedCompositeKey{owned, le.key->search_engine_id, le.key->hash}, cnt_ptr);
			}
		}
	}
}

struct TopRow { int64_t c; int32_t search_engine_id; const string_t *phrase; };
struct TopRowMinCmp { bool operator()(const TopRow &a, const TopRow &b) const { return a.c > b.c; } };

static OperatorFinalizeResultType FnFinalize(ExecutionContext &, TableFunctionInput &in, DataChunk &out) {
	auto &g = in.global_state->Cast<FnGlobalState>();
	auto &l = in.local_state->Cast<FnLocalState>();
	const auto active = g.active_local_states.load(std::memory_order_relaxed);

	// Optimization for single-threaded aggregation
	if (active == 1) {
		std::priority_queue<TopRow, std::vector<TopRow>, TopRowMinCmp> pq;
		for (auto &kv : l.map) {
			if (*kv.second <= 0) continue;
			if (pq.size() < 10) pq.push({*kv.second, kv.first.search_engine_id, &kv.first.phrase});
			else if (*kv.second > pq.top().c) { pq.pop(); pq.push({*kv.second, kv.first.search_engine_id, &kv.first.phrase}); }
		}
		std::vector<TopRow> top; while (!pq.empty()) { top.push_back(pq.top()); pq.pop(); }
		std::sort(top.begin(), top.end(), [&](const TopRow &a, const TopRow &b) {
			if (a.c != b.c) return a.c > b.c; if (a.search_engine_id != b.search_engine_id) return a.search_engine_id < b.search_engine_id;
			return a.phrase->GetString() < b.phrase->GetString();
		});
		for (idx_t i = 0; i < top.size(); ++i) {
			FlatVector::GetData<int32_t>(out.data[0])[i] = top[i].search_engine_id;
			FlatVector::GetData<string_t>(out.data[1])[i] = StringVector::AddString(out.data[1], *top[i].phrase);
			FlatVector::GetData<int64_t>(out.data[2])[i] = top[i].c;
		}
		out.SetCardinality(top.size());
		return OperatorFinalizeResultType::FINISHED;
	}

	if (!l.merged) {
		MergeLocalIntoPartitions(l, g);
		l.merged = true;
		// Use acq_rel to ensure previous writes to partition maps are visible to the last thread
		if (g.merged_local_states.fetch_add(1, std::memory_order_acq_rel) + 1 != active) {
			out.SetCardinality(0);
			return OperatorFinalizeResultType::FINISHED;
		}
	}

	// Last thread collects the top results from all partitions
	std::priority_queue<TopRow, std::vector<TopRow>, TopRowMinCmp> pq;
	for (int i = 0; i < 256; ++i) {
		for (auto &kv : g.partitions[i].map) {
			if (*kv.second <= 0) continue;
			if (pq.size() < 10) pq.push({*kv.second, kv.first.search_engine_id, &kv.first.phrase});
			else if (*kv.second > pq.top().c) { pq.pop(); pq.push({*kv.second, kv.first.search_engine_id, &kv.first.phrase}); }
		}
	}

	std::vector<TopRow> top; while (!pq.empty()) { top.push_back(pq.top()); pq.pop(); }
	std::sort(top.begin(), top.end(), [&](const TopRow &a, const TopRow &b) {
		if (a.c != b.c) return a.c > b.c; if (a.search_engine_id != b.search_engine_id) return a.search_engine_id < b.search_engine_id;
		return a.phrase->GetString() < b.phrase->GetString();
	});
	for (idx_t i = 0; i < top.size(); ++i) {
		FlatVector::GetData<int32_t>(out.data[0])[i] = top[i].search_engine_id;
		FlatVector::GetData<string_t>(out.data[1])[i] = StringVector::AddString(out.data[1], *top[i].phrase);
		FlatVector::GetData<int64_t>(out.data[2])[i] = top[i].c;
	}
	out.SetCardinality(top.size());
	return OperatorFinalizeResultType::FINISHED;
}

static void LoadInternal(ExtensionLoader &loader) {
	TableFunction f("dbweaver", {LogicalType::TABLE}, nullptr, FnBind, FnInit, FnInitLocal);
	f.in_out_function = FnExecute; f.in_out_function_final = FnFinalize; loader.RegisterFunction(f);
}
void DbweaverExtension::Load(ExtensionLoader &loader) { LoadInternal(loader); }
std::string DbweaverExtension::Name() { return "dbweaver"; }
std::string DbweaverExtension::Version() const { return DuckDB::LibraryVersion(); }
} // namespace duckdb

extern "C" { DUCKDB_CPP_EXTENSION_ENTRY(dbweaver, loader) { duckdb::LoadInternal(loader); } }
