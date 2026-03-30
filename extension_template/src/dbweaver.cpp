/*
query_template: SELECT SearchPhrase, COUNT(*) AS c
                FROM hits
                WHERE SearchPhrase <> ''
                GROUP BY SearchPhrase
                ORDER BY c DESC
                LIMIT 10;

split_template: select * from dbweaver((SELECT SearchPhrase FROM hits WHERE (SearchPhrase!='')));

query_example: SELECT SearchPhrase, COUNT(*) AS c
               FROM hits
               WHERE SearchPhrase <> ''
               GROUP BY SearchPhrase
               ORDER BY c DESC
               LIMIT 10;

split_query: select * from dbweaver((SELECT SearchPhrase FROM hits WHERE (SearchPhrase!='')));
*/

#define DUCKDB_EXTENSION_MAIN

#include "dbweaver_extension.hpp"
#include "duckdb.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/common/types/string_heap.hpp" // StringHeap
#include "duckdb/common/types/vector.hpp"      // DictionaryVector

#include <absl/container/flat_hash_map.h>

#include <atomic>
#include <limits>
#include <mutex>
#include <vector>
#include <array>
#include <queue>
#include <algorithm>
#include <cstdint>
#include <cstring> // memcmp

namespace duckdb {

// ============================================================
//  string_t hash/eq (byte-wise)
//  - Key type is duckdb::string_t (non-owning pointer+len).
//  - We ONLY store string_t that points into a StringHeap we own,
//    so pointers stay valid.
// ============================================================

struct StringTHash {
    size_t operator()(const string_t &s) const noexcept {
        return duckdb::Hash(s.GetData(), s.GetSize());
    }
};

struct StringTEq {
    bool operator()(const string_t &a, const string_t &b) const noexcept {
        const auto an = a.GetSize();
        const auto bn = b.GetSize();
        if (an != bn) return false;
        if (an == 0) return true;
        return std::memcmp(a.GetData(), b.GetData(), an) == 0;
    }
};

// ============================================================
//  Hash wrapper for avoiding redundant hash computation
// ============================================================

struct HashedStringT {
    string_t str;
    size_t hash;

    bool operator==(const HashedStringT &other) const noexcept {
        if (hash != other.hash) return false;
        const auto an = str.GetSize();
        const auto bn = other.str.GetSize();
        if (an != bn) return false;
        return !an || std::memcmp(str.GetData(), other.str.GetData(), an) == 0;
    }
};

struct HashedStringTHash {
    size_t operator()(const HashedStringT &k) const noexcept {
        return k.hash;
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
//  Sharded global aggregation
//  - Each shard owns its own StringHeap, so keys are stable.
//  - Merge does lookup by bytes (no copy on hit).
//  - Copy into global heap ONLY on miss.
// ============================================================
struct Shard {
    std::mutex lock;
    StringHeap heap;
    absl::flat_hash_map<HashedStringT, int64_t, HashedStringTHash> map;
};


// 128 shards: good for future 8 threads. For thread=1 it doesn't hurt much,
// but you can set it smaller if you want.
static constexpr size_t SHARD_COUNT = 128; // must be power of 2
static constexpr size_t RADIX_PARTITION_COUNT = SHARD_COUNT; // must be power of 2

// ============================================================
//  Global/Local state
// ============================================================

struct FnGlobalState : public GlobalTableFunctionState {
    std::array<Shard, SHARD_COUNT> shards;

    std::atomic<idx_t> active_local_states {0};
    std::atomic<idx_t> merged_local_states {0};

    idx_t MaxThreads() const override { return std::numeric_limits<idx_t>::max(); }
};

struct LocalRadixPartition {
    StringHeap heap;
    absl::flat_hash_map<HashedStringT, int64_t, HashedStringTHash> map;
};


struct FnLocalState : public LocalTableFunctionState {
    bool merged = false;
    std::array<LocalRadixPartition, RADIX_PARTITION_COUNT> partitions;

    // Reusable per-chunk dictionary counters (indexed by dict select_id).
    std::vector<int64_t> dict_chunk_counts;
    std::vector<uint8_t> dict_chunk_seen;
    std::vector<idx_t> dict_chunk_touched;
    DataChunk dict_unique_values;

    void EnsureDictChunkCapacity(idx_t dict_size) {
        if (dict_size <= dict_chunk_counts.size()) {
            return;
        }
        dict_chunk_counts.resize(dict_size, 0);
        dict_chunk_seen.resize(dict_size, 0);
    }
    void AddCount(const string_t &s, int64_t inc) {
        const size_t hash = duckdb::Hash(s.GetData(), s.GetSize());
        auto &p = partitions[hash & (RADIX_PARTITION_COUNT - 1)];
        HashedStringT tmp_search { s, hash };
        auto it = p.map.find(tmp_search);
        if (it != p.map.end()) {
            it->second += inc;
        } else {
            const auto owned = p.heap.AddString(s);
            HashedStringT owned_hs { owned, hash };
            p.map.emplace(std::move(owned_hs), inc);
        }
    }

};

// ============================================================
//  Init / Bind
// ============================================================

static unique_ptr<GlobalTableFunctionState> FnInit(ClientContext &, TableFunctionInitInput &) {
    auto gs = make_uniq<FnGlobalState>();

    // Reserve shard maps modestly; tune if you know your distinct.
    // For distinct ~6M overall, avg per shard ~46k.
    for (size_t i = 0; i < SHARD_COUNT; ++i) {
        gs->shards[i].map.reserve(1 << 16); // 65536
    }
    return unique_ptr<GlobalTableFunctionState>(std::move(gs));
}

static unique_ptr<LocalTableFunctionState> FnInitLocal(ExecutionContext &, TableFunctionInitInput &,
                                                      GlobalTableFunctionState *global_state) {
    auto &g = global_state->Cast<FnGlobalState>();
    g.active_local_states.fetch_add(1, std::memory_order_relaxed);

    auto ls = make_uniq<FnLocalState>();

    // IMPORTANT:
    // If distinct is huge (e.g., 6M) and threads=1, you should reserve millions here
    // to avoid rehash storms (memory permitting). Example:
    //   ls->map.reserve(7'000'000);
    //
    // For generality, keep a moderate default.
    constexpr idx_t total_reserve = 1 << 18; // 262k
    const idx_t per_partition_reserve =
        std::max<idx_t>(idx_t(1), total_reserve / idx_t(RADIX_PARTITION_COUNT));
    for (size_t p = 0; p < RADIX_PARTITION_COUNT; ++p) {
        ls->partitions[p].map.reserve(per_partition_reserve);
    }

    return unique_ptr<LocalTableFunctionState>(std::move(ls));
}

static unique_ptr<FunctionData> FnBind(ClientContext &, TableFunctionBindInput &,
                                       vector<LogicalType> &return_types,
                                       vector<string> &names) {
    return_types.push_back(LogicalType::VARCHAR);
    return_types.push_back(LogicalType::BIGINT);
    names.push_back("SearchPhrase");
    names.push_back("c");
    return make_uniq<FnBindData>();
}

// ============================================================
//  Dictionary fast path
//  - First pass: classify rows by dict select_id and count per dict entry.
//  - Second pass: probe/insert hash map once per touched dict entry.
// ============================================================

static inline bool TryCountDictionaryChunk(FnLocalState &l, DataChunk &input) {
    auto &col = input.data[0];
    if (col.GetVectorType() != VectorType::DICTIONARY_VECTOR) {
        return false;
    }

    auto opt_dict_size = DictionaryVector::DictionarySize(col);
    if (!opt_dict_size.IsValid()) {
        return false;
    }
    const idx_t dict_size = opt_dict_size.GetIndex();
    if (dict_size == 0) {
        return true;
    }

    l.EnsureDictChunkCapacity(dict_size);
    l.dict_chunk_touched.clear();
    l.dict_chunk_touched.reserve(std::min(dict_size, input.size()));

    auto &offsets = DictionaryVector::SelVector(col);

    // Hot loop: only classify by select_id, no string/hash work.
    for (idx_t r = 0; r < input.size(); ++r) {
        const idx_t dict_idx = offsets.get_index(r);
        if (!l.dict_chunk_seen[dict_idx]) {
            l.dict_chunk_seen[dict_idx] = 1;
            l.dict_chunk_touched.push_back(dict_idx);
        }
        l.dict_chunk_counts[dict_idx] += 1;
    }

    auto &dictionary_vector = DictionaryVector::Child(col);
    const idx_t unique_count = l.dict_chunk_touched.size();
    SelectionVector unique_entries(STANDARD_VECTOR_SIZE);
    for (idx_t i = 0; i < unique_count; ++i) {
        unique_entries.set_index(i, l.dict_chunk_touched[i]);
    }

    auto &unique_values = l.dict_unique_values;
    if (unique_values.ColumnCount() == 0) {
        unique_values.InitializeEmpty(input.GetTypes());
    }

    unique_values.data[0].Slice(dictionary_vector, unique_entries, unique_count);
    unique_values.SetCardinality(unique_count);
    unique_values.data[0].Flatten(unique_count);
    auto *unique_strs = FlatVector::GetData<string_t>(unique_values.data[0]);
    auto &unique_validity = FlatVector::Validity(unique_values.data[0]);

    // Only touched dictionary entries do one map operation each.
    for (idx_t i = 0; i < unique_count; ++i) {
        const idx_t dict_idx = unique_entries.get_index(i);
        const int64_t inc = l.dict_chunk_counts[dict_idx];
        l.dict_chunk_counts[dict_idx] = 0;
        l.dict_chunk_seen[dict_idx] = 0;
        if (inc == 0) {
            continue;
        }

        if (!unique_validity.RowIsValid(i)) {
            continue;
        }
        const string_t s = unique_strs[i];
        if (s.GetSize() == 0) {
            continue;
        }

        l.AddCount(s, inc);
    }

    l.dict_chunk_touched.clear();
    return true;
}

// ============================================================
//  Execute: fast group-by on VARCHAR
//  - No std::string
//  - On hit: no copy
//  - On miss: copy exactly once into local StringHeap
// ============================================================

static OperatorResultType FnExecute(ExecutionContext &, TableFunctionInput &in,
                                    DataChunk &input, DataChunk &) {
    if (input.size() == 0) {
        return OperatorResultType::NEED_MORE_INPUT;
    }
    auto &l = in.local_state->Cast<FnLocalState>();

    if (TryCountDictionaryChunk(l, input)) {
        return OperatorResultType::NEED_MORE_INPUT;
    }

    UnifiedVectorFormat sv_uvf;
    input.data[0].ToUnifiedFormat(input.size(), sv_uvf);
    auto *ptr = (string_t *)sv_uvf.data;

    auto &valid = sv_uvf.validity;
    const bool all_valid = valid.AllValid();

    // split_query already filters SearchPhrase != ''.
    if (all_valid) {
        for (idx_t r = 0; r < input.size(); ++r) {
            const idx_t i = sv_uvf.sel->get_index(r);
            const string_t s = ptr[i];
            if (s.GetSize() == 0) continue;

            l.AddCount(s, 1);
        }
    } else {
        for (idx_t r = 0; r < input.size(); ++r) {
            const idx_t i = sv_uvf.sel->get_index(r);
            if (!valid.RowIsValid(i)) continue;

            const string_t s = ptr[i];
            if (s.GetSize() == 0) continue;

            l.AddCount(s, 1);
        }
    }

    return OperatorResultType::NEED_MORE_INPUT;
}

// ============================================================
//  Finalize: sharded merge + heap Top10
//  - Merge: lookup without copying; copy into global shard heap only on miss
// ============================================================

struct TopRow {
    int64_t c;
    const string_t *key; // points into global shard heap (stable while state lives)
};

// min-heap by count
struct TopRowMinCmp {
    bool operator()(const TopRow &a, const TopRow &b) const {
        return a.c > b.c; // min-heap
    }
};

static OperatorFinalizeResultType FnFinalize(ExecutionContext &, TableFunctionInput &in,
                                            DataChunk &out) {
    auto &g = in.global_state->Cast<FnGlobalState>();
    auto &l = in.local_state->Cast<FnLocalState>();

    // Merge local -> global exactly once
    if (!l.merged) {
        for (size_t p = 0; p < RADIX_PARTITION_COUNT; ++p) {
            auto &part = l.partitions[p];
            if (part.map.empty()) {
                continue;
            }
            const size_t shard_idx = p & (SHARD_COUNT - 1);
            auto &shard = g.shards[shard_idx];
            std::lock_guard<std::mutex> guard(shard.lock);
            for (auto &kv : part.map) {
                const HashedStringT &k_hs = kv.first;
                const string_t &k = k_hs.str;
                const size_t hash = k_hs.hash;
                const int64_t v = kv.second;
                auto it = shard.map.find(k_hs);
                if (it != shard.map.end()) {
                    it->second += v;
                } else {
                    const auto owned = shard.heap.AddString(k);
                    HashedStringT owned_hs { owned, hash };
                    shard.map.emplace(std::move(owned_hs), v);
                }
            }


        }
        l.merged = true;
        g.merged_local_states.fetch_add(1, std::memory_order_relaxed);
    }

    const auto merged = g.merged_local_states.load(std::memory_order_relaxed);
    const auto active = g.active_local_states.load(std::memory_order_relaxed);

    // Only last finisher produces output
    if (!(active > 0 && merged == active)) {
        out.SetCardinality(0);
        return OperatorFinalizeResultType::FINISHED;
    }

    // Top-10 via min-heap
    std::priority_queue<TopRow, std::vector<TopRow>, TopRowMinCmp> heap;
    for (size_t i = 0; i < SHARD_COUNT; ++i) {
        auto &shard = g.shards[i];
        std::lock_guard<std::mutex> guard(shard.lock);

        for (auto &kv : shard.map) {
            const int64_t c = kv.second;
            if (c <= 0) continue;

            if (heap.size() < 10) {
                heap.push(TopRow{c, &kv.first.str});
            } else if (c > heap.top().c) {
                heap.pop();
                heap.push(TopRow{c, &kv.first.str});
            }
        }
    }


    // Extract + sort desc (tie-break lexicographically for determinism)
    std::vector<TopRow> top;
    top.reserve(heap.size());
    while (!heap.empty()) {
        top.push_back(heap.top());
        heap.pop();
    }

    auto less_lex = [](const string_t &a, const string_t &b) {
        const auto an = a.GetSize();
        const auto bn = b.GetSize();
        const auto n = std::min(an, bn);
        int cmp = n ? std::memcmp(a.GetData(), b.GetData(), n) : 0;
        if (cmp != 0) return cmp < 0;
        return an < bn;
    };

    std::sort(top.begin(), top.end(), [&](const TopRow &a, const TopRow &b) {
        if (a.c != b.c) return a.c > b.c;
        return less_lex(*a.key, *b.key);
    });
    idx_t out_idx = 0;
    auto *out_keys = FlatVector::GetData<string_t>(out.data[0]);
    auto *out_counts = FlatVector::GetData<int64_t>(out.data[1]);
    for (auto &r : top) {
        out_keys[out_idx] = StringVector::AddString(out.data[0], *r.key);
        out_counts[out_idx] = r.c;
        out_idx++;
    }
    out.SetCardinality(out_idx);


    return OperatorFinalizeResultType::FINISHED;
}

// ============================================================
//  Extension load
// ============================================================

static void LoadInternal(ExtensionLoader &loader) {
    TableFunction f("dbweaver", {LogicalType::TABLE}, nullptr, FnBind, FnInit, FnInitLocal);
    f.in_out_function       = FnExecute;
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