#define DUCKDB_EXTENSION_MAIN

#include "dbweaver_extension.hpp"
#include "duckdb.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/common/types/string_heap.hpp"
#include "duckdb/common/vector_operations/vector_operations.hpp"
#include "duckdb/common/types/hash.hpp"

#include <absl/container/flat_hash_set.h>
#include <atomic>
#include <limits>
#include <mutex>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <array>
#include <string>
#include <thread>

namespace duckdb {

// ------------------------
// Utilities and Types
// ------------------------
struct FnBindData : public FunctionData {
    unique_ptr<FunctionData> Copy() const override { return make_uniq<FnBindData>(); }
    bool Equals(const FunctionData &) const override { return true; }
};

static constexpr uint32_t RADIX_BITS = 12;                 // 2048 partitions
static constexpr idx_t RADIX_PARTITIONS = 1u << RADIX_BITS;
static_assert((RADIX_PARTITIONS & (RADIX_PARTITIONS - 1)) == 0, "RADIX_PARTITIONS must be power-of-two");

// Structure to store a string and its pre-computed hash
struct HashedString {
    string_t s;
    hash_t h;

    bool operator<(const HashedString &other) const {
        return s < other.s;
    }
    bool operator==(const HashedString &other) const {
        return s == other.s;
    }
};

// Custom hasher that returns the pre-computed hash
struct PrecomputedHasher {
    size_t operator()(const HashedString &v) const {
        return (size_t)v.h;
    }
};

struct HashedStringEq {
    bool operator()(const HashedString &a, const HashedString &b) const {
        return a.s == b.s;
    }
};

// ------------------------
// Local state
// ------------------------
struct FnLocalState : public LocalTableFunctionState {
    StringHeap heap;
    Vector hash_vec;

    // Each partition has its own vector of HashedString
    std::vector<std::vector<HashedString>> buf;

    // Track which partitions were touched in this thread
    std::vector<uint16_t> touched;
    std::vector<uint8_t> touched_flag;

    // Reusable set for chunk-level dedup
    absl::flat_hash_set<HashedString, PrecomputedHasher, HashedStringEq> chunk_unique;

    FnLocalState()
        : hash_vec(LogicalType::HASH),
          buf(RADIX_PARTITIONS),
          touched(),
          touched_flag(RADIX_PARTITIONS, 0) {
        touched.reserve(256);
    }
};

// ------------------------
// Global state
// ------------------------
struct GlobalPartition {
    StringHeap heap;
    absl::flat_hash_set<HashedString, PrecomputedHasher, HashedStringEq> set;
};

struct FnGlobalState : public GlobalTableFunctionState {
    std::atomic<idx_t> active_local_states {0};
    std::atomic<idx_t> finished_threads {0};
    std::atomic<idx_t> partition_cursor {0};
    std::atomic<idx_t> threads_completed_merge {0};
    std::atomic<idx_t> total_distinct_count {0};

    std::mutex local_states_lock;
    std::vector<FnLocalState*> local_states;

    std::array<GlobalPartition, RADIX_PARTITIONS> parts;

    idx_t MaxThreads() const override {
        return std::numeric_limits<idx_t>::max();
    }
};

static unique_ptr<GlobalTableFunctionState> FnInit(ClientContext &, TableFunctionInitInput &) {
    return make_uniq<FnGlobalState>();
}

static unique_ptr<LocalTableFunctionState> FnInitLocal(ExecutionContext &, TableFunctionInitInput &,
                                                      GlobalTableFunctionState *global_state) {
    auto &g = global_state->Cast<FnGlobalState>();
    auto l = make_uniq<FnLocalState>();
    {
        std::lock_guard<std::mutex> guard(g.local_states_lock);
        g.local_states.push_back(l.get());
        g.active_local_states.fetch_add(1, std::memory_order_relaxed);
    }
    return std::move(l);
}

// ------------------------
// Bind
// ------------------------
static unique_ptr<FunctionData> FnBind(ClientContext &, TableFunctionBindInput &,
                                      vector<LogicalType> &return_types,
                                      vector<string> &names) {
    return_types.push_back(LogicalType::BIGINT);
    names.push_back("cnt_distinct_searchphrase");
    return make_uniq<FnBindData>();
}

// ------------------------
// Execute: vectorized hashing and radix partitioning
// ------------------------
static constexpr bool EARLY_LOCAL_DEDUP = true;

static OperatorResultType FnExecute(ExecutionContext &, TableFunctionInput &in,
                                    DataChunk &input, DataChunk &) {
    auto &l = in.local_state->Cast<FnLocalState>();
    const idx_t n = input.size();
    if (n == 0) {
        return OperatorResultType::NEED_MORE_INPUT;
    }

    VectorOperations::Hash(input.data[0], l.hash_vec, n);
    auto *hashes = FlatVector::GetData<hash_t>(l.hash_vec);

    UnifiedVectorFormat uvf;
    input.data[0].ToUnifiedFormat(n, uvf);
    auto *data_ptr = (string_t *)uvf.data;
    const auto &valid = uvf.validity;
    const bool all_valid = valid.AllValid();
    const auto *sel = uvf.sel;

    if (EARLY_LOCAL_DEDUP) {
        l.chunk_unique.clear();
        for (idx_t r = 0; r < n; r++) {
            const idx_t i = sel->get_index(r);
            if (!all_valid && !valid.RowIsValid(i)) {
                continue;
            }
            string_t v = data_ptr[i];
            hash_t h = hashes[r];
            HashedString hs {v, h};

            if (l.chunk_unique.find(hs) == l.chunk_unique.end()) {
                string_t hv = l.heap.AddString(v);
                HashedString hsv {hv, h};
                l.chunk_unique.insert(hsv);

                const idx_t part = (idx_t)(h >> (64 - RADIX_BITS));
                if (!l.touched_flag[part]) {
                    l.touched_flag[part] = 1;
                    l.touched.push_back((uint16_t)part);
                }
                l.buf[part].push_back(hsv);
            }
        }
    } else {
        for (idx_t r = 0; r < n; r++) {
            const idx_t i = sel->get_index(r);
            if (!all_valid && !valid.RowIsValid(i)) {
                continue;
            }
            string_t v = l.heap.AddString(data_ptr[i]);
            hash_t h = hashes[r];
            const idx_t part = (idx_t)(h >> (64 - RADIX_BITS));
            if (!l.touched_flag[part]) {
                l.touched_flag[part] = 1;
                l.touched.push_back((uint16_t)part);
            }
            l.buf[part].push_back({v, h});
        }
    }
    return OperatorResultType::NEED_MORE_INPUT;
}

// ------------------------
// Finalize: Phased global merge via partition ownership (Lock-Free)
// ------------------------
static OperatorFinalizeResultType FnFinalize(ExecutionContext &, TableFunctionInput &in,
                                            DataChunk &out) {
    auto &g = in.global_state->Cast<FnGlobalState>();
    auto &l = in.local_state->Cast<FnLocalState>();

    // Phase 1: Local Pre-sort and Dedup in parallel
    for (auto p : l.touched) {
        auto &v_vec = l.buf[p];
        if (v_vec.empty()) continue;
        std::sort(v_vec.begin(), v_vec.end());
        v_vec.erase(std::unique(v_vec.begin(), v_vec.end()), v_vec.end());
    }

    // Barrier 1: Wait for all threads to finish execution and local dedup
    idx_t my_idx = g.finished_threads.fetch_add(1);
    const idx_t total_active = g.active_local_states.load(std::memory_order_seq_cst);

    while (g.finished_threads.load(std::memory_order_seq_cst) < total_active) {
        std::this_thread::yield();
    }

    // Phase 2: Global Merge with Partition Ownership
    // Each thread dynamically picks a block of partitions to own and merge
    idx_t p_start;
    const idx_t block_size = 64;
    idx_t my_partition_sum = 0;

    while ((p_start = g.partition_cursor.fetch_add(block_size, std::memory_order_seq_cst)) < RADIX_PARTITIONS) {
        idx_t p_end = std::min(p_start + block_size, (idx_t)RADIX_PARTITIONS);
        for (idx_t p = p_start; p < p_end; p++) {
            auto &gp = g.parts[p];
            // This thread owns partition 'p', merge data from all local states
            for (auto *ls : g.local_states) {
                auto &v_vec = ls->buf[p];
                if (v_vec.empty()) continue;

                gp.set.reserve(gp.set.size() + v_vec.size());
                for (const auto &hs : v_vec) {
                    if (gp.set.find(hs) == gp.set.end()) {
                        string_t global_v = gp.heap.AddString(hs.s);
                        gp.set.insert({global_v, hs.h});
                    }
                }
                // Free local memory as we go
                std::vector<HashedString>().swap(v_vec);
            }
            my_partition_sum += gp.set.size();
        }
    }
    g.total_distinct_count.fetch_add(my_partition_sum, std::memory_order_seq_cst);

    // Barrier 2: Wait for all partitions to be merged
    g.threads_completed_merge.fetch_add(1, std::memory_order_seq_cst);
    while (g.threads_completed_merge.load(std::memory_order_seq_cst) < total_active) {
        std::this_thread::yield();
    }

    // Only one thread returns the result
    if (my_idx == 0) {
        out.SetCardinality(1);
        out.SetValue(0, 0, Value::BIGINT(g.total_distinct_count.load(std::memory_order_relaxed)));
    } else {
        out.SetCardinality(0);
    }

    return OperatorFinalizeResultType::FINISHED;
}

// ------------------------
// Load
// ------------------------
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
