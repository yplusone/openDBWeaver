#define DUCKDB_EXTENSION_MAIN

#include "dbweaver_extension.hpp"
#include "duckdb.hpp"
#include "duckdb/function/table_function.hpp"
#include <atomic>

static inline uint64_t NextPowerOfTwo(uint64_t n) {
    if (n <= 1) return 1;
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n |= n >> 32;
    n++;
    return n;
}

#include <limits>
#include <mutex>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <array>
#include <thread>

namespace duckdb {

// ------------------------
// Utilities
// ------------------------
struct FnBindData : public FunctionData {
    unique_ptr<FunctionData> Copy() const override { return make_uniq<FnBindData>(); }
    bool Equals(const FunctionData &) const override { return true; }
};

static inline uint64_t ToKey(int64_t v) {
    return (uint64_t)v ^ (1ULL << 63);
}

// ------------------------
// Partitioning
// ------------------------
static constexpr uint32_t RADIX_BITS = 10;                 // 1024 partitions
static constexpr idx_t RADIX_PARTITIONS = 1u << RADIX_BITS;

static inline idx_t PartOf(int64_t v) {
    return (idx_t)(ToKey(v) >> (64 - RADIX_BITS));
}

// ------------------------
// Local state
// ------------------------
struct FnLocalState : public LocalTableFunctionState {
    bool finalized = false;
    static constexpr idx_t CACHE_SIZE = 4096;
    int64_t cache[CACHE_SIZE];
    bool cache_valid[CACHE_SIZE];
    // thread-local radix buffers
    std::vector<std::vector<int64_t>> buf;
    std::vector<int64_t> hash_table_buffer;

    FnLocalState()

        : buf(RADIX_PARTITIONS) {
        std::fill(cache_valid, cache_valid + CACHE_SIZE, false);
    }
};


// ------------------------
// Global state
// ------------------------
struct FnGlobalState : public GlobalTableFunctionState {
    std::atomic<idx_t> active_local_states {0};
    std::atomic<idx_t> threads_reached_finalize {0};
    std::atomic<idx_t> finalize_partition_idx {0};
    std::atomic<idx_t> finalize_threads_finished {0};
    std::atomic<uint64_t> total_distinct_count {0};

    std::mutex local_states_lock;
    std::vector<FnLocalState*> local_states;

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
    g.active_local_states.fetch_add(1, std::memory_order_relaxed);
    
    auto l = make_uniq<FnLocalState>();
    {
        std::lock_guard<std::mutex> guard(g.local_states_lock);
        g.local_states.push_back(l.get());
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
    names.push_back("cnt_distinct_userid");
    return make_uniq<FnBindData>();
}

// ------------------------
// Execute: radix partition into thread-local buffers
// ------------------------
static OperatorResultType FnExecute(ExecutionContext &, TableFunctionInput &in,
                                    DataChunk &input, DataChunk &) {
    auto &l = in.local_state->Cast<FnLocalState>();

    const idx_t n = input.size();
    if (n == 0) {
        return OperatorResultType::NEED_MORE_INPUT;
    }

    UnifiedVectorFormat uvf;
    input.data[0].ToUnifiedFormat(n, uvf);

    auto *data_ptr = (int64_t *)uvf.data;
    const auto &valid = uvf.validity;
    const auto *sel = uvf.sel;
    if (valid.AllValid()) {
        for (idx_t r = 0; r < n; r++) {
            const idx_t i = sel->get_index(r);
            const int64_t v = data_ptr[i];
            const idx_t cache_idx = (idx_t)(ToKey(v) & (FnLocalState::CACHE_SIZE - 1));
            if (!l.cache_valid[cache_idx] || l.cache[cache_idx] != v) {
                l.cache[cache_idx] = v;
                l.cache_valid[cache_idx] = true;
                l.buf[PartOf(v)].push_back(v);
            }
        }
    } else {
        for (idx_t r = 0; r < n; r++) {
            const idx_t i = sel->get_index(r);
            if (valid.RowIsValid(i)) {
                const int64_t v = data_ptr[i];
                const idx_t cache_idx = (idx_t)(ToKey(v) & (FnLocalState::CACHE_SIZE - 1));
                if (!l.cache_valid[cache_idx] || l.cache[cache_idx] != v) {
                    l.cache[cache_idx] = v;
                    l.cache_valid[cache_idx] = true;
                    l.buf[PartOf(v)].push_back(v);
                }
            }
        }
    }


    return OperatorResultType::NEED_MORE_INPUT;
}

// ------------------------
// Finalize: Pull-based cooperative merge
// ------------------------
static OperatorFinalizeResultType FnFinalize(ExecutionContext &, TableFunctionInput &in,
                                            DataChunk &out) {
    auto &g = in.global_state->Cast<FnGlobalState>();
    auto &l = in.local_state->Cast<FnLocalState>();

    out.SetCardinality(0);

    if (!l.finalized) {
        l.finalized = true;
        g.threads_reached_finalize.fetch_add(1, std::memory_order_release);
    }

    const idx_t num_threads = g.active_local_states.load(std::memory_order_acquire);

    // Barrier 1: Wait for all threads to arrive before starting the pull-merge phase
    // This ensures all local buffers are complete.
    while (g.threads_reached_finalize.load(std::memory_order_acquire) < num_threads) {
        std::this_thread::yield();
    }

    // Cooperative Pull Phase: Threads take turns processing partitions
    while (true) {
        idx_t p = g.finalize_partition_idx.fetch_add(1, std::memory_order_relaxed);
        if (p >= RADIX_PARTITIONS) {
            break;
        }
        // Sum total capacity to determine hash table size
        idx_t total_elements = 0;
        for (auto *other_l : g.local_states) {
            total_elements += other_l->buf[p].size();
        }

        if (total_elements > 0) {
            idx_t table_size = NextPowerOfTwo(total_elements * 2);
            if (table_size < 32) table_size = 32;
            
            const int64_t sentinel = std::numeric_limits<int64_t>::min();
            l.hash_table_buffer.assign(table_size, sentinel);
            auto &table = l.hash_table_buffer;
            const idx_t mask = table_size - 1;
            idx_t distinct_in_partition = 0;
            bool sentinel_seen = false;

            for (auto *other_l : g.local_states) {
                auto &b = other_l->buf[p];
                for (int64_t v : b) {
                    if (v == sentinel) {
                        if (!sentinel_seen) {
                            sentinel_seen = true;
                            distinct_in_partition++;
                        }
                        continue;
                    }
                    
                    // Identity Hash: Use raw bits (transformed by ToKey) as index
                    idx_t idx = (idx_t)ToKey(v) & mask;
                    while (table[idx] != sentinel) {
                        if (table[idx] == v) goto next_val;
                        idx = (idx + 1) & mask;
                    }
                    table[idx] = v;
                    distinct_in_partition++;
                    next_val:;
                }
                // Free local memory once it's merged into the partition table
                std::vector<int64_t>().swap(b);
            }
            g.total_distinct_count.fetch_add(distinct_in_partition, std::memory_order_relaxed);
        }

    }

    // Barrier 2: Wait for all threads to finish processing all partitions
    // This ensures no thread exits and destroys its local state while others are still pulling from it.
    const idx_t finished_idx = g.finalize_threads_finished.fetch_add(1, std::memory_order_acq_rel) + 1;
    
    if (finished_idx == num_threads) {
        // Last finisher emits the final global sum
        out.SetCardinality(1);
        out.SetValue(0, 0, Value::BIGINT(g.total_distinct_count.load(std::memory_order_relaxed)));
    } else {
        while (g.finalize_threads_finished.load(std::memory_order_acquire) < num_threads) {
            std::this_thread::yield();
        }
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
