/*
query_template: SELECT COUNT(DISTINCT UserID) AS cnt_distinct_userid FROM hits;

split_template: SELECT cnt_distinct_userid
FROM dbweaver((
  SELECT UserID
  FROM hits
));
query_example: SELECT COUNT(DISTINCT UserID) AS cnt_distinct_userid FROM hits;

split_query: SELECT cnt_distinct_userid
FROM dbweaver((
  SELECT UserID
  FROM hits
));
*/
#define DUCKDB_EXTENSION_MAIN

#include "dbweaver_extension.hpp"
#include "duckdb.hpp"
#include "duckdb/function/table_function.hpp"

#include <absl/container/flat_hash_set.h>

#include <atomic>
#include <limits>
#include <mutex>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <array>

namespace duckdb {

// ------------------------
// Utilities
// ------------------------
struct FnBindData : public FunctionData {
    unique_ptr<FunctionData> Copy() const override { return make_uniq<FnBindData>(); }
    bool Equals(const FunctionData &) const override { return true; }
};

// Order-preserving signed->unsigned mapping (cheaper than Mix64 and helps stable partitioning)
static inline uint64_t ToKey(int64_t v) {
    return (uint64_t)v ^ (1ULL << 63);
}

// ------------------------
// Partitioning
// ------------------------
// Strategy: increase partitions to reduce lock contention and per-set size.
// Use high-bit radix partition on ToKey(v) (fast, deterministic).
static constexpr uint32_t RADIX_BITS = 12;                 // 2048 partitions
static constexpr idx_t RADIX_PARTITIONS = 1u << RADIX_BITS;
static_assert((RADIX_PARTITIONS & (RADIX_PARTITIONS - 1)) == 0, "RADIX_PARTITIONS must be power-of-two");

static inline idx_t PartOf(int64_t v) {
    return (idx_t)(ToKey(v) >> (64 - RADIX_BITS));
}

// ------------------------
// Global state
// ------------------------
struct GlobalPartition {
    std::mutex lock;
    absl::flat_hash_set<int64_t> set;
};

struct FnGlobalState : public GlobalTableFunctionState {
    std::atomic<idx_t> active_local_states {0};
    std::atomic<idx_t> merged_local_states {0};

    std::array<GlobalPartition, RADIX_PARTITIONS> parts;

    idx_t MaxThreads() const override {
        return std::numeric_limits<idx_t>::max();
    }
};

static unique_ptr<GlobalTableFunctionState> FnInit(ClientContext &, TableFunctionInitInput &) {
    return make_uniq<FnGlobalState>();
}

// ------------------------
// Local state
// ------------------------
struct FnLocalState : public LocalTableFunctionState {
    bool merged = false;

    // thread-local radix buffers
    std::vector<std::vector<int64_t>> buf;

    // track touched partitions to avoid scanning all partitions in finalize
    std::vector<uint16_t> touched;
    std::vector<uint8_t> touched_flag;

    FnLocalState()
        : buf(RADIX_PARTITIONS),
          touched(),
          touched_flag(RADIX_PARTITIONS, 0) {
        touched.reserve(256); // typical chunk touches far less than 2048 partitions
    }
};

static unique_ptr<LocalTableFunctionState> FnInitLocal(ExecutionContext &, TableFunctionInitInput &,
                                                      GlobalTableFunctionState *global_state) {
    auto &g = global_state->Cast<FnGlobalState>();
    g.active_local_states.fetch_add(1, std::memory_order_relaxed);
    return make_uniq<FnLocalState>();
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
    const bool all_valid = valid.AllValid();
    const auto *sel = uvf.sel;

    // Reserve heuristic:
    // avoid scanning all partitions; reserve only touched partitions after first pass.
    // For large chunks, we can reserve a bit for touched partitions based on expected hit rate.
    // (We'll do this lazily during pushing for simplicity.)

    auto push_value = [&](int64_t v) {
        const idx_t part = PartOf(v);

        // track touched partitions (cheap)
        if (!l.touched_flag[part]) {
            l.touched_flag[part] = 1;
            l.touched.push_back((uint16_t)part);

            // small first-touch reserve to reduce frequent realloc on hot partitions
            // (tuned for chunked ingestion)
            auto &b = l.buf[part];
            if (b.capacity() < b.size() + 64) {
                b.reserve(b.size() + 256);
            }
        }

        l.buf[part].push_back(v);
    };

    if (all_valid) {
        for (idx_t r = 0; r < n; r++) {
            const idx_t i = sel->get_index(r);
            push_value(data_ptr[i]);
        }
    } else {
        for (idx_t r = 0; r < n; r++) {
            const idx_t i = sel->get_index(r);
            if (!valid.RowIsValid(i)) {
                continue;
            }
            push_value(data_ptr[i]);
        }
    }

    return OperatorResultType::NEED_MORE_INPUT;
}

// ------------------------
// Finalize: local sort+unique on touched partitions then merge into global sets
// ------------------------
static OperatorFinalizeResultType FnFinalize(ExecutionContext &, TableFunctionInput &in,
                                            DataChunk &out) {
    auto &g = *reinterpret_cast<FnGlobalState *>(in.global_state.get());
    auto &l = in.local_state->Cast<FnLocalState>();

    if (!l.merged) {
        // Only iterate partitions that were touched by this thread
        for (idx_t ti = 0; ti < l.touched.size(); ti++) {
            const idx_t p = (idx_t)l.touched[ti];
            auto &v = l.buf[p];
            if (v.empty()) {
                continue;
            }

            // local batch dedup
            std::sort(v.begin(), v.end());
            v.erase(std::unique(v.begin(), v.end()), v.end());

            auto &gp = g.parts[p];
            {
                std::lock_guard<std::mutex> guard(gp.lock);

                // more aggressive reserve to avoid rehash
                // (flat_hash_set uses open addressing; keeping load factor low helps)
                gp.set.reserve(gp.set.size() + v.size() + (v.size() >> 1));

                // range insert is typically faster than per-element insert loops
                gp.set.insert(v.begin(), v.end());
            }

            // release local memory for this partition
            std::vector<int64_t>().swap(v);
        }

        // reset touched flags (not strictly needed since finalize runs once, but keep state clean)
        // also release touched memory
        std::vector<uint16_t>().swap(l.touched);
        std::vector<uint8_t>().swap(l.touched_flag);

        l.merged = true;
        g.merged_local_states.fetch_add(1, std::memory_order_relaxed);
    }

    // only last finisher emits output
    const auto merged = g.merged_local_states.load(std::memory_order_relaxed);
    const auto active = g.active_local_states.load(std::memory_order_relaxed);

    if (active > 0 && merged == active) {
        idx_t total = 0;
        for (idx_t p = 0; p < RADIX_PARTITIONS; p++) {
            auto &gp = g.parts[p];
            // After all locals have merged, locks are uncontended; still lock for safety.
            std::lock_guard<std::mutex> guard(gp.lock);
            total += gp.set.size();
        }
        out.SetCardinality(1);
        out.SetValue(0, 0, Value::BIGINT(total));
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
