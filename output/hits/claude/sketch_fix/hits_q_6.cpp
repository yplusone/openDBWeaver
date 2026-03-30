/*
query_template: SELECT COUNT(DISTINCT SearchPhrase) AS cnt_distinct_searchphrase FROM hits;

split_template: select * from dbweaver((SELECT SearchPhrase FROM hits));
query_example: SELECT COUNT(DISTINCT SearchPhrase) AS cnt_distinct_searchphrase FROM hits;

split_query: select * from dbweaver((SELECT SearchPhrase FROM hits));
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
#include <string>

namespace duckdb {

// ------------------------
// Utilities and Types
// ------------------------
struct FnBindData : public FunctionData {
    unique_ptr<FunctionData> Copy() const override { return make_uniq<FnBindData>(); }
    bool Equals(const FunctionData &) const override { return true; }
};

// Use 12 radix bits for string hashing (same partitions as BIGINT code)
static constexpr uint32_t RADIX_BITS = 12;                 // 2048 partitions
static constexpr idx_t RADIX_PARTITIONS = 1u << RADIX_BITS;
static_assert((RADIX_PARTITIONS & (RADIX_PARTITIONS - 1)) == 0, "RADIX_PARTITIONS must be power-of-two");

// High-quality string hash function for partitioning (Keeps partitioning deterministic)
static inline uint64_t StringRadixHash(const char *data, size_t len) {
    // FNV-1a 64-bit
    uint64_t hash = 14695981039346656037ULL;
    for (size_t i = 0; i < len; i++) {
        hash ^= (unsigned char)data[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

static inline idx_t PartOf(const char *data, size_t len) {
    return (idx_t)(StringRadixHash(data, len) >> (64 - RADIX_BITS));
}

// ------------------------
// Global state
// ------------------------
struct GlobalPartition {
    std::mutex lock;
    absl::flat_hash_set<std::string> set;
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

    // Each partition has its own vector of string values
    std::vector<std::vector<std::string>> buf;

    // Track which partitions were touched in this thread/chunk
    std::vector<uint16_t> touched;
    std::vector<uint8_t> touched_flag;

    FnLocalState()
        : buf(RADIX_PARTITIONS),
          touched(),
          touched_flag(RADIX_PARTITIONS, 0) {
        touched.reserve(256); // Heuristic: usually far less than 2048 partitions per chunk
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
    names.push_back("cnt_distinct_searchphrase");
    return make_uniq<FnBindData>();
}

// ------------------------
// Execute: radix partition into thread-local buffers; early local dedup
// ------------------------
static constexpr bool EARLY_LOCAL_DEDUP = true;

static OperatorResultType FnExecute(ExecutionContext &, TableFunctionInput &in,
                                    DataChunk &input, DataChunk &) {
    auto &l = in.local_state->Cast<FnLocalState>();
    const idx_t n = input.size();
    if (n == 0) {
        return OperatorResultType::NEED_MORE_INPUT;
    }

    UnifiedVectorFormat uvf;
    input.data[0].ToUnifiedFormat(n, uvf);
    auto *data_ptr = (string_t *)uvf.data;
    const auto &valid = uvf.validity;
    const bool all_valid = valid.AllValid();
    const auto *sel = uvf.sel;

    if (EARLY_LOCAL_DEDUP) {
        absl::flat_hash_set<std::string> local_unique;
        local_unique.reserve(n < 128 ? n : 128); // heuristics for expected cardinality
        if (all_valid) {
            for (idx_t r = 0; r < n; r++) {
                const idx_t i = sel->get_index(r);
                string_t v = data_ptr[i];
                local_unique.insert(std::string(v.GetDataUnsafe(), v.GetSize()));
            }
        } else {
            for (idx_t r = 0; r < n; r++) {
                const idx_t i = sel->get_index(r);
                if (!valid.RowIsValid(i)) {
                    continue;
                }
                string_t v = data_ptr[i];
                local_unique.insert(std::string(v.GetDataUnsafe(), v.GetSize()));
            }
        }
        for (const auto &v : local_unique) {
            const idx_t part = PartOf(v.data(), v.size());
            // track touched partitions
            if (!l.touched_flag[part]) {
                l.touched_flag[part] = 1;
                l.touched.push_back((uint16_t)part);
                auto &b = l.buf[part];
                if (b.capacity() < b.size() + 16) {
                    b.reserve(b.size() + 64);
                }
            }
            l.buf[part].push_back(v);
        }
    } else {
        auto push_value = [&](const std::string &v) {
            const idx_t part = PartOf(v.data(), v.size());
            if (!l.touched_flag[part]) {
                l.touched_flag[part] = 1;
                l.touched.push_back((uint16_t)part);
                auto &b = l.buf[part];
                if (b.capacity() < b.size() + 16) {
                    b.reserve(b.size() + 64);
                }
            }
            l.buf[part].push_back(v);
        };
        if (all_valid) {
            for (idx_t r = 0; r < n; r++) {
                const idx_t i = sel->get_index(r);
                string_t v = data_ptr[i];
                push_value(std::string(v.GetDataUnsafe(), v.GetSize()));
            }
        } else {
            for (idx_t r = 0; r < n; r++) {
                const idx_t i = sel->get_index(r);
                if (!valid.RowIsValid(i)) {
                    continue;
                }
                string_t v = data_ptr[i];
                push_value(std::string(v.GetDataUnsafe(), v.GetSize()));
            }
        }
    }
    return OperatorResultType::NEED_MORE_INPUT;
}

// ------------------------
// Finalize: dedup and merge touched partitions
// ------------------------
static OperatorFinalizeResultType FnFinalize(ExecutionContext &, TableFunctionInput &in,
                                            DataChunk &out) {
    auto &g = *reinterpret_cast<FnGlobalState *>(in.global_state.get());
    auto &l = in.local_state->Cast<FnLocalState>();

    if (!l.merged) {
        for (idx_t ti = 0; ti < l.touched.size(); ti++) {
            const idx_t p = (idx_t)l.touched[ti];
            auto &v = l.buf[p];
            if (v.empty()) {
                continue;
            }

            // Local batch dedup in this partition
            std::sort(v.begin(), v.end());
            v.erase(std::unique(v.begin(), v.end()), v.end());

            auto &gp = g.parts[p];
            {
                std::lock_guard<std::mutex> guard(gp.lock);

                gp.set.reserve(gp.set.size() + v.size() + (v.size() >> 1));
                gp.set.insert(v.begin(), v.end());
            }

            std::vector<std::string>().swap(v);
        }

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