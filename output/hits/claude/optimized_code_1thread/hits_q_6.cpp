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
#include <absl/strings/string_view.h>
#include <absl/hash/hash.h>
#include <atomic>
#include <limits>
#include <mutex>
#include <vector>
#include <string>
#include <memory>
#include <cstring>

namespace duckdb {

// ------------------------
// Optimized wyhash Implementation (v3/v4 hybrid for quality and speed)
// ------------------------
static inline uint64_t _wyread8(const uint8_t *p) { uint64_t v; std::memcpy(&v, p, 8); return v; }
static inline uint64_t _wyread4(const uint8_t *p) { uint32_t v; std::memcpy(&v, p, 4); return v; }
static inline uint64_t _wyread3(const uint8_t *p, size_t k) { 
    return (((uint64_t)p[0]) << 16) | (((uint64_t)p[k >> 1]) << 8) | p[k - 1]; 
}

static inline uint64_t _wymum(uint64_t A, uint64_t B) {
#ifdef __SIZEOF_INT128__
    __uint128_t r = A;
    r *= B;
    return (uint64_t)r ^ (uint64_t)(r >> 64);
#else
    uint64_t ha = A >> 32, la = (uint32_t)A, hb = B >> 32, lb = (uint32_t)B;
    uint64_t rh = ha * hb, rm0 = ha * lb, rm1 = hb * la, rl = la * lb;
    uint64_t t = rl + (rm0 << 32), rh2 = rh + (rm0 >> 32);
    uint64_t res = t + (rm1 << 32);
    return res ^ (rh2 + (rm1 >> 32) + (t < rl));
#endif
}

static inline uint64_t wyhash(const void *key, size_t len, uint64_t seed) {
    const uint8_t *p = (const uint8_t *)key;
    static const uint64_t secret[4] = {0xa0761d6478bd642fULL, 0xe7037ed1a0b428dbULL, 0x8ebc6af09c88c6e3ULL, 0x589965cc75374cc3ULL};
    seed ^= secret[0];
    uint64_t a, b;
    if (len <= 16) {
        if (len >= 8) {
            a = _wyread8(p);
            b = _wyread8(p + len - 8);
        } else if (len >= 4) {
            a = _wyread4(p);
            b = _wyread4(p + len - 4);
        } else if (len > 0) {
            a = _wyread3(p, len);
            b = 0;
        } else {
            a = b = 0;
        }
    } else {
        size_t i = len;
        if (i > 48) {
            uint64_t see1 = seed, see2 = seed;
            do {
                seed = _wymum(_wyread8(p) ^ secret[1], _wyread8(p + 8) ^ seed);
                see1 = _wymum(_wyread8(p + 16) ^ secret[2], _wyread8(p + 24) ^ see1);
                see2 = _wymum(_wyread8(p + 32) ^ secret[3], _wyread8(p + 40) ^ see2);
                p += 48;
                i -= 48;
            } while (i > 48);
            seed ^= see1 ^ see2;
        }
        while (i > 16) {
            seed = _wymum(_wyread8(p) ^ secret[1], _wyread8(p + 8) ^ seed);
            p += 16;
            i -= 16;
        }
        a = _wyread8(p + i - 16);
        b = _wyread8(p + i - 8);
    }
    return _wymum(secret[1] ^ len, _wymum(a ^ secret[1], b ^ seed));
}

// ------------------------
// Utilities and Types
// ------------------------
struct FnBindData : public FunctionData {
    unique_ptr<FunctionData> Copy() const override { return make_uniq<FnBindData>(); }
    bool Equals(const FunctionData &) const override { return true; }
};

// ------------------------
// Global state
// ------------------------
struct FnGlobalState : public GlobalTableFunctionState {
    std::mutex lock;
    absl::flat_hash_set<uint64_t> global_set;

    std::atomic<idx_t> active_local_states {0};
    std::atomic<idx_t> merged_local_states {0};

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
    absl::flat_hash_set<uint64_t> local_set;

    FnLocalState() : merged(false) {
        local_set.reserve(8192);
    }
};

static unique_ptr<LocalTableFunctionState> FnInitLocal(ExecutionContext &, TableFunctionInitInput &, 
                                                      GlobalTableFunctionState *global_state) {
    auto &g = global_state->Cast<FnGlobalState>();
    g.active_local_states.fetch_add(1, std::memory_order_relaxed);
    return make_uniq<FnLocalState>();
}

static unique_ptr<FunctionData> FnBind(ClientContext &, TableFunctionBindInput &, 
                                      vector<LogicalType> &return_types, 
                                      vector<string> &names) {
    return_types.push_back(LogicalType::BIGINT);
    names.push_back("cnt_distinct_searchphrase");
    return make_uniq<FnBindData>();
}

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
    const auto *sel = uvf.sel;

    for (idx_t r = 0; r < n; r++) {
        const idx_t i = sel->get_index(r);
        if (!valid.RowIsValid(i)) {
            continue;
        }

        string_t v = data_ptr[i];
        // Using wyhash with a proper small-string handling to minimize collisions
        uint64_t hash = wyhash(v.GetDataUnsafe(), v.GetSize(), 0);
        l.local_set.insert(hash);
    }

    return OperatorResultType::NEED_MORE_INPUT;
}

static OperatorFinalizeResultType FnFinalize(ExecutionContext &, TableFunctionInput &in, 
                                            DataChunk &out) {
    auto &g = in.global_state->Cast<FnGlobalState>();
    auto &l = in.local_state->Cast<FnLocalState>();

    if (!l.merged) {
        std::lock_guard<std::mutex> guard(g.lock);
        if (g.global_set.empty()) {
            g.global_set = std::move(l.local_set);
        } else {
            g.global_set.insert(l.local_set.begin(), l.local_set.end());
        }
        l.merged = true;
        g.merged_local_states.fetch_add(1, std::memory_order_acq_rel);
    }

    const auto merged = g.merged_local_states.load(std::memory_order_acquire);
    const auto active = g.active_local_states.load(std::memory_order_acquire);

    if (active > 0 && merged == active) {
        out.SetCardinality(1);
        out.SetValue(0, 0, Value::BIGINT(g.global_set.size()));
        
        g.merged_local_states.store(0, std::memory_order_release);
        g.active_local_states.store(0, std::memory_order_release);
    } else {
        out.SetCardinality(0);
    }
    
    return OperatorFinalizeResultType::FINISHED;
}

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
