/*
query_template: SELECT UserID, COUNT(*) AS cnt FROM hits GROUP BY UserID ORDER BY COUNT(*) DESC LIMIT 10;

split_template: select * from dbweaver((SELECT UserID FROM hits));
query_example: SELECT UserID, COUNT(*) AS cnt FROM hits GROUP BY UserID ORDER BY COUNT(*) DESC LIMIT 10;

split_query: select * from dbweaver((SELECT UserID FROM hits));
*/
#define DUCKDB_EXTENSION_MAIN

#include "dbweaver_extension.hpp"
#include "duckdb.hpp"
#include "duckdb/function/table_function.hpp"
#include <atomic>
#include <limits>
#include <mutex>
#include <vector>
#include <algorithm>
#include <cstring>

namespace duckdb {

struct FnBindData : public FunctionData {
    vector<LogicalType> return_types;

    unique_ptr<FunctionData> Copy() const override {
        auto copy = make_uniq<FnBindData>();
        copy->return_types = return_types;
        return std::move(copy);
    }
    bool Equals(const FunctionData &) const override { return true; }
};

// A high-performance flat-array hash table entry
struct TableEntry {
    int64_t key;
    int64_t count;
};

// Simple flat-array hash table using linear probing.
struct FlatHashTable {
    static constexpr uint64_t CAPACITY = 1 << 25; 
    static constexpr uint64_t MASK = CAPACITY - 1;
    static constexpr int64_t EMPTY_KEY = std::numeric_limits<int64_t>::min();

    std::vector<TableEntry> entries;

    FlatHashTable() {
        entries.resize(CAPACITY, {EMPTY_KEY, 0});
    }

    // Optimized hash function for int64_t keys
    inline uint64_t Hash(int64_t v) const {
        uint64_t x = static_cast<uint64_t>(v);
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
        x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
        x = x ^ (x >> 31);
        return x & MASK;
    }

    void Increment(int64_t key, uint64_t h) {
        while (true) {
            if (entries[h].key == key) {
                entries[h].count++;
                return;
            } else if (entries[h].key == EMPTY_KEY) {
                entries[h].key = key;
                entries[h].count = 1;
                return;
            }
            h = (h + 1) & MASK;
        }
    }

    void Increment(int64_t key) {
        Increment(key, Hash(key));
    }

    void Merge(const FlatHashTable &other) {
        for (uint64_t i = 0; i < CAPACITY; ++i) {
            if (other.entries[i].key != EMPTY_KEY) {
                uint64_t h = Hash(other.entries[i].key);
                while (true) {
                    if (entries[h].key == other.entries[i].key) {
                        entries[h].count += other.entries[i].count;
                        break;
                    } else if (entries[h].key == EMPTY_KEY) {
                        entries[h].key = other.entries[i].key;
                        entries[h].count = other.entries[i].count;
                        break;
                    }
                    h = (h + 1) & MASK;
                }
            }
        }
    }
};

struct GroupKey {
    int64_t UserID;
    bool operator==(const GroupKey& other) const { return UserID == other.UserID; }
};

struct SortRow {
    int64_t UserID;
    int64_t cnt;
};

struct SortRowComparator {
    bool operator()(const SortRow &a, const SortRow &b) const {
        if (a.cnt != b.cnt) return a.cnt > b.cnt;
        return a.UserID < b.UserID;
    }
};

struct FnGlobalState : public GlobalTableFunctionState {
    FlatHashTable global_table;
    std::vector<SortRow> sorted_rows;
    std::mutex lock;
    std::atomic<idx_t> active_local_states {0};
    std::atomic<idx_t> merged_local_states {0};
    bool results_ready = false;

    idx_t MaxThreads() const override {
        return std::numeric_limits<idx_t>::max();
    }
};

static unique_ptr<GlobalTableFunctionState> FnInit(ClientContext &, TableFunctionInitInput &) {
    return make_uniq<FnGlobalState>();
}

struct FnLocalState : public LocalTableFunctionState {
    bool merged = false;
    FlatHashTable local_table;
};

static unique_ptr<LocalTableFunctionState> FnInitLocal(ExecutionContext &, TableFunctionInitInput &,GlobalTableFunctionState *global_state) {
    auto &g = global_state->Cast<FnGlobalState>();
    g.active_local_states.fetch_add(1, std::memory_order_relaxed);
    return make_uniq<FnLocalState>();
}

static unique_ptr<FunctionData> FnBind(ClientContext &, TableFunctionBindInput &,
                                            vector<LogicalType> &return_types,
                                            vector<string> &names) {
    names.push_back("UserID");
    names.push_back("cnt");
    return_types.push_back(LogicalType::BIGINT);
    return_types.push_back(LogicalType::BIGINT);

    auto bind_data = make_uniq<FnBindData>();
    bind_data->return_types = return_types;
    return std::move(bind_data);
}

static OperatorResultType FnExecute(ExecutionContext &, TableFunctionInput &in, DataChunk &input, DataChunk &) {
    auto &l = in.local_state->Cast<FnLocalState>();
    if (input.size() == 0) return OperatorResultType::NEED_MORE_INPUT;

    UnifiedVectorFormat UserID_uvf;
    input.data[0].ToUnifiedFormat(input.size(), UserID_uvf);
    int64_t* UserID_ptr = (int64_t*)UserID_uvf.data;

    auto &valid_UserID  = UserID_uvf.validity;
    const bool UserID_all_valid = valid_UserID.AllValid();

    constexpr idx_t PREFETCH_DIST = 16;
    uint64_t hash_cache[PREFETCH_DIST];

    if (UserID_all_valid) {
        // Warm up the prefetch cache
        for (idx_t i = 0; i < PREFETCH_DIST && i < input.size(); ++i) {
            idx_t idx = UserID_uvf.sel->get_index(i);
            hash_cache[i] = l.local_table.Hash(UserID_ptr[idx]);
            __builtin_prefetch(&l.local_table.entries[hash_cache[i]], 1, 1);
        }
        // Main loop
        for (idx_t i = 0; i < input.size(); ++i) {
            idx_t cur_idx = UserID_uvf.sel->get_index(i);
            l.local_table.Increment(UserID_ptr[cur_idx], hash_cache[i % PREFETCH_DIST]);

            if (i + PREFETCH_DIST < input.size()) {
                idx_t next_idx = UserID_uvf.sel->get_index(i + PREFETCH_DIST);
                uint64_t h = l.local_table.Hash(UserID_ptr[next_idx]);
                hash_cache[(i + PREFETCH_DIST) % PREFETCH_DIST] = h;
                __builtin_prefetch(&l.local_table.entries[h], 1, 1);
            }
        }
    } else {
        // Handle NULLs
        for (idx_t i = 0; i < PREFETCH_DIST && i < input.size(); ++i) {
            idx_t idx = UserID_uvf.sel->get_index(i);
            if (valid_UserID.RowIsValid(idx)) {
                hash_cache[i] = l.local_table.Hash(UserID_ptr[idx]);
                __builtin_prefetch(&l.local_table.entries[hash_cache[i]], 1, 1);
            }
        }
        for (idx_t i = 0; i < input.size(); ++i) {
            idx_t cur_idx = UserID_uvf.sel->get_index(i);
            if (valid_UserID.RowIsValid(cur_idx)) {
                l.local_table.Increment(UserID_ptr[cur_idx], hash_cache[i % PREFETCH_DIST]);
            }

            if (i + PREFETCH_DIST < input.size()) {
                idx_t next_idx = UserID_uvf.sel->get_index(i + PREFETCH_DIST);
                if (valid_UserID.RowIsValid(next_idx)) {
                    uint64_t h = l.local_table.Hash(UserID_ptr[next_idx]);
                    hash_cache[(i + PREFETCH_DIST) % PREFETCH_DIST] = h;
                    __builtin_prefetch(&l.local_table.entries[h], 1, 1);
                }
            }
        }
    }

    return OperatorResultType::NEED_MORE_INPUT; 
}

static OperatorFinalizeResultType FnFinalize(ExecutionContext &context, TableFunctionInput &in, DataChunk &out) {
    auto &g = *reinterpret_cast<FnGlobalState *>(in.global_state.get());
    auto &l = in.local_state->Cast<FnLocalState>();

    out.Reset();

    if (!l.merged) {
        {
            std::lock_guard<std::mutex> guard(g.lock);
            g.global_table.Merge(l.local_table);
        }
        l.merged = true;
        g.merged_local_states.fetch_add(1, std::memory_order_relaxed);
    }

    const auto merged = g.merged_local_states.load(std::memory_order_relaxed);
    const auto active = g.active_local_states.load(std::memory_order_relaxed);
    if (active > 0 && merged == active) {
        std::lock_guard<std::mutex> guard(g.lock);
        if (!g.results_ready) {
            for (uint64_t i = 0; i < FlatHashTable::CAPACITY; ++i) {
                if (g.global_table.entries[i].key != FlatHashTable::EMPTY_KEY) {
                    g.sorted_rows.push_back({g.global_table.entries[i].key, g.global_table.entries[i].count});
                }
            }
            idx_t sort_limit = MinValue<idx_t>((idx_t)10, g.sorted_rows.size());
            std::partial_sort(g.sorted_rows.begin(), g.sorted_rows.begin() + sort_limit, g.sorted_rows.end(), SortRowComparator{});
            g.results_ready = true;
        }

        idx_t total_rows = g.sorted_rows.size();
        idx_t max_rows = MinValue<idx_t>(MinValue<idx_t>((idx_t)10, (idx_t)STANDARD_VECTOR_SIZE), total_rows);
        out.SetCardinality(max_rows);
        auto *uid_data = FlatVector::GetData<int64_t>(out.data[0]);
        auto *cnt_data = FlatVector::GetData<int64_t>(out.data[1]);
        for (idx_t i = 0; i < max_rows; ++i) {
            uid_data[i] = g.sorted_rows[i].UserID;
            cnt_data[i] = g.sorted_rows[i].cnt;
        }
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

} 

extern "C" {
    DUCKDB_CPP_EXTENSION_ENTRY(dbweaver, loader) {
        duckdb::LoadInternal(loader);
    }
}
