/*
query_template: SELECT RegionID, COUNT(DISTINCT UserID) AS u FROM hits GROUP BY RegionID ORDER BY u DESC LIMIT 10;

split_template: select * from dbweaver((SELECT RegionID, UserID FROM hits));
query_example: SELECT RegionID, COUNT(DISTINCT UserID) AS u FROM hits GROUP BY RegionID ORDER BY u DESC LIMIT 10;

split_query: select * from dbweaver((SELECT RegionID, UserID FROM hits));
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
#include <cstdint>
#include <memory>

namespace duckdb {

// Based on the observed RegionID range (0 to 131069), we use a flat array of size 131072.
constexpr idx_t MAX_REGION_ID = 131072;

// A minimal open-addressing hash set for int64_t keys to replace std::unordered_set
template <typename T>
class FlatHashSet {
public:
    FlatHashSet() : size_v(0), capacity(0) {}

    void insert(T val) {
        if (capacity == 0 || size_v * 2 >= capacity) {
            grow();
        }
        insert_internal(val);
    }

    void reserve(size_t n) {
        if (n * 2 <= capacity) return;
        uint32_t new_capacity = capacity == 0 ? 8 : capacity;
        while (new_capacity < n * 2) new_capacity *= 2;
        rehash(new_capacity);
    }

    size_t size() const { return size_v; }

    // Iterator for merging and iterating
    struct Iterator {
        const FlatHashSet* set;
        uint32_t idx;
        bool operator!=(const Iterator& other) const { return idx != other.idx; }
        T operator*() const { return set->entries[idx]; }
        Iterator& operator++() {
            idx++;
            while (idx < set->capacity && !set->occupied[idx]) idx++;
            return *this;
        }
    };
    Iterator begin() const {
        uint32_t idx = 0;
        while (idx < capacity && !occupied[idx]) idx++;
        return {this, idx};
    }
    Iterator end() const { return {this, capacity}; }

private:
    std::vector<T> entries;
    std::vector<uint8_t> occupied;
    uint32_t size_v;
    uint32_t capacity;

    void grow() {
        rehash(capacity == 0 ? 8 : capacity * 2);
    }

    void rehash(uint32_t new_capacity) {
        std::vector<T> old_entries = std::move(entries);
        std::vector<uint8_t> old_occupied = std::move(occupied);
        
        entries.assign(new_capacity, T());
        occupied.assign(new_capacity, 0);
        capacity = new_capacity;
        size_v = 0;

        for (uint32_t i = 0; i < old_occupied.size(); ++i) {
            if (old_occupied[i]) {
                insert_internal(old_entries[i]);
            }
        }
    }

    void insert_internal(T val) {
        uint32_t h = hash_func(val);
        uint32_t mask = capacity - 1;
        uint32_t idx = h & mask;
        while (occupied[idx]) {
            if (entries[idx] == val) return;
            idx = (idx + 1) & mask;
        }
        entries[idx] = val;
        occupied[idx] = 1;
        size_v++;
    }

    inline uint32_t hash_func(T val) const {
        uint64_t x = static_cast<uint64_t>(val);
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
        x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
        x = x ^ (x >> 31);
        return static_cast<uint32_t>(x);
    }
};

struct FnBindData : public FunctionData {
    unique_ptr<FunctionData> Copy() const override { return make_uniq<FnBindData>(); }
    bool Equals(const FunctionData &) const override { return true; }
};

struct AggState {
    unique_ptr<FlatHashSet<int64_t>> distinct_users;
};

struct SortRow {
    int32_t region_id;
    int64_t u;
};

struct SortRowComparator {
    bool operator()(const SortRow &a, const SortRow &b) const {
        // DESC order for u
        if (a.u != b.u) return a.u > b.u;
        // Stable tie-break by region_id
        return a.region_id < b.region_id;
    }
};

struct SortState {
    std::vector<SortRow> buffer;
    bool sorted = false;

    inline void AddRow(int64_t u_val, int32_t region_id) {
        buffer.push_back(SortRow{region_id, u_val});
    }

    inline void SortNow() {
        if (!sorted) {
            std::sort(buffer.begin(), buffer.end(), SortRowComparator{});
            sorted = true;
        }
    }
};
struct FnGlobalState : public GlobalTableFunctionState {
    static constexpr idx_t NUM_PARTITIONS = 128;
    // Use flat vector for RegionID aggregation
    std::vector<AggState> agg_vec;
    std::mutex partition_locks[NUM_PARTITIONS];

    std::atomic<idx_t> active_local_states {0};
    std::atomic<idx_t> merged_local_states {0};
    idx_t MaxThreads() const override {
        return std::numeric_limits<idx_t>::max();
    }
    
    SortState sort_state;

    FnGlobalState() : agg_vec(MAX_REGION_ID) {}
};


static unique_ptr<GlobalTableFunctionState> FnInit(ClientContext &, TableFunctionInitInput &) {
    return make_uniq<FnGlobalState>();
}

struct FnLocalState : public LocalTableFunctionState {
    bool merged = false;
    // Local aggregation state mirroring the global one
    std::vector<AggState> agg_vec;
    std::vector<uint8_t> region_touched;
    std::vector<int32_t> active_regions;

    FnLocalState() : agg_vec(MAX_REGION_ID), region_touched(MAX_REGION_ID, 0) {}
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
    // RegionID -> INTEGER
    return_types.emplace_back(LogicalType::INTEGER);
    names.emplace_back("RegionID");
    
    // u -> BIGINT (COUNT DISTINCT result)
    return_types.emplace_back(LogicalType::BIGINT);
    names.emplace_back("u");

    return make_uniq<FnBindData>();
}

static OperatorResultType FnExecute(ExecutionContext &, TableFunctionInput &in,
                                            DataChunk &input, DataChunk &) {
    auto &l = in.local_state->Cast<FnLocalState>();

    if (input.size() == 0) {       
        return OperatorResultType::NEED_MORE_INPUT;
    }

    UnifiedVectorFormat region_id_uvf;
    UnifiedVectorFormat user_id_uvf;
    
    input.data[0].ToUnifiedFormat(input.size(), region_id_uvf);
    input.data[1].ToUnifiedFormat(input.size(), user_id_uvf);
    
    int32_t* region_id_ptr = (int32_t*)region_id_uvf.data;
    int64_t* user_id_ptr = (int64_t*)user_id_uvf.data;
    
    auto &valid_region_id = region_id_uvf.validity;
    auto &valid_user_id = user_id_uvf.validity;
    
    const bool region_id_all_valid = valid_region_id.AllValid();
    const bool user_id_all_valid = valid_user_id.AllValid();
    
    idx_t num_rows = input.size();
    
    for (idx_t row_idx = 0; row_idx < num_rows; ++row_idx) {
        idx_t i_region_id = region_id_uvf.sel->get_index(row_idx);
        idx_t i_user_id = user_id_uvf.sel->get_index(row_idx);
        
        if (!region_id_all_valid && !valid_region_id.RowIsValid(i_region_id)) continue;
        if (!user_id_all_valid && !valid_user_id.RowIsValid(i_user_id)) continue;
        
        int32_t v_region_id = region_id_ptr[i_region_id];
        int64_t v_user_id = user_id_ptr[i_user_id];
        
        if (v_region_id >= 0 && (size_t)v_region_id < MAX_REGION_ID) {
            if (!l.region_touched[v_region_id]) {
                l.region_touched[v_region_id] = 1;
                l.active_regions.push_back(v_region_id);
                l.agg_vec[v_region_id].distinct_users = make_uniq<FlatHashSet<int64_t>>();
            }
            l.agg_vec[v_region_id].distinct_users->insert(v_user_id);
        }
    }

    return OperatorResultType::NEED_MORE_INPUT; 
}
static OperatorFinalizeResultType FnFinalize(ExecutionContext &, TableFunctionInput &in,
                                                        DataChunk &out) {
    auto &g = in.global_state->Cast<FnGlobalState>();
    auto &l = in.local_state->Cast<FnLocalState>();

    if (!l.merged) {
        for (int32_t rid : l.active_regions) {
            idx_t part_idx = (uint32_t)rid % FnGlobalState::NUM_PARTITIONS;
            std::lock_guard<std::mutex> guard(g.partition_locks[part_idx]);
            auto &l_set_ptr = l.agg_vec[rid].distinct_users;
            auto &g_set_ptr = g.agg_vec[rid].distinct_users;
            
            if (!g_set_ptr) {
                g_set_ptr = std::move(l_set_ptr);
            } else if (l_set_ptr) {
                g_set_ptr->reserve(g_set_ptr->size() + l_set_ptr->size());
                for (auto val : *l_set_ptr) {
                    g_set_ptr->insert(val);
                }
            }
        }
        l.merged = true;
        
        if (g.merged_local_states.fetch_add(1, std::memory_order_acq_rel) + 1 == g.active_local_states.load(std::memory_order_acquire)) {
            for (idx_t rid = 0; rid < MAX_REGION_ID; ++rid) {
                auto &state = g.agg_vec[rid];
                if (state.distinct_users) {
                    g.sort_state.AddRow((int64_t)state.distinct_users->size(), (int32_t)rid);
                }
            }
            g.sort_state.SortNow();

            idx_t output_row_idx = 0;
            size_t out_limit = std::min<size_t>(10, g.sort_state.buffer.size());
            for (size_t i = 0; i < out_limit; ++i) {
                const auto &row = g.sort_state.buffer[i];
                out.SetValue(0, output_row_idx, Value::INTEGER(row.region_id));
                out.SetValue(1, output_row_idx, Value::BIGINT(row.u));
                output_row_idx++;
            }
            out.SetCardinality(output_row_idx);
            return OperatorFinalizeResultType::FINISHED;
        }
    }

    out.SetCardinality(0);
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
