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
#include <unordered_set>
#include <vector>
#include <algorithm>

namespace duckdb {

struct FnBindData : public FunctionData {
    unique_ptr<FunctionData> Copy() const override { return make_uniq<FnBindData>(); }
    bool Equals(const FunctionData &) const override { return true; }
};

struct GroupKey {
    int32_t region_id;
    
    bool operator==(const GroupKey& other) const {
        return region_id == other.region_id;
    }
};

struct GroupKeyHash {
    size_t operator()(const GroupKey& k) const {
        size_t h = 0;
        h ^= std::hash<int32_t>{}(k.region_id) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

struct AggState {
    std::unordered_set<int64_t> distinct_users;
    
    AggState() {}
};
struct SortRow {
    int32_t region_id;
    int64_t u;
};

struct SortRowComparator {
    bool operator()(const SortRow &a, const SortRow &b) const {
        // DESC order for u
        if (a.u != b.u) return a.u > b.u;
        return false;
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
    std::unordered_map<GroupKey, AggState, GroupKeyHash> agg_map;
    std::mutex lock;
    std::atomic<idx_t> active_local_states {0};
    std::atomic<idx_t> merged_local_states {0};
    idx_t MaxThreads() const override {
        return std::numeric_limits<idx_t>::max();
    }
    
    SortState sort_state;
};

static unique_ptr<GlobalTableFunctionState> FnInit(ClientContext &, TableFunctionInitInput &) {
    return make_uniq<FnGlobalState>();
}

struct FnLocalState : public LocalTableFunctionState {
    //TODO: initialize local state and other preparations
    bool merged = false;
    std::unordered_map<GroupKey, AggState, GroupKeyHash> agg_map;
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
    // Define the output schema based on authoritative mapping
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

    //TODO: process input chunk and produce output
    
    // UnifiedVectorFormat handles for input columns
    UnifiedVectorFormat region_id_uvf;
    UnifiedVectorFormat user_id_uvf;
    
    // Load input columns into UVF
    input.data[0].ToUnifiedFormat(input.size(), region_id_uvf);
    input.data[1].ToUnifiedFormat(input.size(), user_id_uvf);
    
    // Create typed pointers to physical data
    int32_t* region_id_ptr = (int32_t*)region_id_uvf.data;
    int64_t* user_id_ptr = (int64_t*)user_id_uvf.data;
    
    // Validity bitmaps for NULL checking
    auto &valid_region_id = region_id_uvf.validity;
    auto &valid_user_id = user_id_uvf.validity;
    
    const bool region_id_all_valid = valid_region_id.AllValid();
    const bool user_id_all_valid = valid_user_id.AllValid();
    
    idx_t num_rows = input.size();
    
    // Fast branch: all relevant columns have no NULLs in this batch
    if (region_id_all_valid && user_id_all_valid) {
        // --- Fast path: no per-row NULL checks ---
        for (idx_t row_idx = 0; row_idx < num_rows; ++row_idx) {
            // Directly load values without RowIsValid checks
            idx_t i_region_id = region_id_uvf.sel->get_index(row_idx);
            idx_t i_user_id = user_id_uvf.sel->get_index(row_idx);
            
            int32_t v_region_id = region_id_ptr[i_region_id];
            int64_t v_user_id = user_id_ptr[i_user_id];
            
            // ======================================
            //  Core computation logic (no NULLs)
            GroupKey key;
            key.region_id = v_region_id;
            
            auto &state = l.agg_map[key];
            state.distinct_users.insert(v_user_id);
            //<<CORE_COMPUTE>>
            // ============================
        }
    } else {
        // --- Slow path: at least one column may contain NULLs ---
        for (idx_t row_idx = 0; row_idx < num_rows; ++row_idx) {
            // For each column that is not fully valid, check this row
            idx_t i_region_id = region_id_uvf.sel->get_index(row_idx);
            idx_t i_user_id = user_id_uvf.sel->get_index(row_idx);
            
            if (!region_id_all_valid && !valid_region_id.RowIsValid(i_region_id)) {
                continue; // row is NULL in column RegionID -> skip
            }
            if (!user_id_all_valid && !valid_user_id.RowIsValid(i_user_id)) {
                continue; // row is NULL in column UserID -> skip
            }
            
            // At this point, all required columns are valid for this row
            int32_t v_region_id = region_id_ptr[i_region_id];
            int64_t v_user_id = user_id_ptr[i_user_id];
            
            // ======================================
            //  Core computation logic (NULL-safe)
            GroupKey key;
            key.region_id = v_region_id;
            
            auto &state = l.agg_map[key];
            state.distinct_users.insert(v_user_id);
            //<<CORE_COMPUTE>>
            // ======================================
        }
    }

    return OperatorResultType::NEED_MORE_INPUT; 
}

static OperatorFinalizeResultType FnFinalize(ExecutionContext &, TableFunctionInput &in,
                                                        DataChunk &out) {
    auto &g = *reinterpret_cast<FnGlobalState *>(in.global_state.get());
    auto &l = in.local_state->Cast<FnLocalState>();
    // Merge the local state into the global state exactly once.
    if (!l.merged) {
        {
            std::lock_guard<std::mutex> guard(g.lock);
            //TODO: merge local state with global state
            for (const auto &entry : l.agg_map) {
                const GroupKey &key = entry.first;
                const AggState &state = entry.second;
                auto &global_state = g.agg_map[key];
                for (const auto &user_id : state.distinct_users) {
                    global_state.distinct_users.insert(user_id);
                }
            }
        }
        l.merged = true;
        g.merged_local_states.fetch_add(1, std::memory_order_relaxed);
    }

    // Only the *last* local state to merge emits the final result.
    // All other threads return FINISHED with an empty chunk.
    const auto merged = g.merged_local_states.load(std::memory_order_relaxed);
    const auto active = g.active_local_states.load(std::memory_order_relaxed);
    if (active > 0 && merged == active) {
        {
            std::lock_guard<std::mutex> guard(g.lock);
            //TODO: get result from global state
            // Build sort buffer
            for (const auto &entry : g.agg_map) {
                const GroupKey &key = entry.first;
                const AggState &state = entry.second;
                
                int64_t count_distinct = static_cast<int64_t>(state.distinct_users.size());
                
                g.sort_state.AddRow(count_distinct, key.region_id);
            }
            
            // Sort the data
            g.sort_state.SortNow();
            // Output sorted data with LIMIT 10
            idx_t output_row_idx = 0;
            size_t out_limit = std::min<size_t>(10, g.sort_state.buffer.size());
            for (size_t i = 0; i < out_limit; ++i) {
                const auto &row = g.sort_state.buffer[i];
                out.SetValue(0, output_row_idx, Value::INTEGER(row.region_id));
                out.SetValue(1, output_row_idx, Value::BIGINT(row.u));
                output_row_idx++;
            }
            out.SetCardinality(output_row_idx);


        }
        //TODO: populate out chunk with final results
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