/*
query_template: SELECT MobilePhoneModel, COUNT(DISTINCT UserID) AS u FROM hits WHERE MobilePhoneModel <> '' GROUP BY MobilePhoneModel ORDER BY u DESC LIMIT 10;

split_template: select * from dbweaver((SELECT MobilePhoneModel, UserID FROM hits WHERE (MobilePhoneModel!='')));
query_example: SELECT MobilePhoneModel, COUNT(DISTINCT UserID) AS u FROM hits WHERE MobilePhoneModel <> '' GROUP BY MobilePhoneModel ORDER BY u DESC LIMIT 10;

split_query: select * from dbweaver((SELECT MobilePhoneModel, UserID FROM hits WHERE (MobilePhoneModel!='')));
*/
#define DUCKDB_EXTENSION_MAIN

#include "dbweaver_extension.hpp"
#include "duckdb.hpp"
#include "duckdb/function/table_function.hpp"
#include <atomic>
#include <limits>
#include <mutex>
#include <unordered_map>
#include <functional>
#include <algorithm>
#include <vector>
//TODO: Add more includes as needed

namespace duckdb {

//TODO: Define any helper structs or functions needed for binding/execution

struct FnBindData : public FunctionData {
    unique_ptr<FunctionData> Copy() const override { return make_uniq<FnBindData>(); }
    bool Equals(const FunctionData &) const override { return true; }
};

struct GroupKey {
    string_t mobile_phone_model;
    
    bool operator==(const GroupKey& other) const {
        return mobile_phone_model == other.mobile_phone_model;
    }
};

struct GroupKeyHash {
    size_t operator()(const GroupKey& k) const {
        size_t h = 0;
        h ^= duckdb::Hash(k.mobile_phone_model.GetData(), k.mobile_phone_model.GetSize()) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

struct AggState {
    std::unordered_set<int64_t> distinct_user_ids;
};

struct SortKeyView {
    int64_t u;
    bool u_is_null;
};

struct SortRowComparator {
    bool operator()(const SortKeyView &a, const SortKeyView &b) const {
        // Sort by u in DESC order
        if (a.u_is_null && b.u_is_null) return false;  // Both null, preserve order
        if (a.u_is_null) return false;  // b comes first (not null)
        if (b.u_is_null) return true;   // a comes first (not null)
        if (a.u != b.u) return a.u > b.u;  // Descending order
        return false;  // Equal values
    }
};

struct SortState {
    std::vector<SortKeyView> buffer;
    std::vector<string_t> mobile_phone_models;
    bool sorted = false;
    
    inline void AddRow(int64_t u_val, bool u_is_null, string_t mobile_phone_model) {
        buffer.push_back({u_val, u_is_null});
        mobile_phone_models.push_back(mobile_phone_model);
    }
    
    inline void SortNow() {
        if (!sorted) {
            std::sort(buffer.begin(), buffer.end(), SortRowComparator{});
            sorted = true;
        }
    }
};

struct FnGlobalState : public GlobalTableFunctionState {
    // TODO: Optional accumulators/counters (generator may append to this struct)
    std::mutex lock;
    std::atomic<idx_t> active_local_states {0};
    std::atomic<idx_t> merged_local_states {0};
    std::unordered_map<GroupKey, AggState, GroupKeyHash> agg_map;
    SortState sort_state;
    idx_t MaxThreads() const override {
        return std::numeric_limits<idx_t>::max();
    }
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
    // Set up return types and names according to the output schema
    names.push_back("MobilePhoneModel");
    return_types.push_back(LogicalType::VARCHAR);
    
    names.push_back("u");
    return_types.push_back(LogicalType::BIGINT);

    return make_uniq<FnBindData>();
}

static OperatorResultType FnExecute(ExecutionContext &, TableFunctionInput &in,
                                                    DataChunk &input, DataChunk &) {
    auto &l = in.local_state->Cast<FnLocalState>();

    
    if (input.size() == 0) {       
        return OperatorResultType::NEED_MORE_INPUT;
    }

    // Read input columns using UnifiedVectorFormat
    UnifiedVectorFormat mobile_phone_model_uvf;
    input.data[0].ToUnifiedFormat(input.size(), mobile_phone_model_uvf);
    string_t* mobile_phone_model_ptr = (string_t*)mobile_phone_model_uvf.data;

    UnifiedVectorFormat user_id_uvf;
    input.data[1].ToUnifiedFormat(input.size(), user_id_uvf);
    int64_t* user_id_ptr = (int64_t*)user_id_uvf.data;

    // Validity bitmaps
    auto &valid_mobile_phone_model = mobile_phone_model_uvf.validity;
    auto &valid_user_id = user_id_uvf.validity;

    const bool mobile_phone_model_all_valid = valid_mobile_phone_model.AllValid();
    const bool user_id_all_valid = valid_user_id.AllValid();

    // Process the input chunk row by row
    if (mobile_phone_model_all_valid && user_id_all_valid) {
        // Fast path: no per-row NULL checks
        for (idx_t row_idx = 0; row_idx < input.size(); ++row_idx) {
            idx_t i_mobile_phone_model = mobile_phone_model_uvf.sel->get_index(row_idx);
            idx_t i_user_id = user_id_uvf.sel->get_index(row_idx);

            string_t v_mobile_phone_model = mobile_phone_model_ptr[i_mobile_phone_model];
            int64_t v_user_id = user_id_ptr[i_user_id];

            // Apply filter: MobilePhoneModel <> ''
            if (v_mobile_phone_model.GetSize() == 0) {
                continue; // Skip rows where MobilePhoneModel is empty string
            }

            // Update aggregation state
            GroupKey key;
            key.mobile_phone_model = v_mobile_phone_model;
            auto &agg_state = l.agg_map[key];
            agg_state.distinct_user_ids.insert(v_user_id);
        }
    } else {
        // Slow path: at least one column may contain NULLs
        for (idx_t row_idx = 0; row_idx < input.size(); ++row_idx) {
            idx_t i_mobile_phone_model = mobile_phone_model_uvf.sel->get_index(row_idx);
            idx_t i_user_id = user_id_uvf.sel->get_index(row_idx);

            if (!mobile_phone_model_all_valid && !valid_mobile_phone_model.RowIsValid(i_mobile_phone_model)) {
                continue; // row is NULL in MobilePhoneModel → skip
            }
            if (!user_id_all_valid && !valid_user_id.RowIsValid(i_user_id)) {
                continue; // row is NULL in UserID → skip
            }

            string_t v_mobile_phone_model = mobile_phone_model_ptr[i_mobile_phone_model];
            int64_t v_user_id = user_id_ptr[i_user_id];

            // Apply filter: MobilePhoneModel <> ''
            if (v_mobile_phone_model.GetSize() == 0) {
                continue; // Skip rows where MobilePhoneModel is empty string
            }

            // Update aggregation state
            GroupKey key;
            key.mobile_phone_model = v_mobile_phone_model;
            auto &agg_state = l.agg_map[key];
            agg_state.distinct_user_ids.insert(v_user_id);
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
            // Merge local state with global state
            for (const auto &entry : l.agg_map) {
                const GroupKey &key = entry.first;
                const AggState &local_agg_state = entry.second;
                auto &global_agg_state = g.agg_map[key];
                global_agg_state.distinct_user_ids.insert(
                    local_agg_state.distinct_user_ids.begin(),
                    local_agg_state.distinct_user_ids.end()
                );
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
            // Get result from global state and populate sort buffer
            for (const auto &entry : g.agg_map) {
                const GroupKey &key = entry.first;
                const AggState &agg_state = entry.second;
                
                int64_t count = agg_state.distinct_user_ids.size();
                bool is_null = false;  // COUNT(DISTINCT ...) is never NULL when set
                g.sort_state.AddRow(count, is_null, key.mobile_phone_model);
            }
            
            // Sort the results by u in descending order
            g.sort_state.SortNow();
            
            // Output the sorted results
            idx_t output_idx = 0;
            idx_t total_rows = g.sort_state.buffer.size();
            for (idx_t i = 0; i < total_rows; ++i) {
                const auto &sort_key = g.sort_state.buffer[i];
                const auto &mobile_phone_model = g.sort_state.mobile_phone_models[i];
                
                out.SetValue(0, output_idx, mobile_phone_model);
                out.SetValue(1, output_idx, Value::BIGINT(sort_key.u));
                output_idx++;
            }
            out.SetCardinality(output_idx);
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