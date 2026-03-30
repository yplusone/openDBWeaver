/*
query_template: SELECT RegionID, SUM(AdvEngineID) AS sum_advengineid, COUNT(*) AS c, AVG(ResolutionWidth) AS avg_resolutionwidth, COUNT(DISTINCT UserID) AS cnt_distinct_userid FROM hits GROUP BY RegionID ORDER BY c DESC LIMIT 10;

split_template: select * from dbweaver((SELECT RegionID, AdvEngineID, ResolutionWidth, UserID FROM hits));
query_example: SELECT RegionID, SUM(AdvEngineID) AS sum_advengineid, COUNT(*) AS c, AVG(ResolutionWidth) AS avg_resolutionwidth, COUNT(DISTINCT UserID) AS cnt_distinct_userid FROM hits GROUP BY RegionID ORDER BY c DESC LIMIT 10;

split_query: select * from dbweaver((SELECT RegionID, AdvEngineID, ResolutionWidth, UserID FROM hits));
*/
#define DUCKDB_EXTENSION_MAIN

#include "dbweaver_extension.hpp"
#include "duckdb.hpp"
#include "duckdb/function/table_function.hpp"
#include <atomic>
#include <limits>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <unordered_set>

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
    int64_t sum_advengineid = 0;
    int64_t count_total = 0;
    int64_t sum_resolutionwidth = 0; // for AVG calculation
    std::unordered_set<int64_t> distinct_userids;
};

struct SortKeyView {
    int64_t c;  // The field we're sorting on (count_total)
};

struct SortRowComparator {
    bool operator()(const SortKeyView &a, const SortKeyView &b) const {
        // DESC order for 'c'
        if (a.c != b.c) return a.c > b.c;  // Descending order
        return false; // Equal case
    }
};

struct SortState {
    std::vector<SortKeyView> key_buffer;
    std::vector<GroupKey> group_key_buffer;
    std::vector<AggState> agg_state_buffer;
    bool sorted = false;

    inline void AddRow(const GroupKey &key, const AggState &state) {
        key_buffer.push_back(SortKeyView{state.count_total});
        group_key_buffer.push_back(key);
        agg_state_buffer.push_back(state);
    }

    inline void SortNow() {
        if (!sorted) {
            std::sort(key_buffer.begin(), key_buffer.end(), SortRowComparator{});
            sorted = true;
        }
    }
};

struct FnGlobalState : public GlobalTableFunctionState {
    std::unordered_map<GroupKey, AggState, GroupKeyHash> agg_map;
    std::mutex lock;
    std::atomic<idx_t> active_local_states {0};
    std::atomic<idx_t> merged_local_states {0};
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
    std::unordered_map<GroupKey, AggState, GroupKeyHash> agg_map;
    bool merged = false;
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
    // Set up the return types and names according to the authoritative mapping
    return_types.emplace_back(LogicalType::INTEGER);  // RegionID
    return_types.emplace_back(LogicalType::HUGEINT);  // sum_advengineid
    return_types.emplace_back(LogicalType::BIGINT);   // c
    return_types.emplace_back(LogicalType::DOUBLE);   // avg_resolutionwidth
    return_types.emplace_back(LogicalType::BIGINT);   // cnt_distinct_userid
    
    names.emplace_back("RegionID");
    names.emplace_back("sum_advengineid");
    names.emplace_back("c");
    names.emplace_back("avg_resolutionwidth");
    names.emplace_back("cnt_distinct_userid");

    return make_uniq<FnBindData>();
}

static OperatorResultType FnExecute(ExecutionContext &, TableFunctionInput &in,
                                            DataChunk &input, DataChunk &) {
    auto &l = in.local_state->Cast<FnLocalState>();

    
    if (input.size() == 0) {       
        return OperatorResultType::NEED_MORE_INPUT;
    }

    // Unpack input columns using UnifiedVectorFormat
    UnifiedVectorFormat region_id_uvf;
    input.data[0].ToUnifiedFormat(input.size(), region_id_uvf);
    int32_t* region_id_ptr = (int32_t*)region_id_uvf.data;

    UnifiedVectorFormat adv_engine_id_uvf;
    input.data[1].ToUnifiedFormat(input.size(), adv_engine_id_uvf);
    int16_t* adv_engine_id_ptr = (int16_t*)adv_engine_id_uvf.data;

    UnifiedVectorFormat resolution_width_uvf;
    input.data[2].ToUnifiedFormat(input.size(), resolution_width_uvf);
    int16_t* resolution_width_ptr = (int16_t*)resolution_width_uvf.data;

    UnifiedVectorFormat user_id_uvf;
    input.data[3].ToUnifiedFormat(input.size(), user_id_uvf);
    int64_t* user_id_ptr = (int64_t*)user_id_uvf.data;

    // validity bitmaps
    auto &valid_region_id  = region_id_uvf.validity;
    auto &valid_adv_engine_id  = adv_engine_id_uvf.validity;
    auto &valid_resolution_width  = resolution_width_uvf.validity;
    auto &valid_user_id  = user_id_uvf.validity;
    const bool region_id_all_valid = valid_region_id.AllValid();
    const bool adv_engine_id_all_valid = valid_adv_engine_id.AllValid();
    const bool resolution_width_all_valid = valid_resolution_width.AllValid();
    const bool user_id_all_valid = valid_user_id.AllValid();

    // FAST BRANCH: all relevant columns have no NULLs in this batch
    if (region_id_all_valid && adv_engine_id_all_valid && resolution_width_all_valid && user_id_all_valid) {
        // --- Fast path: no per-row NULL checks ---
        for (idx_t row_idx = 0; row_idx < input.size(); ++row_idx) {
            // Directly load values without RowIsValid checks

            idx_t i_region_id = region_id_uvf.sel->get_index(row_idx);
            int32_t v1 = region_id_ptr[i_region_id];
            idx_t i_adv_engine_id = adv_engine_id_uvf.sel->get_index(row_idx);
            int16_t v2 = adv_engine_id_ptr[i_adv_engine_id];
            idx_t i_resolution_width = resolution_width_uvf.sel->get_index(row_idx);
            int16_t v3 = resolution_width_ptr[i_resolution_width];
            idx_t i_user_id = user_id_uvf.sel->get_index(row_idx);
            int64_t v4 = user_id_ptr[i_user_id];

            // ======================================
            //  Core computation logic (no NULLs)
            GroupKey key;
            key.region_id = v1;
            auto &state = l.agg_map[key];
            state.sum_advengineid += v2;
            state.count_total++;
            state.sum_resolutionwidth += v3;
            state.distinct_userids.insert(v4);
            //<<CORE_COMPUTE>>
            // ============================
        }
    } else {
        // --- Slow path: at least one column may contain NULLs ---
        for (idx_t row_idx = 0; row_idx < input.size(); ++row_idx) {
            // For each column that is not fully valid, check this row
            idx_t i_region_id = region_id_uvf.sel->get_index(row_idx);
            idx_t i_adv_engine_id = adv_engine_id_uvf.sel->get_index(row_idx);
            idx_t i_resolution_width = resolution_width_uvf.sel->get_index(row_idx);
            idx_t i_user_id = user_id_uvf.sel->get_index(row_idx);

            if (!region_id_all_valid && !valid_region_id.RowIsValid(i_region_id)) {
                continue; // row is NULL in column 1 → skip
            }
            if (!adv_engine_id_all_valid && !valid_adv_engine_id.RowIsValid(i_adv_engine_id)) {
                continue;
            }
            if (!resolution_width_all_valid && !valid_resolution_width.RowIsValid(i_resolution_width)) {
                continue;
            }
            if (!user_id_all_valid && !valid_user_id.RowIsValid(i_user_id)) {
                continue;
            }
            // Repeat for additional columns

            // At this point, all required columns are valid for this row

            int32_t v1 = region_id_ptr[i_region_id];
            int16_t v2 = adv_engine_id_ptr[i_adv_engine_id];
            int16_t v3 = resolution_width_ptr[i_resolution_width];
            int64_t v4 = user_id_ptr[i_user_id];

            // ======================================
            //  Core computation logic (NULL-safe)
            GroupKey key;
            key.region_id = v1;
            auto &state = l.agg_map[key];
            state.sum_advengineid += v2;
            state.count_total++;
            state.sum_resolutionwidth += v3;
            state.distinct_userids.insert(v4);
            //<<CORE_COMPUTE>>
            // ======================================
        }
    }

    //TODO: process input chunk and produce output

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
                const AggState &local_state = entry.second;
                auto &global_state = g.agg_map[key];
                global_state.sum_advengineid += local_state.sum_advengineid;
                global_state.count_total += local_state.count_total;
                global_state.sum_resolutionwidth += local_state.sum_resolutionwidth;
                for (const auto &userid : local_state.distinct_userids) {
                    global_state.distinct_userids.insert(userid);
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
    idx_t output_row_idx = 0;
    if (active > 0 && merged == active) {
        {
            std::lock_guard<std::mutex> guard(g.lock);
            // Build the sort buffers before sorting
            for (const auto &entry : g.agg_map) {
                const GroupKey &key = entry.first;
                const AggState &state = entry.second;
                g.sort_state.AddRow(key, state);
            }
            
            // Perform the sort
            g.sort_state.SortNow();
            
            // Output the sorted results
            for (size_t i = 0; i < g.sort_state.key_buffer.size(); ++i) {
                const GroupKey &key = g.sort_state.group_key_buffer[i];
                const AggState &state = g.sort_state.agg_state_buffer[i];
                
                // compute final aggregates
                double avg_resolutionwidth = state.count_total > 0 ? (double)state.sum_resolutionwidth / state.count_total : 0.0;
                int64_t cnt_distinct_userid = state.distinct_userids.size();
                
                // populate out chunk with final results
                out.SetValue(0, output_row_idx, Value::INTEGER(key.region_id));
                out.SetValue(1, output_row_idx, Value::HUGEINT(state.sum_advengineid));
                out.SetValue(2, output_row_idx, Value::BIGINT(state.count_total));
                out.SetValue(3, output_row_idx, Value::DOUBLE(avg_resolutionwidth));
                out.SetValue(4, output_row_idx, Value::BIGINT(cnt_distinct_userid));
                
                output_row_idx++;
                if(output_row_idx >= STANDARD_VECTOR_SIZE) {
                    break;
                }
            }
        }
        out.SetCardinality(output_row_idx);
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