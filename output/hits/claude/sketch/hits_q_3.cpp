/*
query_template: SELECT SUM(AdvEngineID) AS sum_advengineid, COUNT(*) AS cnt, AVG(ResolutionWidth) AS avg_resolutionwidth FROM hits;

split_template: select * from dbweaver((SELECT AdvEngineID, ResolutionWidth FROM hits));
query_example: SELECT SUM(AdvEngineID) AS sum_advengineid, COUNT(*) AS cnt, AVG(ResolutionWidth) AS avg_resolutionwidth FROM hits;

split_query: select * from dbweaver((SELECT AdvEngineID, ResolutionWidth FROM hits));
*/
#define DUCKDB_EXTENSION_MAIN

#include "dbweaver_extension.hpp"
#include "duckdb.hpp"
#include "duckdb/function/table_function.hpp"
#include <atomic>
#include <limits>
#include <mutex>

namespace duckdb {

struct FnBindData : public FunctionData {
    unique_ptr<FunctionData> Copy() const override { return make_uniq<FnBindData>(); }
    bool Equals(const FunctionData &) const override { return true; }
};

struct FnGlobalState : public GlobalTableFunctionState {
    std::mutex lock;
    std::atomic<idx_t> active_local_states {0};
    std::atomic<idx_t> merged_local_states {0};
    idx_t MaxThreads() const override {
        return std::numeric_limits<idx_t>::max();
    }
    
    // Aggregation accumulators for single-group aggregate
    int64_t sum_advengineid = 0;
    idx_t cnt = 0;
    double sum_resolutionwidth = 0.0;
};

static unique_ptr<GlobalTableFunctionState> FnInit(ClientContext &, TableFunctionInitInput &) {
    return make_uniq<FnGlobalState>();
}

struct FnLocalState : public LocalTableFunctionState {
    // Aggregation accumulators for single-group aggregate
    int64_t sum_advengineid = 0;
    idx_t cnt = 0;
    double sum_resolutionwidth = 0.0;
    
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
    //TODO: populate return_types and names
    return_types.push_back(LogicalType::HUGEINT);  // sum_advengineid
    return_types.push_back(LogicalType::BIGINT); // cnt
    return_types.push_back(LogicalType::DOUBLE);  // avg_resolutionwidth
    
    names.push_back("sum_advengineid");
    names.push_back("cnt");
    names.push_back("avg_resolutionwidth");

    return make_uniq<FnBindData>();
}

static OperatorResultType FnExecute(ExecutionContext &, TableFunctionInput &in,
                                    DataChunk &input, DataChunk &) {
    auto &l = in.local_state->Cast<FnLocalState>();

    
    if (input.size() == 0) {       
        return OperatorResultType::NEED_MORE_INPUT;
    }

    //TODO: process input chunk and produce output
    
    // Setup UnifiedVectorFormat for input columns
    UnifiedVectorFormat AdvEngineID_uvf;
    input.data[0].ToUnifiedFormat(input.size(), AdvEngineID_uvf);
    int16_t* AdvEngineID_ptr = (int16_t*)AdvEngineID_uvf.data;
    
    UnifiedVectorFormat ResolutionWidth_uvf;
    input.data[1].ToUnifiedFormat(input.size(), ResolutionWidth_uvf);
    int16_t* ResolutionWidth_ptr = (int16_t*)ResolutionWidth_uvf.data;
    
    // Validity bitmaps
    auto &valid_AdvEngineID = AdvEngineID_uvf.validity;
    auto &valid_ResolutionWidth = ResolutionWidth_uvf.validity;
    
    const bool AdvEngineID_all_valid = valid_AdvEngineID.AllValid();
    const bool ResolutionWidth_all_valid = valid_ResolutionWidth.AllValid();
    
    idx_t num_rows = input.size();
    
    // Fast branch: all relevant columns have no NULLs in this batch
    if (AdvEngineID_all_valid && ResolutionWidth_all_valid) {
        // --- Fast path: no per-row NULL checks ---
        for (idx_t row_idx = 0; row_idx < num_rows; ++row_idx) {
            idx_t i_AdvEngineID = AdvEngineID_uvf.sel->get_index(row_idx);
            idx_t i_ResolutionWidth = ResolutionWidth_uvf.sel->get_index(row_idx);
            
            int16_t v1 = AdvEngineID_ptr[i_AdvEngineID];
            int16_t v2 = ResolutionWidth_ptr[i_ResolutionWidth];
            
            // ======================================
            //  Core computation logic (no NULLs)
            l.sum_advengineid += v1;
            l.cnt++;
            l.sum_resolutionwidth += v2;
            // ============================
        }
    } else {
        // --- Slow path: at least one column may contain NULLs ---
        for (idx_t row_idx = 0; row_idx < num_rows; ++row_idx) {
            idx_t i_AdvEngineID = AdvEngineID_uvf.sel->get_index(row_idx);
            idx_t i_ResolutionWidth = ResolutionWidth_uvf.sel->get_index(row_idx);
            
            if (!AdvEngineID_all_valid && !valid_AdvEngineID.RowIsValid(i_AdvEngineID)) {
                continue; // row is NULL in column AdvEngineID → skip
            }
            if (!ResolutionWidth_all_valid && !valid_ResolutionWidth.RowIsValid(i_ResolutionWidth)) {
                continue;
            }
            
            int16_t v1 = AdvEngineID_ptr[i_AdvEngineID];
            int16_t v2 = ResolutionWidth_ptr[i_ResolutionWidth];
            
            // ======================================
            //  Core computation logic (NULL-safe)
            l.sum_advengineid += v1;
            l.cnt++;
            l.sum_resolutionwidth += v2;
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
            g.sum_advengineid += l.sum_advengineid;
            g.cnt += l.cnt;
            g.sum_resolutionwidth += l.sum_resolutionwidth;
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
        }
        //TODO: populate out chunk with final results
        out.SetCardinality(1);
        hugeint_t sum_advengineid_hugeint;
        sum_advengineid_hugeint.lower = static_cast<uint64_t>(g.sum_advengineid);
        sum_advengineid_hugeint.upper = static_cast<int64_t>(g.sum_advengineid >> 64);
        out.SetValue(0, 0, Value::HUGEINT(sum_advengineid_hugeint));
        out.SetValue(1, 0, Value::BIGINT(g.cnt));
        double avg_res_width = (g.cnt > 0) ? (g.sum_resolutionwidth / g.cnt) : 0.0;
        out.SetValue(2, 0, Value::DOUBLE(avg_res_width));
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