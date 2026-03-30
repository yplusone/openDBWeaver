/*
query_template: SELECT UserID FROM hits WHERE UserID = 435090932899640449;

split_template: select * from dbweaver((SELECT UserID FROM hits WHERE (UserID=435090932899640449)));
query_example: SELECT UserID FROM hits WHERE UserID = 435090932899640449;

split_query: select * from dbweaver((SELECT UserID FROM hits WHERE (UserID=435090932899640449)));
*/
#define DUCKDB_EXTENSION_MAIN

#include "dbweaver_extension.hpp"
#include "duckdb.hpp"
#include "duckdb/function/table_function.hpp"
#include <atomic>
#include <limits>
#include <mutex>

namespace duckdb {

//TODO: Define any helper structs or functions needed for binding/execution

struct FnBindData : public FunctionData {
    unique_ptr<FunctionData> Copy() const override { return make_uniq<FnBindData>(); }
    bool Equals(const FunctionData &) const override { return true; }
};

struct FnGlobalState : public GlobalTableFunctionState {
    // TODO: Optional accumulators/counters (generator may append to this struct)
    std::mutex lock;
    std::atomic<idx_t> active_local_states {0};
    std::atomic<idx_t> merged_local_states {0};
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
    // Output schema definition
    names.push_back("UserID");
    return_types.push_back(LogicalType::BIGINT);

    return make_uniq<FnBindData>();
}
static OperatorResultType FnExecute(ExecutionContext &, TableFunctionInput &in,
                                                DataChunk &input, DataChunk &output) {

    auto &l = in.local_state->Cast<FnLocalState>();

    
    if (input.size() == 0) {       
        return OperatorResultType::NEED_MORE_INPUT;
    }

    // Unpack input columns using UnifiedVectorFormat
    UnifiedVectorFormat UserID_uvf;
    input.data[0].ToUnifiedFormat(input.size(), UserID_uvf);
    int64_t* UserID_ptr = (int64_t*)UserID_uvf.data;

    // Validity bitmaps
    auto &valid_UserID = UserID_uvf.validity;
    const bool UserID_all_valid = valid_UserID.AllValid();

    // Create selection vector to store valid rows after filtering
    SelectionVector sel_vector(STANDARD_VECTOR_SIZE);
    idx_t selected_count = 0;

    // Process rows
    if (UserID_all_valid) {
        // Fast path: no per-row NULL checks
        for (idx_t row_idx = 0; row_idx < input.size(); ++row_idx) {
            idx_t i_UserID = UserID_uvf.sel->get_index(row_idx);
            int64_t v_UserID = UserID_ptr[i_UserID];

            // <<CORE_COMPUTE>>
            // Apply filter: UserID = CAST(435090932899640449 AS BIGINT)
            if (v_UserID == 435090932899640449LL) {
                sel_vector.set_index(selected_count++, row_idx);
            }
        }
    } else {
        // Slow path: at least one column may contain NULLs
        for (idx_t row_idx = 0; row_idx < input.size(); ++row_idx) {
            idx_t i_UserID = UserID_uvf.sel->get_index(row_idx);

            if (!UserID_all_valid && !valid_UserID.RowIsValid(i_UserID)) {
                continue; // row is NULL in column UserID -> skip
            }

            int64_t v_UserID = UserID_ptr[i_UserID];

            // <<CORE_COMPUTE>>
            // Apply filter: UserID = CAST(435090932899640449 AS BIGINT)
            if (v_UserID == 435090932899640449LL) {
                sel_vector.set_index(selected_count++, row_idx);
            }
        }
    }
    // Write output: output chunk should have (selected_count) rows, one column (UserID)
    if (selected_count > 0) {
        // Output is always one column: UserID BIGINT
        Vector &out_vec = output.data[0];
        out_vec.SetVectorType(VectorType::FLAT_VECTOR);
        auto out_ptr = FlatVector::GetData<int64_t>(out_vec);
        auto &out_validity = FlatVector::Validity(out_vec);
        if (UserID_all_valid) {
            for (idx_t i = 0; i < selected_count; ++i) {
                idx_t row_idx = sel_vector.get_index(i);
                idx_t i_UserID = UserID_uvf.sel->get_index(row_idx);
                out_ptr[i] = UserID_ptr[i_UserID];
                out_validity.Set(i, true);
            }
        } else {
            for (idx_t i = 0; i < selected_count; ++i) {
                idx_t row_idx = sel_vector.get_index(i);
                idx_t i_UserID = UserID_uvf.sel->get_index(row_idx);
                if (!UserID_all_valid && !valid_UserID.RowIsValid(i_UserID)) {
                    out_validity.Set(i, false);
                    out_ptr[i] = 0; // or garbage
                } else {
                    out_ptr[i] = UserID_ptr[i_UserID];
                    out_validity.Set(i, true);
                }
            }
        }
        output.SetCardinality(selected_count);
    } else {
        output.SetCardinality(0);
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
        }
        l.merged = true;
        g.merged_local_states.fetch_add(1, std::memory_order_relaxed);
    }
    // No additional results emitted during finalize for SELECT-only queries
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