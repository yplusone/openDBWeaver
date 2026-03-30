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
#include <atomic>
#include <limits>
#include <mutex>
#include <unordered_set>
//TODO: Add more includes as needed

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
    
    // Accumulator for COUNT(DISTINCT SearchPhrase)
    std::unordered_set<string_t> distinct_values;
};

static unique_ptr<GlobalTableFunctionState> FnInit(ClientContext &, TableFunctionInitInput &) {
    return make_uniq<FnGlobalState>();
}

struct FnLocalState : public LocalTableFunctionState {
    //TODO: initialize local state and other preparations
    bool merged = false;
    
    // Accumulator for COUNT(DISTINCT SearchPhrase)
    std::unordered_set<string_t> distinct_values;
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
    // Output schema: cnt_distinct_searchphrase as BIGINT
    return_types.emplace_back(LogicalType::BIGINT);
    names.emplace_back("cnt_distinct_searchphrase");

    return make_uniq<FnBindData>();
}

static OperatorResultType FnExecute(ExecutionContext &, TableFunctionInput &in,
                                                DataChunk &input, DataChunk &) {
    auto &l = in.local_state->Cast<FnLocalState>();

    
    if (input.size() == 0) {       
        return OperatorResultType::NEED_MORE_INPUT;
    }

    // Unpack input columns using UnifiedVectorFormat
    UnifiedVectorFormat SearchPhrase_uvf;
    input.data[0].ToUnifiedFormat(input.size(), SearchPhrase_uvf);
    string_t* SearchPhrase_ptr = (string_t*)SearchPhrase_uvf.data;

    // validity bitmaps
    auto &valid_SearchPhrase = SearchPhrase_uvf.validity;
    const bool SearchPhrase_all_valid = valid_SearchPhrase.AllValid();

    // Process rows
    if (SearchPhrase_all_valid) {
        // --- Fast path: no per-row NULL checks ---
        for (idx_t row_idx = 0; row_idx < input.size(); ++row_idx) {
            idx_t i_SearchPhrase = SearchPhrase_uvf.sel->get_index(row_idx);
            string_t v_SearchPhrase = SearchPhrase_ptr[i_SearchPhrase];

            // ======================================
            //  Core computation logic (no NULLs)
            l.distinct_values.insert(v_SearchPhrase);
            //<<CORE_COMPUTE>>
            // ============================
        }
    } else {
        // --- Slow path: at least one column may contain NULLs ---
        for (idx_t row_idx = 0; row_idx < input.size(); ++row_idx) {
            idx_t i_SearchPhrase = SearchPhrase_uvf.sel->get_index(row_idx);

            if (!SearchPhrase_all_valid && !valid_SearchPhrase.RowIsValid(i_SearchPhrase)) {
                continue; // row is NULL in column SearchPhrase → skip
            }

            // At this point, all required columns are valid for this row

            string_t v_SearchPhrase = SearchPhrase_ptr[i_SearchPhrase];

            // ======================================
            //  Core computation logic (NULL-safe)
            l.distinct_values.insert(v_SearchPhrase);
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
            // Merge distinct values from local state to global state
            for (const auto& value : l.distinct_values) {
                g.distinct_values.insert(value);
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
        idx_t final_count;
        {
            std::lock_guard<std::mutex> guard(g.lock);
            final_count = g.distinct_values.size();
        }
        out.SetCardinality(1);
        out.SetValue(0, 0, Value::BIGINT(final_count));
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