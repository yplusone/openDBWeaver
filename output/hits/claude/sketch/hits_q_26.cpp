/*
query_template: SELECT SearchPhrase FROM hits WHERE SearchPhrase <> '' ORDER BY SearchPhrase LIMIT 10;

split_template: select * from dbweaver((SELECT SearchPhrase FROM hits WHERE (SearchPhrase!='')));
query_example: SELECT SearchPhrase FROM hits WHERE SearchPhrase <> '' ORDER BY SearchPhrase LIMIT 10;

split_query: select * from dbweaver((SELECT SearchPhrase FROM hits WHERE (SearchPhrase!='')));
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
#include <functional>

namespace duckdb {

struct SortKeyView {
    string_t search_phrase;
};

struct SortRowComparator {
    bool operator()(const SortKeyView &a, const SortKeyView &b) const {
        // Compare SearchPhrase in ascending order
        int cmp = a.search_phrase.GetString().compare(b.search_phrase.GetString());
        if (cmp != 0) {
            return cmp < 0;  // ASC order
        }
        return false; // Equal values, don't change order
    }
};

struct SortState {
    std::vector<SortKeyView> buffer;
    bool sorted = false;
    
    inline void AddRow(string_t search_phrase) {
        buffer.push_back(SortKeyView{search_phrase});
    }
    
    inline void SortNow() {
        if (!sorted) {
            std::sort(buffer.begin(), buffer.end(), SortRowComparator{});
            sorted = true;
        }
    }
};

struct FnBindData : public FunctionData {
    unique_ptr<FunctionData> Copy() const override { return make_uniq<FnBindData>(); }
    bool Equals(const FunctionData &) const override { return true; }
};

struct FnGlobalState : public GlobalTableFunctionState {
    // TODO: Optional accumulators/counters (generator may append to this struct)
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
    SortState sort_state;
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
    // Output schema for the table function
    return_types.emplace_back(LogicalType::VARCHAR);
    names.emplace_back("SearchPhrase");

    return make_uniq<FnBindData>();
}

static OperatorResultType FnExecute(ExecutionContext &, TableFunctionInput &in,
                                                    DataChunk &input, DataChunk &) {
    auto &l = in.local_state->Cast<FnLocalState>();

    
    if (input.size() == 0) {       
        return OperatorResultType::NEED_MORE_INPUT;
    }

    //TODO: process input chunk and produce output
    
    // UnifiedVectorFormat for input columns
    UnifiedVectorFormat SearchPhrase_uvf;
    input.data[0].ToUnifiedFormat(input.size(), SearchPhrase_uvf);
    string_t* SearchPhrase_ptr = (string_t*)SearchPhrase_uvf.data;

    // validity bitmaps
    auto &valid_SearchPhrase = SearchPhrase_uvf.validity;
    const bool SearchPhrase_all_valid = valid_SearchPhrase.AllValid();

    // Process input rows
    if (SearchPhrase_all_valid) {
        // --- Fast path: no per-row NULL checks ---
        for (idx_t row_idx = 0; row_idx < input.size(); ++row_idx) {
            // Directly load values without RowIsValid checks

            idx_t i_SearchPhrase = SearchPhrase_uvf.sel->get_index(row_idx);
            string_t v_SearchPhrase = SearchPhrase_ptr[i_SearchPhrase];

            // ======================================
            //  Core computation logic (no NULLs)
            // Apply filter: SearchPhrase <> CAST('' AS VARCHAR)
            if (v_SearchPhrase.GetSize() == 0) {
                continue; // Skip rows where SearchPhrase is empty string
            }
            //<<CORE_COMPUTE>>
            l.sort_state.AddRow(v_SearchPhrase);
            // ============================
        }
    } else {
        // --- Slow path: at least one column may contain NULLs ---
        for (idx_t row_idx = 0; row_idx < input.size(); ++row_idx) {
            // For each column that is not fully valid, check this row
            idx_t i_SearchPhrase = SearchPhrase_uvf.sel->get_index(row_idx);

            if (!SearchPhrase_all_valid && !valid_SearchPhrase.RowIsValid(i_SearchPhrase)) {
                continue; // row is NULL in column 1 → skip
            }

            // At this point, all required columns are valid for this row

            string_t v_SearchPhrase = SearchPhrase_ptr[i_SearchPhrase];

            // ======================================
            //  Core computation logic (NULL-safe)
            // Apply filter: SearchPhrase <> CAST('' AS VARCHAR)
            if (v_SearchPhrase.GetSize() == 0) {
                continue; // Skip rows where SearchPhrase is empty string
            }
            //<<CORE_COMPUTE>>
            l.sort_state.AddRow(v_SearchPhrase);
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
            for (auto& item : l.sort_state.buffer) {
                g.sort_state.AddRow(item.search_phrase);
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
            g.sort_state.SortNow();
        }
        //TODO: populate out chunk with final results
        idx_t output_size = g.sort_state.buffer.size();
        out.SetCardinality(output_size);
        
        auto &result_col = out.data[0];
        result_col.Resize(output_size);
        auto result_ptr = FlatVector::GetData<string_t>(result_col);
        
        for(idx_t i = 0; i < output_size; i++) {
            result_ptr[i] = g.sort_state.buffer[i].search_phrase;
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