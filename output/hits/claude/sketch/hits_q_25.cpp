/*
query_template: SELECT SearchPhrase FROM hits WHERE SearchPhrase <> '' ORDER BY EventTime LIMIT 10;

split_template: select * from dbweaver((SELECT SearchPhrase, EventTime FROM hits));
query_example: SELECT SearchPhrase FROM hits WHERE SearchPhrase <> '' ORDER BY EventTime LIMIT 10;

split_query: select * from dbweaver((SELECT SearchPhrase, EventTime FROM hits));
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

struct SortKeyView {
    timestamp_t event_time;
};

struct SortRowComparator {
    bool operator()(const SortKeyView &a, const SortKeyView &b) const {
        // Sort by EventTime ascending
        return a.event_time < b.event_time;
    }
};

struct SortState {
    std::vector<SortKeyView> key_buffer;
    std::vector<string_t> search_phrase_buffer;
    std::vector<timestamp_t> event_time_buffer;
    bool sorted = false;

    inline void AddRow(string_t search_phrase, timestamp_t event_time) {
        key_buffer.push_back(SortKeyView{event_time});
        search_phrase_buffer.push_back(search_phrase);
        event_time_buffer.push_back(event_time);
    }

    inline void SortNow() {
        if (!sorted && !key_buffer.empty()) {
            // Create index vector for stable sorting
            std::vector<size_t> indices(key_buffer.size());
            std::iota(indices.begin(), indices.end(), 0);
            
            // Sort indices based on key values
            std::sort(indices.begin(), indices.end(), [this](size_t i, size_t j) {
                return SortRowComparator{}(key_buffer[i], key_buffer[j]);
            });
            
            // Reorder buffers based on sorted indices
            std::vector<SortKeyView> sorted_key_buffer;
            std::vector<string_t> sorted_search_phrase_buffer;
            std::vector<timestamp_t> sorted_event_time_buffer;
            
            sorted_key_buffer.reserve(key_buffer.size());
            sorted_search_phrase_buffer.reserve(search_phrase_buffer.size());
            sorted_event_time_buffer.reserve(event_time_buffer.size());
            
            for (size_t idx : indices) {
                sorted_key_buffer.push_back(key_buffer[idx]);
                sorted_search_phrase_buffer.push_back(search_phrase_buffer[idx]);
                sorted_event_time_buffer.push_back(event_time_buffer[idx]);
            }
            
            key_buffer = std::move(sorted_key_buffer);
            search_phrase_buffer = std::move(sorted_search_phrase_buffer);
            event_time_buffer = std::move(sorted_event_time_buffer);
            
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
    
    SortState sort_state;
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
    return_types.emplace_back(LogicalType::VARCHAR);
    names.emplace_back("SearchPhrase");
    
    return_types.emplace_back(LogicalType::TIMESTAMP);
    names.emplace_back("EventTime");

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
    UnifiedVectorFormat SearchPhrase_uvf;
    input.data[0].ToUnifiedFormat(input.size(), SearchPhrase_uvf);
    string_t* SearchPhrase_ptr = (string_t*)SearchPhrase_uvf.data;
    
    UnifiedVectorFormat EventTime_uvf;
    input.data[1].ToUnifiedFormat(input.size(), EventTime_uvf);
    timestamp_t* EventTime_ptr = (timestamp_t*)EventTime_uvf.data;
    
    // Validity bitmaps for NULL checking
    auto &valid_SearchPhrase = SearchPhrase_uvf.validity;
    auto &valid_EventTime = EventTime_uvf.validity;
    
    const bool SearchPhrase_all_valid = valid_SearchPhrase.AllValid();
    const bool EventTime_all_valid = valid_EventTime.AllValid();
    
    // Process rows in the input chunk
    idx_t num_rows = input.size();
    
    // Fast path: when no NULLs exist in either column
    if (SearchPhrase_all_valid && EventTime_all_valid) {
        for (idx_t row_idx = 0; row_idx < num_rows; ++row_idx) {
            idx_t i_SearchPhrase = SearchPhrase_uvf.sel->get_index(row_idx);
            idx_t i_EventTime = EventTime_uvf.sel->get_index(row_idx);
            
            string_t v_SearchPhrase = SearchPhrase_ptr[i_SearchPhrase];
            timestamp_t v_EventTime = EventTime_ptr[i_EventTime];
            
            // Check filter: SearchPhrase != ''
            if (v_SearchPhrase.GetSize() == 0) {
                continue; // Skip this row as it doesn't match the filter condition
            }
            
            // ======================================
            //  Core computation logic (no NULLs)
            // Add row to local sort buffer
            l.sort_state.AddRow(v_SearchPhrase, v_EventTime);
            // ======================================
        }
    } else {
        // Slow path: at least one column may contain NULLs
        for (idx_t row_idx = 0; row_idx < num_rows; ++row_idx) {
            idx_t i_SearchPhrase = SearchPhrase_uvf.sel->get_index(row_idx);
            idx_t i_EventTime = EventTime_uvf.sel->get_index(row_idx);
            
            if (!SearchPhrase_all_valid && !valid_SearchPhrase.RowIsValid(i_SearchPhrase)) {
                continue; // row is NULL in SearchPhrase column → skip
            }
            if (!EventTime_all_valid && !valid_EventTime.RowIsValid(i_EventTime)) {
                continue; // row is NULL in EventTime column → skip
            }
            
            string_t v_SearchPhrase = SearchPhrase_ptr[i_SearchPhrase];
            timestamp_t v_EventTime = EventTime_ptr[i_EventTime];
            
            // Check filter: SearchPhrase != ''
            if (v_SearchPhrase.GetSize() == 0) {
                continue; // Skip this row as it doesn't match the filter condition
            }
            
            // ======================================
            //  Core computation logic (NULL-safe)
            // Add row to local sort buffer
            l.sort_state.AddRow(v_SearchPhrase, v_EventTime);
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
            // Merge local sort state with global sort state
            for(size_t i = 0; i < l.sort_state.key_buffer.size(); ++i) {
                g.sort_state.AddRow(l.sort_state.search_phrase_buffer[i], l.sort_state.event_time_buffer[i]);
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
            // Perform the sort operation on global state
            g.sort_state.SortNow();
        }
        
        // Calculate how many rows to output in this call
        idx_t total_rows = g.sort_state.search_phrase_buffer.size();
        idx_t start_idx = 0;
        
        // Output up to STANDARD_VECTOR_SIZE rows per call
        idx_t output_size = std::min(static_cast<idx_t>(total_rows), static_cast<idx_t>(STANDARD_VECTOR_SIZE));
        
        // Set up output vectors
        out.SetCardinality(output_size);
        
        auto &search_phrase_output = out.data[0];
        auto &event_time_output = out.data[1];
        
        // Fill output vectors with sorted data
        for(idx_t i = 0; i < output_size; ++i) {
            search_phrase_output.SetValue(i, Value(g.sort_state.search_phrase_buffer[start_idx + i]));
            event_time_output.SetValue(i, Value::TIMESTAMP(g.sort_state.event_time_buffer[start_idx + i]));
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