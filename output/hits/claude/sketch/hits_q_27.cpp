/*
query_template: SELECT SearchPhrase FROM hits WHERE SearchPhrase <> '' ORDER BY EventTime, SearchPhrase LIMIT 10;

split_template: select * from dbweaver((SELECT SearchPhrase, EventTime FROM hits WHERE (SearchPhrase!='')));
query_example: SELECT SearchPhrase FROM hits WHERE SearchPhrase <> '' ORDER BY EventTime, SearchPhrase LIMIT 10;

split_query: select * from dbweaver((SELECT SearchPhrase, EventTime FROM hits WHERE (SearchPhrase!='')));
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

namespace duckdb {

struct FnBindData : public FunctionData {
    unique_ptr<FunctionData> Copy() const override { return make_uniq<FnBindData>(); }
    bool Equals(const FunctionData &) const override { return true; }
};

struct SortKeyView {
    timestamp_t event_time;
    string_t search_phrase;
};

struct SortRowComparator {
    bool operator()(const SortKeyView &a, const SortKeyView &b) const {
        // First compare by EventTime (ASC)
        if (a.event_time != b.event_time) {
            return a.event_time < b.event_time;
        }
        // Then compare by SearchPhrase (ASC)
        const char *a_data = a.search_phrase.GetData();
        const char *b_data = b.search_phrase.GetData();
        idx_t a_size = a.search_phrase.GetSize();
        idx_t b_size = b.search_phrase.GetSize();
        idx_t min_size = std::min(a_size, b_size);
        
        for (idx_t i = 0; i < min_size; ++i) {
            if (a_data[i] != b_data[i]) {
                return a_data[i] < b_data[i];
            }
        }
        return a_size < b_size;
    }
};

struct SortState {
    std::vector<SortKeyView> buffer;
    bool sorted = false;
    
    inline void AddRow(timestamp_t event_time, string_t search_phrase) {
        buffer.push_back(SortKeyView{event_time, search_phrase});
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
    // Define output schema based on authoritative mapping
    // Column: SearchPhrase -> VARCHAR
    names.emplace_back("SearchPhrase");
    return_types.emplace_back(LogicalType::VARCHAR);
    
    // Column: EventTime -> TIMESTAMP (from execution logic)
    names.emplace_back("EventTime");
    return_types.emplace_back(LogicalType::TIMESTAMP);

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
    UnifiedVectorFormat SearchPhrase_uvf;
    input.data[0].ToUnifiedFormat(input.size(), SearchPhrase_uvf);
    string_t* SearchPhrase_ptr = (string_t*)SearchPhrase_uvf.data;
    
    UnifiedVectorFormat EventTime_uvf;
    input.data[1].ToUnifiedFormat(input.size(), EventTime_uvf);
    timestamp_t* EventTime_ptr = (timestamp_t*)EventTime_uvf.data;
    
    // Validity bitmaps
    auto &valid_SearchPhrase = SearchPhrase_uvf.validity;
    auto &valid_EventTime = EventTime_uvf.validity;
    
    const bool SearchPhrase_all_valid = valid_SearchPhrase.AllValid();
    const bool EventTime_all_valid = valid_EventTime.AllValid();
    
    // Process rows with NULL handling
    if (SearchPhrase_all_valid && EventTime_all_valid) {
        // Fast path: no per-row NULL checks
        for (idx_t row_idx = 0; row_idx < input.size(); ++row_idx) {
            idx_t i_SearchPhrase = SearchPhrase_uvf.sel->get_index(row_idx);
            idx_t i_EventTime = EventTime_uvf.sel->get_index(row_idx);
            
            string_t search_phrase_val = SearchPhrase_ptr[i_SearchPhrase];
            timestamp_t event_time_val = EventTime_ptr[i_EventTime];
            
            // Check filter: SearchPhrase <> ' '
            if (search_phrase_val.GetSize() == 1 && search_phrase_val.GetData()[0] == ' ') {
                continue; // Skip rows where SearchPhrase equals ' '
            }
            
            // ======================================
            //  Core computation logic (no NULLs)
            l.sort_state.AddRow(event_time_val, search_phrase_val);
            //<<CORE_COMPUTE>>
            // ======================================
        }
    } else {
        // Slow path: at least one column may contain NULLs
        for (idx_t row_idx = 0; row_idx < input.size(); ++row_idx) {
            idx_t i_SearchPhrase = SearchPhrase_uvf.sel->get_index(row_idx);
            idx_t i_EventTime = EventTime_uvf.sel->get_index(row_idx);
            
            if (!SearchPhrase_all_valid && !valid_SearchPhrase.RowIsValid(i_SearchPhrase)) {
                continue; // row is NULL in SearchPhrase → skip
            }
            if (!EventTime_all_valid && !valid_EventTime.RowIsValid(i_EventTime)) {
                continue; // row is NULL in EventTime → skip
            }
            
            string_t search_phrase_val = SearchPhrase_ptr[i_SearchPhrase];
            timestamp_t event_time_val = EventTime_ptr[i_EventTime];
            
            // Check filter: SearchPhrase <> ' '
            if (search_phrase_val.GetSize() == 1 && search_phrase_val.GetData()[0] == ' ') {
                continue; // Skip rows where SearchPhrase equals ' '
            }
            
            // ======================================
            //  Core computation logic (NULL-safe)
            l.sort_state.AddRow(event_time_val, search_phrase_val);
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
            for (auto& item : l.sort_state.buffer) {
                g.sort_state.AddRow(item.event_time, item.search_phrase);
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
        // Output sorted results
        idx_t output_count = 0;
        while (output_count < STANDARD_VECTOR_SIZE && g.sort_state.sorted && 
               g.sort_state.buffer.size() > output_count) {
            auto& sorted_item = g.sort_state.buffer[output_count];
            FlatVector::GetData<string_t>(out.data[0])[output_count] = sorted_item.search_phrase;
            FlatVector::GetData<timestamp_t>(out.data[1])[output_count] = sorted_item.event_time;
            output_count++;
        }
        out.SetCardinality(output_count);
        
        // Mark that we've processed all items
        g.sort_state.buffer.clear();
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