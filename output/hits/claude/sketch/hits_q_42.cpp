/*
query_template: SELECT WindowClientWidth, WindowClientHeight, COUNT(*) AS PageViews FROM hits WHERE CounterID = 62 AND EventDate >= '2013-07-01' AND EventDate <= '2013-07-31' AND IsRefresh = 0 AND DontCountHits = 0 AND URLHash = 2868770270353813622 GROUP BY WindowClientWidth, WindowClientHeight ORDER BY PageViews DESC LIMIT 10 OFFSET 10000;

split_template: select * from dbweaver((SELECT WindowClientWidth, WindowClientHeight FROM hits WHERE (CounterID=62) AND (IsRefresh=0) AND (DontCountHits=0) AND (URLHash=2868770270353813622)));
query_example: SELECT WindowClientWidth, WindowClientHeight, COUNT(*) AS PageViews FROM hits WHERE CounterID = 62 AND EventDate >= '2013-07-01' AND EventDate <= '2013-07-31' AND IsRefresh = 0 AND DontCountHits = 0 AND URLHash = 2868770270353813622 GROUP BY WindowClientWidth, WindowClientHeight ORDER BY PageViews DESC LIMIT 10 OFFSET 10000;

split_query: select * from dbweaver((SELECT WindowClientWidth, WindowClientHeight FROM hits WHERE (CounterID=62) AND (IsRefresh=0) AND (DontCountHits=0) AND (URLHash=2868770270353813622)));
*/
#define DUCKDB_EXTENSION_MAIN

#include "dbweaver_extension.hpp"
#include "duckdb.hpp"
#include "duckdb/function/table_function.hpp"
#include <atomic>
#include <limits>
#include <mutex>
#include <unordered_map>
#include <queue>
#include <algorithm>
//TODO: Add more includes as needed


namespace duckdb {

//TODO: Define any helper structs or functions needed for binding/execution

struct FnBindData : public FunctionData {
    unique_ptr<FunctionData> Copy() const override { return make_uniq<FnBindData>(); }
    bool Equals(const FunctionData &) const override { return true; }
};

struct GroupKey {
    int16_t width;
    int16_t height;
    
    bool operator==(const GroupKey& other) const {
        return width == other.width && height == other.height;
    }
};

struct GroupKeyHash {
    size_t operator()(const GroupKey& k) const {
        size_t h = 0;
        h ^= std::hash<int16_t>{}(k.width) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int16_t>{}(k.height) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

struct AggState {
    int64_t page_views = 0;
};

struct SortKeyView {
    int64_t page_views;
};

struct SortRowComparator {
    bool operator()(const SortKeyView &a, const SortKeyView &b) const {
        // Sort by PageViews in descending order (DESC)
        if (a.page_views != b.page_views) {
            return a.page_views > b.page_views; // DESC order
        }
        return false; // Stable ordering if equal
    }
};

struct SortState {
    std::vector<SortKeyView> buffer;
    std::vector<GroupKey> keys;
    std::vector<AggState> states;
    bool sorted = false;
    idx_t top_k = 10; // Limit 10
    
    inline void AddRow(const GroupKey &key, const AggState &state) {
        SortKeyView view;
        view.page_views = state.page_views;
        
        if (top_k != 0 && buffer.size() >= top_k) {
            // Use a max-heap to maintain top-k smallest elements
            if (SortRowComparator{}(view, buffer[0])) {
                std::pop_heap(buffer.begin(), buffer.end(), SortRowComparator{});
                buffer.pop_back();
                
                buffer.push_back(view);
                std::push_heap(buffer.begin(), buffer.end(), SortRowComparator{});
            }
        } else {
            buffer.push_back(view);
            keys.push_back(key);
            states.push_back(state);
            
            if (top_k != 0 && buffer.size() > top_k) {
                // Maintain heap property and size
                std::make_heap(buffer.begin(), buffer.end(), SortRowComparator{});
                while (buffer.size() > top_k) {
                    std::pop_heap(buffer.begin(), buffer.end(), SortRowComparator{});
                    buffer.pop_back();
                    keys.pop_back();
                    states.pop_back();
                }
            }
        }
    }
    
    inline void SortNow() {
        if (!sorted) {
            if (top_k == 0) {
                // Full sort when no limit
                std::sort(buffer.begin(), buffer.end(), SortRowComparator{});
            } else {
                // Extract elements from heap and sort them
                std::sort(buffer.begin(), buffer.end(), SortRowComparator{});
            }
            sorted = true;
        }
    }
};

struct FnGlobalState : public GlobalTableFunctionState {
    std::unordered_map<GroupKey, AggState, GroupKeyHash> agg_map;
    SortState sort_state;
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
    //TODO: populate return_types and names
    return_types.push_back(LogicalType::SMALLINT);  // WindowClientWidth
    return_types.push_back(LogicalType::SMALLINT);  // WindowClientHeight
    return_types.push_back(LogicalType::BIGINT);    // PageViews
    names.push_back("WindowClientWidth");
    names.push_back("WindowClientHeight");
    names.push_back("PageViews");

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
    UnifiedVectorFormat WindowClientWidth_uvf;
    input.data[0].ToUnifiedFormat(input.size(), WindowClientWidth_uvf);
    int16_t* WindowClientWidth_ptr = (int16_t*)WindowClientWidth_uvf.data;
    
    UnifiedVectorFormat WindowClientHeight_uvf;
    input.data[1].ToUnifiedFormat(input.size(), WindowClientHeight_uvf);
    int16_t* WindowClientHeight_ptr = (int16_t*)WindowClientHeight_uvf.data;
    
    // Validity bitmaps for NULL checking
    auto &valid_WindowClientWidth = WindowClientWidth_uvf.validity;
    auto &valid_WindowClientHeight = WindowClientHeight_uvf.validity;
    
    const bool WindowClientWidth_all_valid = valid_WindowClientWidth.AllValid();
    const bool WindowClientHeight_all_valid = valid_WindowClientHeight.AllValid();
    
    idx_t num_rows = input.size();
    
    // Fast path: when no NULLs exist in either column
    if (WindowClientWidth_all_valid && WindowClientHeight_all_valid) {
        for (idx_t row_idx = 0; row_idx < num_rows; ++row_idx) {
            idx_t i_WindowClientWidth = WindowClientWidth_uvf.sel->get_index(row_idx);
            idx_t i_WindowClientHeight = WindowClientHeight_uvf.sel->get_index(row_idx);
            
            int16_t width_val = WindowClientWidth_ptr[i_WindowClientWidth];
            int16_t height_val = WindowClientHeight_ptr[i_WindowClientHeight];
            
            // ======================================
            //  Core computation logic (no NULLs)
            GroupKey key;
            key.width = width_val;
            key.height = height_val;
            auto &agg_state = l.agg_map[key];
            agg_state.page_views++;
            //<<CORE_COMPUTE>>
            // ============================
        }
    } else {
        // Slow path: at least one column may contain NULLs
        for (idx_t row_idx = 0; row_idx < num_rows; ++row_idx) {
            idx_t i_WindowClientWidth = WindowClientWidth_uvf.sel->get_index(row_idx);
            idx_t i_WindowClientHeight = WindowClientHeight_uvf.sel->get_index(row_idx);
            
            if (!WindowClientWidth_all_valid && !valid_WindowClientWidth.RowIsValid(i_WindowClientWidth)) {
                continue; // row is NULL in WindowClientWidth → skip
            }
            if (!WindowClientHeight_all_valid && !valid_WindowClientHeight.RowIsValid(i_WindowClientHeight)) {
                continue; // row is NULL in WindowClientHeight → skip
            }
            
            int16_t width_val = WindowClientWidth_ptr[i_WindowClientWidth];
            int16_t height_val = WindowClientHeight_ptr[i_WindowClientHeight];
            
            // ======================================
            //  Core computation logic (NULL-safe)
            GroupKey key;
            key.width = width_val;
            key.height = height_val;
            auto &agg_state = l.agg_map[key];
            agg_state.page_views++;
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
                g.agg_map[key].page_views += state.page_views;
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
            // Add all entries to the sort state
            for (const auto &entry : g.agg_map) {
                const GroupKey &key = entry.first;
                const AggState &state = entry.second;
                g.sort_state.AddRow(key, state);
            }
            
            // Perform the sort
            g.sort_state.SortNow();
            
            idx_t output_idx = 0;
            // Output sorted results
            for (size_t i = 0; i < g.sort_state.buffer.size() && output_idx < out.size(); ++i) {
                out.SetValue(0, output_idx, Value::SMALLINT(g.sort_state.keys[i].width));
                out.SetValue(1, output_idx, Value::SMALLINT(g.sort_state.keys[i].height));
                out.SetValue(2, output_idx, Value::BIGINT(g.sort_state.states[i].page_views));
                output_idx++;
            }
            out.SetCardinality(output_idx);
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