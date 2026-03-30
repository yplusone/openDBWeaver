/*
query_template: SELECT Title, COUNT(*) AS PageViews FROM hits WHERE CounterID = 62 AND EventDate >= '2013-07-01' AND EventDate <= '2013-07-31' AND DontCountHits = 0 AND IsRefresh = 0 AND Title <> '' GROUP BY Title ORDER BY PageViews DESC LIMIT 10;

split_template: select * from dbweaver((SELECT Title FROM hits WHERE (CounterID=62) AND (DontCountHits=0) AND (IsRefresh=0) AND (Title!='')));
query_example: SELECT Title, COUNT(*) AS PageViews FROM hits WHERE CounterID = 62 AND EventDate >= '2013-07-01' AND EventDate <= '2013-07-31' AND DontCountHits = 0 AND IsRefresh = 0 AND Title <> '' GROUP BY Title ORDER BY PageViews DESC LIMIT 10;

split_query: select * from dbweaver((SELECT Title FROM hits WHERE (CounterID=62) AND (DontCountHits=0) AND (IsRefresh=0) AND (Title!='')));
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

namespace duckdb {

struct FnBindData : public FunctionData {
    unique_ptr<FunctionData> Copy() const override { return make_uniq<FnBindData>(); }
    bool Equals(const FunctionData &) const override { return true; }
};

struct GroupKey {
    string_t Title;
    
    bool operator==(const GroupKey& other) const {
        return Title == other.Title;
    }
};

struct GroupKeyHash {
    size_t operator()(const GroupKey& k) const {
        size_t h = 0;
        h ^= duckdb::Hash(k.Title.GetData(), k.Title.GetSize()) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

struct AggState {
    int64_t PageViews = 0;
};

struct SortKeyView {
    int64_t PageViews;
    string_t Title;
};

struct SortRowComparator {
    bool operator()(const SortKeyView &a, const SortKeyView &b) const {
        // Sort by PageViews DESC
        if (a.PageViews != b.PageViews) {
            return a.PageViews > b.PageViews;  // DESC order
        }
        // If PageViews are equal, use Title as tie-breaker
        return std::strcmp(a.Title.GetData(), b.Title.GetData()) < 0;
    }
};

struct SortState {
    std::vector<SortKeyView> buffer;
    bool sorted = false;
    
    inline void AddRow(string_t title, int64_t pageViews) {
        buffer.push_back(SortKeyView{pageViews, title});
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
    std::unordered_map<GroupKey, AggState, GroupKeyHash> agg_map;
    SortState sort_state;
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
    // Output schema: Title (VARCHAR) and PageViews (BIGINT)
    return_types.push_back(LogicalType::VARCHAR);  // Title
    return_types.push_back(LogicalType::BIGINT);   // PageViews
    names.push_back("Title");
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
    
    // Setup UnifiedVectorFormat for input columns
    UnifiedVectorFormat Title_uvf;
    input.data[0].ToUnifiedFormat(input.size(), Title_uvf);
    string_t* Title_ptr = (string_t*)Title_uvf.data;
    
    UnifiedVectorFormat PageViews_uvf;
    input.data[1].ToUnifiedFormat(input.size(), PageViews_uvf);
    int64_t* PageViews_ptr = (int64_t*)PageViews_uvf.data;
    
    // 1) Batch-level NULL summary (whether each column is fully non-NULL)
    // validity bitmaps
    auto &valid_Title  = Title_uvf.validity;
    auto &valid_PageViews = PageViews_uvf.validity;
    // Add more columns as needed
    const bool Title_all_valid = valid_Title.AllValid();
    const bool PageViews_all_valid = valid_PageViews.AllValid();
    // Add more columns as needed:
    // const bool colN_all_valid = valid_<colN>.AllValid();
    
    // 2) FAST BRANCH: all relevant columns have no NULLs in this batch
    if (Title_all_valid && PageViews_all_valid /* && colN_all_valid ... */) {
        // --- Fast path: no per-row NULL checks ---
        for (idx_t row_idx = 0; row_idx < input.size(); ++row_idx) {
            // Directly load values without RowIsValid checks

            idx_t i_Title = Title_uvf.sel->get_index(row_idx);
            idx_t i_PageViews = PageViews_uvf.sel->get_index(row_idx);
            string_t v1 = Title_ptr[i_Title];
            int64_t v2 = PageViews_ptr[i_PageViews];
            //  vN = <COL_N>_ptr[i_<COL_N>];
    
            // ======================================
            //  Core computation logic (no NULLs)
            //<<CORE_COMPUTE>>
            GroupKey key;
            key.Title = v1;
            auto &state = l.agg_map[key];
            state.PageViews += v2;
            // ============================
        }
    } else {
        // --- Slow path: at least one column may contain NULLs ---
        for (idx_t row_idx = 0; row_idx < input.size(); ++row_idx) {
            // For each column that is not fully valid, check this row
            idx_t i_Title = Title_uvf.sel->get_index(row_idx);
            idx_t i_PageViews = PageViews_uvf.sel->get_index(row_idx);

            if (!Title_all_valid && !valid_Title.RowIsValid(i_Title)) {
                continue; // row is NULL in column 1 → skip
            }
            if (!PageViews_all_valid && !valid_PageViews.RowIsValid(i_PageViews)) {
                continue; // row is NULL in column 2 → skip
            }
            // Repeat for additional columns

            // At this point, all required columns are valid for this row

            string_t v1 = Title_ptr[i_Title];
            int64_t v2 = PageViews_ptr[i_PageViews];
            // auto vN = <COL_N>_ptr[i_<COL_N>];

            // ======================================
            //  Core computation logic (NULL-safe)
            //<<CORE_COMPUTE>>
            GroupKey key;
            key.Title = v1;
            auto &state = l.agg_map[key];
            state.PageViews += v2;
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
                g.agg_map[key].PageViews += state.PageViews;
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
            // Populate sort buffer with aggregated results
            for (const auto &entry : g.agg_map) {
                const GroupKey &key = entry.first;
                const AggState &state = entry.second;
                g.sort_state.AddRow(key.Title, state.PageViews);
            }
            
            // Sort the data
            g.sort_state.SortNow();
            
            // Output sorted results
            for (size_t i = 0; i < g.sort_state.buffer.size(); ++i) {
                const SortKeyView &view = g.sort_state.buffer[i];
                out.SetValue(0, output_row_idx, Value(view.Title.GetString()));
                out.SetValue(1, output_row_idx, Value::BIGINT(view.PageViews));
                output_row_idx++;
                
                if (output_row_idx >= STANDARD_VECTOR_SIZE) {
                    break;
                }
            }
        }
        out.SetCardinality(output_row_idx);
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