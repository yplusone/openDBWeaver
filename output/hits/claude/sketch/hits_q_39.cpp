/*
query_template: SELECT URL, COUNT(*) AS PageViews FROM hits WHERE CounterID = 62 AND EventDate >= '2013-07-01' AND EventDate <= '2013-07-31' AND IsRefresh = 0 AND IsLink <> 0 AND IsDownload = 0 GROUP BY URL ORDER BY PageViews DESC LIMIT 10 OFFSET 1000;

split_template: select * from dbweaver((SELECT URL FROM hits WHERE (CounterID=62) AND (IsRefresh=0) AND (IsLink!=0) AND (IsDownload=0)));
query_example: SELECT URL, COUNT(*) AS PageViews FROM hits WHERE CounterID = 62 AND EventDate >= '2013-07-01' AND EventDate <= '2013-07-31' AND IsRefresh = 0 AND IsLink <> 0 AND IsDownload = 0 GROUP BY URL ORDER BY PageViews DESC LIMIT 10 OFFSET 1000;

split_query: select * from dbweaver((SELECT URL FROM hits WHERE (CounterID=62) AND (IsRefresh=0) AND (IsLink!=0) AND (IsDownload=0)));
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

namespace duckdb {

struct FnBindData : public FunctionData {
    unique_ptr<FunctionData> Copy() const override { return make_uniq<FnBindData>(); }
    bool Equals(const FunctionData &) const override { return true; }
};

struct GroupKey {
    string_t url;
    
    bool operator==(const GroupKey& other) const {
        return url == other.url;
    }
};

struct GroupKeyHash {
    size_t operator()(const GroupKey& k) const {
        size_t h = 0;
        h ^= duckdb::Hash(k.url.GetData(), k.url.GetSize()) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

struct AggState {
    int64_t count_val = 0;
};

struct SortKeyView {
    int64_t page_views;
    string_t url;
};

struct SortRowComparator {
    bool operator()(const SortKeyView &a, const SortKeyView &b) const {
        if (a.page_views != b.page_views) {
            // DESC order for PageViews
            return a.page_views > b.page_views;
        }
        // If values are equal, compare URL to ensure consistent ordering
        return std::strcmp(a.url.GetData(), b.url.GetData()) < 0;
    }
};

struct SortState {
    std::vector<SortKeyView> buffer;
    bool sorted = false;
    idx_t top_k = 10; // Limit 10 as specified

    inline void AddRow(string_t url, int64_t page_views) {
        buffer.push_back(SortKeyView{page_views, url});
    }

    inline void SortNow() {
        if (!sorted) {
            if (top_k < buffer.size()) {
                // Use partial sort for Top-K
                std::nth_element(buffer.begin(), buffer.begin() + top_k, buffer.end(), SortRowComparator{});
                buffer.resize(top_k);
            }
            std::sort(buffer.begin(), buffer.end(), SortRowComparator{});
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
    // Output schema: URL (VARCHAR), PageViews (BIGINT)
    return_types.emplace_back(LogicalType::VARCHAR);
    return_types.emplace_back(LogicalType::BIGINT);
    
    names.emplace_back("URL");
    names.emplace_back("PageViews");

    return make_uniq<FnBindData>();
}

static OperatorResultType FnExecute(ExecutionContext &, TableFunctionInput &in,
                                                    DataChunk &input, DataChunk &) {
    auto &l = in.local_state->Cast<FnLocalState>();

    
    if (input.size() == 0) {       
        return OperatorResultType::NEED_MORE_INPUT;
    }

    //TODO: process input chunk and produce output
    
    // UnifiedVectorFormat setup for input columns
    UnifiedVectorFormat url_uvf;
    input.data[0].ToUnifiedFormat(input.size(), url_uvf);
    string_t* url_ptr = (string_t*)url_uvf.data;
    
    UnifiedVectorFormat pageviews_uvf;
    input.data[1].ToUnifiedFormat(input.size(), pageviews_uvf);
    int64_t* pageviews_ptr = (int64_t*)pageviews_uvf.data;
    
    // validity bitmaps
    auto &valid_url = url_uvf.validity;
    auto &valid_pageviews = pageviews_uvf.validity;
    const bool url_all_valid = valid_url.AllValid();
    const bool pageviews_all_valid = valid_pageviews.AllValid();
    
    // Process input rows
    if (url_all_valid && pageviews_all_valid) {
        // --- Fast path: no per-row NULL checks ---
        for (idx_t row_idx = 0; row_idx < input.size(); ++row_idx) {
            idx_t i_url = url_uvf.sel->get_index(row_idx);
            idx_t i_pageviews = pageviews_uvf.sel->get_index(row_idx);
            string_t v_url = url_ptr[i_url];
            int64_t v_pageviews = pageviews_ptr[i_pageviews];
            
            // ======================================
            //  Core computation logic (no NULLs)
            GroupKey key;
            key.url = v_url;
            auto &state = l.agg_map[key];
            state.count_val++;
            //<<CORE_COMPUTE>>
            // ============================
        }
    } else {
        // --- Slow path: at least one column may contain NULLs ---
        for (idx_t row_idx = 0; row_idx < input.size(); ++row_idx) {
            idx_t i_url = url_uvf.sel->get_index(row_idx);
            idx_t i_pageviews = pageviews_uvf.sel->get_index(row_idx);
            
            if ((!url_all_valid && !valid_url.RowIsValid(i_url)) || 
                (!pageviews_all_valid && !valid_pageviews.RowIsValid(i_pageviews))) {
                continue; // row has NULL in either column → skip
            }
            
            string_t v_url = url_ptr[i_url];
            int64_t v_pageviews = pageviews_ptr[i_pageviews];
            
            // ======================================
            //  Core computation logic (NULL-safe)
            GroupKey key;
            key.url = v_url;
            auto &state = l.agg_map[key];
            state.count_val++;
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
                g.agg_map[key].count_val += state.count_val;
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
            // Populate sort buffer with aggregated results
            for (const auto &entry : g.agg_map) {
                const GroupKey &key = entry.first;
                const AggState &state = entry.second;
                g.sort_state.AddRow(key.url, state.count_val);
            }
            
            // Sort the results
            g.sort_state.SortNow();
            
            // Output sorted results (up to top-k)
            idx_t output_idx = 0;
            idx_t limit = std::min((idx_t)g.sort_state.buffer.size(), g.sort_state.top_k);
            for (idx_t i = 0; i < limit && output_idx < STANDARD_VECTOR_SIZE; ++i) {
                const SortKeyView &row = g.sort_state.buffer[i];
                out.SetValue(0, output_idx, Value(row.url));
                out.SetValue(1, output_idx, Value::BIGINT(row.page_views));
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