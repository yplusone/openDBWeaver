/*
query_template: SELECT TraficSourceID, SearchEngineID, AdvEngineID, CASE WHEN (SearchEngineID = 0 AND AdvEngineID = 0) THEN Referer ELSE '' END AS Src, URL AS Dst, COUNT(*) AS PageViews FROM hits WHERE CounterID = 62 AND EventDate >= '2013-07-01' AND EventDate <= '2013-07-31' AND IsRefresh = 0 GROUP BY TraficSourceID, SearchEngineID, AdvEngineID, Src, Dst ORDER BY PageViews DESC LIMIT 10 OFFSET 1000;

split_template: select * from dbweaver((SELECT TraficSourceID, SearchEngineID, AdvEngineID, Referer, URL FROM hits WHERE (CounterID=62) AND (IsRefresh=0)));
query_example: SELECT TraficSourceID, SearchEngineID, AdvEngineID, CASE WHEN (SearchEngineID = 0 AND AdvEngineID = 0) THEN Referer ELSE '' END AS Src, URL AS Dst, COUNT(*) AS PageViews FROM hits WHERE CounterID = 62 AND EventDate >= '2013-07-01' AND EventDate <= '2013-07-31' AND IsRefresh = 0 GROUP BY TraficSourceID, SearchEngineID, AdvEngineID, Src, Dst ORDER BY PageViews DESC LIMIT 10 OFFSET 1000;

split_query: select * from dbweaver((SELECT TraficSourceID, SearchEngineID, AdvEngineID, Referer, URL FROM hits WHERE (CounterID=62) AND (IsRefresh=0)));
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
#include <cstring>

namespace duckdb {

//TODO: Define any helper structs or functions needed for binding/execution

struct FnBindData : public FunctionData {
    unique_ptr<FunctionData> Copy() const override { return make_uniq<FnBindData>(); }
    bool Equals(const FunctionData &) const override { return true; }
    idx_t limit = 10; // Top-K limit
};

struct GroupKey {
    int16_t trafic_source_id;
    int16_t search_engine_id;
    int16_t adv_engine_id;
    string_t src;
    string_t dst;
    
    bool operator==(const GroupKey& other) const {
        return trafic_source_id == other.trafic_source_id && 
               search_engine_id == other.search_engine_id && 
               adv_engine_id == other.adv_engine_id &&
               src == other.src &&
               dst == other.dst;
    }
};

struct GroupKeyHash {
    size_t operator()(const GroupKey& k) const {
        size_t h = 0;
        h ^= std::hash<int16_t>{}(k.trafic_source_id) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int16_t>{}(k.search_engine_id) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int16_t>{}(k.adv_engine_id) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= duckdb::Hash(k.src.GetData(), k.src.GetSize()) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= duckdb::Hash(k.dst.GetData(), k.dst.GetSize()) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

struct AggState {
    int64_t page_views = 0;
};

struct SortKeyView {
    int64_t page_views;
    GroupKey key;
};

struct SortRowComparator {
    bool operator()(const SortKeyView &a, const SortKeyView &b) const {
        // PageViews DESC
        if (a.page_views != b.page_views) {
            return a.page_views > b.page_views; // Descending order
        }
        // Tie-breaker: lexicographic order of all other fields to ensure stability
        if (a.key.trafic_source_id != b.key.trafic_source_id) {
            return a.key.trafic_source_id < b.key.trafic_source_id;
        }
        if (a.key.search_engine_id != b.key.search_engine_id) {
            return a.key.search_engine_id < b.key.search_engine_id;
        }
        if (a.key.adv_engine_id != b.key.adv_engine_id) {
            return a.key.adv_engine_id < b.key.adv_engine_id;
        }
        int cmp_src = std::strcmp(a.key.src.GetData(), a.key.src.GetSize(), b.key.src.GetData(), b.key.src.GetSize());
        if (cmp_src != 0) {
            return cmp_src < 0;
        }
        int cmp_dst = std::strcmp(a.key.dst.GetData(), a.key.dst.GetSize(), b.key.dst.GetData(), b.key.dst.GetSize());
        return cmp_dst < 0;
    }
};

struct SortState {
    std::vector<SortKeyView> buffer;
    bool sorted = false;
    idx_t limit;
    
    SortState(idx_t k_limit) : limit(k_limit) {}
    
    inline void AddRow(const GroupKey &key, const AggState &state) {
        SortKeyView view;
        view.page_views = state.page_views;
        view.key = key;
        
        if (limit != 0) {
            // Top-K case
            if (buffer.size() < limit) {
                buffer.push_back(view);
                if (buffer.size() == limit) {
                    std::make_heap(buffer.begin(), buffer.end(), SortRowComparator{});
                }
            } else {
                // Check if current element is smaller than max (top of heap)
                if (SortRowComparator{}(view, buffer[0])) {
                    std::pop_heap(buffer.begin(), buffer.end(), SortRowComparator{});
                    buffer.pop_back();
                    buffer.push_back(view);
                    std::push_heap(buffer.begin(), buffer.end(), SortRowComparator{});
                }
            }
        } else {
            // Full sort case
            buffer.push_back(view);
        }
    }
    
    inline void SortNow() {
        if (!sorted && limit == 0) {
            std::sort(buffer.begin(), buffer.end(), SortRowComparator{});
            sorted = true;
        } else if (!sorted && limit != 0) {
            // For Top-K, we need to sort the final heap to get elements in order
            std::sort_heap(buffer.begin(), buffer.end(), SortRowComparator{});
            sorted = true;
        }
    }
};

struct FnGlobalState : public GlobalTableFunctionState {
    // TODO: Optional accumulators/counters (generator may append to this struct)
    std::unordered_map<GroupKey, AggState, GroupKeyHash> agg_map;
    std::mutex lock;
    std::atomic<idx_t> active_local_states {0};
    std::atomic<idx_t> merged_local_states {0};
    idx_t limit = 0;
    SortState sort_state {0};
    idx_t MaxThreads() const override {
        return std::numeric_limits<idx_t>::max();
    }
};

static unique_ptr<GlobalTableFunctionState> FnInit(ClientContext &, TableFunctionInitInput &input) {
    auto result = make_uniq<FnGlobalState>();
    auto &bind_data = input.bind_data->Cast<FnBindData>();
    result->limit = bind_data.limit;
    result->sort_state = SortState(bind_data.limit);
    return result;
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
    names.emplace_back("TraficSourceID");
    names.emplace_back("SearchEngineID");
    names.emplace_back("AdvEngineID");
    names.emplace_back("Src");
    names.emplace_back("Dst");
    names.emplace_back("PageViews");
    
    return_types.emplace_back(LogicalType::SMALLINT);
    return_types.emplace_back(LogicalType::SMALLINT);
    return_types.emplace_back(LogicalType::SMALLINT);
    return_types.emplace_back(LogicalType::VARCHAR);
    return_types.emplace_back(LogicalType::VARCHAR);
    return_types.emplace_back(LogicalType::BIGINT);

    auto bind_data = make_uniq<FnBindData>();
    bind_data->limit = 10; // Top-K limit
    return std::move(bind_data);
}

static OperatorResultType FnExecute(ExecutionContext &, TableFunctionInput &in,
                                            DataChunk &input, DataChunk &) {
    auto &l = in.local_state->Cast<FnLocalState>();

    
    if (input.size() == 0) {       
        return OperatorResultType::NEED_MORE_INPUT;
    }

    //TODO: process input chunk and produce output
    
    // UnifiedVectorFormat handles for input columns
    UnifiedVectorFormat TraficSourceID_uvf;
    input.data[0].ToUnifiedFormat(input.size(), TraficSourceID_uvf);
    int16_t* TraficSourceID_ptr = (int16_t*)TraficSourceID_uvf.data;
    
    UnifiedVectorFormat SearchEngineID_uvf;
    input.data[1].ToUnifiedFormat(input.size(), SearchEngineID_uvf);
    int16_t* SearchEngineID_ptr = (int16_t*)SearchEngineID_uvf.data;
    
    UnifiedVectorFormat AdvEngineID_uvf;
    input.data[2].ToUnifiedFormat(input.size(), AdvEngineID_uvf);
    int16_t* AdvEngineID_ptr = (int16_t*)AdvEngineID_uvf.data;
    
    UnifiedVectorFormat Referer_uvf;
    input.data[3].ToUnifiedFormat(input.size(), Referer_uvf);
    string_t* Referer_ptr = (string_t*)Referer_uvf.data;
    
    UnifiedVectorFormat URL_uvf;
    input.data[4].ToUnifiedFormat(input.size(), URL_uvf);
    string_t* URL_ptr = (string_t*)URL_uvf.data;
    
    UnifiedVectorFormat PageViews_uvf;
    input.data[5].ToUnifiedFormat(input.size(), PageViews_uvf);
    int64_t* PageViews_ptr = (int64_t*)PageViews_uvf.data;
    
    // Validity bitmaps for NULL checking
    auto &valid_TraficSourceID = TraficSourceID_uvf.validity;
    auto &valid_SearchEngineID = SearchEngineID_uvf.validity;
    auto &valid_AdvEngineID = AdvEngineID_uvf.validity;
    auto &valid_Referer = Referer_uvf.validity;
    auto &valid_URL = URL_uvf.validity;
    auto &valid_PageViews = PageViews_uvf.validity;
    
    const bool TraficSourceID_all_valid = valid_TraficSourceID.AllValid();
    const bool SearchEngineID_all_valid = valid_SearchEngineID.AllValid();
    const bool AdvEngineID_all_valid = valid_AdvEngineID.AllValid();
    const bool Referer_all_valid = valid_Referer.AllValid();
    const bool URL_all_valid = valid_URL.AllValid();
    const bool PageViews_all_valid = valid_PageViews.AllValid();
    
    idx_t num_rows = input.size();
    
    // Fast path: all columns have no NULLs in this batch
    if (TraficSourceID_all_valid && SearchEngineID_all_valid && AdvEngineID_all_valid && 
        Referer_all_valid && URL_all_valid && PageViews_all_valid) {
        for (idx_t row_idx = 0; row_idx < num_rows; ++row_idx) {
            idx_t i_TraficSourceID = TraficSourceID_uvf.sel->get_index(row_idx);
            idx_t i_SearchEngineID = SearchEngineID_uvf.sel->get_index(row_idx);
            idx_t i_AdvEngineID = AdvEngineID_uvf.sel->get_index(row_idx);
            idx_t i_Referer = Referer_uvf.sel->get_index(row_idx);
            idx_t i_URL = URL_uvf.sel->get_index(row_idx);
            idx_t i_PageViews = PageViews_uvf.sel->get_index(row_idx);
            
            int16_t v_TraficSourceID = TraficSourceID_ptr[i_TraficSourceID];
            int16_t v_SearchEngineID = SearchEngineID_ptr[i_SearchEngineID];
            int16_t v_AdvEngineID = AdvEngineID_ptr[i_AdvEngineID];
            string_t v_Referer = Referer_ptr[i_Referer];
            string_t v_URL = URL_ptr[i_URL];
            int64_t v_PageViews = PageViews_ptr[i_PageViews];
            
            // ======================================
            //  Core computation logic (no NULLs)
            GroupKey key;
            key.trafic_source_id = v_TraficSourceID;
            key.search_engine_id = v_SearchEngineID;
            key.adv_engine_id = v_AdvEngineID;
            key.src = v_Referer;
            key.dst = v_URL;
            
            auto &agg_state = l.agg_map[key];
            agg_state.page_views += v_PageViews;
            //<<CORE_COMPUTE>>
            // ============================
        }
    } else {
        // Slow path: at least one column may contain NULLs
        for (idx_t row_idx = 0; row_idx < num_rows; ++row_idx) {
            idx_t i_TraficSourceID = TraficSourceID_uvf.sel->get_index(row_idx);
            idx_t i_SearchEngineID = SearchEngineID_uvf.sel->get_index(row_idx);
            idx_t i_AdvEngineID = AdvEngineID_uvf.sel->get_index(row_idx);
            idx_t i_Referer = Referer_uvf.sel->get_index(row_idx);
            idx_t i_URL = URL_uvf.sel->get_index(row_idx);
            idx_t i_PageViews = PageViews_uvf.sel->get_index(row_idx);
            
            if (!TraficSourceID_all_valid && !valid_TraficSourceID.RowIsValid(i_TraficSourceID)) {
                continue; // row is NULL in TraficSourceID → skip
            }
            if (!SearchEngineID_all_valid && !valid_SearchEngineID.RowIsValid(i_SearchEngineID)) {
                continue; // row is NULL in SearchEngineID → skip
            }
            if (!AdvEngineID_all_valid && !valid_AdvEngineID.RowIsValid(i_AdvEngineID)) {
                continue; // row is NULL in AdvEngineID → skip
            }
            if (!Referer_all_valid && !valid_Referer.RowIsValid(i_Referer)) {
                continue; // row is NULL in Referer → skip
            }
            if (!URL_all_valid && !valid_URL.RowIsValid(i_URL)) {
                continue; // row is NULL in URL → skip
            }
            if (!PageViews_all_valid && !valid_PageViews.RowIsValid(i_PageViews)) {
                continue; // row is NULL in PageViews → skip
            }
            
            int16_t v_TraficSourceID = TraficSourceID_ptr[i_TraficSourceID];
            int16_t v_SearchEngineID = SearchEngineID_ptr[i_SearchEngineID];
            int16_t v_AdvEngineID = AdvEngineID_ptr[i_AdvEngineID];
            string_t v_Referer = Referer_ptr[i_Referer];
            string_t v_URL = URL_ptr[i_URL];
            int64_t v_PageViews = PageViews_ptr[i_PageViews];
            
            // ======================================
            //  Core computation logic (NULL-safe)
            GroupKey key;
            key.trafic_source_id = v_TraficSourceID;
            key.search_engine_id = v_SearchEngineID;
            key.adv_engine_id = v_AdvEngineID;
            key.src = v_Referer;
            key.dst = v_URL;
            
            auto &agg_state = l.agg_map[key];
            agg_state.page_views += v_PageViews;
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
        // Now add all aggregated entries to the sort buffer
        {
            std::lock_guard<std::mutex> guard(g.lock);
            for (const auto &entry : g.agg_map) {
                const GroupKey &key = entry.first;
                const AggState &state = entry.second;
                g.sort_state.AddRow(key, state);
            }
            
            // Perform the sort
            g.sort_state.SortNow();
            
            // Output the sorted results
            idx_t output_row = 0;
            for (const auto &sort_entry : g.sort_state.buffer) {
                if (output_row >= STANDARD_VECTOR_SIZE) {
                    out.SetCardinality(output_row);
                    return OperatorFinalizeResultType::YIELD_RESULT;
                }
                
                out.SetValue(0, output_row, Value::SMALLINT(sort_entry.key.trafic_source_id));
                out.SetValue(1, output_row, Value::SMALLINT(sort_entry.key.search_engine_id));
                out.SetValue(2, output_row, Value::SMALLINT(sort_entry.key.adv_engine_id));
                out.SetValue(3, output_row, Value(sort_entry.key.src));
                out.SetValue(4, output_row, Value(sort_entry.key.dst));
                out.SetValue(5, output_row, Value::BIGINT(sort_entry.page_views));
                
                output_row++;
            }
            out.SetCardinality(output_row);
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