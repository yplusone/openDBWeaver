/*
query_template: SELECT URLHash, EventDate, COUNT(*) AS PageViews FROM hits WHERE CounterID = 62 AND EventDate >= '2013-07-01' AND EventDate <= '2013-07-31' AND IsRefresh = 0 AND TraficSourceID IN (-1, 6) AND RefererHash = 3594120000172545465 GROUP BY URLHash, EventDate ORDER BY PageViews DESC LIMIT 10 OFFSET 100;

split_template: select * from dbweaver((SELECT EventDate, TraficSourceID, URLHash FROM hits WHERE (CounterID=62) AND (IsRefresh=0) AND (optional: TraficSourceID IN (-1, 6)) AND (RefererHash=3594120000172545465)));
query_example: SELECT URLHash, EventDate, COUNT(*) AS PageViews FROM hits WHERE CounterID = 62 AND EventDate >= '2013-07-01' AND EventDate <= '2013-07-31' AND IsRefresh = 0 AND TraficSourceID IN (-1, 6) AND RefererHash = 3594120000172545465 GROUP BY URLHash, EventDate ORDER BY PageViews DESC LIMIT 10 OFFSET 100;

split_query: select * from dbweaver((SELECT EventDate, TraficSourceID, URLHash FROM hits WHERE (CounterID=62) AND (IsRefresh=0) AND (TraficSourceID IN (-1, 6)) AND (RefererHash=3594120000172545465)));
*/
#define DUCKDB_EXTENSION_MAIN

#include "dbweaver_extension.hpp"
#include "duckdb.hpp"
#include "duckdb/function/table_function.hpp"
#include <atomic>
#include <limits>
#include <mutex>
#include <unordered_map>
#include <algorithm>
#include <queue>

namespace duckdb {

//TODO: Define any helper structs or functions needed for binding/execution

struct FnBindData : public FunctionData {
    unique_ptr<FunctionData> Copy() const override { return make_uniq<FnBindData>(); }
    bool Equals(const FunctionData &) const override { return true; }
};

struct GroupKey {
    int64_t url_hash;
    date_t event_date;
    
    bool operator==(const GroupKey& other) const {
        return url_hash == other.url_hash && event_date == other.event_date;
    }
};

struct GroupKeyHash {
    size_t operator()(const GroupKey& k) const {
        size_t h = 0;
        h ^= std::hash<int64_t>{}(k.url_hash) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<date_t>{}(k.event_date) + 0x9e3779b9 + (h << 6) + (h >> 2);
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
        // DESC order for PageViews
        return a.page_views > b.page_views;
    }
};

struct SortedRowRef {
    GroupKey key;
    AggState state;
};

struct SortState {
    std::vector<SortedRowRef> buffer;
    std::vector<SortKeyView> sort_keys;
    bool sorted = false;
    
    inline void AddRow(const GroupKey &key, const AggState &state) {
        buffer.push_back({key, state});
        sort_keys.push_back({state.page_views});
    }
    
    inline void SortNow() {
        if (!sorted) {
            std::sort(buffer.begin(), buffer.end(), [this](const SortedRowRef &a, const SortedRowRef &b) {
                SortRowComparator comp;
                SortKeyView key_a{a.state.page_views};
                SortKeyView key_b{b.state.page_views};
                return comp(key_a, key_b);
            });
            sorted = true;
        }
    }
};

struct FnGlobalState : public GlobalTableFunctionState {
    // TODO: Optional accumulators/counters (generator may append to this struct)
    std::mutex lock;
    std::atomic<idx_t> active_local_states {0};
    std::atomic<idx_t> merged_local_states {0};
    std::unordered_map<GroupKey, AggState, GroupKeyHash> agg_map;
    SortState sort_state;
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
    // Define output schema: URLHash (BIGINT), EventDate (DATE), PageViews (BIGINT)
    return_types.emplace_back(LogicalType::BIGINT);  // URLHash
    return_types.emplace_back(LogicalType::DATE);   // EventDate
    return_types.emplace_back(LogicalType::BIGINT); // PageViews
    names.emplace_back("URLHash");
    names.emplace_back("EventDate");
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
    
    // UnifiedVectorFormat handles for input columns
    UnifiedVectorFormat EventDate_uvf;
    UnifiedVectorFormat TraficSourceID_uvf;
    UnifiedVectorFormat URLHash_uvf;
    
    input.data[0].ToUnifiedFormat(input.size(), EventDate_uvf);
    input.data[1].ToUnifiedFormat(input.size(), TraficSourceID_uvf);
    input.data[2].ToUnifiedFormat(input.size(), URLHash_uvf);
    
    date_t* EventDate_ptr = (date_t*)EventDate_uvf.data;
    int16_t* TraficSourceID_ptr = (int16_t*)TraficSourceID_uvf.data;
    int64_t* URLHash_ptr = (int64_t*)URLHash_uvf.data;
    
    // Validity bitmaps for NULL handling
    auto &valid_EventDate = EventDate_uvf.validity;
    auto &valid_TraficSourceID = TraficSourceID_uvf.validity;
    auto &valid_URLHash = URLHash_uvf.validity;
    
    const bool EventDate_all_valid = valid_EventDate.AllValid();
    const bool TraficSourceID_all_valid = valid_TraficSourceID.AllValid();
    const bool URLHash_all_valid = valid_URLHash.AllValid();
    
    idx_t num_rows = input.size();
    
    // Filter constants
    const int32_t counter_id_value = 62;
    const date_t start_date = Date::FromString("2013-07-01");
    const date_t end_date = Date::FromString("2013-07-31");
    const int16_t is_refresh_value = 0;
    const int16_t traffic_source_1 = -1;
    const int16_t traffic_source_2 = 6;
    const int64_t referer_hash_value = 3594120000172545465LL;
    
    // Fast path: all columns are non-NULL
    if (EventDate_all_valid && TraficSourceID_all_valid && URLHash_all_valid) {
        for (idx_t row_idx = 0; row_idx < num_rows; ++row_idx) {
            idx_t i_EventDate = EventDate_uvf.sel->get_index(row_idx);
            idx_t i_TraficSourceID = TraficSourceID_uvf.sel->get_index(row_idx);
            idx_t i_URLHash = URLHash_uvf.sel->get_index(row_idx);
            
            date_t event_date_val = EventDate_ptr[i_EventDate];
            int16_t trafic_source_id_val = TraficSourceID_ptr[i_TraficSourceID];
            int64_t url_hash_val = URLHash_ptr[i_URLHash];
            
            // <<CORE_COMPUTE>> - Process the values here
            // Apply filters:
            // (CounterID = 62) - ASSUMING CounterID is not in input, so we can't filter on it
            // (EventDate >= '2013-07-01')
            // (EventDate <= '2013-07-31')
            // (IsRefresh = 0) - ASSUMING IsRefresh is not in input
            // (TraficSourceID IN (-1, 6))
            // (RefererHash = 3594120000172545465) - ASSUMING RefererHash is not in input
            
            // Since CounterID, IsRefresh, and RefererHash are not in input, we can't filter on them
            // We can only filter based on the available columns: EventDate, TraficSourceID, URLHash
            // So we apply only the filters we can based on available data:
            // (EventDate >= '2013-07-01' AND EventDate <= '2013-07-31')
            // (TraficSourceID IN (-1, 6))
            
            if (!(event_date_val >= start_date && event_date_val <= end_date && 
                  (trafic_source_id_val == traffic_source_1 || trafic_source_id_val == traffic_source_2))) {
                continue; // Skip this row
            }
            
            GroupKey key;
            key.url_hash = url_hash_val;
            key.event_date = event_date_val;
            
            auto &agg_state = l.agg_map[key];
            agg_state.page_views++;
        }
    } else {
        // Slow path: at least one column has NULLs
        for (idx_t row_idx = 0; row_idx < num_rows; ++row_idx) {
            idx_t i_EventDate = EventDate_uvf.sel->get_index(row_idx);
            idx_t i_TraficSourceID = TraficSourceID_uvf.sel->get_index(row_idx);
            idx_t i_URLHash = URLHash_uvf.sel->get_index(row_idx);
            
            if (!EventDate_all_valid && !valid_EventDate.RowIsValid(i_EventDate)) {
                continue;
            }
            if (!TraficSourceID_all_valid && !valid_TraficSourceID.RowIsValid(i_TraficSourceID)) {
                continue;
            }
            if (!URLHash_all_valid && !valid_URLHash.RowIsValid(i_URLHash)) {
                continue;
            }
            
            date_t event_date_val = EventDate_ptr[i_EventDate];
            int16_t trafic_source_id_val = TraficSourceID_ptr[i_TraficSourceID];
            int64_t url_hash_val = URLHash_ptr[i_URLHash];
            
            // <<CORE_COMPUTE>> - Process the values here
            // Apply filters:
            // (CounterID = 62) - ASSUMING CounterID is not in input, so we can't filter on it
            // (EventDate >= '2013-07-01')
            // (EventDate <= '2013-07-31')
            // (IsRefresh = 0) - ASSUMING IsRefresh is not in input
            // (TraficSourceID IN (-1, 6))
            // (RefererHash = 3594120000172545465) - ASSUMING RefererHash is not in input
            
            // Since CounterID, IsRefresh, and RefererHash are not in input, we can't filter on them
            // We can only filter based on the available columns: EventDate, TraficSourceID, URLHash
            // So we apply only the filters we can based on available data:
            // (EventDate >= '2013-07-01' AND EventDate <= '2013-07-31')
            // (TraficSourceID IN (-1, 6))
            
            if (!(event_date_val >= start_date && event_date_val <= end_date && 
                  (trafic_source_id_val == traffic_source_1 || trafic_source_id_val == traffic_source_2))) {
                continue; // Skip this row
            }
            
            GroupKey key;
            key.url_hash = url_hash_val;
            key.event_date = event_date_val;
            
            auto &agg_state = l.agg_map[key];
            agg_state.page_views++;
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
            // Fill the sort buffer with all aggregated results
            for (const auto &entry : g.agg_map) {
                const GroupKey &key = entry.first;
                const AggState &state = entry.second;
                g.sort_state.AddRow(key, state);
            }
            
            // Perform the sort
            g.sort_state.SortNow();
            
            // Output the sorted results
            idx_t output_idx = 0;
            for (size_t i = 0; i < g.sort_state.buffer.size(); ++i) {
                const GroupKey &key = g.sort_state.buffer[i].key;
                const AggState &state = g.sort_state.buffer[i].state;
                
                if (output_idx >= STANDARD_VECTOR_SIZE) {
                    break; // Need to split output across multiple chunks
                }
                
                out.SetValue(0, output_idx, Value::BIGINT(key.url_hash));
                out.SetValue(1, output_idx, Value::DATE(key.event_date));
                out.SetValue(2, output_idx, Value::BIGINT(state.page_views));
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