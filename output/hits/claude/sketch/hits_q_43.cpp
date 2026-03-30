/*
query_template: SELECT DATE_TRUNC('minute', EventTime) AS M, COUNT(*) AS PageViews FROM hits WHERE CounterID = 62 AND EventDate >= '2013-07-14' AND EventDate <= '2013-07-15' AND IsRefresh = 0 AND DontCountHits = 0 GROUP BY DATE_TRUNC('minute', EventTime) ORDER BY DATE_TRUNC('minute', EventTime) LIMIT 10 OFFSET 1000;

split_template: select * from dbweaver((SELECT EventTime FROM hits WHERE (CounterID=62) AND (EventDate>='2013-07-14'::DATE AND EventDate<='2013-07-15'::DATE) AND (IsRefresh=0) AND (DontCountHits=0)));
query_example: SELECT DATE_TRUNC('minute', EventTime) AS M, COUNT(*) AS PageViews FROM hits WHERE CounterID = 62 AND EventDate >= '2013-07-14' AND EventDate <= '2013-07-15' AND IsRefresh = 0 AND DontCountHits = 0 GROUP BY DATE_TRUNC('minute', EventTime) ORDER BY DATE_TRUNC('minute', EventTime) LIMIT 10 OFFSET 1000;

split_query: select * from dbweaver((SELECT EventTime FROM hits WHERE (CounterID=62) AND (EventDate>='2013-07-14'::DATE AND EventDate<='2013-07-15'::DATE) AND (IsRefresh=0) AND (DontCountHits=0)));
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
    timestamp_t M;  // Result of date_trunc('minute', EventTime)
    
    bool operator==(const GroupKey& other) const {
        return M == other.M;
    }
};

struct GroupKeyHash {
    size_t operator()(const GroupKey& k) const {
        size_t h = 0;
        h ^= std::hash<timestamp_t>{}(k.M) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

struct AggState {
    int64_t PageViews = 0;  // COUNT(*)
};

struct SortKeyView {
    timestamp_t M;
};

struct SortRowComparator {
    bool operator()(const SortKeyView &a, const SortKeyView &b) const {
        return a.M > b.M;  // For max-heap (top-k uses min-heap via reverse comparison)
    }
};

struct SortState {
    std::vector<std::pair<SortKeyView, AggState>> buffer;
    bool sorted = false;
    idx_t top_k = 10;  // Fixed to 10 as per limit

    inline void AddRow(const GroupKey &key, const AggState &state) {
        SortKeyView view;
        view.M = key.M;
        buffer.push_back(std::make_pair(view, state));
    }

    inline void SortNow() {
        if (!sorted) {
            if (top_k > 0 && top_k < buffer.size()) {
                // Use a heap-based top-k approach
                std::priority_queue<std::pair<SortKeyView, AggState>, 
                                   std::vector<std::pair<SortKeyView, AggState>>, 
                                   SortRowComparator> pq;
                
                for (auto &item : buffer) {
                    if (pq.size() < top_k) {
                        pq.push(item);
                    } else if (SortRowComparator()(item.first, pq.top().first)) {
                        pq.pop();
                        pq.push(item);
                    }
                }
                
                // Extract elements from heap in descending order and reverse for ascending
                std::vector<std::pair<SortKeyView, AggState>> temp_result;
                while (!pq.empty()) {
                    temp_result.push_back(pq.top());
                    pq.pop();
                }
                std::reverse(temp_result.begin(), temp_result.end());
                
                buffer = std::move(temp_result);
            } else {
                // Full sort
                std::sort(buffer.begin(), buffer.end(), 
                         [](const std::pair<SortKeyView, AggState> &a, 
                            const std::pair<SortKeyView, AggState> &b) {
                             return a.first.M < b.first.M;
                         });
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
    std::unordered_map<GroupKey, AggState, GroupKeyHash> agg_map;
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
    // Define output schema: M (TIMESTAMP) and PageViews (BIGINT)
    return_types.emplace_back(LogicalType::TIMESTAMP); // M
    return_types.emplace_back(LogicalType::BIGINT);   // PageViews
    names.emplace_back("M");
    names.emplace_back("PageViews");

    return make_uniq<FnBindData>();
}

static OperatorResultType FnExecute(ExecutionContext &, TableFunctionInput &in,
                                                DataChunk &input, DataChunk &) {
    auto &l = in.local_state->Cast<FnLocalState>();

    
    if (input.size() == 0) {       
        return OperatorResultType::NEED_MORE_INPUT;
    }

    // Unpack input columns using UnifiedVectorFormat
    UnifiedVectorFormat EventTime_uvf;
    input.data[0].ToUnifiedFormat(input.size(), EventTime_uvf);
    timestamp_t* EventTime_ptr = (timestamp_t*)EventTime_uvf.data;

    // Validity bitmaps
    auto &valid_EventTime = EventTime_uvf.validity;
    const bool EventTime_all_valid = valid_EventTime.AllValid();

    // Process input rows
    if (EventTime_all_valid) {
        // Fast path: no NULLs in EventTime column
        for (idx_t row_idx = 0; row_idx < input.size(); ++row_idx) {
            idx_t i_EventTime = EventTime_uvf.sel->get_index(row_idx);
            timestamp_t v_EventTime = EventTime_ptr[i_EventTime];

            // Calculate the truncated minute timestamp
            // This is equivalent to date_trunc('minute', EventTime)
            auto micros = v_EventTime.value;
            auto seconds = micros / Interval::MICROS_PER_SEC;
            auto minutes = seconds / 60;
            auto truncated_seconds = minutes * 60;
            auto truncated_micros = truncated_seconds * Interval::MICROS_PER_SEC;
            timestamp_t truncated_minute = timestamp_t(truncated_micros);

            GroupKey key;
            key.M = truncated_minute;
            
            auto &state = l.agg_map[key];
            state.PageViews++; // Increment count for the group
        }
    } else {
        // Slow path: EventTime column may contain NULLs
        for (idx_t row_idx = 0; row_idx < input.size(); ++row_idx) {
            idx_t i_EventTime = EventTime_uvf.sel->get_index(row_idx);

            if (!EventTime_all_valid && !valid_EventTime.RowIsValid(i_EventTime)) {
                continue; // Skip rows where EventTime is NULL
            }

            timestamp_t v_EventTime = EventTime_ptr[i_EventTime];

            // Calculate the truncated minute timestamp
            // This is equivalent to date_trunc('minute', EventTime)
            auto micros = v_EventTime.value;
            auto seconds = micros / Interval::MICROS_PER_SEC;
            auto minutes = seconds / 60;
            auto truncated_seconds = minutes * 60;
            auto truncated_micros = truncated_seconds * Interval::MICROS_PER_SEC;
            timestamp_t truncated_minute = timestamp_t(truncated_micros);

            GroupKey key;
            key.M = truncated_minute;
            
            auto &state = l.agg_map[key];
            state.PageViews++; // Increment count for the group
        }
    }

    //TODO: process input chunk and produce output

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
    if (active > 0 && merged == active) {
        // Transfer global map to sort buffer
        for (const auto &entry : g.agg_map) {
            const GroupKey &key = entry.first;
            const AggState &state = entry.second;
            g.sort_state.AddRow(key, state);
        }
        
        // Perform the sort
        g.sort_state.SortNow();
        
        {
            std::lock_guard<std::mutex> guard(g.lock);
            // Get result from global state
            idx_t output_idx = 0;
            idx_t total_rows = g.sort_state.buffer.size();
            out.SetCardinality(total_rows);
            
            for (const auto &item : g.sort_state.buffer) {
                const SortKeyView &key = item.first;
                const AggState &state = item.second;
                
                out.SetValue(0, output_idx, Value::TIMESTAMP(key.M));
                out.SetValue(1, output_idx, Value::BIGINT(state.PageViews));
                output_idx++;
            }
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