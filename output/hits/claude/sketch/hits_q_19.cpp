/*
query_template: SELECT UserID, extract(minute FROM EventTime) AS m, SearchPhrase, COUNT(*) AS cnt FROM hits GROUP BY UserID, m, SearchPhrase ORDER BY COUNT(*) DESC LIMIT 10;

split_template: select * from dbweaver((SELECT UserID, EventTime, SearchPhrase FROM hits));
query_example: SELECT UserID, extract(minute FROM EventTime) AS m, SearchPhrase, COUNT(*) AS cnt FROM hits GROUP BY UserID, m, SearchPhrase ORDER BY COUNT(*) DESC LIMIT 10;

split_query: select * from dbweaver((SELECT UserID, EventTime, SearchPhrase FROM hits));
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

//TODO: Define any helper structs or functions needed for binding/execution

struct FnBindData : public FunctionData {
    unique_ptr<FunctionData> Copy() const override { return make_uniq<FnBindData>(); }
    bool Equals(const FunctionData &) const override { return true; }
};

struct GroupKey {
    int64_t UserID;
    timestamp_t m;
    string_t SearchPhrase;
    
    bool operator==(const GroupKey& other) const {
        return UserID == other.UserID && m == other.m && SearchPhrase == other.SearchPhrase;
    }
};

struct GroupKeyHash {
    size_t operator()(const GroupKey& k) const {
        size_t h = 0;
        h ^= std::hash<int64_t>{}(k.UserID) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<timestamp_t>{}(k.m) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= duckdb::Hash(k.SearchPhrase.GetData(), k.SearchPhrase.GetSize()) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

struct AggState {
    int64_t cnt = 0;
};

struct SortKeyView {
    int64_t cnt;
};

struct SortRowComparator {
    bool operator()(const SortKeyView &a, const SortKeyView &b) const {
        // Sort by cnt DESC
        if (a.cnt != b.cnt) return a.cnt > b.cnt; // Descending order
        return false; // For stability, we could add a tie-breaker here
    }
};

struct SortEntry {
    GroupKey key;
    AggState state;
    
    SortEntry(const GroupKey& k, const AggState& s) : key(k), state(s) {}
};

struct SortState {
    std::vector<SortEntry> buffer;
    std::vector<SortKeyView> key_views;
    bool sorted = false;
    
    inline void AddRow(const GroupKey& key, const AggState& state) {
        buffer.emplace_back(key, state);
        key_views.push_back(SortKeyView{state.cnt});
    }
    
    inline void SortNow() {
        if (!sorted) {
            // Create indices to maintain mapping between sorted keys and original entries
            std::vector<size_t> indices(buffer.size());
            std::iota(indices.begin(), indices.end(), 0);
            
            // Sort indices based on key views using the comparator
            std::sort(indices.begin(), indices.end(), [this](size_t a, size_t b) {
                return SortRowComparator{}(key_views[a], key_views[b]);
            });
            
            // Reorder buffer according to sorted indices
            std::vector<SortEntry> sorted_buffer;
            sorted_buffer.reserve(buffer.size());
            for (size_t idx : indices) {
                sorted_buffer.push_back(std::move(buffer[idx]));
            }
            buffer = std::move(sorted_buffer);
            
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
    return_types.push_back(LogicalType::BIGINT);  // UserID
    return_types.push_back(LogicalType::BIGINT);  // m
    return_types.push_back(LogicalType::VARCHAR);  // SearchPhrase
    return_types.push_back(LogicalType::BIGINT);  // cnt
    
    names.push_back("UserID");
    names.push_back("m");
    names.push_back("SearchPhrase");
    names.push_back("cnt");

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
    UnifiedVectorFormat UserID_uvf;
    input.data[0].ToUnifiedFormat(input.size(), UserID_uvf);
    int64_t* UserID_ptr = (int64_t*)UserID_uvf.data;
    
    UnifiedVectorFormat EventTime_uvf;
    input.data[1].ToUnifiedFormat(input.size(), EventTime_uvf);
    timestamp_t* EventTime_ptr = (timestamp_t*)EventTime_uvf.data;
    
    UnifiedVectorFormat SearchPhrase_uvf;
    input.data[2].ToUnifiedFormat(input.size(), SearchPhrase_uvf);
    string_t* SearchPhrase_ptr = (string_t*)SearchPhrase_uvf.data;
    
    // Validity bitmaps
    auto &valid_UserID = UserID_uvf.validity;
    auto &valid_EventTime = EventTime_uvf.validity;
    auto &valid_SearchPhrase = SearchPhrase_uvf.validity;
    
    const bool UserID_all_valid = valid_UserID.AllValid();
    const bool EventTime_all_valid = valid_EventTime.AllValid();
    const bool SearchPhrase_all_valid = valid_SearchPhrase.AllValid();
    
    // Process rows
    if (UserID_all_valid && EventTime_all_valid && SearchPhrase_all_valid) {
        // Fast path: no NULLs
        for (idx_t row_idx = 0; row_idx < input.size(); ++row_idx) {
            idx_t i_UserID = UserID_uvf.sel->get_index(row_idx);
            idx_t i_EventTime = EventTime_uvf.sel->get_index(row_idx);
            idx_t i_SearchPhrase = SearchPhrase_uvf.sel->get_index(row_idx);
            
            int64_t v_UserID = UserID_ptr[i_UserID];
            timestamp_t v_EventTime = EventTime_ptr[i_EventTime];
            string_t v_SearchPhrase = SearchPhrase_ptr[i_SearchPhrase];
            
            // Core computation logic goes here
            GroupKey key;
            key.UserID = v_UserID;
            key.m = v_EventTime;
            key.SearchPhrase = v_SearchPhrase;
            
            auto &state = l.agg_map[key];
            state.cnt++;
        }
    } else {
        // Slow path: handle potential NULLs
        for (idx_t row_idx = 0; row_idx < input.size(); ++row_idx) {
            idx_t i_UserID = UserID_uvf.sel->get_index(row_idx);
            idx_t i_EventTime = EventTime_uvf.sel->get_index(row_idx);
            idx_t i_SearchPhrase = SearchPhrase_uvf.sel->get_index(row_idx);
            
            if (!UserID_all_valid && !valid_UserID.RowIsValid(i_UserID)) {
                continue;
            }
            if (!EventTime_all_valid && !valid_EventTime.RowIsValid(i_EventTime)) {
                continue;
            }
            if (!SearchPhrase_all_valid && !valid_SearchPhrase.RowIsValid(i_SearchPhrase)) {
                continue;
            }
            
            int64_t v_UserID = UserID_ptr[i_UserID];
            timestamp_t v_EventTime = EventTime_ptr[i_EventTime];
            string_t v_SearchPhrase = SearchPhrase_ptr[i_SearchPhrase];
            
            // Core computation logic goes here
            GroupKey key;
            key.UserID = v_UserID;
            key.m = v_EventTime;
            key.SearchPhrase = v_SearchPhrase;
            
            auto &state = l.agg_map[key];
            state.cnt++;
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
                g.agg_map[key].cnt += state.cnt;
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
        // Fill sort state with data from global agg_map
        {
            std::lock_guard<std::mutex> guard(g.lock);
            for (const auto &entry : g.agg_map) {
                const GroupKey &key = entry.first;
                const AggState &state = entry.second;
                g.sort_state.AddRow(key, state);
            }
            
            // Perform the sort
            g.sort_state.SortNow();
            
            // Output sorted results
            idx_t output_idx = 0;
            for (const auto &entry : g.sort_state.buffer) {
                const GroupKey &key = entry.key;
                const AggState &state = entry.state;
                
                out.SetValue(0, output_idx, Value::BIGINT(key.UserID));
                out.SetValue(1, output_idx, Value::BIGINT(key.m));
                out.SetValue(2, output_idx, Value(key.SearchPhrase.GetString()));
                out.SetValue(3, output_idx, Value::BIGINT(state.cnt));
                
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