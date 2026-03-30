/*
query_template: SELECT SearchEngineID, ClientIP, COUNT(*) AS c, SUM(IsRefresh) AS sum_isrefresh, AVG(ResolutionWidth) AS avg_resolutionwidth FROM hits WHERE SearchPhrase <> '' GROUP BY SearchEngineID, ClientIP ORDER BY c DESC LIMIT 10;

split_template: select * from dbweaver((SELECT SearchEngineID, ClientIP, IsRefresh, ResolutionWidth FROM hits WHERE (SearchPhrase!='')));
query_example: SELECT SearchEngineID, ClientIP, COUNT(*) AS c, SUM(IsRefresh) AS sum_isrefresh, AVG(ResolutionWidth) AS avg_resolutionwidth FROM hits WHERE SearchPhrase <> '' GROUP BY SearchEngineID, ClientIP ORDER BY c DESC LIMIT 10;

split_query: select * from dbweaver((SELECT SearchEngineID, ClientIP, IsRefresh, ResolutionWidth FROM hits WHERE (SearchPhrase!='')));
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
    int16_t search_engine_id;
    int32_t client_ip;
    
    bool operator==(const GroupKey& other) const {
        return search_engine_id == other.search_engine_id && 
               client_ip == other.client_ip;
    }
};

struct GroupKeyHash {
    size_t operator()(const GroupKey& k) const {
        size_t h = 0;
        h ^= std::hash<int16_t>{}(k.search_engine_id) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int32_t>{}(k.client_ip) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

struct AggState {
    int64_t count = 0;
    int64_t sum_isrefresh = 0;
    int64_t sum_resolutionwidth = 0;
    int64_t count_resolutionwidth = 0;
};

struct SortKeyView {
    int64_t c_value;
    bool c_is_null = false;
};

struct SortRowComparator {
    bool operator()(const SortKeyView &a, const SortKeyView &b) const {
        // Sort by c in DESC order
        if (a.c_is_null && b.c_is_null) return false; // Both null, preserve order
        if (a.c_is_null) return true;   // a is null, b is not -> b comes first
        if (b.c_is_null) return false;  // b is null, a is not -> a comes first
        
        // Both are non-null
        if (a.c_value != b.c_value) {
            return a.c_value > b.c_value;  // DESC order
        }
        return false; // Equal values
    }
};

struct SortRow {
    int16_t search_engine_id;
    int32_t client_ip;
    int64_t c_value;
    int64_t sum_isrefresh;
    double avg_resolutionwidth;
    
    SortKeyView GetSortKey() const {
        return {c_value, false};
    }
};

struct SortState {
    std::vector<SortRow> buffer;
    bool sorted = false;
    idx_t limit = 10; // Fixed limit of 10 for top-k
    
    inline void AddRow(int16_t search_engine_id, int32_t client_ip, int64_t c_value, int64_t sum_isrefresh, double avg_resolutionwidth) {
        buffer.push_back({search_engine_id, client_ip, c_value, sum_isrefresh, avg_resolutionwidth});
    }
    
    inline void SortNow() {
        if (!sorted) {
            // Use bounded top-k approach with a min-heap of size K
            if (buffer.size() <= limit) {
                // If we have less than or equal to K items, just sort the whole thing
                std::sort(buffer.begin(), buffer.end(), [this](const SortRow &a, const SortRow &b) {
                    SortKeyView key_a = a.GetSortKey();
                    SortKeyView key_b = b.GetSortKey();
                    return SortRowComparator{}(key_a, key_b);
                });
            } else {
                // Use partial sort to get the top K elements
                std::nth_element(buffer.begin(), buffer.begin() + limit, buffer.end(), [this](const SortRow &a, const SortRow &b) {
                    SortKeyView key_a = a.GetSortKey();
                    SortKeyView key_b = b.GetSortKey();
                    return SortRowComparator{}(key_a, key_b);
                });
                // Then sort the top K elements to ensure they're in correct order
                std::sort(buffer.begin(), buffer.begin() + limit, [this](const SortRow &a, const SortRow &b) {
                    SortKeyView key_a = a.GetSortKey();
                    SortKeyView key_b = b.GetSortKey();
                    return SortRowComparator{}(key_a, key_b);
                });
                buffer.resize(limit); // Keep only the top K
            }
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

inline hugeint_t ToHugeint(__int128 acc) {
    hugeint_t result;
    result.lower = static_cast<uint64_t>(acc);          // low 64 bits
    result.upper = static_cast<int64_t>(acc >> 64);     // high 64 bits (sign-extended)
    return result;
}

static unique_ptr<FunctionData> FnBind(ClientContext &, TableFunctionBindInput &,
                                                    vector<LogicalType> &return_types,
                                                    vector<string> &names) {
    //TODO: populate return_types and names
    names.push_back("SearchEngineID");
    names.push_back("ClientIP");
    names.push_back("c");
    names.push_back("sum_isrefresh");
    names.push_back("avg_resolutionwidth");
    
    return_types.push_back(LogicalType::SMALLINT);
    return_types.push_back(LogicalType::INTEGER);
    return_types.push_back(LogicalType::BIGINT);
    return_types.push_back(LogicalType::HUGEINT);
    return_types.push_back(LogicalType::DOUBLE);

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
    UnifiedVectorFormat SearchEngineID_uvf;
    UnifiedVectorFormat ClientIP_uvf;
    UnifiedVectorFormat IsRefresh_uvf;
    UnifiedVectorFormat ResolutionWidth_uvf;
    
    input.data[0].ToUnifiedFormat(input.size(), SearchEngineID_uvf);
    input.data[1].ToUnifiedFormat(input.size(), ClientIP_uvf);
    input.data[2].ToUnifiedFormat(input.size(), IsRefresh_uvf);
    input.data[3].ToUnifiedFormat(input.size(), ResolutionWidth_uvf);
    
    int16_t* SearchEngineID_ptr = (int16_t*)SearchEngineID_uvf.data;
    int32_t* ClientIP_ptr = (int32_t*)ClientIP_uvf.data;
    int16_t* IsRefresh_ptr = (int16_t*)IsRefresh_uvf.data;
    int16_t* ResolutionWidth_ptr = (int16_t*)ResolutionWidth_uvf.data;
    
    // Validity bitmaps for NULL checking
    auto &valid_SearchEngineID = SearchEngineID_uvf.validity;
    auto &valid_ClientIP = ClientIP_uvf.validity;
    auto &valid_IsRefresh = IsRefresh_uvf.validity;
    auto &valid_ResolutionWidth = ResolutionWidth_uvf.validity;
    
    const bool SearchEngineID_all_valid = valid_SearchEngineID.AllValid();
    const bool ClientIP_all_valid = valid_ClientIP.AllValid();
    const bool IsRefresh_all_valid = valid_IsRefresh.AllValid();
    const bool ResolutionWidth_all_valid = valid_ResolutionWidth.AllValid();
    
    idx_t num_rows = input.size();
    
    if (SearchEngineID_all_valid && ClientIP_all_valid && IsRefresh_all_valid && ResolutionWidth_all_valid) {
        // Fast path: no NULLs in any column
        for (idx_t row_idx = 0; row_idx < num_rows; ++row_idx) {
            idx_t i_SearchEngineID = SearchEngineID_uvf.sel->get_index(row_idx);
            idx_t i_ClientIP = ClientIP_uvf.sel->get_index(row_idx);
            idx_t i_IsRefresh = IsRefresh_uvf.sel->get_index(row_idx);
            idx_t i_ResolutionWidth = ResolutionWidth_uvf.sel->get_index(row_idx);
            
            int16_t search_engine_id = SearchEngineID_ptr[i_SearchEngineID];
            int32_t client_ip = ClientIP_ptr[i_ClientIP];
            int16_t is_refresh = IsRefresh_ptr[i_IsRefresh];
            int16_t resolution_width = ResolutionWidth_ptr[i_ResolutionWidth];
            
            // TODO: Core computation logic here
            GroupKey key;
            key.search_engine_id = search_engine_id;
            key.client_ip = client_ip;
            
            auto &state = l.agg_map[key];
            state.count++;
            state.sum_isrefresh += is_refresh;
            state.sum_resolutionwidth += resolution_width;
            state.count_resolutionwidth++;
        }
    } else {
        // Slow path: at least one column may contain NULLs
        for (idx_t row_idx = 0; row_idx < num_rows; ++row_idx) {
            idx_t i_SearchEngineID = SearchEngineID_uvf.sel->get_index(row_idx);
            idx_t i_ClientIP = ClientIP_uvf.sel->get_index(row_idx);
            idx_t i_IsRefresh = IsRefresh_uvf.sel->get_index(row_idx);
            idx_t i_ResolutionWidth = ResolutionWidth_uvf.sel->get_index(row_idx);
            
            if (!SearchEngineID_all_valid && !valid_SearchEngineID.RowIsValid(i_SearchEngineID)) {
                continue;
            }
            if (!ClientIP_all_valid && !valid_ClientIP.RowIsValid(i_ClientIP)) {
                continue;
            }
            if (!IsRefresh_all_valid && !valid_IsRefresh.RowIsValid(i_IsRefresh)) {
                continue;
            }
            if (!ResolutionWidth_all_valid && !valid_ResolutionWidth.RowIsValid(i_ResolutionWidth)) {
                continue;
            }
            
            int16_t search_engine_id = SearchEngineID_ptr[i_SearchEngineID];
            int32_t client_ip = ClientIP_ptr[i_ClientIP];
            int16_t is_refresh = IsRefresh_ptr[i_IsRefresh];
            int16_t resolution_width = ResolutionWidth_ptr[i_ResolutionWidth];
            
            // TODO: Core computation logic here
            GroupKey key;
            key.search_engine_id = search_engine_id;
            key.client_ip = client_ip;
            
            auto &state = l.agg_map[key];
            state.count++;
            state.sum_isrefresh += is_refresh;
            state.sum_resolutionwidth += resolution_width;
            state.count_resolutionwidth++;
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
                auto &g_state = g.agg_map[key];
                g_state.count += state.count;
                g_state.sum_isrefresh += state.sum_isrefresh;
                g_state.sum_resolutionwidth += state.sum_resolutionwidth;
                g_state.count_resolutionwidth += state.count_resolutionwidth;
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
            // Populate sort buffer with aggregated data
            for (const auto &entry : g.agg_map) {
                const GroupKey &key = entry.first;
                const AggState &state = entry.second;
                
                double avg_resolutionwidth = state.count_resolutionwidth > 0 ? 
                    (double)state.sum_resolutionwidth / state.count_resolutionwidth : 0.0;
                
                g.sort_state.AddRow(key.search_engine_id, key.client_ip, 
                                     state.count, state.sum_isrefresh, avg_resolutionwidth);
            }
            
            // Now sort the data
            g.sort_state.SortNow();
            
            // Output the sorted/top-k results
            idx_t output_idx = 0;
            idx_t max_output_rows = std::min(g.sort_state.buffer.size(), static_cast<size_t>(g.sort_state.limit));
            for (idx_t i = 0; i < max_output_rows && output_idx < out.size(); ++i) {
                const auto &row = g.sort_state.buffer[i];
                
                out.SetValue(0, output_idx, Value::SMALLINT(row.search_engine_id));
                out.SetValue(1, output_idx, Value::INTEGER(row.client_ip));
                out.SetValue(2, output_idx, Value::BIGINT(row.c_value));
                out.SetValue(3, output_idx, Value::HUGEINT(ToHugeint(static_cast<__int128>(row.sum_isrefresh))));
                out.SetValue(4, output_idx, Value::DOUBLE(row.avg_resolutionwidth));
                
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