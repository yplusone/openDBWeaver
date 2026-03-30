/*
query_template: SELECT WatchID, ClientIP, COUNT(*) AS c, SUM(IsRefresh) AS sum_isrefresh, AVG(ResolutionWidth) AS avg_resolutionwidth FROM hits GROUP BY WatchID, ClientIP ORDER BY c DESC LIMIT 10;

split_template: select * from dbweaver((SELECT WatchID, ClientIP, IsRefresh, ResolutionWidth FROM hits));
query_example: SELECT WatchID, ClientIP, COUNT(*) AS c, SUM(IsRefresh) AS sum_isrefresh, AVG(ResolutionWidth) AS avg_resolutionwidth FROM hits GROUP BY WatchID, ClientIP ORDER BY c DESC LIMIT 10;

split_query: select * from dbweaver((SELECT WatchID, ClientIP, IsRefresh, ResolutionWidth FROM hits));
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
    int64_t WatchID;
    int32_t ClientIP;
    
    bool operator==(const GroupKey& other) const {
        return WatchID == other.WatchID && ClientIP == other.ClientIP;
    }
};

struct GroupKeyHash {
    size_t operator()(const GroupKey& k) const {
        size_t h = 0;
        h ^= std::hash<int64_t>{}(k.WatchID) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int32_t>{}(k.ClientIP) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

struct AggState {
    int64_t count = 0;
    int64_t sum_isrefresh = 0;
    int64_t sum_resolutionwidth = 0;
    int64_t count_resolutionwidth = 0;
};

struct SortedRow {
    int64_t WatchID;
    int32_t ClientIP;
    int64_t c;
    int64_t sum_isrefresh;
    double avg_resolutionwidth;
};

struct SortKeyView {
    int64_t c; // The sort key based on the schema
};

struct SortRowComparator {
    bool operator()(const SortKeyView &a, const SortKeyView &b) const {
        // DESC order for c
        if (a.c != b.c) return a.c > b.c; // Descending order
        return false; // Equal case, no further tiebreaker needed
    }
};

struct SortState {
    std::vector<SortedRow> rows;
    std::vector<SortKeyView> keys;
    bool sorted = false;
    idx_t top_k_limit = 10; // Based on limit 10
    
    inline void AddRow(int64_t watch_id, int32_t client_ip, int64_t c, int64_t sum_isrefresh, double avg_resolutionwidth) {
        SortedRow row;
        row.WatchID = watch_id;
        row.ClientIP = client_ip;
        row.c = c;
        row.sum_isrefresh = sum_isrefresh;
        row.avg_resolutionwidth = avg_resolutionwidth;
        
        SortKeyView key_view;
        key_view.c = c;
        
        rows.push_back(row);
        keys.push_back(key_view);
    }
    
    inline void SortNow() {
        if (!sorted) {
            if (top_k_limit > 0 && top_k_limit < rows.size()) {
                // Use partial sort to get top K elements
                std::vector<std::pair<SortKeyView, idx_t>> key_indices;
                for (idx_t i = 0; i < keys.size(); i++) {
                    key_indices.push_back({keys[i], i});
                }
                
                // Custom comparator for descending order
                auto comp = [](const std::pair<SortKeyView, idx_t>& a, const std::pair<SortKeyView, idx_t>& b) {
                    if (a.first.c != b.first.c) return a.first.c > b.first.c; // Descending
                    return a.second > b.second; // Tie-breaker for stability if needed
                };
                
                std::nth_element(key_indices.begin(), key_indices.begin() + top_k_limit, key_indices.end(), comp);
                
                // Sort the top K elements
                std::sort(key_indices.begin(), key_indices.begin() + top_k_limit, comp);
                
                // Create a new sorted_rows vector with only top K
                std::vector<SortedRow> sorted_rows;
                for (idx_t i = 0; i < top_k_limit && i < key_indices.size(); i++) {
                    sorted_rows.push_back(rows[key_indices[i].second]);
                }
                rows = std::move(sorted_rows);
                sorted = true;
            } else {
                // Full sort
                std::vector<std::pair<SortKeyView, idx_t>> key_indices;
                for (idx_t i = 0; i < keys.size(); i++) {
                    key_indices.push_back({keys[i], i});
                }
                
                auto comp = [](const std::pair<SortKeyView, idx_t>& a, const std::pair<SortKeyView, idx_t>& b) {
                    if (a.first.c != b.first.c) return a.first.c > b.first.c; // Descending
                    return a.second < b.second; // Stable sort using original index
                };
                
                std::sort(key_indices.begin(), key_indices.end(), comp);
                
                // Reorder rows based on sorted indices
                std::vector<SortedRow> sorted_rows;
                for (auto& pair : key_indices) {
                    sorted_rows.push_back(rows[pair.second]);
                }
                rows = std::move(sorted_rows);
                sorted = true;
            }
        }
    }
};

struct FnGlobalState : public GlobalTableFunctionState {
    std::unordered_map<GroupKey, AggState, GroupKeyHash> agg_map;
    // TODO: Optional accumulators/counters (generator may append to this struct)
    std::mutex lock;
    std::atomic<idx_t> active_local_states {0};
    std::atomic<idx_t> merged_local_states {0};
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
    names.push_back("WatchID");
    names.push_back("ClientIP");
    names.push_back("c");
    names.push_back("sum_isrefresh");
    names.push_back("avg_resolutionwidth");
    
    return_types.push_back(LogicalType::BIGINT); // WatchID
    return_types.push_back(LogicalType::INTEGER); // ClientIP
    return_types.push_back(LogicalType::BIGINT);  // c
    return_types.push_back(LogicalType::HUGEINT);  // sum_isrefresh (from authoritative mapping)
    return_types.push_back(LogicalType::DOUBLE);  // avg_resolutionwidth

    return make_uniq<FnBindData>();
}

static OperatorResultType FnExecute(ExecutionContext &, TableFunctionInput &in,
                                                    DataChunk &input, DataChunk &) {
    auto &l = in.local_state->Cast<FnLocalState>();

    
    if (input.size() == 0) {       
        return OperatorResultType::NEED_MORE_INPUT;
    }

    // Unpack input columns using UnifiedVectorFormat
    UnifiedVectorFormat WatchID_uvf;
    input.data[0].ToUnifiedFormat(input.size(), WatchID_uvf);
    int64_t* WatchID_ptr = (int64_t*)WatchID_uvf.data;
    
    UnifiedVectorFormat ClientIP_uvf;
    input.data[1].ToUnifiedFormat(input.size(), ClientIP_uvf);
    int32_t* ClientIP_ptr = (int32_t*)ClientIP_uvf.data;
    
    UnifiedVectorFormat IsRefresh_uvf;
    input.data[2].ToUnifiedFormat(input.size(), IsRefresh_uvf);
    int16_t* IsRefresh_ptr = (int16_t*)IsRefresh_uvf.data;
    
    UnifiedVectorFormat ResolutionWidth_uvf;
    input.data[3].ToUnifiedFormat(input.size(), ResolutionWidth_uvf);
    int16_t* ResolutionWidth_ptr = (int16_t*)ResolutionWidth_uvf.data;
    
    // Validity bitmaps
    auto &valid_WatchID = WatchID_uvf.validity;
    auto &valid_ClientIP = ClientIP_uvf.validity;
    auto &valid_IsRefresh = IsRefresh_uvf.validity;
    auto &valid_ResolutionWidth = ResolutionWidth_uvf.validity;
    
    const bool WatchID_all_valid = valid_WatchID.AllValid();
    const bool ClientIP_all_valid = valid_ClientIP.AllValid();
    const bool IsRefresh_all_valid = valid_IsRefresh.AllValid();
    const bool ResolutionWidth_all_valid = valid_ResolutionWidth.AllValid();
    
    // Process rows
    if (WatchID_all_valid && ClientIP_all_valid && IsRefresh_all_valid && ResolutionWidth_all_valid) {
        // Fast path: no NULLs
        for (idx_t row_idx = 0; row_idx < input.size(); ++row_idx) {
            idx_t i_WatchID = WatchID_uvf.sel->get_index(row_idx);
            idx_t i_ClientIP = ClientIP_uvf.sel->get_index(row_idx);
            idx_t i_IsRefresh = IsRefresh_uvf.sel->get_index(row_idx);
            idx_t i_ResolutionWidth = ResolutionWidth_uvf.sel->get_index(row_idx);
            
            int64_t v_WatchID = WatchID_ptr[i_WatchID];
            int32_t v_ClientIP = ClientIP_ptr[i_ClientIP];
            int16_t v_IsRefresh = IsRefresh_ptr[i_IsRefresh];
            int16_t v_ResolutionWidth = ResolutionWidth_ptr[i_ResolutionWidth];
            
            // TODO: Process the values in the fast path
            GroupKey key;
            key.WatchID = v_WatchID;
            key.ClientIP = v_ClientIP;
            
            auto &state = l.agg_map[key];
            state.count++;
            state.sum_isrefresh += v_IsRefresh;
            state.sum_resolutionwidth += v_ResolutionWidth;
            state.count_resolutionwidth++;
        }
    } else {
        // Slow path: at least one column may contain NULLs
        for (idx_t row_idx = 0; row_idx < input.size(); ++row_idx) {
            idx_t i_WatchID = WatchID_uvf.sel->get_index(row_idx);
            idx_t i_ClientIP = ClientIP_uvf.sel->get_index(row_idx);
            idx_t i_IsRefresh = IsRefresh_uvf.sel->get_index(row_idx);
            idx_t i_ResolutionWidth = ResolutionWidth_uvf.sel->get_index(row_idx);
            
            if (!WatchID_all_valid && !valid_WatchID.RowIsValid(i_WatchID)) {
                continue; // Skip row if WatchID is NULL
            }
            if (!ClientIP_all_valid && !valid_ClientIP.RowIsValid(i_ClientIP)) {
                continue; // Skip row if ClientIP is NULL
            }
            if (!IsRefresh_all_valid && !valid_IsRefresh.RowIsValid(i_IsRefresh)) {
                continue; // Skip row if IsRefresh is NULL
            }
            if (!ResolutionWidth_all_valid && !valid_ResolutionWidth.RowIsValid(i_ResolutionWidth)) {
                continue; // Skip row if ResolutionWidth is NULL
            }
            
            int64_t v_WatchID = WatchID_ptr[i_WatchID];
            int32_t v_ClientIP = ClientIP_ptr[i_ClientIP];
            int16_t v_IsRefresh = IsRefresh_ptr[i_IsRefresh];
            int16_t v_ResolutionWidth = ResolutionWidth_ptr[i_ResolutionWidth];
            
            // TODO: Process the values in the NULL-safe path
            GroupKey key;
            key.WatchID = v_WatchID;
            key.ClientIP = v_ClientIP;
            
            auto &state = l.agg_map[key];
            state.count++;
            state.sum_isrefresh += v_IsRefresh;
            state.sum_resolutionwidth += v_ResolutionWidth;
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
                auto &global_state = g.agg_map[key];
                global_state.count += state.count;
                global_state.sum_isrefresh += state.sum_isrefresh;
                global_state.sum_resolutionwidth += state.sum_resolutionwidth;
                global_state.count_resolutionwidth += state.count_resolutionwidth;
            }
            
            // Add rows to sort state
            for (const auto &entry : g.agg_map) {
                const GroupKey &key = entry.first;
                const AggState &state = entry.second;
                
                double avg_resolutionwidth = state.count_resolutionwidth > 0 ? 
                    (double)state.sum_resolutionwidth / state.count_resolutionwidth : 0.0;
                
                g.sort_state.AddRow(key.WatchID, key.ClientIP, state.count, state.sum_isrefresh, avg_resolutionwidth);
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
            // Sort the data
            g.sort_state.SortNow();
            
            // Output sorted results
            idx_t output_row = 0;
            for (const auto &row : g.sort_state.rows) {
                if (output_row >= out.size()) break; // Check if chunk is full
                
                out.SetValue(0, output_row, Value::BIGINT(row.WatchID));
                out.SetValue(1, output_row, Value::INTEGER(row.ClientIP));
                out.SetValue(2, output_row, Value::BIGINT(row.c));
                out.SetValue(3, output_row, Value::HUGEINT(row.sum_isrefresh));
                out.SetValue(4, output_row, Value::DOUBLE(row.avg_resolutionwidth));
                
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