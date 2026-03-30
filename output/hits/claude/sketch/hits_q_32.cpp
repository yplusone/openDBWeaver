/*
query_template: SELECT WatchID, ClientIP, COUNT(*) AS c, SUM(IsRefresh) AS sum_isrefresh, AVG(ResolutionWidth) AS avg_resolutionwidth FROM hits WHERE SearchPhrase <> '' GROUP BY WatchID, ClientIP ORDER BY c DESC LIMIT 10;

split_template: select * from dbweaver((SELECT WatchID, ClientIP, IsRefresh, ResolutionWidth FROM hits WHERE (SearchPhrase!='')));
query_example: SELECT WatchID, ClientIP, COUNT(*) AS c, SUM(IsRefresh) AS sum_isrefresh, AVG(ResolutionWidth) AS avg_resolutionwidth FROM hits WHERE SearchPhrase <> '' GROUP BY WatchID, ClientIP ORDER BY c DESC LIMIT 10;

split_query: select * from dbweaver((SELECT WatchID, ClientIP, IsRefresh, ResolutionWidth FROM hits WHERE (SearchPhrase!='')));
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
#include <queue>

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
    int64_t count_val = 0;
    int64_t sum_isrefresh = 0;
    int64_t sum_resolutionwidth = 0;
    int64_t count_resolutionwidth = 0;
};

struct SortKeyView {
    int64_t c;
};

struct SortRowComparator {
    bool operator()(const SortKeyView &a, const SortKeyView &b) const {
        if (a.c != b.c) return a.c > b.c; // DESC order
        return false; // No secondary key needed
    }
};

struct SortState {
    std::vector<SortKeyView> sort_keys;
    std::vector<GroupKey> group_keys;
    std::vector<AggState> agg_states;
    bool sorted = false;
    idx_t top_k_limit = 10; // limit 10

    inline void AddRow(const GroupKey &key, const AggState &state) {
        SortKeyView skv;
        skv.c = state.count_val; // 'c' corresponds to count_val
        
        if (top_k_limit != 0) {
            // Top-K case: use max-heap to keep smallest K elements
            if (sort_keys.size() < top_k_limit) {
                sort_keys.push_back(skv);
                group_keys.push_back(key);
                agg_states.push_back(state);
                
                if (sort_keys.size() == top_k_limit) {
                    std::make_heap(sort_keys.begin(), sort_keys.end(), SortRowComparator{});
                    // Create corresponding heap for group_keys and agg_states
                    std::vector<idx_t> indices(group_keys.size());
                    std::iota(indices.begin(), indices.end(), 0);
                    std::make_heap(indices.begin(), indices.end(), [this](idx_t i, idx_t j) {
                        return SortRowComparator{}(sort_keys[i], sort_keys[j]);
                    });
                    
                    // Reorder group_keys and agg_states based on indices
                    std::vector<GroupKey> temp_group_keys = group_keys;
                    std::vector<AggState> temp_agg_states = agg_states;
                    for(idx_t i = 0; i < group_keys.size(); ++i) {
                        group_keys[i] = temp_group_keys[indices[i]];
                        agg_states[i] = temp_agg_states[indices[i]];
                    }
                }
            } else {
                // Check if current element should replace the largest in heap
                if(SortRowComparator{}(skv, sort_keys[0])) {
                    sort_keys[0] = skv;
                    group_keys[0] = key;
                    agg_states[0] = state;
                    std::pop_heap(sort_keys.begin(), sort_keys.end(), SortRowComparator{});
                    std::push_heap(sort_keys.begin(), sort_keys.end(), SortRowComparator{});
                    
                    // Also need to reorder group_keys and agg_states accordingly
                    std::vector<idx_t> indices(group_keys.size());
                    std::iota(indices.begin(), indices.end(), 0);
                    std::make_heap(indices.begin(), indices.end(), [this](idx_t i, idx_t j) {
                        return SortRowComparator{}(sort_keys[i], sort_keys[j]);
                    });
                    
                    std::vector<GroupKey> temp_group_keys = group_keys;
                    std::vector<AggState> temp_agg_states = agg_states;
                    for(idx_t i = 0; i < group_keys.size(); ++i) {
                        group_keys[i] = temp_group_keys[indices[i]];
                        agg_states[i] = temp_agg_states[indices[i]];
                    }
                }
            }
        } else {
            // Full sort case
            sort_keys.push_back(skv);
            group_keys.push_back(key);
            agg_states.push_back(state);
        }
    }

    inline void SortNow() {
        if (!sorted) {
            if (top_k_limit != 0) {
                // Already maintained heap of top-k elements
                // Just need to sort the remaining heap
                std::sort_heap(sort_keys.begin(), sort_keys.end(), SortRowComparator{});
                
                // Sort group_keys and agg_states correspondingly
                std::vector<idx_t> indices(sort_keys.size());
                std::iota(indices.begin(), indices.end(), 0);
                std::sort(indices.begin(), indices.end(), [this](idx_t i, idx_t j) {
                    return SortRowComparator{}(sort_keys[i], sort_keys[j]);
                });
                
                std::vector<GroupKey> temp_group_keys = group_keys;
                std::vector<AggState> temp_agg_states = agg_states;
                for(idx_t i = 0; i < group_keys.size(); ++i) {
                    group_keys[i] = temp_group_keys[indices[i]];
                    agg_states[i] = temp_agg_states[indices[i]];
                }
            } else {
                // Full sort
                std::vector<idx_t> indices(sort_keys.size());
                std::iota(indices.begin(), indices.end(), 0);
                
                std::sort(indices.begin(), indices.end(), [this](idx_t i, idx_t j) {
                    return SortRowComparator{}(sort_keys[i], sort_keys[j]);
                });
                
                std::vector<GroupKey> temp_group_keys = group_keys;
                std::vector<AggState> temp_agg_states = agg_states;
                
                group_keys.clear();
                agg_states.clear();
                
                for(idx_t i = 0; i < indices.size(); ++i) {
                    idx_t src_idx = indices[i];
                    group_keys.push_back(temp_group_keys[src_idx]);
                    agg_states.push_back(temp_agg_states[src_idx]);
                }
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
    std::unordered_map<GroupKey, AggState, GroupKeyHash> agg_map;
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
    // Define the output schema based on the authoritative mapping
    names.emplace_back("WatchID");
    names.emplace_back("ClientIP");
    names.emplace_back("c");
    names.emplace_back("sum_isrefresh");
    names.emplace_back("avg_resolutionwidth");
    
    return_types.emplace_back(LogicalType::BIGINT);   // WatchID
    return_types.emplace_back(LogicalType::INTEGER);   // ClientIP
    return_types.emplace_back(LogicalType::BIGINT);    // c
    return_types.emplace_back(LogicalType::HUGEINT);   // sum_isrefresh
    return_types.emplace_back(LogicalType::DOUBLE);    // avg_resolutionwidth
    
    return make_uniq<FnBindData>();
}

static OperatorResultType FnExecute(ExecutionContext &, TableFunctionInput &in,
                                            DataChunk &input, DataChunk &) {
    auto &l = in.local_state->Cast<FnLocalState>();

    
    if (input.size() == 0) {       
        return OperatorResultType::NEED_MORE_INPUT;
    }

    //TODO: process input chunk and produce output
    
    // Declare and load UnifiedVectorFormat handles for input columns
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
    
    // Validity bitmaps for NULL checking
    auto &valid_WatchID = WatchID_uvf.validity;
    auto &valid_ClientIP = ClientIP_uvf.validity;
    auto &valid_IsRefresh = IsRefresh_uvf.validity;
    auto &valid_ResolutionWidth = ResolutionWidth_uvf.validity;
    
    const bool WatchID_all_valid = valid_WatchID.AllValid();
    const bool ClientIP_all_valid = valid_ClientIP.AllValid();
    const bool IsRefresh_all_valid = valid_IsRefresh.AllValid();
    const bool ResolutionWidth_all_valid = valid_ResolutionWidth.AllValid();
    
    // Process rows with optimized NULL handling
    if (WatchID_all_valid && ClientIP_all_valid && IsRefresh_all_valid && ResolutionWidth_all_valid) {
        // Fast path: no per-row NULL checks needed
        for (idx_t row_idx = 0; row_idx < input.size(); ++row_idx) {
            idx_t i_WatchID = WatchID_uvf.sel->get_index(row_idx);
            idx_t i_ClientIP = ClientIP_uvf.sel->get_index(row_idx);
            idx_t i_IsRefresh = IsRefresh_uvf.sel->get_index(row_idx);
            idx_t i_ResolutionWidth = ResolutionWidth_uvf.sel->get_index(row_idx);
            
            int64_t v_WatchID = WatchID_ptr[i_WatchID];
            int32_t v_ClientIP = ClientIP_ptr[i_ClientIP];
            int16_t v_IsRefresh = IsRefresh_ptr[i_IsRefresh];
            int16_t v_ResolutionWidth = ResolutionWidth_ptr[i_ResolutionWidth];
            
            // ======================================
            //  Core computation logic (no NULLs)
            //<<CORE_COMPUTE>>
            GroupKey key;
            key.WatchID = v_WatchID;
            key.ClientIP = v_ClientIP;
            
            auto &agg_state = l.agg_map[key];
            agg_state.count_val++;
            agg_state.sum_isrefresh += v_IsRefresh;
            agg_state.sum_resolutionwidth += v_ResolutionWidth;
            agg_state.count_resolutionwidth++;
            // ======================================
        }
    } else {
        // Slow path: at least one column may contain NULLs
        for (idx_t row_idx = 0; row_idx < input.size(); ++row_idx) {
            idx_t i_WatchID = WatchID_uvf.sel->get_index(row_idx);
            idx_t i_ClientIP = ClientIP_uvf.sel->get_index(row_idx);
            idx_t i_IsRefresh = IsRefresh_uvf.sel->get_index(row_idx);
            idx_t i_ResolutionWidth = ResolutionWidth_uvf.sel->get_index(row_idx);
            
            if (!WatchID_all_valid && !valid_WatchID.RowIsValid(i_WatchID)) {
                continue; // row is NULL in WatchID → skip
            }
            if (!ClientIP_all_valid && !valid_ClientIP.RowIsValid(i_ClientIP)) {
                continue; // row is NULL in ClientIP → skip
            }
            if (!IsRefresh_all_valid && !valid_IsRefresh.RowIsValid(i_IsRefresh)) {
                continue; // row is NULL in IsRefresh → skip
            }
            if (!ResolutionWidth_all_valid && !valid_ResolutionWidth.RowIsValid(i_ResolutionWidth)) {
                continue; // row is NULL in ResolutionWidth → skip
            }
            
            int64_t v_WatchID = WatchID_ptr[i_WatchID];
            int32_t v_ClientIP = ClientIP_ptr[i_ClientIP];
            int16_t v_IsRefresh = IsRefresh_ptr[i_IsRefresh];
            int16_t v_ResolutionWidth = ResolutionWidth_ptr[i_ResolutionWidth];
            
            // ======================================
            //  Core computation logic (NULL-safe)
            //<<CORE_COMPUTE>>
            GroupKey key;
            key.WatchID = v_WatchID;
            key.ClientIP = v_ClientIP;
            
            auto &agg_state = l.agg_map[key];
            agg_state.count_val++;
            agg_state.sum_isrefresh += v_IsRefresh;
            agg_state.sum_resolutionwidth += v_ResolutionWidth;
            agg_state.count_resolutionwidth++;
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
                
                auto &global_agg_state = g.agg_map[key];
                global_agg_state.count_val += state.count_val;
                global_agg_state.sum_isrefresh += state.sum_isrefresh;
                global_agg_state.sum_resolutionwidth += state.sum_resolutionwidth;
                global_agg_state.count_resolutionwidth += state.count_resolutionwidth;
                
                // Add to sort state
                g.sort_state.AddRow(key, global_agg_state);
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
            // Perform the sort operation
            g.sort_state.SortNow();
            
            // Output the sorted results
            idx_t output_row = 0;
            for (idx_t i = 0; i < g.sort_state.group_keys.size() && output_row < g.sort_state.top_k_limit; ++i) {
                const GroupKey &key = g.sort_state.group_keys[i];
                const AggState &state = g.sort_state.agg_states[i];
                
                double avg_resolutionwidth = state.count_resolutionwidth > 0 ? 
                    (double)state.sum_resolutionwidth / state.count_resolutionwidth : 0.0;
                
                // Set values according to the correct output types
                out.SetValue(0, output_row, Value::BIGINT(key.WatchID));
                out.SetValue(1, output_row, Value::INTEGER(key.ClientIP));
                out.SetValue(2, output_row, Value::BIGINT(state.count_val));
                out.SetValue(3, output_row, Value::HUGEINT(state.sum_isrefresh));  // Changed to HUGEINT as per mapping
                out.SetValue(4, output_row, Value::DOUBLE(avg_resolutionwidth));
                
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