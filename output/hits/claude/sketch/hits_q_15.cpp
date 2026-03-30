/*
query_template: SELECT SearchEngineID, SearchPhrase, COUNT(*) AS c FROM hits WHERE SearchPhrase <> '' GROUP BY SearchEngineID, SearchPhrase ORDER BY c DESC LIMIT 10;

split_template: select * from dbweaver((SELECT SearchPhrase, SearchEngineID FROM hits WHERE (SearchPhrase!='')));
query_example: SELECT SearchEngineID, SearchPhrase, COUNT(*) AS c FROM hits WHERE SearchPhrase <> '' GROUP BY SearchEngineID, SearchPhrase ORDER BY c DESC LIMIT 10;

split_query: select * from dbweaver((SELECT SearchPhrase, SearchEngineID FROM hits WHERE (SearchPhrase!='')));
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

//TODO: Define any helper structs or functions needed for binding/execution

struct FnBindData : public FunctionData {
    unique_ptr<FunctionData> Copy() const override { return make_uniq<FnBindData>(); }
    bool Equals(const FunctionData &) const override { return true; }
};

struct GroupKey {
    int16_t SearchEngineID;
    string_t SearchPhrase;
    
    bool operator==(const GroupKey& other) const {
        return SearchEngineID == other.SearchEngineID && SearchPhrase == other.SearchPhrase;
    }
};

struct GroupKeyHash {
    size_t operator()(const GroupKey& k) const {
        size_t h = 0;
        h ^= std::hash<int16_t>{}(k.SearchEngineID) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= duckdb::Hash(k.SearchPhrase.GetData(), k.SearchPhrase.GetSize()) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

struct AggState {
    int64_t count = 0;
};

struct SortKeyView {
    int64_t c;
};

struct SortRowComparator {
    bool operator()(const SortKeyView &a, const SortKeyView &b) const {
        // DESC order for c
        if (a.c != b.c) return a.c > b.c;  // Greater means descending
        return false; // No tie-breaker since stable is false
    }
};

struct SortRowRef {
    GroupKey key;
    AggState state;
};

struct SortState {
    std::vector<SortRowRef> buffer;
    std::vector<SortKeyView> key_views;
    bool sorted = false;
    idx_t top_k_limit = 10; // limit 10
    
    inline void AddRow(const GroupKey &k, const AggState &s) {
        SortRowRef row_ref;
        row_ref.key = k;
        row_ref.state = s;
        buffer.push_back(row_ref);
        
        SortKeyView key_view;
        key_view.c = s.count;
        key_views.push_back(key_view);
    }
    
    inline void SortNow() {
        if (!sorted) {
            if (top_k_limit > 0 && top_k_limit < buffer.size()) {
                // Use nth_element to find top-K elements
                std::vector<size_t> indices(buffer.size());
                std::iota(indices.begin(), indices.end(), 0);
                
                std::nth_element(indices.begin(), 
                               indices.begin() + top_k_limit, 
                               indices.end(), 
                               [this](size_t i, size_t j) {
                                   return SortRowComparator{}(key_views[i], key_views[j]);
                               });
                
                // Keep only top K elements
                std::vector<SortRowRef> new_buffer;
                std::vector<SortKeyView> new_key_views;
                
                for (size_t i = 0; i < top_k_limit && i < indices.size(); ++i) {
                    new_buffer.push_back(buffer[indices[i]]);
                    new_key_views.push_back(key_views[indices[i]]);
                }
                
                buffer = std::move(new_buffer);
                key_views = std::move(new_key_views);
                
                // Now sort the top K
                std::sort(key_views.begin(), key_views.end(), SortRowComparator{});
                
                // Reorder buffer according to sorted key_views
                std::vector<SortRowRef> ordered_buffer;
                for (size_t i = 0; i < key_views.size(); ++i) {
                    for (size_t j = 0; j < buffer.size(); ++j) {
                        if (key_views[i].c == SortKeyView{buffer[j].state.count}.c) {
                            ordered_buffer.push_back(buffer[j]);
                            break;
                        }
                    }
                }
                buffer = std::move(ordered_buffer);
            } else {
                // Full sort
                std::vector<std::pair<SortKeyView, size_t>> pairs;
                for (size_t i = 0; i < key_views.size(); ++i) {
                    pairs.push_back({key_views[i], i});
                }
                
                std::sort(pairs.begin(), pairs.end(), [](const std::pair<SortKeyView, size_t> &a, 
                                                     const std::pair<SortKeyView, size_t> &b) {
                    return SortRowComparator{}(a.first, b.first);
                });
                
                // Reorder buffer based on sorted indices
                std::vector<SortRowRef> ordered_buffer;
                for (auto &p : pairs) {
                    ordered_buffer.push_back(buffer[p.second]);
                }
                buffer = std::move(ordered_buffer);
            }
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
    // Define output schema based on authoritative mapping
    names.push_back("SearchEngineID");
    names.push_back("SearchPhrase");
    names.push_back("c");
    return_types.push_back(LogicalType::SMALLINT);
    return_types.push_back(LogicalType::VARCHAR);
    return_types.push_back(LogicalType::BIGINT);

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
    UnifiedVectorFormat SearchPhrase_uvf;
    UnifiedVectorFormat SearchEngineID_uvf;
    
    // Load input columns into UVF
    input.data[0].ToUnifiedFormat(input.size(), SearchPhrase_uvf);
    input.data[1].ToUnifiedFormat(input.size(), SearchEngineID_uvf);
    
    // Create typed pointers to physical data
    string_t* SearchPhrase_ptr = (string_t*)SearchPhrase_uvf.data;
    int16_t* SearchEngineID_ptr = (int16_t*)SearchEngineID_uvf.data;
    
    // Validity bitmaps for NULL checking
    auto &valid_SearchPhrase = SearchPhrase_uvf.validity;
    auto &valid_SearchEngineID = SearchEngineID_uvf.validity;
    
    const bool SearchPhrase_all_valid = valid_SearchPhrase.AllValid();
    const bool SearchEngineID_all_valid = valid_SearchEngineID.AllValid();
    
    // Process the input chunk with NULL-aware loop
    if (SearchPhrase_all_valid && SearchEngineID_all_valid) {
        // Fast path: no per-row NULL checks
        for (idx_t row_idx = 0; row_idx < input.size(); ++row_idx) {
            idx_t i_SearchPhrase = SearchPhrase_uvf.sel->get_index(row_idx);
            idx_t i_SearchEngineID = SearchEngineID_uvf.sel->get_index(row_idx);
            
            string_t v_SearchPhrase = SearchPhrase_ptr[i_SearchPhrase];
            int16_t v_SearchEngineID = SearchEngineID_ptr[i_SearchEngineID];
            
            // Core computation logic goes here
            GroupKey key;
            key.SearchEngineID = v_SearchEngineID;
            key.SearchPhrase = v_SearchPhrase;
            
            auto &state = l.agg_map[key];
            state.count++;
        }
    } else {
        // Slow path: at least one column may contain NULLs
        for (idx_t row_idx = 0; row_idx < input.size(); ++row_idx) {
            idx_t i_SearchPhrase = SearchPhrase_uvf.sel->get_index(row_idx);
            idx_t i_SearchEngineID = SearchEngineID_uvf.sel->get_index(row_idx);
            
            if (!SearchPhrase_all_valid && !valid_SearchPhrase.RowIsValid(i_SearchPhrase)) {
                continue; // row is NULL in SearchPhrase → skip
            }
            if (!SearchEngineID_all_valid && !valid_SearchEngineID.RowIsValid(i_SearchEngineID)) {
                continue; // row is NULL in SearchEngineID → skip
            }
            
            string_t v_SearchPhrase = SearchPhrase_ptr[i_SearchPhrase];
            int16_t v_SearchEngineID = SearchEngineID_ptr[i_SearchEngineID];
            
            // Core computation logic goes here
            GroupKey key;
            key.SearchEngineID = v_SearchEngineID;
            key.SearchPhrase = v_SearchPhrase;
            
            auto &state = l.agg_map[key];
            state.count++;
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
                g.agg_map[key].count += state.count;
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
            // Build sort buffer from global agg_map
            for (const auto &entry : g.agg_map) {
                const GroupKey &key = entry.first;
                const AggState &state = entry.second;
                g.sort_state.AddRow(key, state);
            }
            
            // Perform the sort
            g.sort_state.SortNow();
            
            // Output the sorted results
            idx_t output_row_idx = 0;
            for (size_t i = 0; i < g.sort_state.buffer.size() && output_row_idx < out.size(); ++i) {
                const GroupKey &key = g.sort_state.buffer[i].key;
                const AggState &state = g.sort_state.buffer[i].state;
                
                out.SetValue(0, output_row_idx, Value::SMALLINT(key.SearchEngineID));
                out.SetValue(1, output_row_idx, Value(key.SearchPhrase));
                out.SetValue(2, output_row_idx, Value::BIGINT(state.count));
                output_row_idx++;
            }
            out.SetCardinality(output_row_idx);
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