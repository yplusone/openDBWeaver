/*
query_template: SELECT SearchPhrase, MIN(URL) AS min_url, COUNT(*) AS c FROM hits WHERE URL LIKE '%google%' AND SearchPhrase <> '' GROUP BY SearchPhrase ORDER BY c DESC LIMIT 10;

split_template: select * from dbweaver((SELECT URL, SearchPhrase FROM hits WHERE (contains(URL, 'google')) AND (SearchPhrase!='')));
query_example: SELECT SearchPhrase, MIN(URL) AS min_url, COUNT(*) AS c FROM hits WHERE URL LIKE '%google%' AND SearchPhrase <> '' GROUP BY SearchPhrase ORDER BY c DESC LIMIT 10;

split_query: select * from dbweaver((SELECT URL, SearchPhrase FROM hits WHERE (contains(URL, 'google')) AND (SearchPhrase!='')));
*/
#define DUCKDB_EXTENSION_MAIN

#include "dbweaver_extension.hpp"
#include "duckdb.hpp"
#include "duckdb/function/table_function.hpp"
#include <atomic>
#include <limits>
#include <mutex>
#include <unordered_map>
#include <functional>
#include <vector>
#include <algorithm>

namespace duckdb {

//TODO: Define any helper structs or functions needed for binding/execution

struct FnBindData : public FunctionData {
    unique_ptr<FunctionData> Copy() const override { return make_uniq<FnBindData>(); }
    bool Equals(const FunctionData &) const override { return true; }
};

struct GroupKey {
    string_t searchphrase;
    
    bool operator==(const GroupKey& other) const {
        return searchphrase == other.searchphrase;
    }
};

struct GroupKeyHash {
    size_t operator()(const GroupKey& k) const {
        size_t h = 0;
        h ^= duckdb::Hash(k.searchphrase.GetData(), k.searchphrase.GetSize()) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

struct AggState {
    string_t min_url;
    int64_t count = 0;
    bool initialized = false;
};

struct SortKeyView {
    int64_t c;
};

struct SortRowComparator {
    bool operator()(const SortKeyView &a, const SortKeyView &b) const {
        // DESC order: b comes before a means a > b
        return a.c > b.c;
    }
};

struct SortState {
    std::vector<SortKeyView> buffer;
    std::vector<GroupKey> group_keys;
    std::vector<AggState> agg_states;
    bool sorted = false;
    
    inline void AddRow(const GroupKey &key, const AggState &state) {
        buffer.push_back(SortKeyView{state.count});
        group_keys.push_back(key);
        agg_states.push_back(state);
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
    std::mutex lock;
    std::atomic<idx_t> active_local_states {0};
    std::atomic<idx_t> merged_local_states {0};
    std::unordered_map<GroupKey, AggState, GroupKeyHash> agg_map;
    idx_t MaxThreads() const override {
        return std::numeric_limits<idx_t>::max();
    }
    SortState sort_state;
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
    names.push_back("SearchPhrase");
    names.push_back("min_url");
    names.push_back("c");
    return_types.push_back(LogicalType::VARCHAR);
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
    UnifiedVectorFormat url_uvf;
    UnifiedVectorFormat searchphrase_uvf;
    
    // Load columns into UVF
    input.data[0].ToUnifiedFormat(input.size(), url_uvf);
    input.data[1].ToUnifiedFormat(input.size(), searchphrase_uvf);
    
    // Create typed pointers to physical data
    string_t* url_ptr = (string_t*)url_uvf.data;
    string_t* searchphrase_ptr = (string_t*)searchphrase_uvf.data;
    
    // Batch-level NULL summary (whether each column is fully non-NULL)
    // validity bitmaps
    auto &valid_url  = url_uvf.validity;
    auto &valid_searchphrase  = searchphrase_uvf.validity;
    // Add more columns as needed
    const bool url_all_valid = valid_url.AllValid();
    const bool searchphrase_all_valid = valid_searchphrase.AllValid();
    // Add more columns as needed:
    // const bool colN_all_valid = valid_<colN>.AllValid();
    
    const idx_t num_rows = input.size();
    
    // 2) FAST BRANCH: all relevant columns have no NULLs in this batch
    if (url_all_valid && searchphrase_all_valid /* && colN_all_valid ... */) {
        // --- Fast path: no per-row NULL checks ---
        for (idx_t row_idx = 0; row_idx < num_rows; ++row_idx) {
            // Directly load values without RowIsValid checks

            idx_t i_url = url_uvf.sel->get_index(row_idx);
            string_t v1 = url_ptr[i_url];
            idx_t i_searchphrase = searchphrase_uvf.sel->get_index(row_idx);
            string_t v2 = searchphrase_ptr[i_searchphrase];
            //  vN = <COL_N>_ptr[i_<COL_N>];
            
            // ======================================
            //  Core computation logic (no NULLs)
            //<<CORE_COMPUTE>>
            // Check filters: URL LIKE '%google%' AND SearchPhrase <> ''
            const char* url_str = v1.GetData();
            idx_t url_len = v1.GetSize();
            const char* searchphrase_str = v2.GetData();
            idx_t searchphrase_len = v2.GetSize();
            
            // Check URL LIKE '%google%'
            bool contains_google = false;
            for (idx_t i = 0; i <= url_len - 6; i++) {
                if (strncmp(url_str + i, "google", 6) == 0) {
                    contains_google = true;
                    break;
                }
            }
            
            // Check SearchPhrase <> ''
            bool searchphrase_not_empty = (searchphrase_len > 0);
            
            // Combined filter check
            if (!contains_google || !searchphrase_not_empty) {
                continue; // Skip this row
            }
            
            // Process aggregation
            GroupKey key;
            key.searchphrase = v2;
            
            auto &agg_state = l.agg_map[key];
            if (!agg_state.initialized) {
                agg_state.min_url = v1;
                agg_state.count = 1;
                agg_state.initialized = true;
            } else {
                // Update MIN
                if (v1 < agg_state.min_url) {
                    agg_state.min_url = v1;
                }
                agg_state.count++;
            }
            // ============================
        }
    } else {
        // --- Slow path: at least one column may contain NULLs ---
        for (idx_t row_idx = 0; row_idx < num_rows; ++row_idx) {
            // For each column that is not fully valid, check this row
            idx_t i_url = url_uvf.sel->get_index(row_idx);
            idx_t i_searchphrase = searchphrase_uvf.sel->get_index(row_idx);

            if (!url_all_valid && !valid_url.RowIsValid(i_url)) {
                continue; // row is NULL in column 1 → skip
            }
            if (!searchphrase_all_valid && !valid_searchphrase.RowIsValid(i_searchphrase)) {
                continue;
            }
            // Repeat for additional columns

            // At this point, all required columns are valid for this row

            string_t v1 = url_ptr[i_url];
            string_t v2 = searchphrase_ptr[i_searchphrase];
            // auto vN = <COL_N>_ptr[i_<COL_N>];
            
            // ======================================
            //  Core computation logic (NULL-safe)
            //<<CORE_COMPUTE>>
            // Check filters: URL LIKE '%google%' AND SearchPhrase <> ''
            const char* url_str = v1.GetData();
            idx_t url_len = v1.GetSize();
            const char* searchphrase_str = v2.GetData();
            idx_t searchphrase_len = v2.GetSize();
            
            // Check URL LIKE '%google%'
            bool contains_google = false;
            for (idx_t i = 0; i <= url_len - 6; i++) {
                if (strncmp(url_str + i, "google", 6) == 0) {
                    contains_google = true;
                    break;
                }
            }
            
            // Check SearchPhrase <> ''
            bool searchphrase_not_empty = (searchphrase_len > 0);
            
            // Combined filter check
            if (!contains_google || !searchphrase_not_empty) {
                continue; // Skip this row
            }
            
            // Process aggregation
            GroupKey key;
            key.searchphrase = v2;
            
            auto &agg_state = l.agg_map[key];
            if (!agg_state.initialized) {
                agg_state.min_url = v1;
                agg_state.count = 1;
                agg_state.initialized = true;
            } else {
                // Update MIN
                if (v1 < agg_state.min_url) {
                    agg_state.min_url = v1;
                }
                agg_state.count++;
            }
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
                auto &global_state = g.agg_map[key];
                if (!global_state.initialized) {
                    global_state = state;
                } else {
                    // Update MIN
                    if (state.min_url < global_state.min_url) {
                        global_state.min_url = state.min_url;
                    }
                    global_state.count += state.count;
                }
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
            // Add all aggregated results to sort state
            for (const auto &entry : g.agg_map) {
                const GroupKey &key = entry.first;
                const AggState &state = entry.second;
                g.sort_state.AddRow(key, state);
            }
            
            // Sort the results based on the sort keys
            g.sort_state.SortNow();
            
            // Output sorted results
            idx_t output_idx = 0;
            for (size_t i = 0; i < g.sort_state.buffer.size(); ++i) {
                const GroupKey &key = g.sort_state.group_keys[i];
                const AggState &state = g.sort_state.agg_states[i];
                
                out.SetValue(0, output_idx, Value(key.searchphrase.GetString()));
                out.SetValue(1, output_idx, Value(state.min_url.GetString()));
                out.SetValue(2, output_idx, Value::BIGINT(state.count));
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