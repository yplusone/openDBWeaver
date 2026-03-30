/*
query_template: SELECT SearchPhrase, MIN(URL) AS min_url, MIN(Title) AS min_title, COUNT(*) AS c, COUNT(DISTINCT UserID) AS cnt_distinct_userid FROM hits WHERE Title LIKE '%Google%' AND URL NOT LIKE '%.google.%' AND SearchPhrase <> '' GROUP BY SearchPhrase ORDER BY c DESC LIMIT 10;

split_template: select * from dbweaver((SELECT Title, URL, SearchPhrase, UserID FROM hits WHERE (contains(Title, 'Google')) AND ((NOT contains(URL, '.google.'))) AND (SearchPhrase!='')));
query_example: SELECT SearchPhrase, MIN(URL) AS min_url, MIN(Title) AS min_title, COUNT(*) AS c, COUNT(DISTINCT UserID) AS cnt_distinct_userid FROM hits WHERE Title LIKE '%Google%' AND URL NOT LIKE '%.google.%' AND SearchPhrase <> '' GROUP BY SearchPhrase ORDER BY c DESC LIMIT 10;

split_query: select * from dbweaver((SELECT Title, URL, SearchPhrase, UserID FROM hits WHERE (contains(Title, 'Google')) AND ((NOT contains(URL, '.google.'))) AND (SearchPhrase!='')));
*/
#define DUCKDB_EXTENSION_MAIN

#include "dbweaver_extension.hpp"
#include "duckdb.hpp"
#include "duckdb/function/table_function.hpp"
#include <atomic>
#include <limits>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <queue>

namespace duckdb {

//TODO: Define any helper structs or functions needed for binding/execution

struct FnBindData : public FunctionData {
    unique_ptr<FunctionData> Copy() const override { return make_uniq<FnBindData>(); }
    bool Equals(const FunctionData &) const override { return true; }
};

struct GroupKey {
    string_t search_phrase;
    
    bool operator==(const GroupKey& other) const {
        return search_phrase == other.search_phrase;
    }
};

struct GroupKeyHash {
    size_t operator()(const GroupKey& k) const {
        size_t h = 0;
        h ^= duckdb::Hash(k.search_phrase.GetData(), k.search_phrase.GetSize()) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

struct AggState {
    string_t min_url;
    bool min_url_initialized = false;
    string_t min_title;
    bool min_title_initialized = false;
    int64_t count = 0;
    std::unordered_set<int64_t> distinct_userids;
};

struct SortKeyView {
    int64_t c;
};

struct SortRowComparator {
    bool operator()(const SortKeyView &a, const SortKeyView &b) const {
        // DESC order for 'c'
        if (a.c != b.c) return a.c > b.c;
        return false;
    }
};

struct SortState {
    std::vector<SortKeyView> sort_buffer;
    std::vector<GroupKey> key_buffer;
    std::vector<AggState> value_buffer;
    bool sorted = false;
    
    inline void AddRow(const GroupKey &key, const AggState &state) {
        sort_buffer.push_back(SortKeyView{state.count});
        key_buffer.push_back(key);
        value_buffer.push_back(state);
    }
    
    inline void SortNow() {
        if (!sorted) {
            // Create index vector for stable sorting
            std::vector<size_t> indices(sort_buffer.size());
            std::iota(indices.begin(), indices.end(), 0);
            
            // Sort indices based on sort_buffer values
            std::sort(indices.begin(), indices.end(), [this](size_t a, size_t b) {
                return SortRowComparator{}(sort_buffer[a], sort_buffer[b]);
            });
            
            // Reorder all buffers based on sorted indices
            std::vector<SortKeyView> temp_sort(sort_buffer.size());
            std::vector<GroupKey> temp_key(key_buffer.size());
            std::vector<AggState> temp_value(value_buffer.size());
            
            for(size_t i = 0; i < indices.size(); ++i) {
                temp_sort[i] = sort_buffer[indices[i]];
                temp_key[i] = key_buffer[indices[i]];
                temp_value[i] = value_buffer[indices[i]];
            }
            
            sort_buffer = std::move(temp_sort);
            key_buffer = std::move(temp_key);
            value_buffer = std::move(temp_value);
            
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
    //TODO: populate return_types and names
    names.push_back("SearchPhrase");
    names.push_back("min_url");
    names.push_back("min_title");
    names.push_back("c");
    names.push_back("cnt_distinct_userid");
    
    return_types.push_back(LogicalType::VARCHAR); // SearchPhrase
    return_types.push_back(LogicalType::VARCHAR); // min_url
    return_types.push_back(LogicalType::VARCHAR); // min_title
    return_types.push_back(LogicalType::BIGINT);  // c
    return_types.push_back(LogicalType::BIGINT);  // cnt_distinct_userid

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
    UnifiedVectorFormat title_uvf;
    input.data[0].ToUnifiedFormat(input.size(), title_uvf);
    string_t* title_ptr = (string_t*)title_uvf.data;

    UnifiedVectorFormat url_uvf;
    input.data[1].ToUnifiedFormat(input.size(), url_uvf);
    string_t* url_ptr = (string_t*)url_uvf.data;

    UnifiedVectorFormat searchphrase_uvf;
    input.data[2].ToUnifiedFormat(input.size(), searchphrase_uvf);
    string_t* searchphrase_ptr = (string_t*)searchphrase_uvf.data;

    UnifiedVectorFormat userid_uvf;
    input.data[3].ToUnifiedFormat(input.size(), userid_uvf);
    int64_t* userid_ptr = (int64_t*)userid_uvf.data;

    // Validity bitmaps
    auto &valid_title = title_uvf.validity;
    auto &valid_url = url_uvf.validity;
    auto &valid_searchphrase = searchphrase_uvf.validity;
    auto &valid_userid = userid_uvf.validity;

    const bool title_all_valid = valid_title.AllValid();
    const bool url_all_valid = valid_url.AllValid();
    const bool searchphrase_all_valid = valid_searchphrase.AllValid();
    const bool userid_all_valid = valid_userid.AllValid();

    idx_t num_rows = input.size();

    if (title_all_valid && url_all_valid && searchphrase_all_valid && userid_all_valid) {
        // Fast path: no per-row NULL checks
        for (idx_t row_idx = 0; row_idx < num_rows; ++row_idx) {
            idx_t i_title = title_uvf.sel->get_index(row_idx);
            idx_t i_url = url_uvf.sel->get_index(row_idx);
            idx_t i_searchphrase = searchphrase_uvf.sel->get_index(row_idx);
            idx_t i_userid = userid_uvf.sel->get_index(row_idx);

            string_t v_title = title_ptr[i_title];
            string_t v_url = url_ptr[i_url];
            string_t v_searchphrase = searchphrase_ptr[i_searchphrase];
            int64_t v_userid = userid_ptr[i_userid];

            // Core computation logic (no NULLs)
            GroupKey key;
            key.search_phrase = v_searchphrase;
            
            auto &agg_state = l.agg_map[key];
            
            // Update min_url
            if (!agg_state.min_url_initialized || v_url < agg_state.min_url) {
                agg_state.min_url = v_url;
                agg_state.min_url_initialized = true;
            }
            
            // Update min_title
            if (!agg_state.min_title_initialized || v_title < agg_state.min_title) {
                agg_state.min_title = v_title;
                agg_state.min_title_initialized = true;
            }
            
            // Update count
            agg_state.count++;
            
            // Update distinct userid set
            agg_state.distinct_userids.insert(v_userid);
        }
    } else {
        // Slow path: at least one column may contain NULLs
        for (idx_t row_idx = 0; row_idx < num_rows; ++row_idx) {
            idx_t i_title = title_uvf.sel->get_index(row_idx);
            idx_t i_url = url_uvf.sel->get_index(row_idx);
            idx_t i_searchphrase = searchphrase_uvf.sel->get_index(row_idx);
            idx_t i_userid = userid_uvf.sel->get_index(row_idx);

            if (!title_all_valid && !valid_title.RowIsValid(i_title)) {
                continue; // row is NULL in column 1 → skip
            }
            if (!url_all_valid && !valid_url.RowIsValid(i_url)) {
                continue;
            }
            if (!searchphrase_all_valid && !valid_searchphrase.RowIsValid(i_searchphrase)) {
                continue;
            }
            if (!userid_all_valid && !valid_userid.RowIsValid(i_userid)) {
                continue;
            }

            string_t v_title = title_ptr[i_title];
            string_t v_url = url_ptr[i_url];
            string_t v_searchphrase = searchphrase_ptr[i_searchphrase];
            int64_t v_userid = userid_ptr[i_userid];

            // Core computation logic (NULL-safe)
            GroupKey key;
            key.search_phrase = v_searchphrase;
            
            auto &agg_state = l.agg_map[key];
            
            // Update min_url
            if (!agg_state.min_url_initialized || v_url < agg_state.min_url) {
                agg_state.min_url = v_url;
                agg_state.min_url_initialized = true;
            }
            
            // Update min_title
            if (!agg_state.min_title_initialized || v_title < agg_state.min_title) {
                agg_state.min_title = v_title;
                agg_state.min_title_initialized = true;
            }
            
            // Update count
            agg_state.count++;
            
            // Update distinct userid set
            agg_state.distinct_userids.insert(v_userid);
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
                const AggState &local_state = entry.second;
                
                auto &global_state = g.agg_map[key];
                
                // Update min_url
                if (!global_state.min_url_initialized || 
                    (local_state.min_url_initialized && local_state.min_url < global_state.min_url)) {
                    global_state.min_url = local_state.min_url;
                    global_state.min_url_initialized = true;
                }
                
                // Update min_title
                if (!global_state.min_title_initialized || 
                    (local_state.min_title_initialized && local_state.min_title < global_state.min_title)) {
                    global_state.min_title = local_state.min_title;
                    global_state.min_title_initialized = true;
                }
                
                // Update count
                global_state.count += local_state.count;
                
                // Update distinct userids
                for (const auto &uid : local_state.distinct_userids) {
                    global_state.distinct_userids.insert(uid);
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
            // Populate sort state with global agg map data
            for (const auto &entry : g.agg_map) {
                const GroupKey &key = entry.first;
                const AggState &state = entry.second;
                g.sort_state.AddRow(key, state);
            }
            
            // Perform the sort
            g.sort_state.SortNow();
            
            // Output sorted results
            idx_t output_idx = 0;
            for (size_t i = 0; i < g.sort_state.key_buffer.size(); ++i) {
                const GroupKey &key = g.sort_state.key_buffer[i];
                const AggState &state = g.sort_state.value_buffer[i];
                
                // Set output values
                out.SetValue(0, output_idx, key.search_phrase);
                
                if (state.min_url_initialized) {
                    out.SetValue(1, output_idx, state.min_url);
                } else {
                    out.SetValue(1, output_idx, Value(LogicalType::VARCHAR));
                }
                
                if (state.min_title_initialized) {
                    out.SetValue(2, output_idx, state.min_title);
                } else {
                    out.SetValue(2, output_idx, Value(LogicalType::VARCHAR));
                }
                
                out.SetValue(3, output_idx, state.count);
                out.SetValue(4, output_idx, (int64_t)state.distinct_userids.size());
                
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