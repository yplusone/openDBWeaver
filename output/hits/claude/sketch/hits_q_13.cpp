/*
query_template: SELECT SearchPhrase, COUNT(*) AS c FROM hits WHERE SearchPhrase <> '' GROUP BY SearchPhrase ORDER BY c DESC LIMIT 10;

split_template: select * from dbweaver((SELECT SearchPhrase FROM hits WHERE (SearchPhrase!='')));
query_example: SELECT SearchPhrase, COUNT(*) AS c FROM hits WHERE SearchPhrase <> '' GROUP BY SearchPhrase ORDER BY c DESC LIMIT 10;

split_query: select * from dbweaver((SELECT SearchPhrase FROM hits WHERE (SearchPhrase!='')));
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
    int64_t count_val = 0;
};

struct SortKeyView {
    string_t search_phrase;
    int64_t c;
};

struct SortRowComparator {
    bool operator()(const SortKeyView &a, const SortKeyView &b) const {
        // Compare by 'c' in DESC order
        if (a.c != b.c) {
            return a.c > b.c; // DESC order
        }
        // If 'c' values are equal, compare by search_phrase for stability
        return std::strcmp(a.search_phrase.GetData(), a.search_phrase.GetSize(),
                          b.search_phrase.GetData(), b.search_phrase.GetSize()) < 0;
    }
};

struct SortState {
    std::vector<SortKeyView> buffer;
    bool sorted = false;
    const idx_t k_limit = 10; // Top-K limit
    
    inline void AddRow(string_t search_phrase, int64_t c) {
        buffer.push_back(SortKeyView{search_phrase, c});
    }
    
    inline void SortNow() {
        if (!sorted) {
            if (k_limit != 0 && buffer.size() > k_limit) {
                // Use partial sort to get top-k elements
                std::partial_sort(buffer.begin(), 
                                buffer.begin() + std::min((size_t)k_limit, buffer.size()), 
                                buffer.end(), 
                                SortRowComparator{});
                // Resize to keep only top-k elements
                if (buffer.size() > k_limit) {
                    buffer.resize(k_limit);
                }
            } else {
                // Full sort
                std::sort(buffer.begin(), buffer.end(), SortRowComparator{});
            }
            sorted = true;
        }
    }
};

struct FnGlobalState : public GlobalTableFunctionState {
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
    // Set up return types and names according to output schema
    return_types.push_back(LogicalType::VARCHAR);
    return_types.push_back(LogicalType::BIGINT);
    names.push_back("SearchPhrase");
    names.push_back("c");

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
    input.data[0].ToUnifiedFormat(input.size(), SearchPhrase_uvf);
    string_t* SearchPhrase_ptr = (string_t*)SearchPhrase_uvf.data;

    UnifiedVectorFormat c_uvf;
    input.data[1].ToUnifiedFormat(input.size(), c_uvf);
    int64_t* c_ptr = (int64_t*)c_uvf.data;

    // 1) Batch-level NULL summary (whether each column is fully non-NULL)
    // validity bitmaps
    auto &valid_SearchPhrase = SearchPhrase_uvf.validity;
    const bool SearchPhrase_all_valid = valid_SearchPhrase.AllValid();
    auto &valid_c = c_uvf.validity;
    const bool c_all_valid = valid_c.AllValid();

    // 2) FAST BRANCH: all relevant columns have no NULLs in this batch
    if (SearchPhrase_all_valid && c_all_valid) {
        // --- Fast path: no per-row NULL checks ---
        for (idx_t row_idx = 0; row_idx < input.size(); ++row_idx) {
            // Directly load values without RowIsValid checks

            idx_t i_SearchPhrase = SearchPhrase_uvf.sel->get_index(row_idx);
            idx_t i_c = c_uvf.sel->get_index(row_idx);
            string_t v1 = SearchPhrase_ptr[i_SearchPhrase];
            int64_t v2 = c_ptr[i_c];
            //  vN = <COL_N>_ptr[i_<COL_N>];

            // ======================================
            //  Core computation logic (no NULLs)
            //<<CORE_COMPUTE>>
            GroupKey key;
            key.search_phrase = v1;
            auto &state = l.agg_map[key];
            state.count_val++;
            // ============================
        }
    } else {
        // --- Slow path: at least one column may contain NULLs ---
        for (idx_t row_idx = 0; row_idx < input.size(); ++row_idx) {
            // For each column that is not fully valid, check this row
            idx_t i_SearchPhrase = SearchPhrase_uvf.sel->get_index(row_idx);
            idx_t i_c = c_uvf.sel->get_index(row_idx);

            if (!SearchPhrase_all_valid && !valid_SearchPhrase.RowIsValid(i_SearchPhrase)) {
                continue; // row is NULL in column 1 → skip
            }
            if (!c_all_valid && !valid_c.RowIsValid(i_c)) {
                continue; // row is NULL in column 2 → skip
            }
            // Repeat for additional columns

            // At this point, all required columns are valid for this row

            string_t v1 = SearchPhrase_ptr[i_SearchPhrase];
            int64_t v2 = c_ptr[i_c];
            // auto vN = <COL_N>_ptr[i_<COL_N>];

            // ======================================
            //  Core computation logic (NULL-safe)
            //<<CORE_COMPUTE>>
            GroupKey key;
            key.search_phrase = v1;
            auto &state = l.agg_map[key];
            state.count_val++;
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
                g.agg_map[key].count_val += state.count_val;
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
            // Transfer aggregated data to sort buffer
            for (const auto &entry : g.agg_map) {
                const GroupKey &key = entry.first;
                const AggState &state = entry.second;
                g.sort_state.AddRow(key.search_phrase, state.count_val);
            }
            
            // Sort the data
            g.sort_state.SortNow();
            
            // Output sorted data
            idx_t output_idx = 0;
            for (const auto &item : g.sort_state.buffer) {
                out.SetValue(0, output_idx, Value(item.search_phrase.GetString()));
                out.SetValue(1, output_idx, Value::BIGINT(item.c));
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