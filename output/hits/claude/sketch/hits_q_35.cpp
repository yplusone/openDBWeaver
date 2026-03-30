/*
query_template: SELECT 1, URL, COUNT(*) AS c FROM hits GROUP BY 1, URL ORDER BY c DESC LIMIT 10;

split_template: select * from dbweaver((SELECT URL FROM hits));
query_example: SELECT 1, URL, COUNT(*) AS c FROM hits GROUP BY 1, URL ORDER BY c DESC LIMIT 10;

split_query: select * from dbweaver((SELECT URL FROM hits));
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
#include <functional>
//TODO: Add more includes as needed

namespace duckdb {

//TODO: Define any helper structs or functions needed for binding/execution

struct FnBindData : public FunctionData {
    unique_ptr<FunctionData> Copy() const override { return make_uniq<FnBindData>(); }
    bool Equals(const FunctionData &) const override { return true; }
};

struct GroupKey {
    string_t url;
    
    bool operator==(const GroupKey& other) const {
        return url == other.url;
    }
};

struct GroupKeyHash {
    size_t operator()(const GroupKey& k) const {
        size_t h = 0;
        h ^= duckdb::Hash(k.url.GetData(), k.url.GetSize()) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

struct AggState {
    int64_t count_val = 0;
};

struct SortKeyView {
    int64_t c;
    string_t url;
};

struct SortRowComparator {
    bool operator()(const SortKeyView &a, const SortKeyView &b) const {
        // Sort by c DESC
        if (a.c != b.c) {
            return a.c > b.c; // DESC order
        }
        // Then by url for stability
        return std::strcmp(a.url.GetData(), b.url.GetData()) < 0;
    }
};

struct SortState {
    std::vector<SortKeyView> buffer;
    bool sorted = false;
    idx_t top_k = 10; // Limit is 10
    
    inline void AddRow(string_t url, int64_t c) {
        buffer.push_back(SortKeyView{c, url});
    }
    
    inline void SortNow() {
        if (!sorted) {
            // Use bounded top-k approach with a max-heap of size K
            if (top_k > 0 && buffer.size() > top_k) {
                // Create a max-heap of size K
                std::make_heap(buffer.begin(), buffer.begin() + top_k, SortRowComparator{});
                
                for (size_t i = top_k; i < buffer.size(); i++) {
                    // If current element is smaller than heap top, replace it
                    if (SortRowComparator{}(buffer[i], buffer[0])) {
                        std::pop_heap(buffer.begin(), buffer.begin() + i + 1, SortRowComparator{});
                        std::swap(buffer[0], buffer[i]);
                        std::push_heap(buffer.begin(), buffer.begin() + top_k, SortRowComparator{});
                    }
                }
                
                // Now we have the smallest K elements at the beginning
                // Sort them in descending order
                std::sort(buffer.begin(), buffer.begin() + top_k, SortRowComparator{});
            } else {
                // Full sort if less than K elements
                std::sort(buffer.begin(), buffer.end(), SortRowComparator{});
                if (buffer.size() > top_k) {
                    buffer.resize(top_k);
                }
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
    idx_t MaxThreads() const override {
        return std::numeric_limits<idx_t>::max();
    }
    
    SortState sort_state;
};

static unique_ptr<GlobalTableFunctionState> FnInit(ClientContext &, TableFunctionInitInput &) {
    return make_uniq<FnGlobalState>();
}

struct FnLocalState : public LocalTableFunctionState {
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
    // Define the output schema based on authoritative mapping
    // "1" -> INTEGER, "URL" -> VARCHAR, "c" -> BIGINT
    
    // Add the INTEGER column (1)
    return_types.emplace_back(LogicalType::INTEGER);
    names.emplace_back("1");
    
    // Add the VARCHAR column (URL)
    return_types.emplace_back(LogicalType::VARCHAR);
    names.emplace_back("URL");
    
    // Add the BIGINT column (c)
    return_types.emplace_back(LogicalType::BIGINT);
    names.emplace_back("c");

    return make_uniq<FnBindData>();
}

static OperatorResultType FnExecute(ExecutionContext &, TableFunctionInput &in,
                                            DataChunk &input, DataChunk &) {
    auto &l = in.local_state->Cast<FnLocalState>();

    
    if (input.size() == 0) {       
        return OperatorResultType::NEED_MORE_INPUT;
    }

    // Unpack input columns using UnifiedVectorFormat
    UnifiedVectorFormat url_uvf;
    input.data[0].ToUnifiedFormat(input.size(), url_uvf);
    string_t* url_ptr = (string_t*)url_uvf.data;

    UnifiedVectorFormat c_uvf;
    input.data[1].ToUnifiedFormat(input.size(), c_uvf);
    int64_t* c_ptr = (int64_t*)c_uvf.data;

    // 1) Batch-level NULL summary (whether each column is fully non-NULL)
    // validity bitmaps
    auto &valid_url  = url_uvf.validity;
    auto &valid_c  = c_uvf.validity;
    // Add more columns as needed
    const bool url_all_valid = valid_url.AllValid();
    const bool c_all_valid = valid_c.AllValid();
    // Add more columns as needed:
    // const bool colN_all_valid = valid_<colN>.AllValid();

    // 2) FAST BRANCH: all relevant columns have no NULLs in this batch
    if (url_all_valid && c_all_valid /* && colN_all_valid ... */) {
        // --- Fast path: no per-row NULL checks ---
        for (idx_t row_idx = 0; row_idx < input.size(); ++row_idx) {
            // Directly load values without RowIsValid checks

            idx_t i_url = url_uvf.sel->get_index(row_idx);
            idx_t i_c = c_uvf.sel->get_index(row_idx);
            string_t v_url = url_ptr[i_url];
            int64_t v_c = c_ptr[i_c];
            //  vN = <COL_N>_ptr[i_<COL_N>];

            // ======================================
            //  Core computation logic (no NULLs)
            GroupKey key;
            key.url = v_url;
            auto &state = l.agg_map[key];
            state.count_val++;
            
            // Add to sort state
            auto &g = in.global_state->Cast<FnGlobalState>();
            {
                std::lock_guard<std::mutex> guard(g.lock);
                g.sort_state.AddRow(v_url, v_c);
            }
            //<<CORE_COMPUTE>>
            // ============================
        }
    } else {
        // --- Slow path: at least one column may contain NULLs ---
        for (idx_t row_idx = 0; row_idx < input.size(); ++row_idx) {
            // For each column that is not fully valid, check this row
            idx_t i_url = url_uvf.sel->get_index(row_idx);
            idx_t i_c = c_uvf.sel->get_index(row_idx);

            if (!url_all_valid && !valid_url.RowIsValid(i_url)) {
                continue; // row is NULL in column 1 → skip
            }
            if (!c_all_valid && !valid_c.RowIsValid(i_c)) {
                continue; // row is NULL in column 2 → skip
            }
            // Repeat for additional columns

            // At this point, all required columns are valid for this row

            string_t v_url = url_ptr[i_url];
            int64_t v_c = c_ptr[i_c];
            // auto vN = <COL_N>_ptr[i_<COL_N>];

            // ======================================
            //  Core computation logic (NULL-safe)
            GroupKey key;
            key.url = v_url;
            auto &state = l.agg_map[key];
            state.count_val++;
            
            // Add to sort state
            auto &g = in.global_state->Cast<FnGlobalState>();
            {
                std::lock_guard<std::mutex> guard(g.lock);
                g.sort_state.AddRow(v_url, v_c);
            }
            //<<CORE_COMPUTE>>
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
        // Perform the sort operation
        g.sort_state.SortNow();
        
        {
            std::lock_guard<std::mutex> guard(g.lock);
            idx_t out_idx = 0;
            
            // Output the sorted results up to the limit
            idx_t limit = std::min(static_cast<idx_t>(g.sort_state.buffer.size()), static_cast<idx_t>(g.sort_state.top_k));
            for (idx_t i = 0; i < limit && out_idx < out.size(); ++i) {
                const SortKeyView &row = g.sort_state.buffer[i];
                // Set the first column ("1") to index value (INTEGER)
                out.SetValue(0, out_idx, Value::INTEGER(static_cast<int32_t>(out_idx + 1)));
                // Set the second column ("URL") to the URL value (VARCHAR)
                out.SetValue(1, out_idx, Value(row.url));
                // Set the third column ("c") to the count value (BIGINT)
                out.SetValue(2, out_idx, Value::BIGINT(row.c));
                out_idx++;
            }
            out.SetCardinality(out_idx);
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