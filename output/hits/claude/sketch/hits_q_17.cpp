/*
query_template: SELECT UserID, SearchPhrase, COUNT(*) AS cnt FROM hits GROUP BY UserID, SearchPhrase ORDER BY COUNT(*) DESC LIMIT 10;

split_template: select * from dbweaver((SELECT UserID, SearchPhrase FROM hits));
query_example: SELECT UserID, SearchPhrase, COUNT(*) AS cnt FROM hits GROUP BY UserID, SearchPhrase ORDER BY COUNT(*) DESC LIMIT 10;

split_query: select * from dbweaver((SELECT UserID, SearchPhrase FROM hits));
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

namespace duckdb {

struct FnBindData : public FunctionData {
    unique_ptr<FunctionData> Copy() const override { return make_uniq<FnBindData>(); }
    bool Equals(const FunctionData &) const override { return true; }
};

struct GroupKey {
    int64_t UserID;
    string_t SearchPhrase;
    
    bool operator==(const GroupKey& other) const {
        return UserID == other.UserID && SearchPhrase == other.SearchPhrase;
    }
};

struct GroupKeyHash {
    size_t operator()(const GroupKey& k) const {
        size_t h = 0;
        h ^= std::hash<int64_t>{}(k.UserID) + 0x9e3779b9 + (h << 6) + (h >> 2);
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
        // Descending order for cnt
        if (a.cnt != b.cnt) return a.cnt > b.cnt;
        return false;
    }
};

struct SortState {
    std::vector<SortKeyView> sort_buffer;
    std::vector<GroupKey> key_buffer;
    std::vector<AggState> value_buffer;
    bool sorted = false;
    
    inline void AddRow(const GroupKey &key, const AggState &value) {
        sort_buffer.push_back(SortKeyView{value.cnt});
        key_buffer.push_back(key);
        value_buffer.push_back(value);
    }
    
    inline void SortNow() {
        if (!sorted) {
            std::vector<size_t> indices(sort_buffer.size());
            std::iota(indices.begin(), indices.end(), 0);
            
            std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
                return SortRowComparator{}(sort_buffer[a], sort_buffer[b]);
            });
            
            std::vector<SortKeyView> sorted_sort_buffer;
            std::vector<GroupKey> sorted_key_buffer;
            std::vector<AggState> sorted_value_buffer;
            
            for (auto idx : indices) {
                sorted_sort_buffer.push_back(sort_buffer[idx]);
                sorted_key_buffer.push_back(key_buffer[idx]);
                sorted_value_buffer.push_back(value_buffer[idx]);
            }
            
            sort_buffer = std::move(sorted_sort_buffer);
            key_buffer = std::move(sorted_key_buffer);
            value_buffer = std::move(sorted_value_buffer);
            
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
    // Set up the output schema based on the authoritative mapping
    return_types.push_back(LogicalType::BIGINT);    // UserID
    return_types.push_back(LogicalType::VARCHAR);   // SearchPhrase
    return_types.push_back(LogicalType::BIGINT);    // cnt
    names.push_back("UserID");
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
    
    // Input column processing using UnifiedVectorFormat
    UnifiedVectorFormat UserID_uvf;
    input.data[0].ToUnifiedFormat(input.size(), UserID_uvf);
    int64_t* UserID_ptr = (int64_t*)UserID_uvf.data;
    
    UnifiedVectorFormat SearchPhrase_uvf;
    input.data[1].ToUnifiedFormat(input.size(), SearchPhrase_uvf);
    string_t* SearchPhrase_ptr = (string_t*)SearchPhrase_uvf.data;
    
    // validity bitmaps
    auto &valid_UserID     = UserID_uvf.validity;
    auto &valid_SearchPhrase = SearchPhrase_uvf.validity;
    
    const bool UserID_all_valid     = valid_UserID.AllValid();
    const bool SearchPhrase_all_valid = valid_SearchPhrase.AllValid();
    
    const idx_t num_rows = input.size();
    
    // Fast branch: all relevant columns have no NULLs in this batch
    if (UserID_all_valid && SearchPhrase_all_valid) {
        // --- Fast path: no per-row NULL checks ---
        for (idx_t row_idx = 0; row_idx < num_rows; ++row_idx) {
            // Directly load values without RowIsValid checks
            idx_t i_UserID = UserID_uvf.sel->get_index(row_idx);
            idx_t i_SearchPhrase = SearchPhrase_uvf.sel->get_index(row_idx);
            
            int64_t v_UserID = UserID_ptr[i_UserID];
            string_t v_SearchPhrase = SearchPhrase_ptr[i_SearchPhrase];
            
            // ======================================
            //  Core computation logic (no NULLs)
            //<<CORE_COMPUTE>>
            GroupKey key;
            key.UserID = v_UserID;
            key.SearchPhrase = v_SearchPhrase;
            l.agg_map[key].cnt++;
            // ============================
        }
    } else {
        // --- Slow path: at least one column may contain NULLs ---
        for (idx_t row_idx = 0; row_idx < num_rows; ++row_idx) {
            // For each column that is not fully valid, check this row
            idx_t i_UserID = UserID_uvf.sel->get_index(row_idx);
            idx_t i_SearchPhrase = SearchPhrase_uvf.sel->get_index(row_idx);
            
            if (!UserID_all_valid && !valid_UserID.RowIsValid(i_UserID)) {
                continue; // row is NULL in column UserID → skip
            }
            if (!SearchPhrase_all_valid && !valid_SearchPhrase.RowIsValid(i_SearchPhrase)) {
                continue;
            }
            
            // At this point, all required columns are valid for this row
            
            int64_t v_UserID = UserID_ptr[i_UserID];
            string_t v_SearchPhrase = SearchPhrase_ptr[i_SearchPhrase];
            
            // ======================================
            //  Core computation logic (NULL-safe)
            //<<CORE_COMPUTE>>
            GroupKey key;
            key.UserID = v_UserID;
            key.SearchPhrase = v_SearchPhrase;
            l.agg_map[key].cnt++;
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
        {
            std::lock_guard<std::mutex> guard(g.lock);
            // Transfer aggregated results to sort buffers
            for (const auto &entry : g.agg_map) {
                const GroupKey &key = entry.first;
                const AggState &state = entry.second;
                g.sort_state.AddRow(key, state);
            }
            
            // Perform the sort
            g.sort_state.SortNow();
            
            // Output sorted results
            idx_t output_row = 0;
            for (size_t i = 0; i < g.sort_state.key_buffer.size(); ++i) {
                if (output_row >= STANDARD_VECTOR_SIZE) {
                    out.SetCardinality(output_row);
                    return OperatorFinalizeResultType::HAVE_RESULT;
                }
                const GroupKey &key = g.sort_state.key_buffer[i];
                const AggState &state = g.sort_state.value_buffer[i];
                out.SetValue(0, output_row, Value::BIGINT(key.UserID));
                out.SetValue(1, output_row, Value(key.SearchPhrase));
                out.SetValue(2, output_row, Value::BIGINT(state.cnt));
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