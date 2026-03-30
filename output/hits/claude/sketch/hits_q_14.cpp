/*
query_template: SELECT SearchPhrase, COUNT(DISTINCT UserID) AS u FROM hits WHERE SearchPhrase <> '' GROUP BY SearchPhrase ORDER BY u DESC LIMIT 10;

split_template: select * from dbweaver((SELECT SearchPhrase, UserID FROM hits WHERE (SearchPhrase!='')));
query_example: SELECT SearchPhrase, COUNT(DISTINCT UserID) AS u FROM hits WHERE SearchPhrase <> '' GROUP BY SearchPhrase ORDER BY u DESC LIMIT 10;

split_query: select * from dbweaver((SELECT SearchPhrase, UserID FROM hits WHERE (SearchPhrase!='')));
*/
#define DUCKDB_EXTENSION_MAIN

#include "dbweaver_extension.hpp"
#include "duckdb.hpp"
#include "duckdb/function/table_function.hpp"
#include <string>
#include <atomic>
#include <limits>
#include <mutex>
#include <unordered_map>
#include <functional>
#include <vector>
#include <queue>
#include <algorithm>

namespace duckdb {

//TODO: Define any helper structs or functions needed for binding/execution
            
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
    std::unordered_set<int64_t> distinct_user_ids;
};

struct SortKeyView {
    int64_t u;
    string_t SearchPhrase;
};

struct SortRowComparator {
    bool operator()(const SortKeyView &a, const SortKeyView &b) const {
        // Sort by 'u' DESC
        if (a.u != b.u) return a.u > b.u;  // DESC order
        return false; // Tie-breaker if needed but not specified in sort keys
    }
};

struct SortState {
    std::vector<SortKeyView> buffer;
    bool sorted = false;
    idx_t top_k_limit = 0;

    SortState(idx_t limit = 0) : top_k_limit(limit) {}

    inline void AddRow(string_t search_phrase, int64_t u_value) {
        SortKeyView view;
        view.SearchPhrase = search_phrase;
        view.u = u_value;
        
        if (top_k_limit > 0) {
            // For Top-K, maintain a max-heap of size K
            if (buffer.size() < top_k_limit) {
                buffer.push_back(view);
                if (buffer.size() == top_k_limit) {
                    // Convert to max-heap when we reach the limit
                    std::make_heap(buffer.begin(), buffer.end(), SortRowComparator{});
                }
            } else {
                // If current value is smaller than max in heap, replace it
                if (SortRowComparator{}(view, buffer[0])) {
                    std::pop_heap(buffer.begin(), buffer.end(), SortRowComparator{});
                    buffer.back() = view;
                    std::push_heap(buffer.begin(), buffer.end(), SortRowComparator{});
                }
            }
        } else {
            // For full sort, just add to buffer
            buffer.push_back(view);
        }
    }

    inline void SortNow() {
        if (!sorted) {
            if (top_k_limit == 0) {
                // Full sort
                std::sort(buffer.begin(), buffer.end(), SortRowComparator{});
            } else {
                // For Top-K, we have a max-heap, so we need to sort in ascending order
                // to get the smallest at the end, then reverse
                std::sort_heap(buffer.begin(), buffer.end(), SortRowComparator{});
            }
            sorted = true;
        }
    }
};

struct FnBindData : public FunctionData {
    idx_t limit = 0;  // For Top-K functionality
    unique_ptr<FunctionData> Copy() const override { return make_uniq<FnBindData>(*this); }
    bool Equals(const FunctionData &other) const override { 
        auto &casted = other.Cast<FnBindData>();
        return limit == casted.limit;
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
    
    FnGlobalState(idx_t limit = 0) : sort_state(limit) {}
};

static unique_ptr<GlobalTableFunctionState> FnInit(ClientContext &, TableFunctionInitInput &input) {
    auto &bind_data = input.bind_data->Cast<FnBindData>();
    return make_uniq<FnGlobalState>(bind_data.limit);
}

struct FnLocalState : public LocalTableFunctionState {
    //TODO: initialize local state and other preparations
    bool merged = false;
    std::unordered_map<GroupKey, AggState, GroupKeyHash> agg_map;
    SortState sort_state;
    
    FnLocalState(idx_t limit = 0) : sort_state(limit) {}
};

static unique_ptr<LocalTableFunctionState> FnInitLocal(ExecutionContext &, TableFunctionInitInput &input,
                                                      GlobalTableFunctionState *global_state) {
    auto &bind_data = input.bind_data->Cast<FnBindData>();
    auto &g = global_state->Cast<FnGlobalState>();
    g.active_local_states.fetch_add(1, std::memory_order_relaxed);
    return make_uniq<FnLocalState>(bind_data.limit);
}

static unique_ptr<FunctionData> FnBind(ClientContext &context, TableFunctionBindInput &input,
                                            vector<LogicalType> &return_types,
                                            vector<string> &names) {
    // Set up the output schema for the final result
    return_types.push_back(LogicalType::VARCHAR);
    return_types.push_back(LogicalType::BIGINT);
    names.push_back("SearchPhrase");
    names.push_back("u");

    auto bind_data = make_uniq<FnBindData>();
    // Extract LIMIT if present
    if (input.inputs.size() > 2 && input.inputs[2].type().id() == LogicalTypeId::BIGINT) {
        bind_data->limit = input.inputs[2].GetValue<int64_t>();
    }

    return std::move(bind_data);
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
    
    UnifiedVectorFormat UserID_uvf;
    input.data[1].ToUnifiedFormat(input.size(), UserID_uvf);
    int64_t* UserID_ptr = (int64_t*)UserID_uvf.data;
    
    // Validity bitmaps for NULL checking
    auto &valid_SearchPhrase = SearchPhrase_uvf.validity;
    auto &valid_UserID = UserID_uvf.validity;
    
    const bool SearchPhrase_all_valid = valid_SearchPhrase.AllValid();
    const bool UserID_all_valid = valid_UserID.AllValid();
    
    idx_t num_rows = input.size();
    
    if (SearchPhrase_all_valid && UserID_all_valid) {
        // Fast path: no NULLs in either column
        for (idx_t row_idx = 0; row_idx < num_rows; ++row_idx) {
            idx_t i_SearchPhrase = SearchPhrase_uvf.sel->get_index(row_idx);
            idx_t i_UserID = UserID_uvf.sel->get_index(row_idx);
            
            string_t search_phrase_val = SearchPhrase_ptr[i_SearchPhrase];
            int64_t user_id_val = UserID_ptr[i_UserID];
            
            // TODO: Core processing logic goes here
            GroupKey key;
            key.search_phrase = search_phrase_val;
            
            auto &state = l.agg_map[key];
            state.distinct_user_ids.insert(user_id_val);
            
            // Add to sort buffer
            l.sort_state.AddRow(search_phrase_val, user_id_val);
        }
    } else {
        // Slow path: potential NULLs in one or both columns
        for (idx_t row_idx = 0; row_idx < num_rows; ++row_idx) {
            idx_t i_SearchPhrase = SearchPhrase_uvf.sel->get_index(row_idx);
            idx_t i_UserID = UserID_uvf.sel->get_index(row_idx);
            
            if (!SearchPhrase_all_valid && !valid_SearchPhrase.RowIsValid(i_SearchPhrase)) {
                continue; // Skip row if SearchPhrase is NULL
            }
            if (!UserID_all_valid && !valid_UserID.RowIsValid(i_UserID)) {
                continue; // Skip row if UserID is NULL
            }
            
            string_t search_phrase_val = SearchPhrase_ptr[i_SearchPhrase];
            int64_t user_id_val = UserID_ptr[i_UserID];
            
            // TODO: Core processing logic goes here
            GroupKey key;
            key.search_phrase = search_phrase_val;
            
            auto &state = l.agg_map[key];
            state.distinct_user_ids.insert(user_id_val);
            
            // Add to sort buffer
            l.sort_state.AddRow(search_phrase_val, user_id_val);
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
                global_state.distinct_user_ids.insert(local_state.distinct_user_ids.begin(), local_state.distinct_user_ids.end());
            }
            
            // Also merge the sort state
            for (const auto &sort_entry : l.sort_state.buffer) {
                g.sort_state.AddRow(sort_entry.SearchPhrase, sort_entry.u);
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
            
            // Get the total number of rows to output
            idx_t total_rows = g.sort_state.buffer.size();
            
            // Set the cardinality of the output chunk
            out.SetCardinality(total_rows);
            
            // Write sorted data to output chunk
            for (idx_t output_row_idx = 0; output_row_idx < total_rows; ++output_row_idx) {
                const auto &sort_view = g.sort_state.buffer[output_row_idx];
                out.SetValue(0, output_row_idx, sort_view.SearchPhrase);
                out.SetValue(1, output_row_idx, sort_view.u);
            }
        }
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