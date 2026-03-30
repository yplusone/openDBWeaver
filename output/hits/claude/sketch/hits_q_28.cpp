/*
query_template: SELECT CounterID, AVG(STRLEN(URL)) AS l, COUNT(*) AS c FROM hits WHERE URL <> '' GROUP BY CounterID HAVING COUNT(*) > 100000 ORDER BY l DESC LIMIT 25;

split_template: select * from dbweaver((SELECT URL, CounterID FROM hits WHERE (URL!='')));
query_example: SELECT CounterID, AVG(STRLEN(URL)) AS l, COUNT(*) AS c FROM hits WHERE URL <> '' GROUP BY CounterID HAVING COUNT(*) > 100000 ORDER BY l DESC LIMIT 25;

split_query: select * from dbweaver((SELECT URL, CounterID FROM hits WHERE (URL!='')));
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
#include <vector>
#include <algorithm>

namespace duckdb {

//TODO: Define any helper structs or functions needed for binding/execution

struct FnBindData : public FunctionData {
    unique_ptr<FunctionData> Copy() const override { return make_uniq<FnBindData>(); }
    bool Equals(const FunctionData &) const override { return true; }
};

struct GroupKey {
    int32_t counterid;
    
    bool operator==(const GroupKey& other) const {
        return counterid == other.counterid;
    }
};

struct GroupKeyHash {
    size_t operator()(const GroupKey& k) const {
        size_t h = 0;
        h ^= std::hash<int32_t>{}(k.counterid) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

struct AggState {
    int64_t sum_strlen = 0;
    int64_t count_rows = 0;
};

struct SortKeyView {
    double l;
    bool l_is_null;
};

struct SortRowComparator {
    bool operator()(const SortKeyView &a, const SortKeyView &b) const {
        // Handle NULLs first: since nulls_first is not specified, we default to NULLS LAST behavior
        if (a.l_is_null && b.l_is_null) {
            return false;  // Both are null, consider them equal
        }
        if (a.l_is_null) {
            return false;  // a is null, b is not -> a comes after b
        }
        if (b.l_is_null) {
            return true;   // b is null, a is not -> a comes before b
        }
        // Both are non-null, compare values in DESC order
        if (a.l != b.l) {
            return a.l > b.l;  // Descending order
        }
        return false;  // Equal values
    }
};

struct SortState {
    std::vector<SortKeyView> buffer;
    std::vector<GroupKey> keys;
    std::vector<AggState> states;
    bool sorted = false;

    inline void AddRow(const GroupKey &key, const AggState &state, double l_val, bool l_is_null) {
        buffer.push_back(SortKeyView{l_val, l_is_null});
        keys.push_back(key);
        states.push_back(state);
    }

    inline void SortNow() {
        if (!sorted && !buffer.empty()) {
            std::sort(buffer.begin(), buffer.end(), 
                      [this](const SortKeyView &a, const SortKeyView &b) {
                          return SortRowComparator{}(a, b);
                      });
            // Also sort the keys and states arrays in parallel based on the buffer's new order
            std::vector<size_t> indices(buffer.size());
            std::iota(indices.begin(), indices.end(), 0);
            std::sort(indices.begin(), indices.end(), 
                      [this](size_t i, size_t j) {
                          return SortRowComparator{}(buffer[i], buffer[j]);
                      });
            
            std::vector<GroupKey> new_keys(keys.size());
            std::vector<AggState> new_states(states.size());
            for(size_t i = 0; i < indices.size(); i++) {
                new_keys[i] = keys[indices[i]];
                new_states[i] = states[indices[i]];
            }
            keys = std::move(new_keys);
            states = std::move(new_states);
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
    // Define output schema according to authoritative mapping:
    // CounterID -> INTEGER
    // l -> DOUBLE
    // c -> BIGINT
    return_types.push_back(LogicalType::INTEGER); // CounterID
    return_types.push_back(LogicalType::DOUBLE); // l (avg strlen)
    return_types.push_back(LogicalType::BIGINT); // c (count)
    
    names.push_back("CounterID");
    names.push_back("l");
    names.push_back("c");

    return make_uniq<FnBindData>();
}

static OperatorResultType FnExecute(ExecutionContext &, TableFunctionInput &in,
                                                    DataChunk &input, DataChunk &) {
    auto &l = in.local_state->Cast<FnLocalState>();

    
    if (input.size() == 0) {       
        return OperatorResultType::NEED_MORE_INPUT;
    }

    // Unpack input columns using UnifiedVectorFormat
    UnifiedVectorFormat counterid_uvf;
    input.data[0].ToUnifiedFormat(input.size(), counterid_uvf);
    int32_t* counterid_ptr = (int32_t*)counterid_uvf.data;

    UnifiedVectorFormat l_uvf;
    input.data[1].ToUnifiedFormat(input.size(), l_uvf);
    double* l_ptr = (double*)l_uvf.data;

    UnifiedVectorFormat c_uvf;
    input.data[2].ToUnifiedFormat(input.size(), c_uvf);
    int64_t* c_ptr = (int64_t*)c_uvf.data;

    // validity bitmaps
    auto &valid_l = l_uvf.validity;
    auto &valid_c = c_uvf.validity;
    const bool l_all_valid = valid_l.AllValid();
    const bool c_all_valid = valid_c.AllValid();

    // Process rows in the input chunk
    if (l_all_valid && c_all_valid) {
        // --- Fast path: no per-row NULL checks ---
        for (idx_t row_idx = 0; row_idx < input.size(); ++row_idx) {
            idx_t i_l = l_uvf.sel->get_index(row_idx);
            idx_t i_c = c_uvf.sel->get_index(row_idx);

            double v_l = l_ptr[i_l];
            int64_t v_c = c_ptr[i_c];

            // Apply filter: (COUNT(*) > 100000)
            if (!(v_c > 100000)) {
                continue; // Skip this row
            }

            // ======================================
            //  Core computation logic (no NULLs)
            GroupKey key;
            key.counterid = counterid_ptr[l_uvf.sel->get_index(row_idx)];
            auto &state = l.agg_map[key];
            state.sum_strlen += (int64_t)v_l;
            state.count_rows++;
            //<<CORE_COMPUTE>>
            // ======================================
        }
    } else {
        // --- Slow path: at least one column may contain NULLs ---
        for (idx_t row_idx = 0; row_idx < input.size(); ++row_idx) {
            idx_t i_l = l_uvf.sel->get_index(row_idx);
            idx_t i_c = c_uvf.sel->get_index(row_idx);

            if (!l_all_valid && !valid_l.RowIsValid(i_l)) {
                continue;
            }
            if (!c_all_valid && !valid_c.RowIsValid(i_c)) {
                continue;
            }

            double v_l = l_ptr[i_l];
            int64_t v_c = c_ptr[i_c];

            // Apply filter: (COUNT(*) > 100000)
            if (!(v_c > 100000)) {
                continue; // Skip this row
            }

            // ======================================
            //  Core computation logic (NULL-safe)
            GroupKey key;
            key.counterid = counterid_ptr[l_uvf.sel->get_index(row_idx)];
            auto &state = l.agg_map[key];
            state.sum_strlen += (int64_t)v_l;
            state.count_rows++;
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
                g.agg_map[key].sum_strlen += state.sum_strlen;
                g.agg_map[key].count_rows += state.count_rows;
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
            // Build the sort buffer from the global map
            for (const auto &entry : g.agg_map) {
                const GroupKey &key = entry.first;
                const AggState &state = entry.second;
                
                double avg_strlen = state.count_rows > 0 ? (double)state.sum_strlen / state.count_rows : 0.0;
                
                g.sort_state.AddRow(key, state, avg_strlen, false);
            }
            
            // Perform the sort
            g.sort_state.SortNow();
            
            // Output the sorted results
            idx_t output_idx = 0;
            for (size_t i = 0; i < g.sort_state.buffer.size(); ++i) {
                const GroupKey &key = g.sort_state.keys[i];
                const AggState &state = g.sort_state.states[i];
                
                double avg_strlen = state.count_rows > 0 ? (double)state.sum_strlen / state.count_rows : 0.0;
                
                out.SetValue(0, output_idx, Value::INTEGER(key.counterid));
                out.SetValue(1, output_idx, Value::DOUBLE(avg_strlen));
                out.SetValue(2, output_idx, Value::BIGINT(state.count_rows));
                
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