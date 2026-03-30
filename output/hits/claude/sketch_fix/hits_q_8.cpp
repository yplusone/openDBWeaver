/*
query_template: SELECT AdvEngineID, COUNT(*) AS cnt FROM hits WHERE AdvEngineID <> 0 GROUP BY AdvEngineID ORDER BY COUNT(*) DESC;

split_template: select * from dbweaver((SELECT AdvEngineID FROM hits WHERE (AdvEngineID!=0)));
query_example: SELECT AdvEngineID, COUNT(*) AS cnt FROM hits WHERE AdvEngineID <> 0 GROUP BY AdvEngineID ORDER BY COUNT(*) DESC;

split_query: select * from dbweaver((SELECT AdvEngineID FROM hits WHERE (AdvEngineID!=0)));
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

//TODO: Define any helper structs or functions needed for binding/execution

struct FnBindData : public FunctionData {
    unique_ptr<FunctionData> Copy() const override { return make_uniq<FnBindData>(); }
    bool Equals(const FunctionData &) const override { return true; }
};

struct GroupKey {
    int16_t AdvEngineID;
    
    bool operator==(const GroupKey& other) const {
        return AdvEngineID == other.AdvEngineID;
    }
};

struct GroupKeyHash {
    size_t operator()(const GroupKey& k) const {
        size_t h = 0;
        h ^= std::hash<int16_t>{}(k.AdvEngineID) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

struct AggState {
    int64_t cnt = 0;
};
struct RowEntry {
    GroupKey key;
    AggState state;
};

struct RowEntryComparator {
    // DESC order of count
    bool operator()(const RowEntry &a, const RowEntry &b) const {
        return a.state.cnt > b.state.cnt;
    }
};

struct SortState {
    std::vector<RowEntry> rows;
    bool sorted = false;

    inline void AddRow(const GroupKey &key, const AggState &state) {
        rows.push_back(RowEntry{key, state});
    }

    inline void SortNow() {
        if (!sorted) {
            std::sort(rows.begin(), rows.end(), RowEntryComparator{});
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
    // Set up output schema based on authoritative mapping
    return_types.emplace_back(LogicalType::SMALLINT);  // AdvEngineID
    return_types.emplace_back(LogicalType::BIGINT);   // cnt
    names.emplace_back("AdvEngineID");
    names.emplace_back("cnt");

    return make_uniq<FnBindData>();
}

static OperatorResultType FnExecute(ExecutionContext &, TableFunctionInput &in,
                                            DataChunk &input, DataChunk &) {
    auto &l = in.local_state->Cast<FnLocalState>();

    
    if (input.size() == 0) {       
        return OperatorResultType::NEED_MORE_INPUT;
    }

    //TODO: process input chunk and produce output

    // Declare UnifiedVectorFormat handles for input columns
    UnifiedVectorFormat AdvEngineID_uvf;
    
    // Load columns into UnifiedVectorFormat
    input.data[0].ToUnifiedFormat(input.size(), AdvEngineID_uvf);
    
    // Create typed pointers to physical data
    int16_t* AdvEngineID_ptr = (int16_t*)AdvEngineID_uvf.data;
    
    // Get validity bitmaps
    auto &valid_AdvEngineID = AdvEngineID_uvf.validity;
    
    // Check if all columns are fully valid
    const bool AdvEngineID_all_valid = valid_AdvEngineID.AllValid();
    
    // Process rows in fast path (no NULLs)
    if (AdvEngineID_all_valid) {
        for (idx_t row_idx = 0; row_idx < input.size(); ++row_idx) {
            idx_t i_AdvEngineID = AdvEngineID_uvf.sel->get_index(row_idx);
            int16_t v_AdvEngineID = AdvEngineID_ptr[i_AdvEngineID];
            
            // ======================================
            //  Core computation logic (no NULLs)
            GroupKey key;
            key.AdvEngineID = v_AdvEngineID;
            auto &state = l.agg_map[key];
            state.cnt++;
            //<<CORE_COMPUTE>>
            // ============================
        }
    } else {
        // Slow path: at least one column has NULLs
        for (idx_t row_idx = 0; row_idx < input.size(); ++row_idx) {
            idx_t i_AdvEngineID = AdvEngineID_uvf.sel->get_index(row_idx);
            
            if (!AdvEngineID_all_valid && !valid_AdvEngineID.RowIsValid(i_AdvEngineID)) {
                continue; // row is NULL in AdvEngineID → skip
            }
            
            int16_t v_AdvEngineID = AdvEngineID_ptr[i_AdvEngineID];
            
            // ======================================
            //  Core computation logic (NULL-safe)
            GroupKey key;
            key.AdvEngineID = v_AdvEngineID;
            auto &state = l.agg_map[key];
            state.cnt++;
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
        // Populate sort buffer
        {
            std::lock_guard<std::mutex> guard(g.lock);
            for (const auto &entry : g.agg_map) {
                const GroupKey &key = entry.first;
                const AggState &state = entry.second;
                g.sort_state.AddRow(key, state);
            }
            // Sort the data
            g.sort_state.SortNow();
            
            // Output sorted data
            idx_t output_idx = 0;
            for (size_t i = 0; i < g.sort_state.rows.size(); ++i) {
                const RowEntry &row = g.sort_state.rows[i];
                out.SetValue(0, output_idx, Value::SMALLINT(row.key.AdvEngineID));
                out.SetValue(1, output_idx, Value::BIGINT(row.state.cnt));
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