/*
query_template: SELECT UserID, COUNT(*) AS cnt FROM hits GROUP BY UserID ORDER BY COUNT(*) DESC LIMIT 10;

split_template: select * from dbweaver((SELECT UserID FROM hits));
query_example: SELECT UserID, COUNT(*) AS cnt FROM hits GROUP BY UserID ORDER BY COUNT(*) DESC LIMIT 10;

split_query: select * from dbweaver((SELECT UserID FROM hits));
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
    vector<LogicalType> return_types;

    unique_ptr<FunctionData> Copy() const override {
        auto copy = make_uniq<FnBindData>();
        copy->return_types = return_types;
        return std::move(copy);
    }
    bool Equals(const FunctionData &) const override { return true; }
};



struct GroupKey {
    int64_t UserID;
    
    bool operator==(const GroupKey& other) const {
        return UserID == other.UserID;
    }
};

struct GroupKeyHash {
    size_t operator()(const GroupKey& k) const {
        size_t h = 0;
        h ^= std::hash<int64_t>{}(k.UserID) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

struct AggState {
    int64_t cnt = 0;
};
struct SortRow {
    GroupKey key;
    int64_t cnt;
};

struct SortRowComparator {
    bool operator()(const SortRow &a, const SortRow &b) const {
        // DESC order: larger values come first
        if (a.cnt != b.cnt) {
            return a.cnt > b.cnt;
        }
        // Optional: deterministic tie-breaker on UserID ASC
        return a.key.UserID < b.key.UserID;
    }
};

struct SortState {
    std::vector<SortRow> rows;
    bool sorted = false;

    inline void AddRow(int64_t cnt_val, const GroupKey &key) {
        rows.push_back(SortRow{key, cnt_val});
    }

    inline void SortNow() {
        if (!sorted) {
            std::sort(rows.begin(), rows.end(), SortRowComparator{});
            sorted = true;
        }
    }
};


struct FnGlobalState : public GlobalTableFunctionState {
    // TODO: Optional accumulators/counters (generator may append to this struct)
    std::unordered_map<GroupKey, AggState, GroupKeyHash> agg_map;
    SortState sort_state;
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
    // Output schema: UserID (BIGINT), cnt (BIGINT)
    names.push_back("UserID");
    names.push_back("cnt");
    return_types.push_back(LogicalType::BIGINT);
    return_types.push_back(LogicalType::BIGINT);

    auto bind_data = make_uniq<FnBindData>();
    bind_data->return_types = return_types;
    return std::move(bind_data);
}



static OperatorResultType FnExecute(ExecutionContext &, TableFunctionInput &in,
                                    DataChunk &input, DataChunk &) {
    auto &l = in.local_state->Cast<FnLocalState>();

    
    if (input.size() == 0) {       
        return OperatorResultType::NEED_MORE_INPUT;
    }

    //TODO: process input chunk and produce output

    // UnifiedVectorFormat setup for input columns
    UnifiedVectorFormat UserID_uvf;
    input.data[0].ToUnifiedFormat(input.size(), UserID_uvf);
    int64_t* UserID_ptr = (int64_t*)UserID_uvf.data;

    // Batch-level NULL summary (whether each column is fully non-NULL)
    // validity bitmaps
    auto &valid_UserID  = UserID_uvf.validity;
    const bool UserID_all_valid = valid_UserID.AllValid();

    // FAST BRANCH: all relevant columns have no NULLs in this batch
    if (UserID_all_valid) {
        // --- Fast path: no per-row NULL checks ---
        for (idx_t row_idx = 0; row_idx < input.size(); ++row_idx) {
            // Directly load values without RowIsValid checks

            idx_t i_UserID = UserID_uvf.sel->get_index(row_idx);
            int64_t v1 = UserID_ptr[i_UserID];

            // ======================================
            //  Core computation logic (no NULLs)
            GroupKey key;
            key.UserID = v1;
            auto &state = l.agg_map[key];
            state.cnt++;
            //<<CORE_COMPUTE>>
            // ============================
        }
    } else {
        // --- Slow path: at least one column may contain NULLs ---
        for (idx_t row_idx = 0; row_idx < input.size(); ++row_idx) {
            // For each column that is not fully valid, check this row
            idx_t i_UserID = UserID_uvf.sel->get_index(row_idx);

            if (!UserID_all_valid && !valid_UserID.RowIsValid(i_UserID)) {
                continue; // row is NULL in column 1 → skip
            }

            // At this point, all required columns are valid for this row

            int64_t v1 = UserID_ptr[i_UserID];

            // ======================================
            //  Core computation logic (NULL-safe)
            GroupKey key;
            key.UserID = v1;
            auto &state = l.agg_map[key];
            state.cnt++;
            //<<CORE_COMPUTE>>
            // ======================================
        }
    }

    return OperatorResultType::NEED_MORE_INPUT; 
}
static OperatorFinalizeResultType FnFinalize(ExecutionContext &context, TableFunctionInput &in,
                                            DataChunk &out) {
    auto &g = *reinterpret_cast<FnGlobalState *>(in.global_state.get());
    auto &l = in.local_state->Cast<FnLocalState>();
    auto &bind = in.bind_data->Cast<FnBindData>();

    auto &client = context.client;
    out.Initialize(client, bind.return_types, STANDARD_VECTOR_SIZE);




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
            // Build sort buffer from global aggregation map
            for (const auto &entry : g.agg_map) {
                const GroupKey &key = entry.first;
                const AggState &state = entry.second;
                g.sort_state.AddRow(state.cnt, key);
            }
            
            // Sort the data now
            g.sort_state.SortNow();
            // Output top-N sorted data, respecting LIMIT 10 and STANDARD_VECTOR_SIZE
            idx_t total_rows = g.sort_state.rows.size();
            idx_t max_rows = MinValue<idx_t>(MinValue<idx_t>((idx_t)10, (idx_t)STANDARD_VECTOR_SIZE), total_rows);
            out.SetCardinality(max_rows);


            for (idx_t i = 0; i < max_rows; ++i) {
                const SortRow &row = g.sort_state.rows[i];
                out.SetValue(0, i, Value::BIGINT(row.key.UserID));
                out.SetValue(1, i, Value::BIGINT(row.cnt));
            }


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