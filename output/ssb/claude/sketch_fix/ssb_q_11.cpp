/*
query_template: SELECT
  EXTRACT(YEAR FROM LO_ORDERDATE) AS year,
  C_NATION,
  SUM(LO_REVENUE - LO_SUPPLYCOST) AS profit
FROM lineorder_flat
WHERE C_REGION = :c_region
  AND S_REGION = :s_region
  AND P_MFGR IN (:mfgr_1, :mfgr_2)
GROUP BY
  EXTRACT(YEAR FROM LO_ORDERDATE),
  C_NATION
ORDER BY
  year ASC,
  C_NATION ASC

split_template: SELECT year, C_NATION, profit
FROM dbweaver((
  SELECT LO_ORDERDATE, C_NATION, P_MFGR, LO_REVENUE, LO_SUPPLYCOST
  FROM lineorder_flat
  WHERE C_REGION = :c_region
    AND S_REGION = :s_region
), :mfgr_1, :mfgr_2) ORDER BY year ASC, C_NATION ASC;
query_example: SELECT
  EXTRACT(YEAR FROM LO_ORDERDATE) AS year,
  C_NATION,
  SUM(LO_REVENUE - LO_SUPPLYCOST) AS profit
FROM lineorder_flat
WHERE
  C_REGION = 'AMERICA'
  AND S_REGION = 'AMERICA'
  AND (P_MFGR = 'MFGR#1' OR P_MFGR = 'MFGR#2')
GROUP BY
  EXTRACT(YEAR FROM LO_ORDERDATE),
  C_NATION
ORDER BY
  year ASC,
  C_NATION ASC

split_query: SELECT year, C_NATION, profit
FROM dbweaver((
  SELECT LO_ORDERDATE, C_NATION, P_MFGR, LO_REVENUE, LO_SUPPLYCOST
  FROM lineorder_flat
  WHERE C_REGION = 'AMERICA'
    AND S_REGION = 'AMERICA'
), 'MFGR#1', 'MFGR#2') ORDER BY year ASC, C_NATION ASC;
*/
#define DUCKDB_EXTENSION_MAIN

#include "dbweaver_extension.hpp"
#include "duckdb.hpp"
#include "duckdb/function/table_function.hpp"
#include <atomic>
#include <limits>
#include <mutex>
#include <unordered_map>

namespace duckdb {

struct FnBindData : public FunctionData {
    string mfgr_1;
    string mfgr_2;

    explicit FnBindData(string mfgr_1_p, string mfgr_2_p) : mfgr_1(mfgr_1_p), mfgr_2(mfgr_2_p) {}

    unique_ptr<FunctionData> Copy() const override { 
        return make_uniq<FnBindData>(mfgr_1, mfgr_2); 
    }
    bool Equals(const FunctionData &other_p) const override { 
        auto &other = other_p.Cast<FnBindData>();
        return mfgr_1 == other.mfgr_1 && mfgr_2 == other.mfgr_2;
    }
};

struct GroupKey {
    int32_t year;
    string_t nation;
    
    bool operator==(const GroupKey& other) const {
        return year == other.year && nation == other.nation;
    }
};

struct GroupKeyHash {
    size_t operator()(const GroupKey& k) const {
        size_t h = 0;
        h ^= std::hash<int32_t>{}(k.year) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= duckdb::Hash(k.nation.GetData(), k.nation.GetSize()) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

struct AggState {
    int64_t profit_sum = 0;
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
};

static unique_ptr<GlobalTableFunctionState> FnInit(ClientContext &context, TableFunctionInitInput &input) {
    //auto &bind_data = input.bind_data->Cast<FnBindData>();
    //string mfgr_1 = bind_data.mfgr_1;
    //string mfgr_2 = bind_data.mfgr_2;
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

static unique_ptr<FunctionData> FnBind(ClientContext &, TableFunctionBindInput &input,
                                            vector<LogicalType> &return_types,
                                            vector<string> &names) {
    //TODO: populate return_types and names
    string mfgr_1 = input.inputs[1].GetValue<string>();
    string mfgr_2 = input.inputs[2].GetValue<string>();
    
    return_types.push_back(LogicalType::BIGINT);
    return_types.push_back(LogicalType::VARCHAR);
    return_types.push_back(LogicalType::HUGEINT);
    
    names.push_back("year");
    names.push_back("C_NATION");
    names.push_back("profit");

    return make_uniq<FnBindData>(mfgr_1, mfgr_2);
}

static OperatorResultType FnExecute(ExecutionContext &, TableFunctionInput &in,
                                                    DataChunk &input, DataChunk &) {
    auto &l = in.local_state->Cast<FnLocalState>();

    // Access bound parameters if needed
    auto &bind_data = in.bind_data->Cast<FnBindData>();
    string mfgr_1 = bind_data.mfgr_1;
    string mfgr_2 = bind_data.mfgr_2;
    
    if (input.size() == 0) {       
        return OperatorResultType::NEED_MORE_INPUT;
    }

    // Unpack input columns using UnifiedVectorFormat
    UnifiedVectorFormat lo_orderdate_uvf;
    input.data[0].ToUnifiedFormat(input.size(), lo_orderdate_uvf);
    date_t* lo_orderdate_ptr = (date_t*)lo_orderdate_uvf.data;

    UnifiedVectorFormat c_nation_uvf;
    input.data[1].ToUnifiedFormat(input.size(), c_nation_uvf);
    string_t* c_nation_ptr = (string_t*)c_nation_uvf.data;

    UnifiedVectorFormat p_mfgr_uvf;
    input.data[2].ToUnifiedFormat(input.size(), p_mfgr_uvf);
    string_t* p_mfgr_ptr = (string_t*)p_mfgr_uvf.data;

    UnifiedVectorFormat lo_revenue_uvf;
    input.data[3].ToUnifiedFormat(input.size(), lo_revenue_uvf);
    uint32_t* lo_revenue_ptr = (uint32_t*)lo_revenue_uvf.data;

    UnifiedVectorFormat lo_supplycost_uvf;
    input.data[4].ToUnifiedFormat(input.size(), lo_supplycost_uvf);
    uint32_t* lo_supplycost_ptr = (uint32_t*)lo_supplycost_uvf.data;

    // validity bitmaps
    auto &valid_lo_orderdate = lo_orderdate_uvf.validity;
    auto &valid_c_nation = c_nation_uvf.validity;
    auto &valid_p_mfgr = p_mfgr_uvf.validity;
    auto &valid_lo_revenue = lo_revenue_uvf.validity;
    auto &valid_lo_supplycost = lo_supplycost_uvf.validity;

    const bool lo_orderdate_all_valid = valid_lo_orderdate.AllValid();
    const bool c_nation_all_valid = valid_c_nation.AllValid();
    const bool p_mfgr_all_valid = valid_p_mfgr.AllValid();
    const bool lo_revenue_all_valid = valid_lo_revenue.AllValid();
    const bool lo_supplycost_all_valid = valid_lo_supplycost.AllValid();

    // FAST BRANCH: all relevant columns have no NULLs in this batch
    if (lo_orderdate_all_valid && c_nation_all_valid && p_mfgr_all_valid && lo_revenue_all_valid && lo_supplycost_all_valid) {
        // --- Fast path: no per-row NULL checks ---
        for (idx_t row_idx = 0; row_idx < input.size(); ++row_idx) {
            // Directly load values without RowIsValid checks

            idx_t i_lo_orderdate = lo_orderdate_uvf.sel->get_index(row_idx);
            idx_t i_c_nation = c_nation_uvf.sel->get_index(row_idx);
            idx_t i_p_mfgr = p_mfgr_uvf.sel->get_index(row_idx);
            idx_t i_lo_revenue = lo_revenue_uvf.sel->get_index(row_idx);
            idx_t i_lo_supplycost = lo_supplycost_uvf.sel->get_index(row_idx);

            date_t v_lo_orderdate = lo_orderdate_ptr[i_lo_orderdate];
            string_t v_c_nation = c_nation_ptr[i_c_nation];
            string_t v_p_mfgr = p_mfgr_ptr[i_p_mfgr];
            uint32_t v_lo_revenue = lo_revenue_ptr[i_lo_revenue];
            uint32_t v_lo_supplycost = lo_supplycost_ptr[i_lo_supplycost];
            // Apply filter: (P_MFGR = {mfgr_1} OR P_MFGR = {mfgr_2})
            auto p_mfgr_str = v_p_mfgr.GetString();
            if (!(p_mfgr_str == mfgr_1 || p_mfgr_str == mfgr_2)) {
                continue; // Skip row if it doesn't match either manufacturer
            }



            // ======================================
            //  Core computation logic (no NULLs)
            int32_t year = Date::ExtractYear(v_lo_orderdate);
            GroupKey key;
            key.year = year;
            key.nation = v_c_nation;
            auto &state = l.agg_map[key];
            state.profit_sum += (int64_t)v_lo_revenue - (int64_t)v_lo_supplycost;
            // ============================
        }
    } else {
        // --- Slow path: at least one column may contain NULLs ---
        for (idx_t row_idx = 0; row_idx < input.size(); ++row_idx) {
            // For each column that is not fully valid, check this row
            idx_t i_lo_orderdate = lo_orderdate_uvf.sel->get_index(row_idx);
            idx_t i_c_nation = c_nation_uvf.sel->get_index(row_idx);
            idx_t i_p_mfgr = p_mfgr_uvf.sel->get_index(row_idx);
            idx_t i_lo_revenue = lo_revenue_uvf.sel->get_index(row_idx);
            idx_t i_lo_supplycost = lo_supplycost_uvf.sel->get_index(row_idx);

            if (!lo_orderdate_all_valid && !valid_lo_orderdate.RowIsValid(i_lo_orderdate)) {
                continue; // row is NULL in column lo_orderdate → skip
            }
            if (!c_nation_all_valid && !valid_c_nation.RowIsValid(i_c_nation)) {
                continue;
            }
            if (!p_mfgr_all_valid && !valid_p_mfgr.RowIsValid(i_p_mfgr)) {
                continue;
            }
            if (!lo_revenue_all_valid && !valid_lo_revenue.RowIsValid(i_lo_revenue)) {
                continue;
            }
            if (!lo_supplycost_all_valid && !valid_lo_supplycost.RowIsValid(i_lo_supplycost)) {
                continue;
            }

            // At this point, all required columns are valid for this row

            date_t v_lo_orderdate = lo_orderdate_ptr[i_lo_orderdate];
            string_t v_c_nation = c_nation_ptr[i_c_nation];
            string_t v_p_mfgr = p_mfgr_ptr[i_p_mfgr];
            uint32_t v_lo_revenue = lo_revenue_ptr[i_lo_revenue];
            uint32_t v_lo_supplycost = lo_supplycost_ptr[i_lo_supplycost];
            // Apply filter: (P_MFGR = {mfgr_1} OR P_MFGR = {mfgr_2})
            auto p_mfgr_str = v_p_mfgr.GetString();
            if (!(p_mfgr_str == mfgr_1 || p_mfgr_str == mfgr_2)) {
                continue; // Skip row if it doesn't match either manufacturer
            }



            // ======================================
            //  Core computation logic (NULL-safe)
            int32_t year = Date::ExtractYear(v_lo_orderdate);
            GroupKey key;
            key.year = year;
            key.nation = v_c_nation;
            auto &state = l.agg_map[key];
            state.profit_sum += (int64_t)v_lo_revenue - (int64_t)v_lo_supplycost;
            // ======================================
        }
    }

    //TODO: process input chunk and produce output

    return OperatorResultType::NEED_MORE_INPUT; 
}

static OperatorFinalizeResultType FnFinalize(ExecutionContext &, TableFunctionInput &in,
                                                            DataChunk &out) {
    auto &g = *reinterpret_cast<FnGlobalState *>(in.global_state.get());
    auto &l = in.local_state->Cast<FnLocalState>();

    // Access bound parameters if needed
    //auto &bind_data = in.bind_data->Cast<FnBindData>();
    //string mfgr_1 = bind_data.mfgr_1;
    //string mfgr_2 = bind_data.mfgr_2;

    // Merge the local state into the global state exactly once.
    if (!l.merged) {
        {
            std::lock_guard<std::mutex> guard(g.lock);
            //TODO: merge local state with global state
            for (const auto &entry : l.agg_map) {
                const GroupKey &key = entry.first;
                const AggState &state = entry.second;
                g.agg_map[key].profit_sum += state.profit_sum;
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
            //TODO: get result from global state
            idx_t output_row = 0;
            for (const auto &entry : g.agg_map) {
                const GroupKey &key = entry.first;
                const AggState &state = entry.second;
                
                out.SetValue(0, output_row, Value::BIGINT(key.year));
                out.SetValue(1, output_row, Value(key.nation));
                out.SetValue(2, output_row, Value::HUGEINT(state.profit_sum));
                
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
    TableFunction f("dbweaver", {LogicalType::TABLE, LogicalType::VARCHAR, LogicalType::VARCHAR}, nullptr, FnBind, FnInit, FnInitLocal);
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