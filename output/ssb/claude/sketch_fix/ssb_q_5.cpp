/*
query_template: SELECT
  EXTRACT(YEAR FROM LO_ORDERDATE) AS year,
  P_BRAND,
  SUM(LO_REVENUE) AS sum_revenue
FROM lineorder_flat
WHERE P_BRAND BETWEEN :brand_lo AND :brand_hi
  AND S_REGION = :s_region
GROUP BY year, P_BRAND
ORDER BY year, P_BRAND

split_template: SELECT year, P_BRAND, sum_revenue
FROM dbweaver((
  SELECT LO_ORDERDATE, P_BRAND, LO_REVENUE
  FROM lineorder_flat
  WHERE P_BRAND BETWEEN :brand_lo AND :brand_hi
    AND S_REGION = :s_region
)) ORDER BY year, P_BRAND;
query_example: SELECT
  EXTRACT(YEAR FROM LO_ORDERDATE) AS year,
  P_BRAND,
  SUM(LO_REVENUE) AS sum_revenue
FROM lineorder_flat
WHERE P_BRAND >= 'MFGR#2221' AND P_BRAND <= 'MFGR#2228'
  AND S_REGION = 'ASIA'
GROUP BY year, P_BRAND
ORDER BY year, P_BRAND

split_query: SELECT year, P_BRAND, sum_revenue
FROM dbweaver((
  SELECT LO_ORDERDATE, P_BRAND, LO_REVENUE
  FROM lineorder_flat
  WHERE P_BRAND >= 'MFGR#2221' AND P_BRAND <= 'MFGR#2228'
    AND S_REGION = 'ASIA'
)) ORDER BY year, P_BRAND;
*/
#define DUCKDB_EXTENSION_MAIN

#include "dbweaver_extension.hpp"
#include "duckdb.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/common/types/date.hpp"
#include <atomic>
#include <limits>
#include <mutex>
#include <unordered_map>
//TODO: Add more includes as needed



namespace duckdb {

//TODO: Define any helper structs or functions needed for binding/execution

struct FnBindData : public FunctionData {
    unique_ptr<FunctionData> Copy() const override { return make_uniq<FnBindData>(); }
    bool Equals(const FunctionData &) const override { return true; }
};

struct GroupKey {
    int64_t year;
    string_t brand;
    
    bool operator==(const GroupKey& other) const {
        return year == other.year && brand == other.brand;
    }
};

struct GroupKeyHash {
    size_t operator()(const GroupKey& k) const {
        size_t h = 0;
        h ^= std::hash<int64_t>{}(k.year) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= duckdb::Hash(k.brand.GetData(), k.brand.GetSize()) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

struct AggState {
    __int128 sum_revenue = 0;
};

struct FnGlobalState : public GlobalTableFunctionState {
    std::unordered_map<GroupKey, AggState, GroupKeyHash> agg_map;
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
    std::unordered_map<GroupKey, AggState, GroupKeyHash> agg_map;
    bool merged = false;
};

static unique_ptr<LocalTableFunctionState> FnInitLocal(ExecutionContext &, TableFunctionInitInput &,
                                                      GlobalTableFunctionState *global_state) {
    auto &g = global_state->Cast<FnGlobalState>();
    g.active_local_states.fetch_add(1, std::memory_order_relaxed);
    return make_uniq<FnLocalState>();
}

inline hugeint_t ToHugeint(__int128 acc) {
    hugeint_t result;
    result.lower = static_cast<uint64_t>(acc);          // low 64 bits
    result.upper = static_cast<int64_t>(acc >> 64);     // high 64 bits (sign-extended)
    return result;
}

static unique_ptr<FunctionData> FnBind(ClientContext &, TableFunctionBindInput &,
                                            vector<LogicalType> &return_types,
                                            vector<string> &names) {
    //TODO: populate return_types and names
    return_types.push_back(LogicalType::BIGINT);
    return_types.push_back(LogicalType::VARCHAR);
    return_types.push_back(LogicalType::HUGEINT);
    names.push_back("year");
    names.push_back("P_BRAND");
    names.push_back("sum_revenue");

    return make_uniq<FnBindData>();
}

static OperatorResultType FnExecute(ExecutionContext &, TableFunctionInput &in,
                                            DataChunk &input, DataChunk &) {
    auto &l = in.local_state->Cast<FnLocalState>();

    
    if (input.size() == 0) {       
        return OperatorResultType::NEED_MORE_INPUT;
    }

    // Unpack input columns using UnifiedVectorFormat
    UnifiedVectorFormat LO_ORDERDATE_uvf;
    input.data[0].ToUnifiedFormat(input.size(), LO_ORDERDATE_uvf);
    date_t* LO_ORDERDATE_ptr = (date_t*)LO_ORDERDATE_uvf.data;
    
    UnifiedVectorFormat P_BRAND_uvf;
    input.data[1].ToUnifiedFormat(input.size(), P_BRAND_uvf);
    string_t* P_BRAND_ptr = (string_t*)P_BRAND_uvf.data;
    
    UnifiedVectorFormat LO_REVENUE_uvf;
    input.data[2].ToUnifiedFormat(input.size(), LO_REVENUE_uvf);
    uint32_t* LO_REVENUE_ptr = (uint32_t*)LO_REVENUE_uvf.data;
    
    // Validity bitmaps for NULL handling
    auto &valid_LO_ORDERDATE = LO_ORDERDATE_uvf.validity;
    auto &valid_P_BRAND = P_BRAND_uvf.validity;
    auto &valid_LO_REVENUE = LO_REVENUE_uvf.validity;
    
    const bool LO_ORDERDATE_all_valid = valid_LO_ORDERDATE.AllValid();
    const bool P_BRAND_all_valid = valid_P_BRAND.AllValid();
    const bool LO_REVENUE_all_valid = valid_LO_REVENUE.AllValid();
    
    // Process rows with optimized NULL handling
    if (LO_ORDERDATE_all_valid && P_BRAND_all_valid && LO_REVENUE_all_valid) {
        // Fast path: no NULLs in any column
        for (idx_t row_idx = 0; row_idx < input.size(); ++row_idx) {
            idx_t i_LO_ORDERDATE = LO_ORDERDATE_uvf.sel->get_index(row_idx);
            idx_t i_P_BRAND = P_BRAND_uvf.sel->get_index(row_idx);
            idx_t i_LO_REVENUE = LO_REVENUE_uvf.sel->get_index(row_idx);
            
            date_t v_LO_ORDERDATE = LO_ORDERDATE_ptr[i_LO_ORDERDATE];
            string_t v_P_BRAND = P_BRAND_ptr[i_P_BRAND];
            uint32_t v_LO_REVENUE = LO_REVENUE_ptr[i_LO_REVENUE];
            // Extract year from date using DuckDB Date helper
            int32_t year_i32, month_i32, day_i32;
            Date::Convert(v_LO_ORDERDATE, year_i32, month_i32, day_i32);
            int64_t year = static_cast<int64_t>(year_i32);
            
            GroupKey key;




            key.year = year;
            key.brand = v_P_BRAND;
            
            auto &state = l.agg_map[key];
            state.sum_revenue += v_LO_REVENUE;
        }
    } else {
        // Slow path: at least one column has potential NULLs
        for (idx_t row_idx = 0; row_idx < input.size(); ++row_idx) {
            idx_t i_LO_ORDERDATE = LO_ORDERDATE_uvf.sel->get_index(row_idx);
            idx_t i_P_BRAND = P_BRAND_uvf.sel->get_index(row_idx);
            idx_t i_LO_REVENUE = LO_REVENUE_uvf.sel->get_index(row_idx);
            
            if (!LO_ORDERDATE_all_valid && !valid_LO_ORDERDATE.RowIsValid(i_LO_ORDERDATE)) {
                continue;
            }
            if (!P_BRAND_all_valid && !valid_P_BRAND.RowIsValid(i_P_BRAND)) {
                continue;
            }
            if (!LO_REVENUE_all_valid && !valid_LO_REVENUE.RowIsValid(i_LO_REVENUE)) {
                continue;
            }
            
            date_t v_LO_ORDERDATE = LO_ORDERDATE_ptr[i_LO_ORDERDATE];
            string_t v_P_BRAND = P_BRAND_ptr[i_P_BRAND];
            uint32_t v_LO_REVENUE = LO_REVENUE_ptr[i_LO_REVENUE];
            // Extract year from date using DuckDB Date helper
            int32_t year_i32, month_i32, day_i32;
            Date::Convert(v_LO_ORDERDATE, year_i32, month_i32, day_i32);
            int64_t year = static_cast<int64_t>(year_i32);
            
            GroupKey key;




            key.year = year;
            key.brand = v_P_BRAND;
            
            auto &state = l.agg_map[key];
            state.sum_revenue += v_LO_REVENUE;
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
                auto &g_state = g.agg_map[key];
                g_state.sum_revenue += state.sum_revenue;
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
                out.SetValue(1, output_row, Value(key.brand));
                hugeint_t huge_val = ToHugeint(state.sum_revenue);
                out.SetValue(2, output_row, Value::HUGEINT(huge_val));
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