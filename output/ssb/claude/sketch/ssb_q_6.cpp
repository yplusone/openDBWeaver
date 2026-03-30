/*
query_template: SELECT
  SUM(LO_REVENUE) AS sum_revenue,
  EXTRACT(YEAR FROM LO_ORDERDATE) AS year,
  P_BRAND
FROM lineorder_flat
WHERE P_BRAND = :brand
  AND S_REGION = :s_region
GROUP BY year, P_BRAND
ORDER BY year, P_BRAND

split_template: SELECT sum_revenue, year, P_BRAND
FROM dbweaver((
  SELECT lo_orderdate, lo_revenue, P_BRAND
  FROM lineorder_flat
  WHERE P_BRAND = :brand AND S_REGION = :s_region
)) ORDER BY year, P_BRAND;
query_example: SELECT
  sum(LO_REVENUE),
  EXTRACT(YEAR FROM LO_ORDERDATE) AS year,
  P_BRAND
FROM lineorder_flat
WHERE P_BRAND = 'MFGR#2239' AND S_REGION = 'EUROPE'
GROUP BY year, P_BRAND
ORDER BY year, P_BRAND

split_query: SELECT sum_revenue, year, P_BRAND
FROM dbweaver((
  SELECT lo_orderdate, lo_revenue, P_BRAND
  FROM lineorder_flat
  WHERE P_BRAND = 'MFGR#2239' AND S_REGION = 'EUROPE'
)) ORDER BY year, P_BRAND;
*/
#define DUCKDB_EXTENSION_MAIN

#include "dbweaver_extension.hpp"
#include "duckdb.hpp"
#include "duckdb/function/table_function.hpp"
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
    int32_t year;
    string_t brand;
    
    bool operator==(const GroupKey& other) const {
        return year == other.year && brand == other.brand;
    }
};

struct GroupKeyHash {
    size_t operator()(const GroupKey& k) const {
        size_t h = 0;
        h ^= std::hash<int32_t>{}(k.year) + 0x9e3779b9 + (h << 6) + (h >> 2);
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
    return_types.emplace_back(LogicalType::HUGEINT); // sum_revenue
    return_types.emplace_back(LogicalType::BIGINT);  // year
    return_types.emplace_back(LogicalType::VARCHAR); // P_BRAND
    
    names.emplace_back("sum_revenue");
    names.emplace_back("year");
    names.emplace_back("P_BRAND");

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
    UnifiedVectorFormat lo_orderdate_uvf;
    input.data[0].ToUnifiedFormat(input.size(), lo_orderdate_uvf);
    date_t* lo_orderdate_ptr = (date_t*)lo_orderdate_uvf.data;
    
    UnifiedVectorFormat lo_revenue_uvf;
    input.data[1].ToUnifiedFormat(input.size(), lo_revenue_uvf);
    uint32_t* lo_revenue_ptr = (uint32_t*)lo_revenue_uvf.data;
    
    UnifiedVectorFormat P_BRAND_uvf;
    input.data[2].ToUnifiedFormat(input.size(), P_BRAND_uvf);
    string_t* P_BRAND_ptr = (string_t*)P_BRAND_uvf.data;
    
    // Validity bitmaps
    auto &valid_lo_orderdate = lo_orderdate_uvf.validity;
    auto &valid_lo_revenue = lo_revenue_uvf.validity;
    auto &valid_P_BRAND = P_BRAND_uvf.validity;
    
    const bool lo_orderdate_all_valid = valid_lo_orderdate.AllValid();
    const bool lo_revenue_all_valid = valid_lo_revenue.AllValid();
    const bool P_BRAND_all_valid = valid_P_BRAND.AllValid();
    
    // Process rows with null handling
    if (lo_orderdate_all_valid && lo_revenue_all_valid && P_BRAND_all_valid) {
        // Fast path: no per-row NULL checks
        for (idx_t row_idx = 0; row_idx < input.size(); ++row_idx) {
            idx_t i_lo_orderdate = lo_orderdate_uvf.sel->get_index(row_idx);
            idx_t i_lo_revenue = lo_revenue_uvf.sel->get_index(row_idx);
            idx_t i_P_BRAND = P_BRAND_uvf.sel->get_index(row_idx);
            
            date_t v_lo_orderdate = lo_orderdate_ptr[i_lo_orderdate];
            uint32_t v_lo_revenue = lo_revenue_ptr[i_lo_revenue];
            string_t v_P_BRAND = P_BRAND_ptr[i_P_BRAND];
            
            // Core computation logic (no NULLs)
            // Extract year from date
            date_seconds_t epoch_time = Interval::GetEpochSeconds(v_lo_orderdate);
            struct tm tm_struct;
            time_t t = epoch_time;
            localtime_r(&t, &tm_struct);
            int32_t year = tm_struct.tm_year + 1900;
            
            GroupKey key;
            key.year = year;
            key.brand = v_P_BRAND;
            
            auto &state = l.agg_map[key];
            state.sum_revenue += v_lo_revenue;
        }
    } else {
        // Slow path: at least one column may contain NULLs
        for (idx_t row_idx = 0; row_idx < input.size(); ++row_idx) {
            idx_t i_lo_orderdate = lo_orderdate_uvf.sel->get_index(row_idx);
            idx_t i_lo_revenue = lo_revenue_uvf.sel->get_index(row_idx);
            idx_t i_P_BRAND = P_BRAND_uvf.sel->get_index(row_idx);
            
            if (!lo_orderdate_all_valid && !valid_lo_orderdate.RowIsValid(i_lo_orderdate)) {
                continue; // row is NULL in column lo_orderdate → skip
            }
            if (!lo_revenue_all_valid && !valid_lo_revenue.RowIsValid(i_lo_revenue)) {
                continue; // row is NULL in column lo_revenue → skip
            }
            if (!P_BRAND_all_valid && !valid_P_BRAND.RowIsValid(i_P_BRAND)) {
                continue; // row is NULL in column P_BRAND → skip
            }
            
            date_t v_lo_orderdate = lo_orderdate_ptr[i_lo_orderdate];
            uint32_t v_lo_revenue = lo_revenue_ptr[i_lo_revenue];
            string_t v_P_BRAND = P_BRAND_ptr[i_P_BRAND];
            
            // Core computation logic (NULL-safe)
            // Extract year from date
            date_seconds_t epoch_time = Interval::GetEpochSeconds(v_lo_orderdate);
            struct tm tm_struct;
            time_t t = epoch_time;
            localtime_r(&t, &tm_struct);
            int32_t year = tm_struct.tm_year + 1900;
            
            GroupKey key;
            key.year = year;
            key.brand = v_P_BRAND;
            
            auto &state = l.agg_map[key];
            state.sum_revenue += v_lo_revenue;
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
                g.agg_map[key].sum_revenue += state.sum_revenue;
            }
        }
        l.merged = true;
        g.merged_local_states.fetch_add(1, std::memory_order_relaxed);
    }

    // Only the *last* local state to merge emits the final result.
    // All other threads return FINISHED with an empty chunk.
    const auto merged = g.merged_local_states.load(std::memory_order_relaxed);
    const auto active = g.active_local_states.load(std::memory_order_relaxed);
    idx_t output_row = 0;
    if (active > 0 && merged == active) {
        {
            std::lock_guard<std::mutex> guard(g.lock);
            //TODO: get result from global state
            for (const auto &entry : g.agg_map) {
                const GroupKey &key = entry.first;
                const AggState &state = entry.second;
                
                //TODO: populate out chunk with final results
                hugeint_t sum_hugeint = ToHugeint(state.sum_revenue);
                out.SetValue(0, output_row, Value::HUGEINT(sum_hugeint));
                out.SetValue(1, output_row, Value::BIGINT(key.year));
                out.SetValue(2, output_row, Value(key.brand.ToString()));
                
                output_row++;
                if(output_row >= STANDARD_VECTOR_SIZE) {
                    break;
                }
            }
            out.SetCardinality(output_row);
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