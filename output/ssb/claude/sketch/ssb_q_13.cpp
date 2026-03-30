/*
query_template: SELECT
  EXTRACT(YEAR FROM LO_ORDERDATE) AS year,
  S_CITY,
  P_BRAND,
  SUM(LO_REVENUE - LO_SUPPLYCOST) AS profit
FROM lineorder_flat
WHERE S_NATION = :s_nation
  AND EXTRACT(YEAR FROM LO_ORDERDATE) IN (:year_1, :year_2)
  AND P_CATEGORY = :p_category
GROUP BY
  EXTRACT(YEAR FROM LO_ORDERDATE),
  S_CITY,
  P_BRAND
ORDER BY
  year ASC,
  S_CITY ASC,
  P_BRAND ASC

split_template: SELECT year, S_CITY, P_BRAND, profit
FROM dbweaver((
  SELECT lo_orderdate, S_CITY, P_BRAND, lo_revenue, lo_supplycost
  FROM lineorder_flat
  WHERE S_NATION = :s_nation
    AND P_CATEGORY = :p_category
),:year_1,:year_2) ORDER BY year ASC, S_CITY ASC, P_BRAND ASC;
query_example: SELECT
  EXTRACT(YEAR FROM LO_ORDERDATE) AS year,
  S_CITY,
  P_BRAND,
  SUM(LO_REVENUE - LO_SUPPLYCOST) AS profit
FROM lineorder_flat
WHERE
  S_NATION = 'UNITED STATES'
  AND (EXTRACT(YEAR FROM LO_ORDERDATE) = 1997 OR EXTRACT(YEAR FROM LO_ORDERDATE) = 1998)
  AND P_CATEGORY = 'MFGR#14'
GROUP BY
  EXTRACT(YEAR FROM LO_ORDERDATE),
  S_CITY,
  P_BRAND
ORDER BY
  year ASC,
  S_CITY ASC,
  P_BRAND ASC

split_query: SELECT year, S_CITY, P_BRAND, profit
FROM dbweaver((
  SELECT lo_orderdate, S_CITY, P_BRAND, lo_revenue, lo_supplycost
  FROM lineorder_flat
  WHERE S_NATION = 'UNITED STATES'
    AND P_CATEGORY = 'MFGR#14'
),1997,1998) ORDER BY year ASC, S_CITY ASC, P_BRAND ASC;
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

namespace duckdb {

struct FnBindData : public FunctionData {
    int32_t year_1;
    int32_t year_2;

    explicit FnBindData(int32_t year_1_p, int32_t year_2_p) : year_1(year_1_p), year_2(year_2_p) {}
    
    unique_ptr<FunctionData> Copy() const override { 
        return make_uniq<FnBindData>(year_1, year_2); 
    }
    bool Equals(const FunctionData &other_p) const override { 
        auto &other = other_p.Cast<FnBindData>();
        return year_1 == other.year_1 && year_2 == other.year_2;
    }
};

struct GroupKey {
    int32_t year;
    string_t s_city;
    string_t p_brand;
    
    bool operator==(const GroupKey& other) const {
        return year == other.year && 
               s_city == other.s_city && 
               p_brand == other.p_brand;
    }
};

struct GroupKeyHash {
    size_t operator()(const GroupKey& k) const {
        size_t h = 0;
        h ^= std::hash<int32_t>{}(k.year) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= duckdb::Hash(k.s_city.GetData(), k.s_city.GetSize()) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= duckdb::Hash(k.p_brand.GetData(), k.p_brand.GetSize()) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

struct AggState {
    __int128 profit = 0;
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

static unique_ptr<GlobalTableFunctionState> FnInit(ClientContext &context, TableFunctionInitInput &in) {
    //auto &bind_data = in.bind_data->Cast<FnBindData>();
    //int32_t year_1 = bind_data.year_1;
    //int32_t year_2 = bind_data.year_2;
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

inline hugeint_t ToHugeint(__int128 acc) {
    hugeint_t result;
    result.lower = static_cast<uint64_t>(acc);          // low 64 bits
    result.upper = static_cast<int64_t>(acc >> 64);     // high 64 bits (sign-extended)
    return result;
}

static unique_ptr<FunctionData> FnBind(ClientContext &, TableFunctionBindInput &input,
                                            vector<LogicalType> &return_types,
                                            vector<string> &names) {
    // Extract parameters from input
    int32_t year_1 = input.inputs[1].GetValue<int32_t>();
    int32_t year_2 = input.inputs[2].GetValue<int32_t>();
    
    //TODO: populate return_types and names
    return_types.emplace_back(LogicalType::BIGINT); // year
    return_types.emplace_back(LogicalType::VARCHAR); // S_CITY
    return_types.emplace_back(LogicalType::VARCHAR); // P_BRAND
    return_types.emplace_back(LogicalType::HUGEINT); // profit
    
    names.emplace_back("year");
    names.emplace_back("S_CITY");
    names.emplace_back("P_BRAND");
    names.emplace_back("profit");

    return make_uniq<FnBindData>(year_1, year_2);
}

static OperatorResultType FnExecute(ExecutionContext &, TableFunctionInput &in,
                                                    DataChunk &input, DataChunk &) {
    auto &l = in.local_state->Cast<FnLocalState>();

    // Access bound parameters
    auto &bind_data = in.bind_data->Cast<FnBindData>();
    int32_t year_1 = bind_data.year_1;
    int32_t year_2 = bind_data.year_2;
    
    if (input.size() == 0) {       
        return OperatorResultType::NEED_MORE_INPUT;
    }

    // Process input columns using UnifiedVectorFormat
    UnifiedVectorFormat lo_orderdate_uvf;
    input.data[0].ToUnifiedFormat(input.size(), lo_orderdate_uvf);
    date_t* lo_orderdate_ptr = (date_t*)lo_orderdate_uvf.data;

    UnifiedVectorFormat s_city_uvf;
    input.data[1].ToUnifiedFormat(input.size(), s_city_uvf);
    string_t* s_city_ptr = (string_t*)s_city_uvf.data;

    UnifiedVectorFormat p_brand_uvf;
    input.data[2].ToUnifiedFormat(input.size(), p_brand_uvf);
    string_t* p_brand_ptr = (string_t*)p_brand_uvf.data;

    UnifiedVectorFormat lo_revenue_uvf;
    input.data[3].ToUnifiedFormat(input.size(), lo_revenue_uvf);
    uint32_t* lo_revenue_ptr = (uint32_t*)lo_revenue_uvf.data;

    UnifiedVectorFormat lo_supplycost_uvf;
    input.data[4].ToUnifiedFormat(input.size(), lo_supplycost_uvf);
    uint32_t* lo_supplycost_ptr = (uint32_t*)lo_supplycost_uvf.data;

    // Validity bitmaps
    auto &valid_lo_orderdate = lo_orderdate_uvf.validity;
    auto &valid_s_city = s_city_uvf.validity;
    auto &valid_p_brand = p_brand_uvf.validity;
    auto &valid_lo_revenue = lo_revenue_uvf.validity;
    auto &valid_lo_supplycost = lo_supplycost_uvf.validity;

    const bool lo_orderdate_all_valid = valid_lo_orderdate.AllValid();
    const bool s_city_all_valid = valid_s_city.AllValid();
    const bool p_brand_all_valid = valid_p_brand.AllValid();
    const bool lo_revenue_all_valid = valid_lo_revenue.AllValid();
    const bool lo_supplycost_all_valid = valid_lo_supplycost.AllValid();

    const idx_t num_rows = input.size();

    // Prepare output chunk for filtered rows
    SelectionVector sel_vector(num_rows);
    idx_t output_count = 0;

    if (lo_orderdate_all_valid && s_city_all_valid && p_brand_all_valid && lo_revenue_all_valid && lo_supplycost_all_valid) {
        // Fast path: no per-row NULL checks
        for (idx_t row_idx = 0; row_idx < num_rows; ++row_idx) {
            idx_t i_lo_orderdate = lo_orderdate_uvf.sel->get_index(row_idx);
            idx_t i_s_city = s_city_uvf.sel->get_index(row_idx);
            idx_t i_p_brand = p_brand_uvf.sel->get_index(row_idx);
            idx_t i_lo_revenue = lo_revenue_uvf.sel->get_index(row_idx);
            idx_t i_lo_supplycost = lo_supplycost_uvf.sel->get_index(row_idx);

            date_t v_lo_orderdate = lo_orderdate_ptr[i_lo_orderdate];
            string_t v_s_city = s_city_ptr[i_s_city];
            string_t v_p_brand = p_brand_ptr[i_p_brand];
            uint32_t v_lo_revenue = lo_revenue_ptr[i_lo_revenue];
            uint32_t v_lo_supplycost = lo_supplycost_ptr[i_lo_supplycost];

            // Apply filter: EXTRACT(YEAR FROM lo_orderdate) IN ({year_1}, {year_2})
            int32_t extracted_year = Date::ExtractYear(v_lo_orderdate);
            if (extracted_year != year_1 && extracted_year != year_2) {
                continue; // Row does not match the filter condition
            }

            // Create group key
            GroupKey key;
            key.year = extracted_year;
            key.s_city = v_s_city;
            key.p_brand = v_p_brand;
            
            // Update aggregation state
            auto &state = l.agg_map[key];
            state.profit += static_cast<__int128>(v_lo_revenue) - static_cast<__int128>(v_lo_supplycost);

            // Row passed the filter, add to selection vector
            sel_vector.set_index(output_count++, row_idx);
        }
    } else {
        // Slow path: at least one column may contain NULLs
        for (idx_t row_idx = 0; row_idx < num_rows; ++row_idx) {
            idx_t i_lo_orderdate = lo_orderdate_uvf.sel->get_index(row_idx);
            idx_t i_s_city = s_city_uvf.sel->get_index(row_idx);
            idx_t i_p_brand = p_brand_uvf.sel->get_index(row_idx);
            idx_t i_lo_revenue = lo_revenue_uvf.sel->get_index(row_idx);
            idx_t i_lo_supplycost = lo_supplycost_uvf.sel->get_index(row_idx);

            if (!lo_orderdate_all_valid && !valid_lo_orderdate.RowIsValid(i_lo_orderdate)) {
                continue; // row is NULL in column lo_orderdate → skip
            }
            if (!s_city_all_valid && !valid_s_city.RowIsValid(i_s_city)) {
                continue;
            }
            if (!p_brand_all_valid && !valid_p_brand.RowIsValid(i_p_brand)) {
                continue;
            }
            if (!lo_revenue_all_valid && !valid_lo_revenue.RowIsValid(i_lo_revenue)) {
                continue;
            }
            if (!lo_supplycost_all_valid && !valid_lo_supplycost.RowIsValid(i_lo_supplycost)) {
                continue;
            }

            date_t v_lo_orderdate = lo_orderdate_ptr[i_lo_orderdate];
            string_t v_s_city = s_city_ptr[i_s_city];
            string_t v_p_brand = p_brand_ptr[i_p_brand];
            uint32_t v_lo_revenue = lo_revenue_ptr[i_lo_revenue];
            uint32_t v_lo_supplycost = lo_supplycost_ptr[i_lo_supplycost];

            // Apply filter: EXTRACT(YEAR FROM lo_orderdate) IN ({year_1}, {year_2})
            int32_t extracted_year = Date::ExtractYear(v_lo_orderdate);
            if (extracted_year != year_1 && extracted_year != year_2) {
                continue; // Row does not match the filter condition
            }

            // Create group key
            GroupKey key;
            key.year = extracted_year;
            key.s_city = v_s_city;
            key.p_brand = v_p_brand;
            
            // Update aggregation state
            auto &state = l.agg_map[key];
            state.profit += static_cast<__int128>(v_lo_revenue) - static_cast<__int128>(v_lo_supplycost);

            // Row passed the filter, add to selection vector
            sel_vector.set_index(output_count++, row_idx);
        }
    }

    // Set the selection vector on the input chunk to only include rows that passed the filter
    if (output_count != num_rows) {
        input.Slice(sel_vector, output_count);
    }

    return OperatorResultType::NEED_MORE_INPUT; 
}

static OperatorFinalizeResultType FnFinalize(ExecutionContext &, TableFunctionInput &in,
                                                            DataChunk &out) {
    auto &g = *reinterpret_cast<FnGlobalState *>(in.global_state.get());
    auto &l = in.local_state->Cast<FnLocalState>();
    
    // Access bound parameters
    //auto &bind_data = in.bind_data->Cast<FnBindData>();
    //int32_t year_1 = bind_data.year_1;
    //int32_t year_2 = bind_data.year_2;
    
    // Merge the local state into the global state exactly once.
    if (!l.merged) {
        {
            std::lock_guard<std::mutex> guard(g.lock);
            //TODO: merge local state with global state
            for (const auto &entry : l.agg_map) {
                const GroupKey &key = entry.first;
                const AggState &local_state = entry.second;
                auto &global_state = g.agg_map[key];
                global_state.profit += local_state.profit;
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
            idx_t total_rows = g.agg_map.size();
            if (total_rows > 0) {
                out.SetCardinality(total_rows);
                for (const auto &entry : g.agg_map) {
                    const GroupKey &key = entry.first;
                    const AggState &state = entry.second;
                    
                    // Populate output chunk
                    out.SetValue(0, output_row, Value::BIGINT(static_cast<int64_t>(key.year)));
                    out.SetValue(1, output_row, Value(key.s_city));
                    out.SetValue(2, output_row, Value(key.p_brand));
                    hugeint_t profit_hugeint = ToHugeint(state.profit);
                    out.SetValue(3, output_row, Value::HUGEINT(profit_hugeint));
                    
                    output_row++;
                }
            } else {
                out.SetCardinality(0);
            }
        }
        //TODO: populate out chunk with final results
    } else {
        out.SetCardinality(0);
    }
    
    return OperatorFinalizeResultType::FINISHED;
}

static void LoadInternal(ExtensionLoader &loader) {
    TableFunction f("dbweaver", {LogicalType::TABLE, LogicalType::INTEGER, LogicalType::INTEGER}, nullptr, FnBind, FnInit, FnInitLocal);
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