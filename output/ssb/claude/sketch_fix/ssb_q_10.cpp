/*
query_template: SELECT
  C_CITY,
  S_CITY,
  EXTRACT(YEAR FROM LO_ORDERDATE) AS year,
  SUM(LO_REVENUE) AS revenue
FROM lineorder_flat
WHERE C_CITY IN (:c_city_1, :c_city_2)
  AND S_CITY IN (:s_city_1, :s_city_2)
  AND CAST(strftime('%Y%m', LO_ORDERDATE) AS INTEGER) = :yyyymm
GROUP BY
  C_CITY,
  S_CITY,
  EXTRACT(YEAR FROM LO_ORDERDATE)
ORDER BY
  year ASC,
  revenue DESC

split_template: SELECT C_CITY, S_CITY, year, revenue
FROM dbweaver((
  SELECT C_CITY, S_CITY, LO_ORDERDATE, LO_REVENUE
  FROM lineorder_flat
),:c_city_1,:c_city_2,:s_city_1,:s_city_2,:yyyymm) ORDER BY year ASC, revenue DESC;
query_example: SELECT
  C_CITY,
  S_CITY,
  EXTRACT(YEAR FROM LO_ORDERDATE) AS year,
  SUM(LO_REVENUE) AS revenue
FROM lineorder_flat
WHERE
  (C_CITY = 'UNITED KI1' OR C_CITY = 'UNITED KI5')
  AND (S_CITY = 'UNITED KI1' OR S_CITY = 'UNITED KI5')
  AND CAST(strftime('%Y%m', LO_ORDERDATE) AS INTEGER) = 199712
GROUP BY
  C_CITY,
  S_CITY,
  EXTRACT(YEAR FROM LO_ORDERDATE)
ORDER BY
  year ASC,
  revenue DESC

split_query: SELECT C_CITY, S_CITY, year, revenue
FROM dbweaver((
  SELECT C_CITY, S_CITY, LO_ORDERDATE, LO_REVENUE
  FROM lineorder_flat
),'UNITED KI1','UNITED KI5','UNITED KI1','UNITED KI5',199712) ORDER BY year ASC, revenue DESC;
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
    string c_city_1;
    string c_city_2;
    string s_city_1;
    string s_city_2;
    int32_t yyyymm;

    explicit FnBindData(string c_city_1_p, string c_city_2_p, string s_city_1_p, string s_city_2_p, int32_t yyyymm_p) 
        : c_city_1(c_city_1_p), c_city_2(c_city_2_p), s_city_1(s_city_1_p), s_city_2(s_city_2_p), yyyymm(yyyymm_p) {}

    unique_ptr<FunctionData> Copy() const override { 
        return make_uniq<FnBindData>(c_city_1, c_city_2, s_city_1, s_city_2, yyyymm); 
    }
    bool Equals(const FunctionData &other_p) const override { 
        auto &other = other_p.Cast<FnBindData>();
        return c_city_1 == other.c_city_1 && c_city_2 == other.c_city_2 && 
               s_city_1 == other.s_city_1 && s_city_2 == other.s_city_2 && 
               yyyymm == other.yyyymm;
    }
};



struct GroupKey {
    string_t c_city;
    string_t s_city;
    int32_t year;
    
    bool operator==(const GroupKey& other) const {
        return c_city == other.c_city && s_city == other.s_city && year == other.year;
    }
};

struct GroupKeyHash {
    size_t operator()(const GroupKey& k) const {
        size_t h = 0;
        h ^= duckdb::Hash(k.c_city.GetData(), k.c_city.GetSize()) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= duckdb::Hash(k.s_city.GetData(), k.s_city.GetSize()) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int32_t>{}(k.year) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

struct AggState {
    __int128 sum_revenue = 0;
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
    //string c_city_1 = bind_data.c_city_1;
    //string c_city_2 = bind_data.c_city_2;
    //string s_city_1 = bind_data.s_city_1;
    //string s_city_2 = bind_data.s_city_2;
    //string yyyymm = bind_data.yyyymm;
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
    // Define output schema according to authoritative mapping
    names.emplace_back("C_CITY");
    names.emplace_back("S_CITY");
    names.emplace_back("year");
    names.emplace_back("revenue");
    
    return_types.emplace_back(LogicalType::VARCHAR);
    return_types.emplace_back(LogicalType::VARCHAR);
    return_types.emplace_back(LogicalType::BIGINT);
    return_types.emplace_back(LogicalType::HUGEINT);
    string c_city_1 = input.inputs[1].GetValue<string>();
    string c_city_2 = input.inputs[2].GetValue<string>();
    string s_city_1 = input.inputs[3].GetValue<string>();
    string s_city_2 = input.inputs[4].GetValue<string>();
    int32_t yyyymm = static_cast<int32_t>(input.inputs[5].GetValue<int64_t>());

    return make_uniq<FnBindData>(c_city_1, c_city_2, s_city_1, s_city_2, yyyymm);
}


static OperatorResultType FnExecute(ExecutionContext &, TableFunctionInput &in,
                                    DataChunk &input, DataChunk &) {
    auto &l = in.local_state->Cast<FnLocalState>();

    // Access bind data parameters
    auto &bind_data = in.bind_data->Cast<FnBindData>();
    string c_city_1 = bind_data.c_city_1;
    string c_city_2 = bind_data.c_city_2;
    string s_city_1 = bind_data.s_city_1;
    string s_city_2 = bind_data.s_city_2;
    int32_t yyyymm = bind_data.yyyymm;
    


    
    if (input.size() == 0) {       
        return OperatorResultType::NEED_MORE_INPUT;
    }

    // Extract input columns using UnifiedVectorFormat
    UnifiedVectorFormat c_city_uvf;
    input.data[0].ToUnifiedFormat(input.size(), c_city_uvf);
    string_t* c_city_ptr = (string_t*)c_city_uvf.data;

    UnifiedVectorFormat s_city_uvf;
    input.data[1].ToUnifiedFormat(input.size(), s_city_uvf);
    string_t* s_city_ptr = (string_t*)s_city_uvf.data;

    UnifiedVectorFormat lo_orderdate_uvf;
    input.data[2].ToUnifiedFormat(input.size(), lo_orderdate_uvf);
    date_t* lo_orderdate_ptr = (date_t*)lo_orderdate_uvf.data;

    UnifiedVectorFormat lo_revenue_uvf;
    input.data[3].ToUnifiedFormat(input.size(), lo_revenue_uvf);
    uint32_t* lo_revenue_ptr = (uint32_t*)lo_revenue_uvf.data;

    // Validity bitmaps
    auto &valid_c_city = c_city_uvf.validity;
    auto &valid_s_city = s_city_uvf.validity;
    auto &valid_lo_orderdate = lo_orderdate_uvf.validity;
    auto &valid_lo_revenue = lo_revenue_uvf.validity;

    const bool c_city_all_valid = valid_c_city.AllValid();
    const bool s_city_all_valid = valid_s_city.AllValid();
    const bool lo_orderdate_all_valid = valid_lo_orderdate.AllValid();
    const bool lo_revenue_all_valid = valid_lo_revenue.AllValid();

    // Use bound integer yyyymm directly for comparison
    int32_t target_month = yyyymm;

    // Process rows


    idx_t num_rows = input.size();
    if (c_city_all_valid && s_city_all_valid && lo_orderdate_all_valid && lo_revenue_all_valid) {
        // Fast path: no nulls
        for (idx_t row_idx = 0; row_idx < num_rows; ++row_idx) {
            idx_t i_c_city = c_city_uvf.sel->get_index(row_idx);
            idx_t i_s_city = s_city_uvf.sel->get_index(row_idx);
            idx_t i_lo_orderdate = lo_orderdate_uvf.sel->get_index(row_idx);
            idx_t i_lo_revenue = lo_revenue_uvf.sel->get_index(row_idx);

            string_t c_city_val = c_city_ptr[i_c_city];
            string_t s_city_val = s_city_ptr[i_s_city];
            date_t lo_orderdate_val = lo_orderdate_ptr[i_lo_orderdate];
            uint32_t lo_revenue_val = lo_revenue_ptr[i_lo_revenue];

            // Apply filter predicates
            string c_city_str = c_city_val.GetString();
            string s_city_str = s_city_val.GetString();
            
            // Check C_CITY condition
            if (c_city_str != c_city_1 && c_city_str != c_city_2) {
                continue;  // Row fails filter
            }
            
            // Check S_CITY condition
            if (s_city_str != s_city_1 && s_city_str != s_city_2) {
                continue;  // Row fails filter
            }
            // Extract year and month from date
            int32_t year, month, day;
            Date::Convert(lo_orderdate_val, year, month, day);
            int date_month = year * 100 + month;
            
            // Check date condition
            if (date_month != target_month) {
                continue;  // Row fails filter
            }
            
            // Row passed all filters - process the row values


            GroupKey key;
            key.c_city = c_city_val;
            key.s_city = s_city_val;
            key.year = year;
            
            auto &agg_state = l.agg_map[key];
            agg_state.sum_revenue += lo_revenue_val;
        }
    } else {
        // Slow path: handle possible nulls
        for (idx_t row_idx = 0; row_idx < num_rows; ++row_idx) {
            idx_t i_c_city = c_city_uvf.sel->get_index(row_idx);
            idx_t i_s_city = s_city_uvf.sel->get_index(row_idx);
            idx_t i_lo_orderdate = lo_orderdate_uvf.sel->get_index(row_idx);
            idx_t i_lo_revenue = lo_revenue_uvf.sel->get_index(row_idx);

            if (!c_city_all_valid && !valid_c_city.RowIsValid(i_c_city)) {
                continue;
            }
            if (!s_city_all_valid && !valid_s_city.RowIsValid(i_s_city)) {
                continue;
            }
            if (!lo_orderdate_all_valid && !valid_lo_orderdate.RowIsValid(i_lo_orderdate)) {
                continue;
            }
            if (!lo_revenue_all_valid && !valid_lo_revenue.RowIsValid(i_lo_revenue)) {
                continue;
            }

            string_t c_city_val = c_city_ptr[i_c_city];
            string_t s_city_val = s_city_ptr[i_s_city];
            date_t lo_orderdate_val = lo_orderdate_ptr[i_lo_orderdate];
            uint32_t lo_revenue_val = lo_revenue_ptr[i_lo_revenue];

            // Apply filter predicates
            string c_city_str = c_city_val.GetString();
            string s_city_str = s_city_val.GetString();
            
            // Check C_CITY condition
            if (c_city_str != c_city_1 && c_city_str != c_city_2) {
                continue;  // Row fails filter
            }
            
            // Check S_CITY condition
            if (s_city_str != s_city_1 && s_city_str != s_city_2) {
                continue;  // Row fails filter
            }
            // Extract year and month from date
            int32_t year, month, day;
            Date::Convert(lo_orderdate_val, year, month, day);
            int date_month = year * 100 + month;
            
            // Check date condition
            if (date_month != target_month) {
                continue;  // Row fails filter
            }
            
            // Row passed all filters - process the row values


            GroupKey key;
            key.c_city = c_city_val;
            key.s_city = s_city_val;
            key.year = year;
            
            auto &agg_state = l.agg_map[key];
            agg_state.sum_revenue += lo_revenue_val;
        }
    }

    return OperatorResultType::NEED_MORE_INPUT; 
}

static OperatorFinalizeResultType FnFinalize(ExecutionContext &, TableFunctionInput &in,
                                            DataChunk &out) {
    auto &g = *reinterpret_cast<FnGlobalState *>(in.global_state.get());
    auto &l = in.local_state->Cast<FnLocalState>();
    // Access bind data parameters
    // auto &bind_data = in.bind_data->Cast<FnBindData>();
    // string c_city_1 = bind_data.c_city_1;
    // string c_city_2 = bind_data.c_city_2;
    // string s_city_1 = bind_data.s_city_1;
    // string s_city_2 = bind_data.s_city_2;
    // string yyyymm = bind_data.yyyymm;
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
    if (active > 0 && merged == active) {
        {
            std::lock_guard<std::mutex> guard(g.lock);
            //TODO: get result from global state
            idx_t output_row = 0;
            for (const auto &entry : g.agg_map) {
                const GroupKey &key = entry.first;
                const AggState &state = entry.second;
                
                out.SetValue(0, output_row, key.c_city);
                out.SetValue(1, output_row, key.s_city);
                out.SetValue(2, output_row, Value::BIGINT(key.year));
                out.SetValue(3, output_row, Value::HUGEINT(ToHugeint(state.sum_revenue)));
                
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
    TableFunction f("dbweaver", {LogicalType::TABLE, LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::INTEGER}, nullptr, FnBind, FnInit, FnInitLocal);
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