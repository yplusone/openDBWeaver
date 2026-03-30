/*
query_template: SELECT
  C_CITY,
  S_CITY,
  EXTRACT(YEAR FROM LO_ORDERDATE) AS year,
  SUM(LO_REVENUE) AS revenue
FROM lineorder_flat
WHERE C_CITY IN (:c_city_1, :c_city_2)
  AND S_CITY IN (:s_city_1, :s_city_2)
  AND EXTRACT(YEAR FROM LO_ORDERDATE) BETWEEN :year_lo AND :year_hi
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
  FROM lineorder_flat WHERE (C_CITY = :c_city_1 OR C_CITY = :c_city_2) AND (S_CITY = :s_city_1 OR S_CITY = :s_city_2)
), :year_lo, :year_hi) ORDER BY year ASC, revenue DESC;
query_example: SELECT
  C_CITY,
  S_CITY,
  EXTRACT(YEAR FROM LO_ORDERDATE) AS year,
  SUM(LO_REVENUE) AS revenue
FROM lineorder_flat
WHERE
  (C_CITY = 'UNITED KI1' OR C_CITY = 'UNITED KI5')
  AND (S_CITY = 'UNITED KI1' OR S_CITY = 'UNITED KI5')
  AND EXTRACT(YEAR FROM LO_ORDERDATE) BETWEEN 1992 AND 1997
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
  FROM lineorder_flat WHERE (C_CITY = 'UNITED KI1' OR C_CITY = 'UNITED KI5') AND (S_CITY = 'UNITED KI1' OR S_CITY = 'UNITED KI5')
), 1992, 1997) ORDER BY year ASC, revenue DESC;
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
            // Helper function to extract year from date
            static int32_t ExtractYearFromDate(date_t date) {
                int32_t year, month, day;
                Date::Convert(date, year, month, day);
                return year;
            }
            


            

            
            // Grouping key for hash aggregation
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
            
            // Aggregate state for SUM(LO_REVENUE)
            struct AggState {
                __int128 revenue_sum = 0;
            };
            
            //TODO: Define any helper structs or functions needed for binding/execution
            
            struct FnBindData : public FunctionData {
                int32_t year_lo;
                int32_t year_hi;
                unique_ptr<FunctionData> Copy() const override { 
                    auto copy = make_uniq<FnBindData>();
                    copy->year_lo = year_lo;
                    copy->year_hi = year_hi;
                    return std::move(copy);
                }
                bool Equals(const FunctionData &other) const override { 
                    auto &cast = other.Cast<FnBindData>();
                    return year_lo == cast.year_lo && year_hi == cast.year_hi;
                }
            };

            struct FnGlobalState : public GlobalTableFunctionState {
                std::unordered_map<GroupKey, AggState, GroupKeyHash> agg_map;
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
                std::unordered_map<GroupKey, AggState, GroupKeyHash> agg_map;
                //TODO: initialize local state and other preparations
                bool merged = false;
            };

            static unique_ptr<LocalTableFunctionState> FnInitLocal(ExecutionContext &, TableFunctionInitInput &,
                                                                  GlobalTableFunctionState *global_state) {
                auto &g = global_state->Cast<FnGlobalState>();
                g.active_local_states.fetch_add(1, std::memory_order_relaxed);
                return make_uniq<FnLocalState>();
            }
            
            // Convert __int128 accumulator to DuckDB's hugeint_t
            inline hugeint_t ToHugeint(__int128 acc) {
                hugeint_t result;
                result.lower = static_cast<uint64_t>(acc);          // low 64 bits
                result.upper = static_cast<int64_t>(acc >> 64);     // high 64 bits (sign-extended)
                return result;
            }
                static unique_ptr<FunctionData> FnBind(ClientContext &context, TableFunctionBindInput &input,
                                                    vector<LogicalType> &return_types,
                                                    vector<string> &names) {
                    // Expect a table argument followed by two integer filter parameters
                    // input.inputs[0] is the TABLE argument, scalars start at index 1
                    auto year_lo = input.inputs[1].GetValue<int32_t>();
                    auto year_hi = input.inputs[2].GetValue<int32_t>();
                    
                    // Define output schema based on authoritative mapping
                    return_types.emplace_back(LogicalType::VARCHAR); // C_CITY
                    return_types.emplace_back(LogicalType::VARCHAR); // S_CITY
                    return_types.emplace_back(LogicalType::BIGINT);  // year
                    return_types.emplace_back(LogicalType::HUGEINT); // revenue
                    
                    names.emplace_back("C_CITY");
                    names.emplace_back("S_CITY");
                    names.emplace_back("year");
                    names.emplace_back("revenue");

                    auto bind_data = make_uniq<FnBindData>();
                    bind_data->year_lo = year_lo;
                    bind_data->year_hi = year_hi;
                    return std::move(bind_data);
                }
            

                static OperatorResultType FnExecute(ExecutionContext &, TableFunctionInput &in,
                                                    DataChunk &input, DataChunk &) {
                    auto &l = in.local_state->Cast<FnLocalState>();
                    auto &bind_data = in.bind_data->Cast<FnBindData>();
                    int32_t year_lo = bind_data.year_lo;
                    int32_t year_hi = bind_data.year_hi;



                    
                    if (input.size() == 0) {       
                        return OperatorResultType::NEED_MORE_INPUT;
                    }

                    // Setup UnifiedVectorFormat for input columns
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
                    
                    // validity bitmaps
                    auto &valid_c_city = c_city_uvf.validity;
                    auto &valid_s_city = s_city_uvf.validity;
                    auto &valid_lo_orderdate = lo_orderdate_uvf.validity;
                    auto &valid_lo_revenue = lo_revenue_uvf.validity;
                    
                    const bool c_city_all_valid = valid_c_city.AllValid();
                    const bool s_city_all_valid = valid_s_city.AllValid();
                    const bool lo_orderdate_all_valid = valid_lo_orderdate.AllValid();
                    const bool lo_revenue_all_valid = valid_lo_revenue.AllValid();
                    
                    // Process input chunk using UnifiedVectorFormat
                    idx_t count = input.size();
                    
                    // FAST BRANCH: all relevant columns have no NULLs in this batch
                    if (c_city_all_valid && s_city_all_valid && lo_orderdate_all_valid && lo_revenue_all_valid) {
                        // --- Fast path: no per-row NULL checks ---
                        for (idx_t row_idx = 0; row_idx < count; ++row_idx) {
                            // Directly load values without RowIsValid checks
                            
                            idx_t i_c_city = c_city_uvf.sel->get_index(row_idx);
                            idx_t i_s_city = s_city_uvf.sel->get_index(row_idx);
                            idx_t i_lo_orderdate = lo_orderdate_uvf.sel->get_index(row_idx);
                            idx_t i_lo_revenue = lo_revenue_uvf.sel->get_index(row_idx);
                            
                            string_t c_city_val = c_city_ptr[i_c_city];
                            string_t s_city_val = s_city_ptr[i_s_city];
                            date_t lo_orderdate_val = lo_orderdate_ptr[i_lo_orderdate];
                            uint32_t lo_revenue_val = lo_revenue_ptr[i_lo_revenue];
                            
                            // ======================================
                            //  Core computation logic (no NULLs)
                            // Apply filter: EXTRACT(YEAR FROM LO_ORDERDATE) BETWEEN {year_lo} AND {year_hi}
                            int32_t extracted_year = ExtractYearFromDate(lo_orderdate_val);
                            if (extracted_year < year_lo || extracted_year > year_hi) {
                                continue; // Skip this row as it doesn't satisfy the filter
                            }
                            // Create group key
                            GroupKey key;
                            key.c_city = c_city_val;
                            key.s_city = s_city_val;
                            key.year = extracted_year;
                            
                            // Update aggregation state
                            auto &agg_state = l.agg_map[key];
                            agg_state.revenue_sum += lo_revenue_val;
                            // ============================
                        }
                    } else {
                        // --- Slow path: at least one column may contain NULLs ---
                        for (idx_t row_idx = 0; row_idx < count; ++row_idx) {
                            // For each column that is not fully valid, check this row
                            idx_t i_c_city = c_city_uvf.sel->get_index(row_idx);
                            idx_t i_s_city = s_city_uvf.sel->get_index(row_idx);
                            idx_t i_lo_orderdate = lo_orderdate_uvf.sel->get_index(row_idx);
                            idx_t i_lo_revenue = lo_revenue_uvf.sel->get_index(row_idx);
                            
                            if (!c_city_all_valid && !valid_c_city.RowIsValid(i_c_city)) {
                                continue; // row is NULL in column 0 → skip
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
                            
                            // At this point, all required columns are valid for this row
                            
                            string_t c_city_val = c_city_ptr[i_c_city];
                            string_t s_city_val = s_city_ptr[i_s_city];
                            date_t lo_orderdate_val = lo_orderdate_ptr[i_lo_orderdate];
                            uint32_t lo_revenue_val = lo_revenue_ptr[i_lo_revenue];
                            
                            // ======================================
                            //  Core computation logic (NULL-safe)
                            // Apply filter: EXTRACT(YEAR FROM LO_ORDERDATE) BETWEEN {year_lo} AND {year_hi}
                            int32_t extracted_year = ExtractYearFromDate(lo_orderdate_val);
                            if (extracted_year < year_lo || extracted_year > year_hi) {
                                continue; // Skip this row as it doesn't satisfy the filter
                            }
                            // Create group key
                            GroupKey key;
                            key.c_city = c_city_val;
                            key.s_city = s_city_val;
                            key.year = extracted_year;
                            
                            // Update aggregation state
                            auto &agg_state = l.agg_map[key];
                            agg_state.revenue_sum += lo_revenue_val;
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
                            // Merge local state with global state
                            for (const auto &entry : l.agg_map) {
                                const GroupKey &key = entry.first;
                                const AggState &state = entry.second;
                                auto &global_agg_state = g.agg_map[key];
                                global_agg_state.revenue_sum += state.revenue_sum;
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
                            // Get result from global state
                            idx_t output_row = 0;
                            for (const auto &entry : g.agg_map) {
                                const GroupKey &key = entry.first;
                                const AggState &state = entry.second;
                                
                                // Populate output chunk
                                out.SetValue(0, output_row, Value(key.c_city));
                                out.SetValue(1, output_row, Value(key.s_city));
                                out.SetValue(2, output_row, Value::BIGINT(key.year));
                                // Convert __int128 accumulator to hugeint_t for revenue
                                hugeint_t revenue_hugeint = ToHugeint(state.revenue_sum);
                                out.SetValue(3, output_row, Value::HUGEINT(revenue_hugeint));
                                
                                output_row++;
                                
                                // Check if we need to flush the chunk
                                if (output_row >= STANDARD_VECTOR_SIZE) {
                                    break;
                                }
                            }
                            out.SetCardinality(output_row);
                        }
                        // Additional chunks will be processed in subsequent calls
                        // For now, just emit what we have
                    } else {
                        out.SetCardinality(0);
                    }
                    
                    return OperatorFinalizeResultType::FINISHED;
                }
                static void LoadInternal(ExtensionLoader &loader) {
                    // First argument is a table, followed by two integer parameters (year_lo, year_hi)
                    TableFunction f("dbweaver", {LogicalTypeId::TABLE, LogicalType::INTEGER, LogicalType::INTEGER}, nullptr, FnBind, FnInit, FnInitLocal);
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