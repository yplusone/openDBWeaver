/*
query_template: SELECT
  C_NATION,
  S_NATION,
  EXTRACT(YEAR FROM LO_ORDERDATE) AS year,
  SUM(LO_REVENUE) AS revenue
FROM lineorder_flat
WHERE C_REGION = :c_region
  AND S_REGION = :s_region
  AND EXTRACT(YEAR FROM LO_ORDERDATE) BETWEEN :year_lo AND :year_hi
GROUP BY
  C_NATION,
  S_NATION,
  EXTRACT(YEAR FROM LO_ORDERDATE)
ORDER BY
  year ASC,
  revenue DESC

split_template: SELECT
  C_NATION,
  S_NATION,
  year,
  revenue
FROM dbweaver((
  SELECT C_NATION, S_NATION, LO_ORDERDATE, LO_REVENUE
  FROM lineorder_flat
  WHERE C_REGION = :c_region
    AND S_REGION = :s_region
),:year_lo,:year_hi) ORDER BY year ASC, revenue DESC;
query_example: SELECT
  C_NATION,
  S_NATION,
  EXTRACT(YEAR FROM LO_ORDERDATE) AS year,
  SUM(LO_REVENUE) AS revenue
FROM lineorder_flat
WHERE
  C_REGION = 'ASIA'
  AND S_REGION = 'ASIA'
  AND EXTRACT(YEAR FROM LO_ORDERDATE) BETWEEN 1992 AND 1997
GROUP BY
  C_NATION,
  S_NATION,
  EXTRACT(YEAR FROM LO_ORDERDATE)
ORDER BY
  year ASC,
  revenue DESC

split_query: SELECT
  C_NATION,
  S_NATION,
  year,
  revenue
FROM dbweaver((
  SELECT C_NATION, S_NATION, LO_ORDERDATE, LO_REVENUE
  FROM lineorder_flat
  WHERE C_REGION = 'ASIA'
    AND S_REGION = 'ASIA'
),1992,1997) ORDER BY year ASC, revenue DESC;
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

            //TODO: Define any helper structs or functions needed for binding/execution
            
            struct FnBindData : public FunctionData {
                int32_t year_lo;
                int32_t year_hi;

                explicit FnBindData(int32_t year_lo_p, int32_t year_hi_p) : year_lo(year_lo_p), year_hi(year_hi_p) {}

                unique_ptr<FunctionData> Copy() const override { return make_uniq<FnBindData>(year_lo, year_hi); }
                bool Equals(const FunctionData &other_p) const override {
                    auto &other = other_p.Cast<FnBindData>();
                    return year_lo == other.year_lo && year_hi == other.year_hi;
                }
            };

            struct GroupKey {
                string_t c_nation;
                string_t s_nation;
                int32_t year;
                
                bool operator==(const GroupKey& other) const {
                    return c_nation == other.c_nation && s_nation == other.s_nation && year == other.year;
                }
            };
            
            struct GroupKeyHash {
                size_t operator()(const GroupKey& k) const {
                    size_t h = 0;
                    h ^= duckdb::Hash(k.c_nation.GetData(), k.c_nation.GetSize()) + 0x9e3779b9 + (h << 6) + (h >> 2);
                    h ^= duckdb::Hash(k.s_nation.GetData(), k.s_nation.GetSize()) + 0x9e3779b9 + (h << 6) + (h >> 2);
                    h ^= std::hash<int32_t>{}(k.year) + 0x9e3779b9 + (h << 6) + (h >> 2);
                    return h;
                }
            };
            
            struct AggState {
                __int128_t sum_revenue = 0;
            };

            struct FnGlobalState : public GlobalTableFunctionState {
                // TODO: Optional accumulators/counters (generator may append to this struct)
                std::mutex lock;
                std::atomic<idx_t> active_local_states {0};
                std::atomic<idx_t> merged_local_states {0};
              	idx_t MaxThreads() const override {
                    return std::numeric_limits<idx_t>::max();
                }
                std::unordered_map<GroupKey, AggState, GroupKeyHash> agg_map;
            };

            static unique_ptr<GlobalTableFunctionState> FnInit(ClientContext &context, TableFunctionInitInput &in) {
                //auto &bind_data = in.bind_data->Cast<FnBindData>();
                //int32_t year_lo = bind_data.year_lo;
                //int32_t year_hi = bind_data.year_hi;
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
                    // Add return types and names
                    return_types.emplace_back(LogicalType::VARCHAR);  // C_NATION
                    names.emplace_back("C_NATION");
                    
                    return_types.emplace_back(LogicalType::VARCHAR);  // S_NATION
                    names.emplace_back("S_NATION");
                    
                    return_types.emplace_back(LogicalType::BIGINT);  // year
                    names.emplace_back("year");
                    
                    return_types.emplace_back(LogicalType::HUGEINT);  // revenue
                    names.emplace_back("revenue");

                    int32_t year_lo = input.inputs[1].GetValue<int32_t>();
                    int32_t year_hi = input.inputs[2].GetValue<int32_t>();

                    return make_uniq<FnBindData>(year_lo, year_hi);
                }

                static OperatorResultType FnExecute(ExecutionContext &, TableFunctionInput &in,
                                                    DataChunk &input, DataChunk &) {
                    auto &l = in.local_state->Cast<FnLocalState>();

                    // Access bound parameters
                    auto &bind_data = in.bind_data->Cast<FnBindData>();
                    int32_t year_lo = bind_data.year_lo;
                    int32_t year_hi = bind_data.year_hi;
                    
                    if (input.size() == 0) {       
                        return OperatorResultType::NEED_MORE_INPUT;
                    }

                    // Process input chunk using UnifiedVectorFormat
                    UnifiedVectorFormat c_nation_uvf;
                    UnifiedVectorFormat s_nation_uvf;
                    UnifiedVectorFormat lo_orderdate_uvf;
                    UnifiedVectorFormat lo_revenue_uvf;
                    
                    input.data[0].ToUnifiedFormat(input.size(), c_nation_uvf);
                    input.data[1].ToUnifiedFormat(input.size(), s_nation_uvf);
                    input.data[2].ToUnifiedFormat(input.size(), lo_orderdate_uvf);
                    input.data[3].ToUnifiedFormat(input.size(), lo_revenue_uvf);
                    
                    string_t* c_nation_ptr = (string_t*)c_nation_uvf.data;
                    string_t* s_nation_ptr = (string_t*)s_nation_uvf.data;
                    date_t* lo_orderdate_ptr = (date_t*)lo_orderdate_uvf.data;
                    uint32_t* lo_revenue_ptr = (uint32_t*)lo_revenue_uvf.data;
                    
                    // validity bitmaps
                    auto &valid_c_nation  = c_nation_uvf.validity;
                    auto &valid_s_nation  = s_nation_uvf.validity;
                    auto &valid_lo_orderdate  = lo_orderdate_uvf.validity;
                    auto &valid_lo_revenue  = lo_revenue_uvf.validity;
                    
                    const bool c_nation_all_valid = valid_c_nation.AllValid();
                    const bool s_nation_all_valid = valid_s_nation.AllValid();
                    const bool lo_orderdate_all_valid = valid_lo_orderdate.AllValid();
                    const bool lo_revenue_all_valid = valid_lo_revenue.AllValid();
                    
                    // Create selection vector to track which rows pass the filter
                    SelectionVector sel_vector(STANDARD_VECTOR_SIZE);
                    idx_t selected_count = 0;
                    
                    // Fast path: all relevant columns have no NULLs in this batch
                    if (c_nation_all_valid && s_nation_all_valid && lo_orderdate_all_valid && lo_revenue_all_valid) {
                        for (idx_t row_idx = 0; row_idx < input.size(); ++row_idx) {
                            idx_t i_c_nation = c_nation_uvf.sel->get_index(row_idx);
                            idx_t i_s_nation = s_nation_uvf.sel->get_index(row_idx);
                            idx_t i_lo_orderdate = lo_orderdate_uvf.sel->get_index(row_idx);
                            idx_t i_lo_revenue = lo_revenue_uvf.sel->get_index(row_idx);
                            
                            string_t c_nation_val = c_nation_ptr[i_c_nation];
                            string_t s_nation_val = s_nation_ptr[i_s_nation];
                            date_t lo_orderdate_val = lo_orderdate_ptr[i_lo_orderdate];
                            uint32_t lo_revenue_val = lo_revenue_ptr[i_lo_revenue];
                            
                            // Extract year from date
                            int32_t extracted_year = Date::ExtractYear(lo_orderdate_val);
                            
                            // Evaluate filter predicate: EXTRACT(YEAR FROM LO_ORDERDATE) BETWEEN {year_lo} AND {year_hi}
                            if (extracted_year >= year_lo && extracted_year <= year_hi) {
                                sel_vector.set_index(selected_count++, row_idx);
                                
                                // Update aggregation state
                                GroupKey key;
                                key.c_nation = c_nation_val;
                                key.s_nation = s_nation_val;
                                key.year = extracted_year;
                                
                                auto &state = l.agg_map[key];
                                state.sum_revenue += lo_revenue_val;
                            }
                        }
                    } else {
                        // Slow path: at least one column may contain NULLs
                        for (idx_t row_idx = 0; row_idx < input.size(); ++row_idx) {
                            idx_t i_c_nation = c_nation_uvf.sel->get_index(row_idx);
                            idx_t i_s_nation = s_nation_uvf.sel->get_index(row_idx);
                            idx_t i_lo_orderdate = lo_orderdate_uvf.sel->get_index(row_idx);
                            idx_t i_lo_revenue = lo_revenue_uvf.sel->get_index(row_idx);
                            
                            if (!c_nation_all_valid && !valid_c_nation.RowIsValid(i_c_nation)) {
                                continue; // row is NULL in column C_NATION -> skip
                            }
                            if (!s_nation_all_valid && !valid_s_nation.RowIsValid(i_s_nation)) {
                                continue; // row is NULL in column S_NATION -> skip
                            }
                            if (!lo_orderdate_all_valid && !valid_lo_orderdate.RowIsValid(i_lo_orderdate)) {
                                continue; // row is NULL in column LO_ORDERDATE -> skip
                            }
                            if (!lo_revenue_all_valid && !valid_lo_revenue.RowIsValid(i_lo_revenue)) {
                                continue; // row is NULL in column LO_REVENUE -> skip
                            }
                            
                            string_t c_nation_val = c_nation_ptr[i_c_nation];
                            string_t s_nation_val = s_nation_ptr[i_s_nation];
                            date_t lo_orderdate_val = lo_orderdate_ptr[i_lo_orderdate];
                            uint32_t lo_revenue_val = lo_revenue_ptr[i_lo_revenue];
                            
                            // Extract year from date
                            int32_t extracted_year = Date::ExtractYear(lo_orderdate_val);
                            
                            // Evaluate filter predicate: EXTRACT(YEAR FROM LO_ORDERDATE) BETWEEN {year_lo} AND {year_hi}
                            if (extracted_year >= year_lo && extracted_year <= year_hi) {
                                sel_vector.set_index(selected_count++, row_idx);
                                
                                // Update aggregation state
                                GroupKey key;
                                key.c_nation = c_nation_val;
                                key.s_nation = s_nation_val;
                                key.year = extracted_year;
                                
                                auto &state = l.agg_map[key];
                                state.sum_revenue += lo_revenue_val;
                            }
                        }
                    }
                    
                    // Apply the selection vector to the input chunk to filter rows
                    input.Slice(sel_vector, selected_count);

                    return OperatorResultType::NEED_MORE_INPUT; 
                }

                static OperatorFinalizeResultType FnFinalize(ExecutionContext &, TableFunctionInput &in,
                                                            DataChunk &out) {
                    auto &g = *reinterpret_cast<FnGlobalState *>(in.global_state.get());
                    auto &l = in.local_state->Cast<FnLocalState>();
                    
                    // Access bound parameters
                    //auto &bind_data = in.bind_data->Cast<FnBindData>();
                    //int32_t year_lo = bind_data.year_lo;
                    //int32_t year_hi = bind_data.year_hi;

                    // Merge the local state into the global state exactly once.
                    if (!l.merged) {
                        {
                            std::lock_guard<std::mutex> guard(g.lock);
                            for (const auto &entry : l.agg_map) {
                                const GroupKey &key = entry.first;
                                const AggState &state = entry.second;
                                // merge local state with global state
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
                            idx_t output_row = 0;
                            for (const auto &entry : g.agg_map) {
                                const GroupKey &key = entry.first;
                                const AggState &state = entry.second;
                                
                                // populate out chunk with final results
                                out.SetValue(0, output_row, key.c_nation);
                                out.SetValue(1, output_row, key.s_nation);
                                out.SetValue(2, output_row, Value::BIGINT(key.year));
                                hugeint_t revenue_hugeint = ToHugeint(state.sum_revenue);
                                out.SetValue(3, output_row, Value::HUGEINT(revenue_hugeint));
                                
                                output_row++;
                            }
                            out.SetCardinality(output_row);
                        }
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