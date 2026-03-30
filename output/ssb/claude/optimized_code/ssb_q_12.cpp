/*
query_template: SELECT
  EXTRACT(YEAR FROM LO_ORDERDATE) AS year,
  S_NATION,
  P_CATEGORY,
  SUM(LO_REVENUE - LO_SUPPLYCOST) AS profit
FROM lineorder_flat
WHERE C_REGION = :c_region
  AND S_REGION = :s_region
  AND EXTRACT(YEAR FROM LO_ORDERDATE) IN (:year_1, :year_2)
  AND P_MFGR IN (:mfgr_1, :mfgr_2)
GROUP BY
  EXTRACT(YEAR FROM LO_ORDERDATE),
  S_NATION,
  P_CATEGORY
ORDER BY
  year ASC,
  S_NATION ASC,
  P_CATEGORY ASC

split_template: SELECT year, S_NATION, P_CATEGORY, profit
FROM dbweaver((
  SELECT LO_ORDERDATE, S_NATION, P_CATEGORY, LO_REVENUE, LO_SUPPLYCOST, P_MFGR
  FROM lineorder_flat
  WHERE C_REGION = :c_region
    AND S_REGION = :s_region
),:year_1,:year_2,:mfgr_1,:mfgr_2) ORDER BY year ASC, S_NATION ASC, P_CATEGORY ASC;
query_example: SELECT
  EXTRACT(YEAR FROM LO_ORDERDATE) AS year,
  S_NATION,
  P_CATEGORY,
  SUM(LO_REVENUE - LO_SUPPLYCOST) AS profit
FROM lineorder_flat
WHERE
  C_REGION = 'AMERICA'
  AND S_REGION = 'AMERICA'
  AND (EXTRACT(YEAR FROM LO_ORDERDATE) = 1997 OR EXTRACT(YEAR FROM LO_ORDERDATE) = 1998)
  AND (P_MFGR = 'MFGR#1' OR P_MFGR = 'MFGR#2')
GROUP BY
  EXTRACT(YEAR FROM LO_ORDERDATE),
  S_NATION,
  P_CATEGORY
ORDER BY
  year ASC,
  S_NATION ASC,
  P_CATEGORY ASC

split_query: SELECT year, S_NATION, P_CATEGORY, profit
FROM dbweaver((
  SELECT LO_ORDERDATE, S_NATION, P_CATEGORY, LO_REVENUE, LO_SUPPLYCOST, P_MFGR
  FROM lineorder_flat
  WHERE C_REGION = 'AMERICA'
    AND S_REGION = 'AMERICA'
),1997,1998,'MFGR#1','MFGR#2') ORDER BY year ASC, S_NATION ASC, P_CATEGORY ASC;
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
                int32_t year_1;
                int32_t year_2;
                string mfgr_1;
                string mfgr_2;

                explicit FnBindData(int32_t year_1_p, int32_t year_2_p, string mfgr_1_p, string mfgr_2_p) 
                    : year_1(year_1_p), year_2(year_2_p), mfgr_1(mfgr_1_p), mfgr_2(mfgr_2_p) {}

                unique_ptr<FunctionData> Copy() const override { 
                    return make_uniq<FnBindData>(year_1, year_2, mfgr_1, mfgr_2); 
                }
                bool Equals(const FunctionData &other_p) const override { 
                    auto &other = other_p.Cast<FnBindData>();
                    return year_1 == other.year_1 && year_2 == other.year_2 && 
                           mfgr_1 == other.mfgr_1 && mfgr_2 == other.mfgr_2;
                }
            };

            struct GroupKey {
                int32_t year;
                string_t nation;
                string_t category;
                
                bool operator==(const GroupKey& other) const {
                    return year == other.year && 
                           nation == other.nation && 
                           category == other.category;
                }
            };
            
            struct GroupKeyHash {
                size_t operator()(const GroupKey& k) const {
                    size_t h = 0;
                    h ^= std::hash<int32_t>{}(k.year) + 0x9e3779b9 + (h << 6) + (h >> 2);
                    h ^= duckdb::Hash(k.nation.GetData(), k.nation.GetSize()) + 0x9e3779b9 + (h << 6) + (h >> 2);
                    h ^= duckdb::Hash(k.category.GetData(), k.category.GetSize()) + 0x9e3779b9 + (h << 6) + (h >> 2);
                    return h;
                }
            };
            
            struct AggState {
                __int128 profit_sum = 0;
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

            static unique_ptr<GlobalTableFunctionState> FnInit(ClientContext &context, TableFunctionInitInput &input) {
                //auto &bind_data = input.bind_data->Cast<FnBindData>();
                //int32_t year_1 = bind_data.year_1;
                //int32_t year_2 = bind_data.year_2;
                //string mfgr_1 = bind_data.mfgr_1;
                //string mfgr_2 = bind_data.mfgr_2;
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
                result.lower = static_cast<uint64_t>(acc);
                result.upper = static_cast<int64_t>(acc >> 64);
                return result;
            }
            

                static unique_ptr<FunctionData> FnBind(ClientContext &, TableFunctionBindInput &input,
                                                    vector<LogicalType> &return_types,
                                                    vector<string> &names) {
                    // Extract parameters from input
                    int32_t year_1 = input.inputs[1].GetValue<int32_t>();
                    int32_t year_2 = input.inputs[2].GetValue<int32_t>();
                    string mfgr_1 = input.inputs[3].GetValue<string>();
                    string mfgr_2 = input.inputs[4].GetValue<string>();

                    // Define output schema based on authoritative mapping
                    return_types.push_back(LogicalType::BIGINT);  // year
                    return_types.push_back(LogicalType::VARCHAR); // S_NATION
                    return_types.push_back(LogicalType::VARCHAR); // P_CATEGORY
                    return_types.push_back(LogicalType::HUGEINT); // profit
                    
                    names.push_back("year");
                    names.push_back("S_NATION");
                    names.push_back("P_CATEGORY");
                    names.push_back("profit");

                    return make_uniq<FnBindData>(year_1, year_2, mfgr_1, mfgr_2);
                }
            

                static OperatorResultType FnExecute(ExecutionContext &context, TableFunctionInput &in,
                                                    DataChunk &input, DataChunk &output) {
                    auto &l = in.local_state->Cast<FnLocalState>();
                    
                    // Access bind data parameters
                    auto &bind_data = in.bind_data->Cast<FnBindData>();
                    int32_t year_1 = bind_data.year_1;
                    int32_t year_2 = bind_data.year_2;
                    string mfgr_1 = bind_data.mfgr_1;
                    string mfgr_2 = bind_data.mfgr_2;
                    
                    if (input.size() == 0) {       
                        return OperatorResultType::NEED_MORE_INPUT;
                    }
                    // Set up UnifiedVectorFormat handles for input columns.
                    // First stage (cheap filters) only needs LO_ORDERDATE and P_MFGR.
                    UnifiedVectorFormat lo_orderdate_uvf;
                    UnifiedVectorFormat p_mfgr_uvf;

                    input.data[0].ToUnifiedFormat(input.size(), lo_orderdate_uvf);
                    input.data[5].ToUnifiedFormat(input.size(), p_mfgr_uvf);

                    date_t *lo_orderdate_ptr = (date_t *)lo_orderdate_uvf.data;
                    string_t *p_mfgr_ptr = (string_t *)p_mfgr_uvf.data;

                    // Pre-allocate an array of qualifying row indices for the second stage
                    idx_t selection[STANDARD_VECTOR_SIZE];
                    idx_t selection_count = 0;

                    // Fast path for CONSTANT_VECTOR on LO_ORDERDATE: evaluate year filter once
                    if (input.data[0].GetVectorType() == VectorType::CONSTANT_VECTOR) {

                        // Work directly with the underlying Vector using ConstantVector helpers
                        Vector &orderdate_vec = input.data[0];
                        if (ConstantVector::IsNull(orderdate_vec)) {
                            // Entire chunk has NULL LO_ORDERDATE → no matching rows
                            return OperatorResultType::NEED_MORE_INPUT;
                        }
                        auto const_orderdate_ptr = ConstantVector::GetData<date_t>(orderdate_vec);
                        date_t const_orderdate = const_orderdate_ptr[0];
                        int32_t const_year = Date::ExtractYear(const_orderdate);
                        bool year_match_const = (const_year == year_1) || (const_year == year_2);
                        if (!year_match_const) {
                            // Entire chunk fails the year predicate → skip without touching other columns
                            return OperatorResultType::NEED_MORE_INPUT;
                        }
                        // Constant year for this whole chunk: first build a selection of rows
                        // that also satisfy the P_MFGR filter using only cheap columns.
                        auto &valid_p_mfgr_const = p_mfgr_uvf.validity;
                        const bool p_mfgr_all_valid_const = valid_p_mfgr_const.AllValid();
                        selection_count = 0;
                        if (p_mfgr_all_valid_const) {
                            for (idx_t row_idx = 0; row_idx < input.size(); ++row_idx) {
                                const idx_t logical_index = row_idx;
                                const idx_t base_index = p_mfgr_uvf.sel->get_index(logical_index);
                                const string_t v_p_mfgr = p_mfgr_ptr[base_index];
                                if ((v_p_mfgr == mfgr_1) || (v_p_mfgr == mfgr_2)) {
                                    // store logical row index so all columns use the same sel mapping later
                                    selection[selection_count++] = logical_index;
                                }
                            }
                        } else {
                            for (idx_t row_idx = 0; row_idx < input.size(); ++row_idx) {
                                const idx_t logical_index = row_idx;
                                const idx_t base_index = p_mfgr_uvf.sel->get_index(logical_index);
                                if (!valid_p_mfgr_const.RowIsValid(base_index)) {
                                    continue;
                                }
                                const string_t v_p_mfgr = p_mfgr_ptr[base_index];
                                if ((v_p_mfgr == mfgr_1) || (v_p_mfgr == mfgr_2)) {
                                    selection[selection_count++] = logical_index;
                                }
                            }
                        }



                        // If nothing passed both year and mfgr filters, skip this chunk entirely
                        if (selection_count == 0) {
                            return OperatorResultType::NEED_MORE_INPUT;
                        }

                        // Second stage: load remaining columns only for qualifying rows.
                        UnifiedVectorFormat s_nation_uvf;
                        UnifiedVectorFormat p_category_uvf;
                        UnifiedVectorFormat lo_revenue_uvf;
                        UnifiedVectorFormat lo_supplycost_uvf;

                        input.data[1].ToUnifiedFormat(input.size(), s_nation_uvf);
                        input.data[2].ToUnifiedFormat(input.size(), p_category_uvf);
                        input.data[3].ToUnifiedFormat(input.size(), lo_revenue_uvf);
                        input.data[4].ToUnifiedFormat(input.size(), lo_supplycost_uvf);

                        auto &valid_s_nation_const = s_nation_uvf.validity;
                        auto &valid_p_category_const = p_category_uvf.validity;
                        auto &valid_lo_revenue_const = lo_revenue_uvf.validity;
                        auto &valid_lo_supplycost_const = lo_supplycost_uvf.validity;

                        const bool s_nation_all_valid_const = valid_s_nation_const.AllValid();
                        const bool p_category_all_valid_const = valid_p_category_const.AllValid();
                        const bool lo_revenue_all_valid_const = valid_lo_revenue_const.AllValid();
                        const bool lo_supplycost_all_valid_const = valid_lo_supplycost_const.AllValid();

                        string_t *s_nation_ptr = (string_t *)s_nation_uvf.data;
                        string_t *p_category_ptr = (string_t *)p_category_uvf.data;
                        uint32_t *lo_revenue_ptr = (uint32_t *)lo_revenue_uvf.data;
                        uint32_t *lo_supplycost_ptr = (uint32_t *)lo_supplycost_uvf.data;
                        if (s_nation_all_valid_const && p_category_all_valid_const &&
                            lo_revenue_all_valid_const && lo_supplycost_all_valid_const) {
                            // No NULLs in any of the remaining columns for qualifying rows
                            for (idx_t sel_idx = 0; sel_idx < selection_count; ++sel_idx) {
                                const idx_t logical_index = selection[sel_idx];

                                const idx_t i_s_nation = s_nation_uvf.sel->get_index(logical_index);
                                const idx_t i_p_category = p_category_uvf.sel->get_index(logical_index);
                                const idx_t i_lo_revenue = lo_revenue_uvf.sel->get_index(logical_index);
                                const idx_t i_lo_supplycost = lo_supplycost_uvf.sel->get_index(logical_index);

                                const string_t v_s_nation = s_nation_ptr[i_s_nation];
                                const string_t v_p_category = p_category_ptr[i_p_category];
                                const uint32_t v_lo_revenue = lo_revenue_ptr[i_lo_revenue];
                                const uint32_t v_lo_supplycost = lo_supplycost_ptr[i_lo_supplycost];

                                GroupKey key;
                                key.year = const_year;
                                key.nation = v_s_nation;
                                key.category = v_p_category;

                                AggState &agg_state = l.agg_map[key];
                                agg_state.profit_sum += (int64_t)v_lo_revenue - (int64_t)v_lo_supplycost;
                            }
                        } else {
                            // At least one of the remaining columns may contain NULLs
                            for (idx_t sel_idx = 0; sel_idx < selection_count; ++sel_idx) {
                                const idx_t logical_index = selection[sel_idx];

                                const idx_t i_s_nation = s_nation_uvf.sel->get_index(logical_index);
                                const idx_t i_p_category = p_category_uvf.sel->get_index(logical_index);
                                const idx_t i_lo_revenue = lo_revenue_uvf.sel->get_index(logical_index);
                                const idx_t i_lo_supplycost = lo_supplycost_uvf.sel->get_index(logical_index);

                                if (!s_nation_all_valid_const && !valid_s_nation_const.RowIsValid(i_s_nation)) {
                                    continue;
                                }
                                if (!p_category_all_valid_const && !valid_p_category_const.RowIsValid(i_p_category)) {
                                    continue;
                                }
                                if (!lo_revenue_all_valid_const && !valid_lo_revenue_const.RowIsValid(i_lo_revenue)) {
                                    continue;
                                }
                                if (!lo_supplycost_all_valid_const && !valid_lo_supplycost_const.RowIsValid(i_lo_supplycost)) {
                                    continue;
                                }

                                const string_t v_s_nation = s_nation_ptr[i_s_nation];
                                const string_t v_p_category = p_category_ptr[i_p_category];
                                const uint32_t v_lo_revenue = lo_revenue_ptr[i_lo_revenue];
                                const uint32_t v_lo_supplycost = lo_supplycost_ptr[i_lo_supplycost];

                                GroupKey key;
                                key.year = const_year;
                                key.nation = v_s_nation;
                                key.category = v_p_category;

                                AggState &agg_state = l.agg_map[key];
                                agg_state.profit_sum += (int64_t)v_lo_revenue - (int64_t)v_lo_supplycost;
                            }
                        }


                        return OperatorResultType::NEED_MORE_INPUT;
                    }

                    // General (non-constant) case
                    auto &valid_lo_orderdate = lo_orderdate_uvf.validity;
                    auto &valid_p_mfgr = p_mfgr_uvf.validity;

                    const bool lo_orderdate_all_valid = valid_lo_orderdate.AllValid();
                    const bool p_mfgr_all_valid = valid_p_mfgr.AllValid();

                    // First stage: year and mfgr filters only
                    selection_count = 0;
                    if (lo_orderdate_all_valid && p_mfgr_all_valid) {
                        for (idx_t row_idx = 0; row_idx < input.size(); ++row_idx) {
                            const idx_t logical_index = row_idx;
                            const idx_t base_index = lo_orderdate_uvf.sel->get_index(logical_index);
                            const date_t v_lo_orderdate = lo_orderdate_ptr[base_index];

                            const int32_t extracted_year = Date::ExtractYear(v_lo_orderdate);
                            const bool year_match = (extracted_year == year_1) || (extracted_year == year_2);
                            if (!year_match) {
                                continue;
                            }

                            const idx_t i_p_mfgr = p_mfgr_uvf.sel->get_index(logical_index);
                            const string_t v_p_mfgr = p_mfgr_ptr[i_p_mfgr];
                            const bool mfgr_match = (v_p_mfgr == mfgr_1) || (v_p_mfgr == mfgr_2);
                            if (!mfgr_match) {
                                continue;
                            }

                            // This row passes both filters; remember its logical index
                            selection[selection_count++] = logical_index;
                        }
                    } else {
                        for (idx_t row_idx = 0; row_idx < input.size(); ++row_idx) {
                            const idx_t logical_index = row_idx;
                            const idx_t base_index = lo_orderdate_uvf.sel->get_index(logical_index);
                            if (!lo_orderdate_all_valid && !valid_lo_orderdate.RowIsValid(base_index)) {
                                continue;
                            }

                            const date_t v_lo_orderdate = lo_orderdate_ptr[base_index];
                            const int32_t extracted_year = Date::ExtractYear(v_lo_orderdate);
                            const bool year_match = (extracted_year == year_1) || (extracted_year == year_2);
                            if (!year_match) {
                                continue;
                            }

                            const idx_t i_p_mfgr = p_mfgr_uvf.sel->get_index(logical_index);
                            if (!p_mfgr_all_valid && !valid_p_mfgr.RowIsValid(i_p_mfgr)) {
                                continue;
                            }
                            const string_t v_p_mfgr = p_mfgr_ptr[i_p_mfgr];
                            const bool mfgr_match = (v_p_mfgr == mfgr_1) || (v_p_mfgr == mfgr_2);
                            if (!mfgr_match) {
                                continue;
                            }

                            selection[selection_count++] = logical_index;
                        }
                    }



                    if (selection_count == 0) {
                        return OperatorResultType::NEED_MORE_INPUT;
                    }

                    // Second stage: load and aggregate remaining columns for qualifying rows
                    UnifiedVectorFormat s_nation_uvf;
                    UnifiedVectorFormat p_category_uvf;
                    UnifiedVectorFormat lo_revenue_uvf;
                    UnifiedVectorFormat lo_supplycost_uvf;

                    input.data[1].ToUnifiedFormat(input.size(), s_nation_uvf);
                    input.data[2].ToUnifiedFormat(input.size(), p_category_uvf);
                    input.data[3].ToUnifiedFormat(input.size(), lo_revenue_uvf);
                    input.data[4].ToUnifiedFormat(input.size(), lo_supplycost_uvf);

                    auto &valid_s_nation = s_nation_uvf.validity;
                    auto &valid_p_category = p_category_uvf.validity;
                    auto &valid_lo_revenue = lo_revenue_uvf.validity;
                    auto &valid_lo_supplycost = lo_supplycost_uvf.validity;

                    const bool s_nation_all_valid = valid_s_nation.AllValid();
                    const bool p_category_all_valid = valid_p_category.AllValid();
                    const bool lo_revenue_all_valid = valid_lo_revenue.AllValid();
                    const bool lo_supplycost_all_valid = valid_lo_supplycost.AllValid();

                    string_t *s_nation_ptr = (string_t *)s_nation_uvf.data;
                    string_t *p_category_ptr = (string_t *)p_category_uvf.data;
                    uint32_t *lo_revenue_ptr = (uint32_t *)lo_revenue_uvf.data;
                    uint32_t *lo_supplycost_ptr = (uint32_t *)lo_supplycost_uvf.data;
                    if (lo_orderdate_all_valid && s_nation_all_valid && p_category_all_valid &&
                        lo_revenue_all_valid && lo_supplycost_all_valid) {

                        // Fast path: no per-row NULL checks for qualifying rows
                        for (idx_t sel_idx = 0; sel_idx < selection_count; ++sel_idx) {
                            const idx_t logical_index = selection[sel_idx];

                            const idx_t base_index = lo_orderdate_uvf.sel->get_index(logical_index);
                            const idx_t i_s_nation = s_nation_uvf.sel->get_index(logical_index);
                            const idx_t i_p_category = p_category_uvf.sel->get_index(logical_index);
                            const idx_t i_lo_revenue = lo_revenue_uvf.sel->get_index(logical_index);
                            const idx_t i_lo_supplycost = lo_supplycost_uvf.sel->get_index(logical_index);

                            const date_t v_lo_orderdate = lo_orderdate_ptr[base_index];
                            const string_t v_s_nation = s_nation_ptr[i_s_nation];
                            const string_t v_p_category = p_category_ptr[i_p_category];
                            const uint32_t v_lo_revenue = lo_revenue_ptr[i_lo_revenue];
                            const uint32_t v_lo_supplycost = lo_supplycost_ptr[i_lo_supplycost];

                            const int32_t extracted_year = Date::ExtractYear(v_lo_orderdate);

                            GroupKey key;
                            key.year = extracted_year;
                            key.nation = v_s_nation;
                            key.category = v_p_category;

                            AggState &agg_state = l.agg_map[key];
                            agg_state.profit_sum += (int64_t)v_lo_revenue - (int64_t)v_lo_supplycost;
                        }
                    } else {
                        // Slow path: at least one column may contain NULLs
                        for (idx_t sel_idx = 0; sel_idx < selection_count; ++sel_idx) {
                            const idx_t logical_index = selection[sel_idx];

                            const idx_t base_index = lo_orderdate_uvf.sel->get_index(logical_index);
                            const idx_t i_s_nation = s_nation_uvf.sel->get_index(logical_index);
                            const idx_t i_p_category = p_category_uvf.sel->get_index(logical_index);
                            const idx_t i_lo_revenue = lo_revenue_uvf.sel->get_index(logical_index);
                            const idx_t i_lo_supplycost = lo_supplycost_uvf.sel->get_index(logical_index);

                            if (!s_nation_all_valid && !valid_s_nation.RowIsValid(i_s_nation)) {
                                continue;
                            }
                            if (!p_category_all_valid && !valid_p_category.RowIsValid(i_p_category)) {
                                continue;
                            }
                            if (!lo_revenue_all_valid && !valid_lo_revenue.RowIsValid(i_lo_revenue)) {
                                continue;
                            }
                            if (!lo_supplycost_all_valid && !valid_lo_supplycost.RowIsValid(i_lo_supplycost)) {
                                continue;
                            }

                            const date_t v_lo_orderdate = lo_orderdate_ptr[base_index];
                            const string_t v_s_nation = s_nation_ptr[i_s_nation];
                            const string_t v_p_category = p_category_ptr[i_p_category];
                            const uint32_t v_lo_revenue = lo_revenue_ptr[i_lo_revenue];
                            const uint32_t v_lo_supplycost = lo_supplycost_ptr[i_lo_supplycost];

                            const int32_t extracted_year = Date::ExtractYear(v_lo_orderdate);

                            GroupKey key;
                            key.year = extracted_year;
                            key.nation = v_s_nation;
                            key.category = v_p_category;

                            AggState &agg_state = l.agg_map[key];
                            agg_state.profit_sum += (int64_t)v_lo_revenue - (int64_t)v_lo_supplycost;
                        }
                    }



                    return OperatorResultType::NEED_MORE_INPUT; 
                }

            

                static OperatorFinalizeResultType FnFinalize(ExecutionContext &context, TableFunctionInput &in,
                                                            DataChunk &out) {
                    auto &g = *reinterpret_cast<FnGlobalState *>(in.global_state.get());
                    auto &l = in.local_state->Cast<FnLocalState>();
                    
                    // Access bind data parameters
                    auto &bind_data = in.bind_data->Cast<FnBindData>();
                    int32_t year_1 = bind_data.year_1;
                    int32_t year_2 = bind_data.year_2;
                    string mfgr_1 = bind_data.mfgr_1;
                    string mfgr_2 = bind_data.mfgr_2;
                    
                    // Merge the local state into the global state exactly once.
                    if (!l.merged) {
                        {
                            std::lock_guard<std::mutex> guard(g.lock);
                            for (const auto &entry : l.agg_map) {
                                const GroupKey &key = entry.first;
                                const AggState &state = entry.second;
                                // merge local state with global state
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
                            idx_t output_idx = 0;
                            for (const auto &entry : g.agg_map) {
                                const GroupKey &key = entry.first;
                                const AggState &state = entry.second;
                                
                                // populate out chunk with final results
                                out.SetValue(0, output_idx, Value::BIGINT(key.year));
                                out.SetValue(1, output_idx, Value(key.nation));
                                out.SetValue(2, output_idx, Value(key.category));
                                out.SetValue(3, output_idx, Value::HUGEINT(ToHugeint(state.profit_sum)));
                                
                                output_idx++;
                            }
                            out.SetCardinality(output_idx);
                        }
                    } else {
                        out.SetCardinality(0);
                    }
                    
                    return OperatorFinalizeResultType::FINISHED;
                }
            

                static void LoadInternal(ExtensionLoader &loader) {
                    TableFunction f("dbweaver", {LogicalType::TABLE, LogicalType::INTEGER, LogicalType::INTEGER, LogicalType::VARCHAR, LogicalType::VARCHAR}, nullptr, FnBind, FnInit, FnInitLocal);
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