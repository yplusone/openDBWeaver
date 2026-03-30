/*
query_template: SELECT SUM(LO_EXTENDEDPRICE * LO_DISCOUNT) AS revenue
FROM lineorder_flat
WHERE CAST(strftime('%V', LO_ORDERDATE) AS INTEGER) = :week
  AND EXTRACT(YEAR FROM LO_ORDERDATE) = :year
  AND LO_DISCOUNT BETWEEN :discount_lo AND :discount_hi
  AND LO_QUANTITY BETWEEN :qty_lo AND :qty_hi

split_template: SELECT revenue
FROM dbweaver((
  SELECT LO_EXTENDEDPRICE, LO_DISCOUNT, LO_ORDERDATE
  FROM lineorder_flat
  WHERE LO_DISCOUNT BETWEEN :discount_lo AND :discount_hi
    AND LO_QUANTITY BETWEEN :qty_lo AND :qty_hi
), :week, :year)
query_example: SELECT SUM(LO_EXTENDEDPRICE * LO_DISCOUNT) AS revenue
FROM lineorder_flat
WHERE CAST(strftime('%V', LO_ORDERDATE) AS INTEGER) = 6
  AND EXTRACT(YEAR FROM LO_ORDERDATE) = 1994
  AND LO_DISCOUNT BETWEEN 5 AND 7
  AND LO_QUANTITY BETWEEN 26 AND 35

split_example: 
*/
#define DUCKDB_EXTENSION_MAIN
#include "dbweaver_extension.hpp"
#include "duckdb.hpp"
#include "duckdb/function/table_function.hpp"
#include <atomic>
#include <limits>
#include <mutex>

namespace {
// Compute the date_t day number (days since 1970-01-01) of the Monday and Sunday of ISO week (year, week).
// Uses a constant-time algorithm (no per-row calendar arithmetic).
static inline void ComputeISOWeekDateRange(int32_t year, int32_t week, int32_t &lo_days, int32_t &hi_days) {
    // Convert civil date to days since 1970-01-01.
    auto CivilToDays = [](int32_t y, uint32_t m, uint32_t d) -> int32_t {
        // Howard Hinnant's algorithm
        y -= (m <= 2);
        const int32_t era = (y >= 0 ? y : y - 399) / 400;
        const uint32_t yoe = (uint32_t)(y - era * 400);
        const uint32_t doy = (153 * (m + (m > 2 ? (uint32_t)-3 : (uint32_t)9)) + 2) / 5 + d - 1;
        const uint32_t doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
        return (int32_t)(era * 146097 + (int32_t)doe - 719468);
    };
    auto WeekdayFromDays = [](int32_t z) -> int32_t {
        // Monday=1..Sunday=7, for days since 1970-01-01
        int32_t w = (z + 3) % 7;
        if (w < 0) {
            w += 7;
        }
        return w + 1;
    };

    // ISO week 1 is the week with Jan 4th.
    const int32_t jan4 = CivilToDays(year, 1, 4);
    const int32_t jan4_wd = WeekdayFromDays(jan4); // 1..7
    const int32_t week1_monday = jan4 - (jan4_wd - 1);

    lo_days = week1_monday + (week - 1) * 7;
    hi_days = lo_days + 6;
}
} // namespace


namespace duckdb {

            //TODO: Define any helper structs or functions needed for binding/execution
            struct FnBindData : public FunctionData {
                int32_t week;
                int32_t year;
                int32_t lo_date_days;
                int32_t hi_date_days;

                explicit FnBindData(int32_t week_p, int32_t year_p, int32_t lo_date_days_p, int32_t hi_date_days_p)
                    : week(week_p), year(year_p), lo_date_days(lo_date_days_p), hi_date_days(hi_date_days_p) {
                }

                unique_ptr<FunctionData> Copy() const override {
                    return make_uniq<FnBindData>(week, year, lo_date_days, hi_date_days);
                }
                bool Equals(const FunctionData &other_p) const override {
                    auto &other = other_p.Cast<FnBindData>();
                    return week == other.week && year == other.year && lo_date_days == other.lo_date_days && hi_date_days == other.hi_date_days;
                }
            };


            struct FnGlobalState : public GlobalTableFunctionState {
                // Accumulator for SUM(lo_extendedprice * lo_discount)
                __int128 revenue_sum = 0;
                std::mutex lock;
                std::atomic<idx_t> active_local_states {0};
                std::atomic<idx_t> merged_local_states {0};
              	idx_t MaxThreads() const override {
                    return std::numeric_limits<idx_t>::max();
                }
                
                //auto &bind_data = in.bind_data->Cast<FnBindData>();
                //int32_t week = bind_data.week;
                //int32_t year = bind_data.year;
            };

            static unique_ptr<GlobalTableFunctionState> FnInit(ClientContext &, TableFunctionInitInput &input) {
                //auto &bind_data = input.bind_data->Cast<FnBindData>();
                //int32_t week = bind_data.week;
                //int32_t year = bind_data.year;
                
                return make_uniq<FnGlobalState>();
            }
            
            struct FnLocalState : public LocalTableFunctionState {
                // Accumulator for SUM(lo_extendedprice * lo_discount)
                __int128 revenue_sum = 0;
                bool merged = false;
            };

            static unique_ptr<LocalTableFunctionState> FnInitLocal(ExecutionContext &, TableFunctionInitInput &input,
                                                                  GlobalTableFunctionState *global_state) {
                //auto &bind_data = input.bind_data->Cast<FnBindData>();
                //int32_t week = bind_data.week;
                //int32_t year = bind_data.year;
                
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
                    int32_t week = input.inputs[1].GetValue<int32_t>();
                    int32_t year = input.inputs[2].GetValue<int32_t>();

                    // Precompute exact date range for ISO week/year as date_t day integers
                    int32_t lo_days, hi_days;
                    ComputeISOWeekDateRange(year, week, lo_days, hi_days);

                    // Define output schema based on authoritative mapping
                    names.push_back("revenue");
                    return_types.push_back(LogicalType::HUGEINT);

                    return make_uniq<FnBindData>(week, year, lo_days, hi_days);
                }
                static OperatorResultType FnExecute(ExecutionContext &, TableFunctionInput &in,
                                                    DataChunk &input, DataChunk &) {
                    auto &l = in.local_state->Cast<FnLocalState>();
                    // Access bound parameters
                    auto &bind_data = in.bind_data->Cast<FnBindData>();
                    const int32_t lo_days = bind_data.lo_date_days;
                    const int32_t hi_days = bind_data.hi_date_days;

                    
                    if (input.size() == 0) {       
                        return OperatorResultType::NEED_MORE_INPUT;
                    }

                    // Batch-level optimization: if LO_ORDERDATE is a ConstantVector, check date filter once.
                    if (input.data[0].GetVectorType() == VectorType::CONSTANT_VECTOR) {
                        // The constant may still be NULL
                        if (ConstantVector::IsNull(input.data[0])) {
                            input.SetCardinality(0);
                            return OperatorResultType::NEED_MORE_INPUT;
                        }
                        const auto v_lo_orderdate = ConstantVector::GetData<date_t>(input.data[0])[0];
                        const int32_t od_days = v_lo_orderdate.days;
                        if (od_days < lo_days || od_days > hi_days) {
                            // Entire batch fails the date filter, skip without touching other columns
                            input.SetCardinality(0);
                            return OperatorResultType::NEED_MORE_INPUT;
                        }

                        // Entire batch passes the date filter, only need to accumulate extendedprice * discount
                        UnifiedVectorFormat lo_extendedprice_uvf;
                        input.data[1].ToUnifiedFormat(input.size(), lo_extendedprice_uvf);
                        auto *lo_extendedprice_ptr = (uint32_t *)lo_extendedprice_uvf.data;

                        UnifiedVectorFormat lo_discount_uvf;
                        input.data[2].ToUnifiedFormat(input.size(), lo_discount_uvf);
                        auto *lo_discount_ptr = (uint8_t *)lo_discount_uvf.data;

                        auto &valid_lo_extendedprice = lo_extendedprice_uvf.validity;
                        auto &valid_lo_discount = lo_discount_uvf.validity;
                        const bool lo_extendedprice_all_valid = valid_lo_extendedprice.AllValid();
                        const bool lo_discount_all_valid = valid_lo_discount.AllValid();

                        if (lo_extendedprice_all_valid && lo_discount_all_valid) {
                            for (idx_t row_idx = 0; row_idx < input.size(); ++row_idx) {
                                const idx_t i_lo_extendedprice = lo_extendedprice_uvf.sel->get_index(row_idx);
                                const idx_t i_lo_discount = lo_discount_uvf.sel->get_index(row_idx);
                                const uint32_t v_lo_extendedprice = lo_extendedprice_ptr[i_lo_extendedprice];
                                const uint8_t v_lo_discount = lo_discount_ptr[i_lo_discount];
                                l.revenue_sum += (__int128)v_lo_extendedprice * v_lo_discount;
                            }
                        } else {
                            for (idx_t row_idx = 0; row_idx < input.size(); ++row_idx) {
                                const idx_t i_lo_extendedprice = lo_extendedprice_uvf.sel->get_index(row_idx);
                                const idx_t i_lo_discount = lo_discount_uvf.sel->get_index(row_idx);
                                if (!lo_extendedprice_all_valid && !valid_lo_extendedprice.RowIsValid(i_lo_extendedprice)) {
                                    continue;
                                }
                                if (!lo_discount_all_valid && !valid_lo_discount.RowIsValid(i_lo_discount)) {
                                    continue;
                                }
                                const uint32_t v_lo_extendedprice = lo_extendedprice_ptr[i_lo_extendedprice];
                                const uint8_t v_lo_discount = lo_discount_ptr[i_lo_discount];
                                l.revenue_sum += (__int128)v_lo_extendedprice * v_lo_discount;
                            }
                        }
                        return OperatorResultType::NEED_MORE_INPUT;
                    }

                    // Process input columns using UnifiedVectorFormat
UnifiedVectorFormat lo_orderdate_uvf;
input.data[0].ToUnifiedFormat(input.size(), lo_orderdate_uvf);
date_t* lo_orderdate_ptr = (date_t*)lo_orderdate_uvf.data;

UnifiedVectorFormat lo_extendedprice_uvf;
input.data[1].ToUnifiedFormat(input.size(), lo_extendedprice_uvf);
uint32_t* lo_extendedprice_ptr = (uint32_t*)lo_extendedprice_uvf.data;

UnifiedVectorFormat lo_discount_uvf;
input.data[2].ToUnifiedFormat(input.size(), lo_discount_uvf);
uint8_t* lo_discount_ptr = (uint8_t*)lo_discount_uvf.data;



                    // Validity bitmaps
                    auto &valid_lo_extendedprice = lo_extendedprice_uvf.validity;
                    auto &valid_lo_discount = lo_discount_uvf.validity;
                    auto &valid_lo_orderdate = lo_orderdate_uvf.validity;

                    const bool lo_extendedprice_all_valid = valid_lo_extendedprice.AllValid();
                    const bool lo_discount_all_valid = valid_lo_discount.AllValid();
                    const bool lo_orderdate_all_valid = valid_lo_orderdate.AllValid();
                    // We never output or forward rows from this input chunk; we only accumulate.
                    // Avoid SelectionVector construction and input.Slice() overhead.

                    // Process rows with null handling
                    if (lo_extendedprice_all_valid && lo_discount_all_valid && lo_orderdate_all_valid) {
                        // Fast path: no per-row NULL checks
                        for (idx_t row_idx = 0; row_idx < input.size(); ++row_idx) {
                            const idx_t i_lo_extendedprice = lo_extendedprice_uvf.sel->get_index(row_idx);
                            const idx_t i_lo_discount = lo_discount_uvf.sel->get_index(row_idx);
                            const idx_t i_lo_orderdate = lo_orderdate_uvf.sel->get_index(row_idx);

                            const date_t v_lo_orderdate = lo_orderdate_ptr[i_lo_orderdate];
                            const int32_t od_days = v_lo_orderdate.days;
                            if (od_days < lo_days || od_days > hi_days) {
                                continue;
                            }

                            const uint32_t v_lo_extendedprice = lo_extendedprice_ptr[i_lo_extendedprice];
                            const uint8_t v_lo_discount = lo_discount_ptr[i_lo_discount];
                            l.revenue_sum += (__int128)v_lo_extendedprice * v_lo_discount;
                        }
                    } else {
                        // Slow path: at least one column may contain NULLs
                        for (idx_t row_idx = 0; row_idx < input.size(); ++row_idx) {
                            const idx_t i_lo_extendedprice = lo_extendedprice_uvf.sel->get_index(row_idx);
                            const idx_t i_lo_discount = lo_discount_uvf.sel->get_index(row_idx);
                            const idx_t i_lo_orderdate = lo_orderdate_uvf.sel->get_index(row_idx);

                            if (!lo_orderdate_all_valid && !valid_lo_orderdate.RowIsValid(i_lo_orderdate)) {
                                continue;
                            }
                            const date_t v_lo_orderdate = lo_orderdate_ptr[i_lo_orderdate];
                            const int32_t od_days = v_lo_orderdate.days;
                            if (od_days < lo_days || od_days > hi_days) {
                                continue;
                            }

                            if (!lo_extendedprice_all_valid && !valid_lo_extendedprice.RowIsValid(i_lo_extendedprice)) {
                                continue;
                            }
                            if (!lo_discount_all_valid && !valid_lo_discount.RowIsValid(i_lo_discount)) {
                                continue;
                            }

                            const uint32_t v_lo_extendedprice = lo_extendedprice_ptr[i_lo_extendedprice];
                            const uint8_t v_lo_discount = lo_discount_ptr[i_lo_discount];
                            l.revenue_sum += (__int128)v_lo_extendedprice * v_lo_discount;
                        }
                    }

                    return OperatorResultType::NEED_MORE_INPUT; 
 
                }
            

            

                static OperatorFinalizeResultType FnFinalize(ExecutionContext &, TableFunctionInput &in,
                                                            DataChunk &out) {
                    auto &g = *reinterpret_cast<FnGlobalState *>(in.global_state.get());
                    auto &l = in.local_state->Cast<FnLocalState>();
                    // Access bound parameters
                    auto &bind_data = in.bind_data->Cast<FnBindData>();
                    (void)bind_data;

                    
                    // Merge the local state into the global state exactly once.
                    if (!l.merged) {
                        {
                            std::lock_guard<std::mutex> guard(g.lock);
                            g.revenue_sum += l.revenue_sum;
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
                            // Get final result from global state
                        }
                        
                        // Emit the final result
                        hugeint_t revenue_hugeint = ToHugeint(g.revenue_sum);
                        out.SetCardinality(1);
                        out.SetValue(0, 0, Value::HUGEINT(revenue_hugeint));
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