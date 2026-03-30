/*
query_template: SELECT SUM(LO_EXTENDEDPRICE * LO_DISCOUNT) AS revenue
FROM lineorder_flat
WHERE CAST(strftime('%Y%m', LO_ORDERDATE) AS INTEGER) = :yyyymm
  AND LO_DISCOUNT BETWEEN :discount_lo AND :discount_hi
  AND LO_QUANTITY BETWEEN :qty_lo AND :qty_hi

split_template: SELECT revenue
FROM dbweaver((
  SELECT LO_ORDERDATE, LO_EXTENDEDPRICE, LO_DISCOUNT, LO_QUANTITY
  FROM lineorder_flat
  WHERE LO_DISCOUNT BETWEEN :discount_lo AND :discount_hi
    AND LO_QUANTITY BETWEEN :qty_lo AND :qty_hi
),:yyyymm);
query_example: SELECT SUM(LO_EXTENDEDPRICE * LO_DISCOUNT) AS revenue
FROM lineorder_flat
WHERE CAST(strftime('%Y%m', LO_ORDERDATE) AS INTEGER) = 199401
  AND LO_DISCOUNT BETWEEN 4 AND 6
  AND LO_QUANTITY BETWEEN 26 AND 35

split_example: 
*/
#define DUCKDB_EXTENSION_MAIN

#include "dbweaver_extension.hpp"
#include "duckdb.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/common/types/date.hpp"
#include <atomic>
#include <limits>
#include <mutex>

namespace duckdb {

//TODO: Define any helper structs or functions needed for binding/execution

struct FnBindData : public FunctionData {
    int32_t yyyymm;
    
    explicit FnBindData(int32_t yyyymm_p) : yyyymm(yyyymm_p) {}
    
    unique_ptr<FunctionData> Copy() const override { 
        return make_uniq<FnBindData>(yyyymm); 
    }
    bool Equals(const FunctionData &) const override { 
        return true; 
    }
};

struct FnGlobalState : public GlobalTableFunctionState {
    // Accumulator for SUM of (lo_extendedprice * lo_discount)
    std::atomic<int64_t> revenue_sum {0};
    std::mutex lock;
    std::atomic<idx_t> active_local_states {0};
    std::atomic<idx_t> merged_local_states {0};
    idx_t MaxThreads() const override {
        return std::numeric_limits<idx_t>::max();
    }
    
    //auto &bind_data = in.bind_data->Cast<FnBindData>();
    //int32_t yyyymm = bind_data.yyyymm;
};

static unique_ptr<GlobalTableFunctionState> FnInit(ClientContext &, TableFunctionInitInput &) {
    return make_uniq<FnGlobalState>();
}

struct FnLocalState : public LocalTableFunctionState {
    // Accumulator for SUM of (lo_extendedprice * lo_discount)
    int64_t revenue_sum = 0;
    bool merged = false;
};

static unique_ptr<LocalTableFunctionState> FnInitLocal(ExecutionContext &, TableFunctionInitInput &,
                                                      GlobalTableFunctionState *global_state) {
    auto &g = global_state->Cast<FnGlobalState>();
    g.active_local_states.fetch_add(1, std::memory_order_relaxed);
    return make_uniq<FnLocalState>();
}

// Helper function to extract year-month as integer from a date
static int32_t ExtractYearMonth(date_t date) {
    int32_t year, month, day;
    Date::Convert(date, year, month, day);
    return year * 100 + month;
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
    // Return revenue as HUGEINT
    return_types.emplace_back(LogicalType::HUGEINT);
    names.emplace_back("revenue");
    
    int32_t yyyymm = input.inputs[1].GetValue<int32_t>();

    return make_uniq<FnBindData>(yyyymm);
}

static OperatorResultType FnExecute(ExecutionContext &, TableFunctionInput &in,
                                            DataChunk &input, DataChunk &) {
    auto &l = in.local_state->Cast<FnLocalState>();

    auto &bind_data = in.bind_data->Cast<FnBindData>();
    int32_t yyyymm = bind_data.yyyymm;
    
    if (input.size() == 0) {       
        return OperatorResultType::NEED_MORE_INPUT;
    }

    // Setup UnifiedVectorFormat handles for input columns
    UnifiedVectorFormat lo_orderdate_uvf;
    UnifiedVectorFormat lo_extendedprice_uvf;
    UnifiedVectorFormat lo_discount_uvf;
    UnifiedVectorFormat lo_quantity_uvf;
    
    input.data[0].ToUnifiedFormat(input.size(), lo_orderdate_uvf);
    input.data[1].ToUnifiedFormat(input.size(), lo_extendedprice_uvf);
    input.data[2].ToUnifiedFormat(input.size(), lo_discount_uvf);
    input.data[3].ToUnifiedFormat(input.size(), lo_quantity_uvf);
    
    date_t* lo_orderdate_ptr = (date_t*)lo_orderdate_uvf.data;
    uint32_t* lo_extendedprice_ptr = (uint32_t*)lo_extendedprice_uvf.data;
    uint8_t* lo_discount_ptr = (uint8_t*)lo_discount_uvf.data;
    uint8_t* lo_quantity_ptr = (uint8_t*)lo_quantity_uvf.data;
    
    // Validity bitmaps
    auto &valid_lo_orderdate = lo_orderdate_uvf.validity;
    auto &valid_lo_extendedprice = lo_extendedprice_uvf.validity;
    auto &valid_lo_discount = lo_discount_uvf.validity;
    auto &valid_lo_quantity = lo_quantity_uvf.validity;
    
    const bool lo_orderdate_all_valid = valid_lo_orderdate.AllValid();
    const bool lo_extendedprice_all_valid = valid_lo_extendedprice.AllValid();
    const bool lo_discount_all_valid = valid_lo_discount.AllValid();
    const bool lo_quantity_all_valid = valid_lo_quantity.AllValid();
    
    // Output selection vector to handle filtered rows
    SelectionVector sel_vector(STANDARD_VECTOR_SIZE);
    idx_t output_count = 0;
    
    // Process rows with NULL handling
    if (lo_orderdate_all_valid && lo_extendedprice_all_valid && lo_discount_all_valid && lo_quantity_all_valid) {
        // Fast path: no NULLs in any column
        for (idx_t row_idx = 0; row_idx < input.size(); ++row_idx) {
            idx_t i_lo_orderdate = lo_orderdate_uvf.sel->get_index(row_idx);
            idx_t i_lo_extendedprice = lo_extendedprice_uvf.sel->get_index(row_idx);
            idx_t i_lo_discount = lo_discount_uvf.sel->get_index(row_idx);
            idx_t i_lo_quantity = lo_quantity_uvf.sel->get_index(row_idx);
            
            date_t v_lo_orderdate = lo_orderdate_ptr[i_lo_orderdate];
            uint32_t v_lo_extendedprice = lo_extendedprice_ptr[i_lo_extendedprice];
            uint8_t v_lo_discount = lo_discount_ptr[i_lo_discount];
            uint8_t v_lo_quantity = lo_quantity_ptr[i_lo_quantity];
            
            // Apply filter: (CAST(strftime('%Y%m', lo_orderdate) AS INTEGER) = {yyyymm})
            int32_t extracted_yearmonth = ExtractYearMonth(v_lo_orderdate);
            if (extracted_yearmonth != yyyymm) {
                continue;  // Row fails filter, skip it
            }
            
            // Compute lo_extendedprice * lo_discount and accumulate to revenue_sum
            int64_t price = (int64_t)v_lo_extendedprice;
            int64_t discount = (int64_t)v_lo_discount;
            l.revenue_sum += price * discount;
            
            // Row passes filter, add to output selection
            sel_vector.set_index(output_count++, row_idx);
        }
    } else {
        // Slow path: some columns may have NULLs
        for (idx_t row_idx = 0; row_idx < input.size(); ++row_idx) {
            idx_t i_lo_orderdate = lo_orderdate_uvf.sel->get_index(row_idx);
            idx_t i_lo_extendedprice = lo_extendedprice_uvf.sel->get_index(row_idx);
            idx_t i_lo_discount = lo_discount_uvf.sel->get_index(row_idx);
            idx_t i_lo_quantity = lo_quantity_uvf.sel->get_index(row_idx);
            
            if (!lo_orderdate_all_valid && !valid_lo_orderdate.RowIsValid(i_lo_orderdate)) {
                continue;
            }
            if (!lo_extendedprice_all_valid && !valid_lo_extendedprice.RowIsValid(i_lo_extendedprice)) {
                continue;
            }
            if (!lo_discount_all_valid && !valid_lo_discount.RowIsValid(i_lo_discount)) {
                continue;
            }
            if (!lo_quantity_all_valid && !valid_lo_quantity.RowIsValid(i_lo_quantity)) {
                continue;
            }
            
            date_t v_lo_orderdate = lo_orderdate_ptr[i_lo_orderdate];
            uint32_t v_lo_extendedprice = lo_extendedprice_ptr[i_lo_extendedprice];
            uint8_t v_lo_discount = lo_discount_ptr[i_lo_discount];
            uint8_t v_lo_quantity = lo_quantity_ptr[i_lo_quantity];
            
            // Apply filter: (CAST(strftime('%Y%m', lo_orderdate) AS INTEGER) = {yyyymm})
            int32_t extracted_yearmonth = ExtractYearMonth(v_lo_orderdate);
            if (extracted_yearmonth != yyyymm) {
                continue;  // Row fails filter, skip it
            }
            
            // Compute lo_extendedprice * lo_discount and accumulate to revenue_sum
            int64_t price = (int64_t)v_lo_extendedprice;
            int64_t discount = (int64_t)v_lo_discount;
            l.revenue_sum += price * discount;
            
            // Row passes filter, add to output selection
            sel_vector.set_index(output_count++, row_idx);
        }
    }

    // Update the input chunk with the filtered selection vector
    if (output_count != input.size()) {
        input.Slice(sel_vector, output_count);
    }
    
    // Pass through the filtered data
    return OperatorResultType::NEED_MORE_INPUT; 
}

static OperatorFinalizeResultType FnFinalize(ExecutionContext &, TableFunctionInput &in,
                                                    DataChunk &out) {
    auto &g = *reinterpret_cast<FnGlobalState *>(in.global_state.get());
    auto &l = in.local_state->Cast<FnLocalState>();
    //auto &bind_data = in.bind_data->Cast<FnBindData>();
    //int32_t yyyymm = bind_data.yyyymm;
    
    // Merge the local state into the global state exactly once.
    if (!l.merged) {
        {
            std::lock_guard<std::mutex> guard(g.lock);
            g.revenue_sum.fetch_add(l.revenue_sum, std::memory_order_relaxed);
        }
        l.merged = true;
        g.merged_local_states.fetch_add(1, std::memory_order_relaxed);
    }

    // Only the *last* local state to merge emits the final result.
    // All other threads return FINISHED with an empty chunk.
    const auto merged = g.merged_local_states.load(std::memory_order_relaxed);
    const auto active = g.active_local_states.load(std::memory_order_relaxed);
    if (active > 0 && merged == active) {
        int64_t final_revenue;
        {
            std::lock_guard<std::mutex> guard(g.lock);
            final_revenue = g.revenue_sum.load(std::memory_order_relaxed);
        }
        out.SetCardinality(1);
        // Convert int64_t to hugeint_t for the final output
        __int128 revenue_128 = static_cast<__int128>(final_revenue);
        hugeint_t revenue_hugeint = ToHugeint(revenue_128);
        out.SetValue(0, 0, Value::HUGEINT(revenue_hugeint));
    } else {
        out.SetCardinality(0);
    }
    
    return OperatorFinalizeResultType::FINISHED;
}

static void LoadInternal(ExtensionLoader &loader) {
    TableFunction f("dbweaver", {LogicalType::TABLE, LogicalType::INTEGER}, nullptr, FnBind, FnInit, FnInitLocal);
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