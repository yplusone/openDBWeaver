/*
query_template: SELECT SUM(LO_EXTENDEDPRICE * LO_DISCOUNT) AS revenue
FROM lineorder_flat
WHERE EXTRACT(YEAR FROM LO_ORDERDATE) = :year
  AND LO_DISCOUNT BETWEEN :discount_lo AND :discount_hi
  AND LO_QUANTITY < :qty_lt

split_template: SELECT revenue
FROM dbweaver((
  SELECT LO_ORDERDATE, LO_EXTENDEDPRICE, LO_DISCOUNT, LO_QUANTITY
  FROM lineorder_flat
  WHERE LO_DISCOUNT BETWEEN :discount_lo AND :discount_hi
    AND LO_QUANTITY < :qty_lt
),:year);
query_example: SELECT SUM(LO_EXTENDEDPRICE * LO_DISCOUNT) AS revenue
FROM lineorder_flat
WHERE EXTRACT(YEAR FROM LO_ORDERDATE) = 1993
  AND LO_DISCOUNT BETWEEN 1 AND 3
  AND LO_QUANTITY < 25

split_example: 
*/
#define DUCKDB_EXTENSION_MAIN

#include "dbweaver_extension.hpp"
#include "duckdb.hpp"
#include "duckdb/function/table_function.hpp"
#include <atomic>
#include <limits>
#include <mutex>
#include "duckdb/common/types/date.hpp"
//TODO: Add more includes as needed

namespace duckdb {

//TODO: Define any helper structs or functions needed for binding/execution

struct FnBindData : public FunctionData {
    int32_t year;  // add parameter field
    
    explicit FnBindData(int32_t year_p) : year(year_p) {}  // constructor
    
    unique_ptr<FunctionData> Copy() const override { 
        return make_uniq<FnBindData>(year);  // copy the parameter
    }
    
    bool Equals(const FunctionData &other_p) const override { 
        auto &other = other_p.Cast<FnBindData>();
        return year == other.year;  // compare the parameter
    }
};

struct FnGlobalState : public GlobalTableFunctionState {
    // TODO: Optional accumulators/counters (generator may append to this struct)
    std::mutex lock;
    std::atomic<idx_t> active_local_states {0};
    std::atomic<idx_t> merged_local_states {0};
    idx_t MaxThreads() const override {
        return std::numeric_limits<idx_t>::max();
    }
    
    //auto &bind_data = in.bind_data->Cast<FnBindData>();
    //int32_t year = bind_data.year;
    
    // Accumulator for SUM(lo_extendedprice * lo_discount)
    __int128 revenue = 0;
};

static unique_ptr<GlobalTableFunctionState> FnInit(ClientContext &, TableFunctionInitInput &) {
    return make_uniq<FnGlobalState>();
}

struct FnLocalState : public LocalTableFunctionState {
    //TODO: initialize local state and other preparations
    bool merged = false;
    
    // Accumulator for SUM(lo_extendedprice * lo_discount)
    __int128 revenue = 0;
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
    //TODO: populate return_types and names
    int32_t year = input.inputs[1].GetValue<int32_t>();
    
    // Add return types and names for the aggregate output
    return_types.push_back(LogicalType::HUGEINT);
    names.push_back("revenue");

    return make_uniq<FnBindData>(year);
}

static OperatorResultType FnExecute(ExecutionContext &, TableFunctionInput &in,
                                                    DataChunk &input, DataChunk &) {
    auto &l = in.local_state->Cast<FnLocalState>();

    // Access parameters from bind data if needed
    auto &bind_data = in.bind_data->Cast<FnBindData>();
    int32_t year = bind_data.year;
    
    if (input.size() == 0) {       
        return OperatorResultType::NEED_MORE_INPUT;
    }

    //TODO: process input chunk and produce output
    
    // UnifiedVectorFormat handles for input columns
    UnifiedVectorFormat lo_orderdate_uvf;
    UnifiedVectorFormat lo_extendedprice_uvf;
    UnifiedVectorFormat lo_discount_uvf;
    UnifiedVectorFormat lo_quantity_uvf;
    
    // Load input columns into UnifiedVectorFormat
    input.data[0].ToUnifiedFormat(input.size(), lo_orderdate_uvf);
    input.data[1].ToUnifiedFormat(input.size(), lo_extendedprice_uvf);
    input.data[2].ToUnifiedFormat(input.size(), lo_discount_uvf);
    input.data[3].ToUnifiedFormat(input.size(), lo_quantity_uvf);
    
    // Create typed pointers to physical data
    date_t* lo_orderdate_ptr = (date_t*)lo_orderdate_uvf.data;
    uint32_t* lo_extendedprice_ptr = (uint32_t*)lo_extendedprice_uvf.data;
    uint8_t* lo_discount_ptr = (uint8_t*)lo_discount_uvf.data;
    uint8_t* lo_quantity_ptr = (uint8_t*)lo_quantity_uvf.data;
    
    // Validity bitmaps for NULL checking
    auto &valid_lo_orderdate = lo_orderdate_uvf.validity;
    auto &valid_lo_extendedprice = lo_extendedprice_uvf.validity;
    auto &valid_lo_discount = lo_discount_uvf.validity;
    auto &valid_lo_quantity = lo_quantity_uvf.validity;
    
    // Batch-level NULL summary (whether each column is fully non-NULL)
    const bool lo_orderdate_all_valid = valid_lo_orderdate.AllValid();
    const bool lo_extendedprice_all_valid = valid_lo_extendedprice.AllValid();
    const bool lo_discount_all_valid = valid_lo_discount.AllValid();
    const bool lo_quantity_all_valid = valid_lo_quantity.AllValid();
    
    idx_t num_rows = input.size();
    
    // FAST BRANCH: all relevant columns have no NULLs in this batch
    if (lo_orderdate_all_valid && lo_extendedprice_all_valid && lo_discount_all_valid && lo_quantity_all_valid) {
        // --- Fast path: no per-row NULL checks ---
        for (idx_t row_idx = 0; row_idx < num_rows; ++row_idx) {
            // Directly load values without RowIsValid checks
            
            idx_t i_lo_orderdate = lo_orderdate_uvf.sel->get_index(row_idx);
            idx_t i_lo_extendedprice = lo_extendedprice_uvf.sel->get_index(row_idx);
            idx_t i_lo_discount = lo_discount_uvf.sel->get_index(row_idx);
            idx_t i_lo_quantity = lo_quantity_uvf.sel->get_index(row_idx);
            
            auto v_orderdate = lo_orderdate_ptr[i_lo_orderdate];
            auto v_extendedprice = lo_extendedprice_ptr[i_lo_extendedprice];
            auto v_discount = lo_discount_ptr[i_lo_discount];
            auto v_quantity = lo_quantity_ptr[i_lo_quantity];
            
            // ======================================
            //  Core computation logic (no NULLs)
            //<<CORE_COMPUTE>>
            // Extract year from date and check against parameter
            int32_t extracted_year = Date::ExtractYear(v_orderdate);
            if (extracted_year != year) {
                continue; // Skip this row if year doesn't match
            }
            
            // Update revenue accumulator: SUM(lo_extendedprice * lo_discount)
            l.revenue += static_cast<__int128>(v_extendedprice) * static_cast<__int128>(v_discount);
            // ============================
        }
    } else {
        // --- Slow path: at least one column may contain NULLs ---
        for (idx_t row_idx = 0; row_idx < num_rows; ++row_idx) {
            // For each column that is not fully valid, check this row
            idx_t i_lo_orderdate = lo_orderdate_uvf.sel->get_index(row_idx);
            idx_t i_lo_extendedprice = lo_extendedprice_uvf.sel->get_index(row_idx);
            idx_t i_lo_discount = lo_discount_uvf.sel->get_index(row_idx);
            idx_t i_lo_quantity = lo_quantity_uvf.sel->get_index(row_idx);
            
            if (!lo_orderdate_all_valid && !valid_lo_orderdate.RowIsValid(i_lo_orderdate)) {
                continue; // row is NULL in column lo_orderdate → skip
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
            
            // At this point, all required columns are valid for this row
            
            auto v_orderdate = lo_orderdate_ptr[i_lo_orderdate];
            auto v_extendedprice = lo_extendedprice_ptr[i_lo_extendedprice];
            auto v_discount = lo_discount_ptr[i_lo_discount];
            auto v_quantity = lo_quantity_ptr[i_lo_quantity];
            
            // ======================================
            //  Core computation logic (NULL-safe)
            //<<CORE_COMPUTE>>
            // Extract year from date and check against parameter
            int32_t extracted_year = Date::ExtractYear(v_orderdate);
            if (extracted_year != year) {
                continue; // Skip this row if year doesn't match
            }
            
            // Update revenue accumulator: SUM(lo_extendedprice * lo_discount)
            l.revenue += static_cast<__int128>(v_extendedprice) * static_cast<__int128>(v_discount);
            // ======================================
        }
    }

    return OperatorResultType::NEED_MORE_INPUT; 
}

static OperatorFinalizeResultType FnFinalize(ExecutionContext &, TableFunctionInput &in,
                                                            DataChunk &out) {
    auto &g = *reinterpret_cast<FnGlobalState *>(in.global_state.get());
    auto &l = in.local_state->Cast<FnLocalState>();
    
    // Access parameters from bind data if needed
    //auto &bind_data = in.bind_data->Cast<FnBindData>();
    //int32_t year = bind_data.year;
    
    // Merge the local state into the global state exactly once.
    if (!l.merged) {
        {
            std::lock_guard<std::mutex> guard(g.lock);
            //TODO: merge local state with global state
            g.revenue += l.revenue;
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
        }
        //TODO: populate out chunk with final results
        out.SetCardinality(1);
        hugeint_t revenue_hugeint = ToHugeint(g.revenue);
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