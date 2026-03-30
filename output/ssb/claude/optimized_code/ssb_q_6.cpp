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
#include "duckdb/common/types/date.hpp"
#include <atomic>
#include <limits>
#include <mutex>

namespace duckdb {

static constexpr int32_t YEAR_MIN = 1992;
static constexpr int32_t NUM_YEARS = 7;  // 1992-1998

inline hugeint_t ToHugeint(__int128 acc) {
    hugeint_t result;
    result.lower = static_cast<uint64_t>(acc);
    result.upper = static_cast<int64_t>(acc >> 64);
    return result;
}

struct FnBindData : public FunctionData {
    unique_ptr<FunctionData> Copy() const override { return make_uniq<FnBindData>(); }
    bool Equals(const FunctionData &) const override { return true; }
};

struct FnGlobalState : public GlobalTableFunctionState {
    __int128 sum_revenue[NUM_YEARS] = {};
    string brand_value;
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
    __int128 sum_revenue[NUM_YEARS] = {};
    bool merged = false;
    bool brand_initialized = false;
    string brand_value;
};

static unique_ptr<LocalTableFunctionState> FnInitLocal(ExecutionContext &, TableFunctionInitInput &,
                                                      GlobalTableFunctionState *global_state) {
    auto &g = global_state->Cast<FnGlobalState>();
    g.active_local_states.fetch_add(1, std::memory_order_relaxed);
    return make_uniq<FnLocalState>();
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

    const idx_t count = input.size();
    auto *l_sum = l.sum_revenue;

    // Capture P_BRAND once (all rows share the same brand in this query)
    if (!l.brand_initialized) {
        UnifiedVectorFormat brand_uvf;
        input.data[2].ToUnifiedFormat(count, brand_uvf);
        auto *brand_ptr = reinterpret_cast<string_t *>(brand_uvf.data);
        const idx_t i0 = brand_uvf.sel->get_index(0);
        if (brand_uvf.validity.RowIsValid(i0)) {
            l.brand_value = brand_ptr[i0].GetString();
            l.brand_initialized = true;
        }
    }

    // Fast path: LO_ORDERDATE is a constant vector — extract year once
    if (input.data[0].GetVectorType() == VectorType::CONSTANT_VECTOR) {
        if (ConstantVector::IsNull(input.data[0])) {
            return OperatorResultType::NEED_MORE_INPUT;
        }
        auto const_date = ConstantVector::GetData<date_t>(input.data[0])[0];
        const int year_off = Date::ExtractYear(const_date) - YEAR_MIN;

        UnifiedVectorFormat rev_uvf;
        input.data[1].ToUnifiedFormat(count, rev_uvf);
        auto *rev_ptr = reinterpret_cast<uint32_t *>(rev_uvf.data);
        auto &valid_rev = rev_uvf.validity;
        const bool rev_all_valid = valid_rev.AllValid();

        for (idx_t row_idx = 0; row_idx < count; ++row_idx) {
            const idx_t i_R = rev_uvf.sel->get_index(row_idx);
            if (!rev_all_valid && !valid_rev.RowIsValid(i_R)) continue;
            l_sum[year_off] += rev_ptr[i_R];
        }

        return OperatorResultType::NEED_MORE_INPUT;
    }

    // General path: LO_ORDERDATE is not constant
    UnifiedVectorFormat date_uvf;
    input.data[0].ToUnifiedFormat(count, date_uvf);
    auto *date_ptr = reinterpret_cast<date_t *>(date_uvf.data);

    UnifiedVectorFormat rev_uvf;
    input.data[1].ToUnifiedFormat(count, rev_uvf);
    auto *rev_ptr = reinterpret_cast<uint32_t *>(rev_uvf.data);

    auto &valid_date = date_uvf.validity;
    auto &valid_rev = rev_uvf.validity;
    const bool date_all_valid = valid_date.AllValid();
    const bool rev_all_valid = valid_rev.AllValid();

    if (date_all_valid && rev_all_valid) {
        for (idx_t row_idx = 0; row_idx < count; ++row_idx) {
            const idx_t i_D = date_uvf.sel->get_index(row_idx);
            const idx_t i_R = rev_uvf.sel->get_index(row_idx);
            const int year_off = Date::ExtractYear(date_ptr[i_D]) - YEAR_MIN;
            l_sum[year_off] += rev_ptr[i_R];
        }
    } else {
        for (idx_t row_idx = 0; row_idx < count; ++row_idx) {
            const idx_t i_D = date_uvf.sel->get_index(row_idx);
            const idx_t i_R = rev_uvf.sel->get_index(row_idx);
            if (!date_all_valid && !valid_date.RowIsValid(i_D)) continue;
            if (!rev_all_valid && !valid_rev.RowIsValid(i_R)) continue;
            const int year_off = Date::ExtractYear(date_ptr[i_D]) - YEAR_MIN;
            l_sum[year_off] += rev_ptr[i_R];
        }
    }

    return OperatorResultType::NEED_MORE_INPUT;
}
static OperatorFinalizeResultType FnFinalize(ExecutionContext &, TableFunctionInput &in,
                                                    DataChunk &out) {
    auto &g = *reinterpret_cast<FnGlobalState *>(in.global_state.get());
    auto &l = in.local_state->Cast<FnLocalState>();

    if (!l.merged) {
        {
            std::lock_guard<std::mutex> guard(g.lock);
            for (int i = 0; i < NUM_YEARS; ++i) {
                g.sum_revenue[i] += l.sum_revenue[i];
            }
            if (l.brand_initialized && g.brand_value.empty()) {
                g.brand_value = l.brand_value;
            }
        }
        l.merged = true;
        g.merged_local_states.fetch_add(1, std::memory_order_relaxed);
    }

    const auto merged = g.merged_local_states.load(std::memory_order_relaxed);
    const auto active = g.active_local_states.load(std::memory_order_relaxed);
    if (active > 0 && merged == active) {
        {
            std::lock_guard<std::mutex> guard(g.lock);
            idx_t output_row = 0;
            for (int i = 0; i < NUM_YEARS; ++i) {
                if (g.sum_revenue[i] == 0) continue;
                out.SetValue(0, output_row, Value::HUGEINT(ToHugeint(g.sum_revenue[i])));
                out.SetValue(1, output_row, Value::BIGINT(static_cast<int64_t>(YEAR_MIN + i)));
                out.SetValue(2, output_row, Value(g.brand_value));
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