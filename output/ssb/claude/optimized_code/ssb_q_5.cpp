/*
query_template: SELECT
  EXTRACT(YEAR FROM LO_ORDERDATE) AS year,
  P_BRAND,
  SUM(LO_REVENUE) AS sum_revenue
FROM lineorder_flat
WHERE P_BRAND BETWEEN :brand_lo AND :brand_hi
  AND S_REGION = :s_region
GROUP BY year, P_BRAND
ORDER BY year, P_BRAND

split_template: SELECT year, P_BRAND, sum_revenue
FROM dbweaver((
  SELECT LO_ORDERDATE, P_BRAND, LO_REVENUE
  FROM lineorder_flat
  WHERE P_BRAND BETWEEN :brand_lo AND :brand_hi
    AND S_REGION = :s_region
)) ORDER BY year, P_BRAND;
query_example: SELECT
  EXTRACT(YEAR FROM LO_ORDERDATE) AS year,
  P_BRAND,
  SUM(LO_REVENUE) AS sum_revenue
FROM lineorder_flat
WHERE P_BRAND >= 'MFGR#2221' AND P_BRAND <= 'MFGR#2228'
  AND S_REGION = 'ASIA'
GROUP BY year, P_BRAND
ORDER BY year, P_BRAND

split_query: SELECT year, P_BRAND, sum_revenue
FROM dbweaver((
  SELECT LO_ORDERDATE, P_BRAND, LO_REVENUE
  FROM lineorder_flat
  WHERE P_BRAND >= 'MFGR#2221' AND P_BRAND <= 'MFGR#2228'
    AND S_REGION = 'ASIA'
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
#include <cstring>

namespace duckdb {

static constexpr int64_t YEAR_MIN = 1992;
static constexpr int NUM_YEARS = 7;       // 1992-1998
static constexpr int MAX_BRAND = 5541;    // brand numbers up to 5540
static constexpr int AGG_SIZE = NUM_YEARS * MAX_BRAND;

inline int ParseBrandNum(const string_t &s) {
    const char *p = s.GetData() + 5;     // skip "MFGR#"
    int n = 0;
    while (*p >= '0' && *p <= '9') { n = n * 10 + (*p++ - '0'); }
    return n;
}

inline int AggIdx(int64_t year, int brand_num) {
    return static_cast<int>(year - YEAR_MIN) * MAX_BRAND + brand_num;
}

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
    __int128 sum_revenue[AGG_SIZE] = {};
    uint8_t  populated[AGG_SIZE] = {};
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
    __int128 sum_revenue[AGG_SIZE] = {};
    uint8_t  populated[AGG_SIZE] = {};
    bool merged = false;
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
    return_types.push_back(LogicalType::BIGINT);
    return_types.push_back(LogicalType::VARCHAR);
    return_types.push_back(LogicalType::HUGEINT);
    names.push_back("year");
    names.push_back("P_BRAND");
    names.push_back("sum_revenue");

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
    auto *l_pop = l.populated;

    // Fast path: LO_ORDERDATE is a constant vector — extract year once
    if (input.data[0].GetVectorType() == VectorType::CONSTANT_VECTOR) {
        if (ConstantVector::IsNull(input.data[0])) {
            return OperatorResultType::NEED_MORE_INPUT;
        }
        auto const_date = ConstantVector::GetData<date_t>(input.data[0])[0];
        const int year_off = static_cast<int>(Date::ExtractYear(const_date) - YEAR_MIN);
        const int base = year_off * MAX_BRAND;

        UnifiedVectorFormat P_BRAND_uvf;
        input.data[1].ToUnifiedFormat(count, P_BRAND_uvf);
        auto *P_BRAND_ptr = reinterpret_cast<string_t *>(P_BRAND_uvf.data);

        UnifiedVectorFormat LO_REVENUE_uvf;
        input.data[2].ToUnifiedFormat(count, LO_REVENUE_uvf);
        auto *LO_REVENUE_ptr = reinterpret_cast<uint32_t *>(LO_REVENUE_uvf.data);

        auto &valid_P_BRAND = P_BRAND_uvf.validity;
        auto &valid_LO_REVENUE = LO_REVENUE_uvf.validity;
        const bool P_BRAND_all_valid = valid_P_BRAND.AllValid();
        const bool LO_REVENUE_all_valid = valid_LO_REVENUE.AllValid();

        for (idx_t row_idx = 0; row_idx < count; ++row_idx) {
            const idx_t i_P = P_BRAND_uvf.sel->get_index(row_idx);
            const idx_t i_R = LO_REVENUE_uvf.sel->get_index(row_idx);

            if (!P_BRAND_all_valid && !valid_P_BRAND.RowIsValid(i_P)) continue;
            if (!LO_REVENUE_all_valid && !valid_LO_REVENUE.RowIsValid(i_R)) continue;

            const int idx = base + ParseBrandNum(P_BRAND_ptr[i_P]);
            l_sum[idx] += LO_REVENUE_ptr[i_R];
            l_pop[idx] = 1;
        }

        return OperatorResultType::NEED_MORE_INPUT;
    }

    // General path: LO_ORDERDATE is not constant
    UnifiedVectorFormat LO_ORDERDATE_uvf;
    input.data[0].ToUnifiedFormat(count, LO_ORDERDATE_uvf);
    auto *LO_ORDERDATE_ptr = reinterpret_cast<date_t *>(LO_ORDERDATE_uvf.data);

    UnifiedVectorFormat P_BRAND_uvf;
    input.data[1].ToUnifiedFormat(count, P_BRAND_uvf);
    auto *P_BRAND_ptr = reinterpret_cast<string_t *>(P_BRAND_uvf.data);

    UnifiedVectorFormat LO_REVENUE_uvf;
    input.data[2].ToUnifiedFormat(count, LO_REVENUE_uvf);
    auto *LO_REVENUE_ptr = reinterpret_cast<uint32_t *>(LO_REVENUE_uvf.data);

    auto &valid_LO_ORDERDATE = LO_ORDERDATE_uvf.validity;
    auto &valid_P_BRAND = P_BRAND_uvf.validity;
    auto &valid_LO_REVENUE = LO_REVENUE_uvf.validity;
    const bool LO_ORDERDATE_all_valid = valid_LO_ORDERDATE.AllValid();
    const bool P_BRAND_all_valid = valid_P_BRAND.AllValid();
    const bool LO_REVENUE_all_valid = valid_LO_REVENUE.AllValid();

    if (LO_ORDERDATE_all_valid && P_BRAND_all_valid && LO_REVENUE_all_valid) {
        for (idx_t row_idx = 0; row_idx < count; ++row_idx) {
            const idx_t i_D = LO_ORDERDATE_uvf.sel->get_index(row_idx);
            const idx_t i_P = P_BRAND_uvf.sel->get_index(row_idx);
            const idx_t i_R = LO_REVENUE_uvf.sel->get_index(row_idx);

            const int year_off = static_cast<int>(Date::ExtractYear(LO_ORDERDATE_ptr[i_D]) - YEAR_MIN);
            const int idx = year_off * MAX_BRAND + ParseBrandNum(P_BRAND_ptr[i_P]);
            l_sum[idx] += LO_REVENUE_ptr[i_R];
            l_pop[idx] = 1;
        }
    } else {
        for (idx_t row_idx = 0; row_idx < count; ++row_idx) {
            const idx_t i_D = LO_ORDERDATE_uvf.sel->get_index(row_idx);
            const idx_t i_P = P_BRAND_uvf.sel->get_index(row_idx);
            const idx_t i_R = LO_REVENUE_uvf.sel->get_index(row_idx);

            if (!LO_ORDERDATE_all_valid && !valid_LO_ORDERDATE.RowIsValid(i_D)) continue;
            if (!P_BRAND_all_valid && !valid_P_BRAND.RowIsValid(i_P)) continue;
            if (!LO_REVENUE_all_valid && !valid_LO_REVENUE.RowIsValid(i_R)) continue;

            const int year_off = static_cast<int>(Date::ExtractYear(LO_ORDERDATE_ptr[i_D]) - YEAR_MIN);
            const int idx = year_off * MAX_BRAND + ParseBrandNum(P_BRAND_ptr[i_P]);
            l_sum[idx] += LO_REVENUE_ptr[i_R];
            l_pop[idx] = 1;
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
            for (int i = 0; i < AGG_SIZE; ++i) {
                if (l.populated[i]) {
                    g.sum_revenue[i] += l.sum_revenue[i];
                    g.populated[i] = 1;
                }
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
            for (int i = 0; i < AGG_SIZE; ++i) {
                if (!g.populated[i]) continue;
                const int64_t year = static_cast<int64_t>(i / MAX_BRAND) + YEAR_MIN;
                const int brand_num = i % MAX_BRAND;
                std::string brand_str = "MFGR#" + std::to_string(brand_num);

                out.SetValue(0, output_row, Value::BIGINT(year));
                out.SetValue(1, output_row, Value(brand_str));
                out.SetValue(2, output_row, Value::HUGEINT(ToHugeint(g.sum_revenue[i])));
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