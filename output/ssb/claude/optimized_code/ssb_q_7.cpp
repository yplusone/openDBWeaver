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

split_template: SELECT C_NATION, S_NATION, year, revenue
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

split_query: SELECT C_NATION, S_NATION, year, revenue
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
#include "duckdb/common/types/date.hpp"
#include <atomic>
#include <limits>
#include <mutex>
#include <vector>
#include <algorithm>

namespace duckdb {

static constexpr int32_t YEAR_MIN = 1992;
static constexpr int NUM_YEARS = 7;       // 1992-1998
static constexpr int NUM_NATIONS = 25;
static constexpr int AGG_SIZE = NUM_NATIONS * NUM_NATIONS * NUM_YEARS;  // 4375

static const char * const NATION_NAMES[NUM_NATIONS] = {
    "ALGERIA", "ARGENTINA", "BRAZIL", "CANADA", "CHINA",
    "EGYPT", "ETHIOPIA", "FRANCE", "GERMANY", "INDIA",
    "INDONESIA", "IRAN", "IRAQ", "JAPAN", "JORDAN",
    "KENYA", "MOROCCO", "MOZAMBIQUE", "PERU", "ROMANIA",
    "RUSSIA", "SAUDI ARABIA", "UNITED KINGDOM", "UNITED STATES", "VIETNAM"
};

inline int NationToIdx(const string_t &s) {
    const char *p = s.GetData();
    uint32_t len = s.GetSize();
    switch (len) {
        case 4:
            if (p[0] == 'P') return 18;          // PERU
            return p[3] == 'N' ? 11 : 12;        // IRAN / IRAQ
        case 5:
            switch (p[0]) {
                case 'C': return 4;               // CHINA
                case 'E': return 5;               // EGYPT
                case 'I': return 9;               // INDIA
                case 'J': return 13;              // JAPAN
                default:  return 15;              // KENYA
            }
        case 6:
            switch (p[0]) {
                case 'B': return 2;               // BRAZIL
                case 'C': return 3;               // CANADA
                case 'F': return 7;               // FRANCE
                case 'J': return 14;              // JORDAN
                default:  return 20;              // RUSSIA
            }
        case 7:
            switch (p[0]) {
                case 'A': return 0;               // ALGERIA
                case 'G': return 8;               // GERMANY
                case 'M': return 16;              // MOROCCO
                case 'R': return 19;              // ROMANIA
                default:  return 24;              // VIETNAM
            }
        case 8:  return 6;                        // ETHIOPIA
        case 9:  return p[0] == 'A' ? 1 : 10;    // ARGENTINA / INDONESIA
        case 10: return 17;                       // MOZAMBIQUE
        case 12: return 21;                       // SAUDI ARABIA
        case 13: return 23;                       // UNITED STATES
        default: return 22;                       // UNITED KINGDOM (14)
    }
}

inline int AggIdx(int c, int s, int y) {
    return c * (NUM_NATIONS * NUM_YEARS) + s * NUM_YEARS + y;
}

inline hugeint_t ToHugeint(__int128 acc) {
    hugeint_t result;
    result.lower = static_cast<uint64_t>(acc);
    result.upper = static_cast<int64_t>(acc >> 64);
    return result;
}

struct FnBindData : public FunctionData {
    int32_t year_lo;
    int32_t year_hi;

    explicit FnBindData(int32_t lo, int32_t hi) : year_lo(lo), year_hi(hi) {}
    unique_ptr<FunctionData> Copy() const override { return make_uniq<FnBindData>(year_lo, year_hi); }
    bool Equals(const FunctionData &other_p) const override {
        auto &o = other_p.Cast<FnBindData>();
        return year_lo == o.year_lo && year_hi == o.year_hi;
    }
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

static unique_ptr<FunctionData> FnBind(ClientContext &, TableFunctionBindInput &input,
                                                    vector<LogicalType> &return_types,
                                                    vector<string> &names) {
    //TODO: populate return_types and names
    int32_t year_lo = input.inputs[1].GetValue<int32_t>();
    int32_t year_hi = input.inputs[2].GetValue<int32_t>();
    
    // Set up return types and names for: C_NATION, S_NATION, YEAR, REVENUE
    return_types.push_back(LogicalType::VARCHAR);  // C_NATION
    return_types.push_back(LogicalType::VARCHAR);  // S_NATION
    return_types.push_back(LogicalType::BIGINT);   // YEAR
    return_types.push_back(LogicalType::HUGEINT);  // REVENUE
    
    names.push_back("C_NATION");
    names.push_back("S_NATION");
    names.push_back("year");
    names.push_back("revenue");

    return make_uniq<FnBindData>(year_lo, year_hi);
}

static OperatorResultType FnExecute(ExecutionContext &, TableFunctionInput &in,
                                                    DataChunk &input, DataChunk &) {
    auto &l = in.local_state->Cast<FnLocalState>();
    auto &bind_data = in.bind_data->Cast<FnBindData>();
    const int32_t year_lo = bind_data.year_lo;
    const int32_t year_hi = bind_data.year_hi;

    if (input.size() == 0) {
        return OperatorResultType::NEED_MORE_INPUT;
    }

    const idx_t count = input.size();
    auto *l_sum = l.sum_revenue;
    auto *l_pop = l.populated;

    UnifiedVectorFormat cn_uvf, sn_uvf, rev_uvf;
    input.data[0].ToUnifiedFormat(count, cn_uvf);
    input.data[1].ToUnifiedFormat(count, sn_uvf);
    input.data[3].ToUnifiedFormat(count, rev_uvf);
    auto *cn_ptr  = reinterpret_cast<string_t *>(cn_uvf.data);
    auto *sn_ptr  = reinterpret_cast<string_t *>(sn_uvf.data);
    auto *rev_ptr = reinterpret_cast<uint32_t *>(rev_uvf.data);

    auto &v_cn = cn_uvf.validity;   auto &v_sn = sn_uvf.validity;
    auto &v_rev = rev_uvf.validity;
    const bool cn_ok = v_cn.AllValid(), sn_ok = v_sn.AllValid(), rev_ok = v_rev.AllValid();

    // Fast path: LO_ORDERDATE is constant — extract year once
    if (input.data[2].GetVectorType() == VectorType::CONSTANT_VECTOR) {
        if (ConstantVector::IsNull(input.data[2])) {
            return OperatorResultType::NEED_MORE_INPUT;
        }
        const int32_t year = Date::ExtractYear(ConstantVector::GetData<date_t>(input.data[2])[0]);
        if (year < year_lo || year > year_hi) {
            return OperatorResultType::NEED_MORE_INPUT;
        }
        const int year_off = year - YEAR_MIN;

        if (cn_ok && sn_ok && rev_ok) {
            for (idx_t r = 0; r < count; ++r) {
                const int idx = AggIdx(NationToIdx(cn_ptr[cn_uvf.sel->get_index(r)]),
                                       NationToIdx(sn_ptr[sn_uvf.sel->get_index(r)]),
                                       year_off);
                l_sum[idx] += rev_ptr[rev_uvf.sel->get_index(r)];
                l_pop[idx] = 1;
            }
        } else {
            for (idx_t r = 0; r < count; ++r) {
                const idx_t i_c = cn_uvf.sel->get_index(r);
                const idx_t i_s = sn_uvf.sel->get_index(r);
                const idx_t i_r = rev_uvf.sel->get_index(r);
                if (!cn_ok && !v_cn.RowIsValid(i_c)) continue;
                if (!sn_ok && !v_sn.RowIsValid(i_s)) continue;
                if (!rev_ok && !v_rev.RowIsValid(i_r)) continue;
                const int idx = AggIdx(NationToIdx(cn_ptr[i_c]), NationToIdx(sn_ptr[i_s]), year_off);
                l_sum[idx] += rev_ptr[i_r];
                l_pop[idx] = 1;
            }
        }
        return OperatorResultType::NEED_MORE_INPUT;
    }

    // General path: LO_ORDERDATE is not constant
    UnifiedVectorFormat date_uvf;
    input.data[2].ToUnifiedFormat(count, date_uvf);
    auto *date_ptr = reinterpret_cast<date_t *>(date_uvf.data);
    auto &v_date = date_uvf.validity;
    const bool date_ok = v_date.AllValid();

    if (cn_ok && sn_ok && date_ok && rev_ok) {
        for (idx_t r = 0; r < count; ++r) {
            const idx_t i_d = date_uvf.sel->get_index(r);
            const int32_t year = Date::ExtractYear(date_ptr[i_d]);
            if (year < year_lo || year > year_hi) continue;
            const int idx = AggIdx(NationToIdx(cn_ptr[cn_uvf.sel->get_index(r)]),
                                   NationToIdx(sn_ptr[sn_uvf.sel->get_index(r)]),
                                   year - YEAR_MIN);
            l_sum[idx] += rev_ptr[rev_uvf.sel->get_index(r)];
            l_pop[idx] = 1;
        }
    } else {
        for (idx_t r = 0; r < count; ++r) {
            const idx_t i_c = cn_uvf.sel->get_index(r);
            const idx_t i_s = sn_uvf.sel->get_index(r);
            const idx_t i_d = date_uvf.sel->get_index(r);
            const idx_t i_r = rev_uvf.sel->get_index(r);
            if (!cn_ok   && !v_cn.RowIsValid(i_c))   continue;
            if (!sn_ok   && !v_sn.RowIsValid(i_s))   continue;
            if (!date_ok && !v_date.RowIsValid(i_d))  continue;
            if (!rev_ok  && !v_rev.RowIsValid(i_r))   continue;
            const int32_t year = Date::ExtractYear(date_ptr[i_d]);
            if (year < year_lo || year > year_hi) continue;
            const int idx = AggIdx(NationToIdx(cn_ptr[i_c]), NationToIdx(sn_ptr[i_s]), year - YEAR_MIN);
            l_sum[idx] += rev_ptr[i_r];
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

            struct OutRow { int c; int s; int y; __int128 rev; };
            std::vector<OutRow> rows;
            for (int i = 0; i < AGG_SIZE; ++i) {
                if (!g.populated[i]) continue;
                int c = i / (NUM_NATIONS * NUM_YEARS);
                int rem = i % (NUM_NATIONS * NUM_YEARS);
                int s = rem / NUM_YEARS;
                int y = rem % NUM_YEARS;
                rows.push_back({c, s, y, g.sum_revenue[i]});
            }

            // ORDER BY year ASC, revenue DESC
            std::sort(rows.begin(), rows.end(), [](const OutRow &a, const OutRow &b) {
                if (a.y != b.y) return a.y < b.y;
                return a.rev > b.rev;
            });

            idx_t output_row = 0;
            for (const auto &r : rows) {
                out.SetValue(0, output_row, Value(NATION_NAMES[r.c]));
                out.SetValue(1, output_row, Value(NATION_NAMES[r.s]));
                out.SetValue(2, output_row, Value::BIGINT(static_cast<int64_t>(YEAR_MIN + r.y)));
                out.SetValue(3, output_row, Value::HUGEINT(ToHugeint(r.rev)));
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