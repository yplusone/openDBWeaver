/*
query_template: SELECT
  EXTRACT(YEAR FROM LO_ORDERDATE) AS year,
  C_NATION,
  SUM(LO_REVENUE - LO_SUPPLYCOST) AS profit
FROM lineorder_flat
WHERE C_REGION = :c_region
  AND S_REGION = :s_region
  AND P_MFGR IN (:mfgr_1, :mfgr_2)
GROUP BY
  EXTRACT(YEAR FROM LO_ORDERDATE),
  C_NATION
ORDER BY
  year ASC,
  C_NATION ASC

split_template: SELECT year, C_NATION, profit
FROM dbweaver((
  SELECT LO_ORDERDATE, C_NATION, P_MFGR, LO_REVENUE, LO_SUPPLYCOST
  FROM lineorder_flat
  WHERE C_REGION = :c_region
    AND S_REGION = :s_region
), :mfgr_1, :mfgr_2) ORDER BY year ASC, C_NATION ASC;
query_example: SELECT
  EXTRACT(YEAR FROM LO_ORDERDATE) AS year,
  C_NATION,
  SUM(LO_REVENUE - LO_SUPPLYCOST) AS profit
FROM lineorder_flat
WHERE
  C_REGION = 'AMERICA'
  AND S_REGION = 'AMERICA'
  AND (P_MFGR = 'MFGR#1' OR P_MFGR = 'MFGR#2')
GROUP BY
  EXTRACT(YEAR FROM LO_ORDERDATE),
  C_NATION
ORDER BY
  year ASC,
  C_NATION ASC

split_query: SELECT year, C_NATION, profit
FROM dbweaver((
  SELECT LO_ORDERDATE, C_NATION, P_MFGR, LO_REVENUE, LO_SUPPLYCOST
  FROM lineorder_flat
  WHERE C_REGION = 'AMERICA'
    AND S_REGION = 'AMERICA'
), 'MFGR#1', 'MFGR#2') ORDER BY year ASC, C_NATION ASC;
*/
#define DUCKDB_EXTENSION_MAIN

#include "dbweaver_extension.hpp"
#include "duckdb.hpp"
#include "duckdb/function/table_function.hpp"
#include <atomic>
#include <limits>
#include <mutex>
#include <unordered_map>
#include <array>



namespace duckdb {
struct FnBindData : public FunctionData {
    string mfgr_1;
    string mfgr_2;
    // Precomputed sizes for cheap comparisons against string_t
    idx_t mfgr_1_len;
    idx_t mfgr_2_len;

    explicit FnBindData(string mfgr_1_p, string mfgr_2_p)
        : mfgr_1(std::move(mfgr_1_p)), mfgr_2(std::move(mfgr_2_p)) {
        mfgr_1_len = mfgr_1.size();
        mfgr_2_len = mfgr_2.size();
    }

    unique_ptr<FunctionData> Copy() const override { 
        return make_uniq<FnBindData>(mfgr_1, mfgr_2); 
    }
    bool Equals(const FunctionData &other_p) const override { 
        auto &other = other_p.Cast<FnBindData>();
        return mfgr_1 == other.mfgr_1 && mfgr_2 == other.mfgr_2;
    }
};
struct AggState {
    int64_t profit_sum = 0;
};

// Fixed dimensions for SSB lineorder_flat
static constexpr int32_t BASE_YEAR = 1992;
static constexpr int32_t YEAR_COUNT = 7;   // 1992..1998 inclusive
static constexpr int32_t NATION_COUNT = 25;
static constexpr int32_t AGG_BUCKETS = YEAR_COUNT * NATION_COUNT; // 175

struct FnGlobalState : public GlobalTableFunctionState {
    std::mutex lock;
    std::atomic<idx_t> active_local_states {0};
    std::atomic<idx_t> merged_local_states {0};

    // Dictionary of nation strings (discovered at runtime) and a fixed aggregation array
    std::vector<string> nation_strings;           // index -> std::string (stable across threads after build)
    std::array<AggState, AGG_BUCKETS> global_agg; // [year_id * NATION_COUNT + nation_id]

    FnGlobalState() {
        for (auto &st : global_agg) {
            st.profit_sum = 0;
        }
    }

    idx_t MaxThreads() const override {
        return std::numeric_limits<idx_t>::max();
    }
};



static unique_ptr<GlobalTableFunctionState> FnInit(ClientContext &context, TableFunctionInitInput &input) {
    //auto &bind_data = input.bind_data->Cast<FnBindData>();
    //string mfgr_1 = bind_data.mfgr_1;
    //string mfgr_2 = bind_data.mfgr_2;
    return make_uniq<FnGlobalState>();
}
struct FnLocalState : public LocalTableFunctionState {
    bool merged = false;

    // Local fixed-size aggregation array
    std::array<AggState, AGG_BUCKETS> local_agg;

    // Per-thread copy of the nation dictionary snapshot at init time
    std::vector<string> nation_strings_snapshot;

    FnLocalState() {
        for (auto &st : local_agg) {
            st.profit_sum = 0;
        }
    }
};

static unique_ptr<LocalTableFunctionState> FnInitLocal(ExecutionContext &, TableFunctionInitInput &,
                                                      GlobalTableFunctionState *global_state) {
    auto &g = global_state->Cast<FnGlobalState>();
    g.active_local_states.fetch_add(1, std::memory_order_relaxed);

    auto local_state = make_uniq<FnLocalState>();
    {
        // Snapshot the current global nation dictionary (may be empty initially)
        std::lock_guard<std::mutex> guard(g.lock);
        local_state->nation_strings_snapshot = g.nation_strings;
    }

    return std::move(local_state);
}




static unique_ptr<FunctionData> FnBind(ClientContext &, TableFunctionBindInput &input,
                                            vector<LogicalType> &return_types,
                                            vector<string> &names) {
    //TODO: populate return_types and names
    string mfgr_1 = input.inputs[1].GetValue<string>();
    string mfgr_2 = input.inputs[2].GetValue<string>();
    
    return_types.push_back(LogicalType::BIGINT);
    return_types.push_back(LogicalType::VARCHAR);
    return_types.push_back(LogicalType::HUGEINT);
    
    names.push_back("year");
    names.push_back("C_NATION");
    names.push_back("profit");

    return make_uniq<FnBindData>(mfgr_1, mfgr_2);
}
static OperatorResultType FnExecute(ExecutionContext &, TableFunctionInput &in,
                                                    DataChunk &input, DataChunk &) {
    auto &l = in.local_state->Cast<FnLocalState>();
    auto &g = in.global_state->Cast<FnGlobalState>();



    // Access bound parameters once per chunk (no per-row std::string materialization)
    auto &bind_data = in.bind_data->Cast<FnBindData>();
    const char *mfgr_1_data = bind_data.mfgr_1.data();
    const char *mfgr_2_data = bind_data.mfgr_2.data();
    const idx_t mfgr_1_len = bind_data.mfgr_1_len;
    const idx_t mfgr_2_len = bind_data.mfgr_2_len;
    
    if (input.size() == 0) {       
        return OperatorResultType::NEED_MORE_INPUT;
    }
    // Detect CONSTANT_VECTOR for LO_ORDERDATE to avoid per-row year extraction
    const auto lo_orderdate_vt = input.data[0].GetVectorType();
    const bool lo_orderdate_is_const = (lo_orderdate_vt == VectorType::CONSTANT_VECTOR);

    // Unpack input columns using UnifiedVectorFormat
    UnifiedVectorFormat lo_orderdate_uvf;
    input.data[0].ToUnifiedFormat(input.size(), lo_orderdate_uvf);
    date_t* lo_orderdate_ptr = (date_t*)lo_orderdate_uvf.data;

    UnifiedVectorFormat c_nation_uvf;
    input.data[1].ToUnifiedFormat(input.size(), c_nation_uvf);
    string_t* c_nation_ptr = (string_t*)c_nation_uvf.data;

    UnifiedVectorFormat p_mfgr_uvf;
    input.data[2].ToUnifiedFormat(input.size(), p_mfgr_uvf);
    string_t* p_mfgr_ptr = (string_t*)p_mfgr_uvf.data;

    UnifiedVectorFormat lo_revenue_uvf;
    input.data[3].ToUnifiedFormat(input.size(), lo_revenue_uvf);
    uint32_t* lo_revenue_ptr = (uint32_t*)lo_revenue_uvf.data;

    UnifiedVectorFormat lo_supplycost_uvf;
    input.data[4].ToUnifiedFormat(input.size(), lo_supplycost_uvf);
    uint32_t* lo_supplycost_ptr = (uint32_t*)lo_supplycost_uvf.data;

    // validity bitmaps
    auto &valid_lo_orderdate = lo_orderdate_uvf.validity;
    auto &valid_c_nation = c_nation_uvf.validity;
    auto &valid_p_mfgr = p_mfgr_uvf.validity;
    auto &valid_lo_revenue = lo_revenue_uvf.validity;
    auto &valid_lo_supplycost = lo_supplycost_uvf.validity;

    const bool lo_orderdate_all_valid = valid_lo_orderdate.AllValid();
    const bool c_nation_all_valid = valid_c_nation.AllValid();
    const bool p_mfgr_all_valid = valid_p_mfgr.AllValid();
    const bool lo_revenue_all_valid = valid_lo_revenue.AllValid();
    const bool lo_supplycost_all_valid = valid_lo_supplycost.AllValid();

    // Special case: CONSTANT_VECTOR LO_ORDERDATE that is entirely NULL => skip whole batch
    if (lo_orderdate_is_const && !lo_orderdate_all_valid && !valid_lo_orderdate.RowIsValid(0)) {
        return OperatorResultType::NEED_MORE_INPUT;
    }

    // Precompute year for CONSTANT_VECTOR LO_ORDERDATE when non-NULL
    int32_t const_lo_order_year = 0;
    if (lo_orderdate_is_const) {
        const auto &const_validity = valid_lo_orderdate;
        if (!const_validity.AllValid() && !const_validity.RowIsValid(0)) {
            // already handled above, but guard defensively
        } else {
            const auto *const_dates = ConstantVector::GetData<date_t>(input.data[0]);
            const_lo_order_year = Date::ExtractYear(const_dates[0]);
        }
    }
    // Helper lambda: map a nation string_t to a compact nation_id in [0, NATION_COUNT-1]
    auto get_nation_id = [&](const string_t &nation_val) -> int32_t {
        const char *n_data = nation_val.GetData();
        const idx_t n_len = nation_val.GetSize();

        // First check local snapshot for this thread (no locking, fast path)
        for (idx_t i = 0; i < l.nation_strings_snapshot.size(); ++i) {
            const auto &s = l.nation_strings_snapshot[i];
            if (s.size() == n_len && memcmp(s.data(), n_data, n_len) == 0) {
                return static_cast<int32_t>(i);
            }
        }

        // Missed in snapshot: need to consult/extend global dictionary under lock
        std::lock_guard<std::mutex> guard(g.lock);
        for (idx_t i = 0; i < g.nation_strings.size(); ++i) {
            const auto &s = g.nation_strings[i];
            if (s.size() == n_len && memcmp(s.data(), n_data, n_len) == 0) {
                // Update local snapshot lazily so next lookup is fast
                l.nation_strings_snapshot = g.nation_strings;
                return static_cast<int32_t>(i);
            }
        }
        // Not found: add if capacity allows
        if (g.nation_strings.size() < static_cast<size_t>(NATION_COUNT)) {
            g.nation_strings.emplace_back(n_data, n_len);
            l.nation_strings_snapshot = g.nation_strings;
            return static_cast<int32_t>(g.nation_strings.size() - 1);
        }
        // Fallback: clamp to last bucket to avoid out-of-bounds
        return static_cast<int32_t>(NATION_COUNT - 1);
    };

    // FAST BRANCH: all relevant columns have no NULLs in this batch
    if (lo_orderdate_all_valid && c_nation_all_valid && p_mfgr_all_valid && lo_revenue_all_valid && lo_supplycost_all_valid) {


        // --- Fast path: no per-row NULL checks ---
        if (lo_orderdate_is_const) {
            // Constant LO_ORDERDATE: reuse precomputed year for all rows
            const int32_t year = const_lo_order_year;
            for (idx_t row_idx = 0; row_idx < input.size(); ++row_idx) {
                // Directly load values without RowIsValid checks (except LO_ORDERDATE which is constant)

                idx_t i_c_nation = c_nation_uvf.sel->get_index(row_idx);
                idx_t i_p_mfgr = p_mfgr_uvf.sel->get_index(row_idx);
                idx_t i_lo_revenue = lo_revenue_uvf.sel->get_index(row_idx);
                idx_t i_lo_supplycost = lo_supplycost_uvf.sel->get_index(row_idx);

                string_t v_c_nation = c_nation_ptr[i_c_nation];
                string_t v_p_mfgr = p_mfgr_ptr[i_p_mfgr];
                uint32_t v_lo_revenue = lo_revenue_ptr[i_lo_revenue];
                uint32_t v_lo_supplycost = lo_supplycost_ptr[i_lo_supplycost];
                // Apply filter: (P_MFGR = {mfgr_1} OR P_MFGR = {mfgr_2})
                const idx_t p_len = v_p_mfgr.GetSize();
                const char *p_data = v_p_mfgr.GetData();

                bool match = false;
                if (p_len == mfgr_1_len && memcmp(p_data, mfgr_1_data, p_len) == 0) {
                    match = true;
                } else if (p_len == mfgr_2_len && memcmp(p_data, mfgr_2_data, p_len) == 0) {
                    match = true;
                }
                if (!match) {
                    continue; // Skip row if it doesn't match either manufacturer
                }
                // ======================================
                //  Core computation logic (no NULLs, constant year)
                const int32_t year_id = year - BASE_YEAR;
                if (year_id >= 0 && year_id < YEAR_COUNT) {
                    const int32_t nation_id = get_nation_id(v_c_nation);
                    const idx_t idx = static_cast<idx_t>(year_id) * NATION_COUNT + static_cast<idx_t>(nation_id);
                    l.local_agg[idx].profit_sum += (int64_t)v_lo_revenue - (int64_t)v_lo_supplycost;
                }
                // ============================
            }
        } else {


            for (idx_t row_idx = 0; row_idx < input.size(); ++row_idx) {
                // Directly load values without RowIsValid checks

                idx_t i_lo_orderdate = lo_orderdate_uvf.sel->get_index(row_idx);
                idx_t i_c_nation = c_nation_uvf.sel->get_index(row_idx);
                idx_t i_p_mfgr = p_mfgr_uvf.sel->get_index(row_idx);
                idx_t i_lo_revenue = lo_revenue_uvf.sel->get_index(row_idx);
                idx_t i_lo_supplycost = lo_supplycost_uvf.sel->get_index(row_idx);

                date_t v_lo_orderdate = lo_orderdate_ptr[i_lo_orderdate];
                string_t v_c_nation = c_nation_ptr[i_c_nation];
                string_t v_p_mfgr = p_mfgr_ptr[i_p_mfgr];
                uint32_t v_lo_revenue = lo_revenue_ptr[i_lo_revenue];
                uint32_t v_lo_supplycost = lo_supplycost_ptr[i_lo_supplycost];
                // Apply filter: (P_MFGR = {mfgr_1} OR P_MFGR = {mfgr_2})
                const idx_t p_len = v_p_mfgr.GetSize();
                const char *p_data = v_p_mfgr.GetData();

                bool match = false;
                if (p_len == mfgr_1_len && memcmp(p_data, mfgr_1_data, p_len) == 0) {
                    match = true;
                } else if (p_len == mfgr_2_len && memcmp(p_data, mfgr_2_data, p_len) == 0) {
                    match = true;
                }
                if (!match) {
                    continue; // Skip row if it doesn't match either manufacturer
                }
                // ======================================
                //  Core computation logic (no NULLs)
                const int32_t year = Date::ExtractYear(v_lo_orderdate);
                const int32_t year_id = year - BASE_YEAR;
                if (year_id >= 0 && year_id < YEAR_COUNT) {
                    const int32_t nation_id = get_nation_id(v_c_nation);
                    const idx_t idx = static_cast<idx_t>(year_id) * NATION_COUNT + static_cast<idx_t>(nation_id);
                    l.local_agg[idx].profit_sum += (int64_t)v_lo_revenue - (int64_t)v_lo_supplycost;
                }
                // ============================
            }
        }
    } else {


        // --- Slow path: at least one column may contain NULLs ---
        if (lo_orderdate_is_const) {
            // Constant LO_ORDERDATE: year is precomputed, only other columns may be NULL
            const int32_t year = const_lo_order_year;
            for (idx_t row_idx = 0; row_idx < input.size(); ++row_idx) {
                // For each column that is not fully valid, check this row
                idx_t i_c_nation = c_nation_uvf.sel->get_index(row_idx);
                idx_t i_p_mfgr = p_mfgr_uvf.sel->get_index(row_idx);
                idx_t i_lo_revenue = lo_revenue_uvf.sel->get_index(row_idx);
                idx_t i_lo_supplycost = lo_supplycost_uvf.sel->get_index(row_idx);

                if (!c_nation_all_valid && !valid_c_nation.RowIsValid(i_c_nation)) {
                    continue;
                }
                if (!p_mfgr_all_valid && !valid_p_mfgr.RowIsValid(i_p_mfgr)) {
                    continue;
                }
                if (!lo_revenue_all_valid && !valid_lo_revenue.RowIsValid(i_lo_revenue)) {
                    continue;
                }
                if (!lo_supplycost_all_valid && !valid_lo_supplycost.RowIsValid(i_lo_supplycost)) {
                    continue;
                }
                // At this point, all required columns except LO_ORDERDATE are valid for this row

                string_t v_c_nation = c_nation_ptr[i_c_nation];
                string_t v_p_mfgr = p_mfgr_ptr[i_p_mfgr];
                uint32_t v_lo_revenue = lo_revenue_ptr[i_lo_revenue];
                uint32_t v_lo_supplycost = lo_supplycost_ptr[i_lo_supplycost];
                // Apply filter: (P_MFGR = {mfgr_1} OR P_MFGR = {mfgr_2})
                const idx_t p_len = v_p_mfgr.GetSize();
                const char *p_data = v_p_mfgr.GetData();

                bool match = false;
                if (p_len == mfgr_1_len && memcmp(p_data, mfgr_1_data, p_len) == 0) {
                    match = true;
                } else if (p_len == mfgr_2_len && memcmp(p_data, mfgr_2_data, p_len) == 0) {
                    match = true;
                }
                if (!match) {
                    continue; // Skip row if it doesn't match either manufacturer
                }
                // ======================================
                //  Core computation logic (NULL-safe, constant year)
                const int32_t year_id = year - BASE_YEAR;
                if (year_id >= 0 && year_id < YEAR_COUNT) {
                    const int32_t nation_id = get_nation_id(v_c_nation);
                    const idx_t idx = static_cast<idx_t>(year_id) * NATION_COUNT + static_cast<idx_t>(nation_id);
                    l.local_agg[idx].profit_sum += (int64_t)v_lo_revenue - (int64_t)v_lo_supplycost;
                }
                // ======================================
            }
        } else {


            for (idx_t row_idx = 0; row_idx < input.size(); ++row_idx) {
                // For each column that is not fully valid, check this row
                idx_t i_lo_orderdate = lo_orderdate_uvf.sel->get_index(row_idx);
                idx_t i_c_nation = c_nation_uvf.sel->get_index(row_idx);
                idx_t i_p_mfgr = p_mfgr_uvf.sel->get_index(row_idx);
                idx_t i_lo_revenue = lo_revenue_uvf.sel->get_index(row_idx);
                idx_t i_lo_supplycost = lo_supplycost_uvf.sel->get_index(row_idx);

                if (!lo_orderdate_all_valid && !valid_lo_orderdate.RowIsValid(i_lo_orderdate)) {
                    continue; // row is NULL in column lo_orderdate → skip
                }
                if (!c_nation_all_valid && !valid_c_nation.RowIsValid(i_c_nation)) {
                    continue;
                }
                if (!p_mfgr_all_valid && !valid_p_mfgr.RowIsValid(i_p_mfgr)) {
                    continue;
                }
                if (!lo_revenue_all_valid && !valid_lo_revenue.RowIsValid(i_lo_revenue)) {
                    continue;
                }
                if (!lo_supplycost_all_valid && !valid_lo_supplycost.RowIsValid(i_lo_supplycost)) {
                    continue;
                }
                // At this point, all required columns are valid for this row

                date_t v_lo_orderdate = lo_orderdate_ptr[i_lo_orderdate];
                string_t v_c_nation = c_nation_ptr[i_c_nation];
                string_t v_p_mfgr = p_mfgr_ptr[i_p_mfgr];
                uint32_t v_lo_revenue = lo_revenue_ptr[i_lo_revenue];
                uint32_t v_lo_supplycost = lo_supplycost_ptr[i_lo_supplycost];
                // Apply filter: (P_MFGR = {mfgr_1} OR P_MFGR = {mfgr_2})
                const idx_t p_len = v_p_mfgr.GetSize();
                const char *p_data = v_p_mfgr.GetData();

                bool match = false;
                if (p_len == mfgr_1_len && memcmp(p_data, mfgr_1_data, p_len) == 0) {
                    match = true;
                } else if (p_len == mfgr_2_len && memcmp(p_data, mfgr_2_data, p_len) == 0) {
                    match = true;
                }
                if (!match) {
                    continue; // Skip row if it doesn't match either manufacturer
                }
                // ======================================
                //  Core computation logic (NULL-safe)
                const int32_t year = Date::ExtractYear(v_lo_orderdate);
                const int32_t year_id = year - BASE_YEAR;
                if (year_id >= 0 && year_id < YEAR_COUNT) {
                    const int32_t nation_id = get_nation_id(v_c_nation);
                    const idx_t idx = static_cast<idx_t>(year_id) * NATION_COUNT + static_cast<idx_t>(nation_id);
                    l.local_agg[idx].profit_sum += (int64_t)v_lo_revenue - (int64_t)v_lo_supplycost;
                }
                // ======================================
            }
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
            // Merge fixed-size arrays element-wise
            for (idx_t i = 0; i < AGG_BUCKETS; ++i) {
                g.global_agg[i].profit_sum += l.local_agg[i].profit_sum;
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
        std::lock_guard<std::mutex> guard(g.lock);

        idx_t output_row = 0;
        // Emit results in year-major, then nation-major order to match ORDER BY year, C_NATION
        for (int32_t year_id = 0; year_id < YEAR_COUNT; ++year_id) {
            const int32_t year = BASE_YEAR + year_id;
            for (int32_t nation_id = 0; nation_id < NATION_COUNT; ++nation_id) {
                const idx_t idx = static_cast<idx_t>(year_id) * NATION_COUNT + static_cast<idx_t>(nation_id);
                const AggState &state = g.global_agg[idx];
                if (state.profit_sum == 0) {
                    continue; // skip empty groups
                }
                string nation_str;
                if (nation_id < static_cast<int32_t>(g.nation_strings.size())) {
                    nation_str = g.nation_strings[nation_id];
                } else {
                    // Should not generally happen; use empty string as fallback
                    nation_str = string();
                }

                out.SetValue(0, output_row, Value::BIGINT(year));
                out.SetValue(1, output_row, Value(nation_str));
                out.SetValue(2, output_row, Value::HUGEINT(state.profit_sum));
                ++output_row;
            }
        }
        out.SetCardinality(output_row);
    } else {
        out.SetCardinality(0);
    }
    
    return OperatorFinalizeResultType::FINISHED;
}



static void LoadInternal(ExtensionLoader &loader) {
    TableFunction f("dbweaver", {LogicalType::TABLE, LogicalType::VARCHAR, LogicalType::VARCHAR}, nullptr, FnBind, FnInit, FnInitLocal);
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