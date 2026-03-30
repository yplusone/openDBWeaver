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
#include <atomic>
#include <limits>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <algorithm>

namespace duckdb {

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
        h ^= duckdb::Hash<duckdb::string_t>(k.c_nation) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= duckdb::Hash<duckdb::string_t>(k.s_nation) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int32_t>()(k.year) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};



struct AggState {
    __int128 sum_revenue = 0;
    
    AggState() : sum_revenue(0) {}
};
struct SortRow {
    GroupKey key;
    __int128 revenue;
};

struct SortRowComparator {
    bool operator()(const SortRow &a, const SortRow &b) const {
        // First compare by year (ASC)
        if (a.key.year != b.key.year) {
            return a.key.year < b.key.year;
        }
        // Then compare by revenue (DESC)
        return a.revenue > b.revenue;
    }
};

struct SortState {
    std::vector<SortRow> rows;
    bool sorted = false;

    inline void AddRow(const GroupKey &key, __int128 revenue) {
        rows.push_back(SortRow{key, revenue});
    }

    inline void SortNow() {
        if (!sorted) {
            std::sort(rows.begin(), rows.end(), SortRowComparator{});
            sorted = true;
        }
    }
};


struct FnGlobalState : public GlobalTableFunctionState {
    // TODO: Optional accumulators/counters (generator may append to this struct)
    std::mutex lock;
    std::atomic<idx_t> active_local_states {0};
    std::atomic<idx_t> merged_local_states {0};
    std::unordered_map<GroupKey, AggState, GroupKeyHash> agg_map;
    SortState sort_state;
    idx_t MaxThreads() const override {
        return std::numeric_limits<idx_t>::max();
    }
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

    // Access bound parameters
    auto &bind_data = in.bind_data->Cast<FnBindData>();
    int32_t year_lo = bind_data.year_lo;
    int32_t year_hi = bind_data.year_hi;
    
    if (input.size() == 0) {       
        return OperatorResultType::NEED_MORE_INPUT;
    }

    // Process input chunk using UnifiedVectorFormat
    UnifiedVectorFormat C_NATION_uvf;
    input.data[0].ToUnifiedFormat(input.size(), C_NATION_uvf);
    string_t* C_NATION_ptr = (string_t*)C_NATION_uvf.data;

    UnifiedVectorFormat S_NATION_uvf;
    input.data[1].ToUnifiedFormat(input.size(), S_NATION_uvf);
    string_t* S_NATION_ptr = (string_t*)S_NATION_uvf.data;

    UnifiedVectorFormat lo_orderdate_uvf;
    input.data[2].ToUnifiedFormat(input.size(), lo_orderdate_uvf);
    date_t* lo_orderdate_ptr = (date_t*)lo_orderdate_uvf.data;

    UnifiedVectorFormat lo_revenue_uvf;
    input.data[3].ToUnifiedFormat(input.size(), lo_revenue_uvf);
    uint32_t* lo_revenue_ptr = (uint32_t*)lo_revenue_uvf.data;

    // Validity bitmaps
    auto &valid_C_NATION = C_NATION_uvf.validity;
    auto &valid_S_NATION = S_NATION_uvf.validity;
    auto &valid_lo_orderdate = lo_orderdate_uvf.validity;
    auto &valid_lo_revenue = lo_revenue_uvf.validity;

    const bool C_NATION_all_valid = valid_C_NATION.AllValid();
    const bool S_NATION_all_valid = valid_S_NATION.AllValid();
    const bool lo_orderdate_all_valid = valid_lo_orderdate.AllValid();
    const bool lo_revenue_all_valid = valid_lo_revenue.AllValid();

    // Process rows
    if (C_NATION_all_valid && S_NATION_all_valid && lo_orderdate_all_valid && lo_revenue_all_valid) {
        // Fast path: no per-row NULL checks
        for (idx_t row_idx = 0; row_idx < input.size(); ++row_idx) {
            idx_t i_C_NATION = C_NATION_uvf.sel->get_index(row_idx);
            idx_t i_S_NATION = S_NATION_uvf.sel->get_index(row_idx);
            idx_t i_lo_orderdate = lo_orderdate_uvf.sel->get_index(row_idx);
            idx_t i_lo_revenue = lo_revenue_uvf.sel->get_index(row_idx);

            auto v_C_NATION = C_NATION_ptr[i_C_NATION];
            auto v_S_NATION = S_NATION_ptr[i_S_NATION];
            auto v_lo_orderdate = lo_orderdate_ptr[i_lo_orderdate];
            auto v_lo_revenue = lo_revenue_ptr[i_lo_revenue];

            // Apply filter: date_part('year', lo_orderdate) >= year_lo AND date_part('year', lo_orderdate) <= year_hi
            int32_t year = Date::ExtractYear(v_lo_orderdate);
            if (year < year_lo || year > year_hi) {
                continue; // Row does not satisfy filter conditions
            }
            // Create group key
            GroupKey key;
            key.c_nation = v_C_NATION;
            key.s_nation = v_S_NATION;
            key.year = year;
            
            // Update aggregation state
            auto &state = l.agg_map[key];
            state.sum_revenue += v_lo_revenue;

        }
    } else {
        // Slow path: at least one column may contain NULLs
        for (idx_t row_idx = 0; row_idx < input.size(); ++row_idx) {
            idx_t i_C_NATION = C_NATION_uvf.sel->get_index(row_idx);
            idx_t i_S_NATION = S_NATION_uvf.sel->get_index(row_idx);
            idx_t i_lo_orderdate = lo_orderdate_uvf.sel->get_index(row_idx);
            idx_t i_lo_revenue = lo_revenue_uvf.sel->get_index(row_idx);

            if (!C_NATION_all_valid && !valid_C_NATION.RowIsValid(i_C_NATION)) {
                continue; // row is NULL in column C_NATION → skip
            }
            if (!S_NATION_all_valid && !valid_S_NATION.RowIsValid(i_S_NATION)) {
                continue; // row is NULL in column S_NATION → skip
            }
            if (!lo_orderdate_all_valid && !valid_lo_orderdate.RowIsValid(i_lo_orderdate)) {
                continue; // row is NULL in column lo_orderdate → skip
            }
            if (!lo_revenue_all_valid && !valid_lo_revenue.RowIsValid(i_lo_revenue)) {
                continue; // row is NULL in column lo_revenue → skip
            }

            auto v_C_NATION = C_NATION_ptr[i_C_NATION];
            auto v_S_NATION = S_NATION_ptr[i_S_NATION];
            auto v_lo_orderdate = lo_orderdate_ptr[i_lo_orderdate];
            auto v_lo_revenue = lo_revenue_ptr[i_lo_revenue];

            // Apply filter: date_part('year', lo_orderdate) >= year_lo AND date_part('year', lo_orderdate) <= year_hi
            int32_t year = Date::ExtractYear(v_lo_orderdate);
            if (year < year_lo || year > year_hi) {
                continue; // Row does not satisfy filter conditions
            }
            // Create group key
            GroupKey key;
            key.c_nation = v_C_NATION;
            key.s_nation = v_S_NATION;
            key.year = year;
            
            // Update aggregation state
            auto &state = l.agg_map[key];
            state.sum_revenue += v_lo_revenue;

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
    int32_t year_lo = bind_data.year_lo;
    int32_t year_hi = bind_data.year_hi;
    
    // Merge the local state into the global state exactly once.
    if (!l.merged) {
        {
            std::lock_guard<std::mutex> guard(g.lock);
            // Merge local aggregation map into global map
            for (const auto &entry : l.agg_map) {
                const GroupKey &key = entry.first;
                const AggState &local_state = entry.second;
                
                auto &global_state = g.agg_map[key];
                global_state.sum_revenue += local_state.sum_revenue;
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
            // For each grouped entry in the global map, add to sort state with full grouping (including year)
            for (const auto &entry : g.agg_map) {
                const GroupKey &key = entry.first;
                const AggState &state = entry.second;
                g.sort_state.AddRow(key, state.sum_revenue);
            }

            
            // Perform the sort
            g.sort_state.SortNow();
            idx_t output_row = 0;
            idx_t total_rows = g.sort_state.rows.size();
            
            if (total_rows > 0) {
                out.SetCardinality(total_rows);
                for (size_t i = 0; i < g.sort_state.rows.size(); ++i) {
                    const SortRow &row = g.sort_state.rows[i];
                    // Write C_NATION
                    out.SetValue(0, output_row, row.key.c_nation);
                    // Write S_NATION
                    out.SetValue(1, output_row, row.key.s_nation);
                    // Write year (BIGINT)
                    out.SetValue(2, output_row, Value::BIGINT(row.key.year));
                    // Write revenue (HUGEINT)
                    hugeint_t revenue_val = ToHugeint(row.revenue);
                    out.SetValue(3, output_row, Value::HUGEINT(revenue_val));
                    output_row++;
                }
            } else {
                out.SetCardinality(0);
            }

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