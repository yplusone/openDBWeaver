/*
query_template: SELECT
  EXTRACT(YEAR FROM LO_ORDERDATE) AS year,
  P_BRAND,
  SUM(LO_REVENUE) AS sum_revenue
FROM lineorder_flat
WHERE P_CATEGORY = :p_category
  AND S_REGION  = :s_region
GROUP BY year, P_BRAND
ORDER BY year, P_BRAND

split_template: SELECT year, P_BRAND, sum_revenue
FROM dbweaver((
  SELECT LO_ORDERDATE, P_BRAND, LO_REVENUE
  FROM lineorder_flat
  WHERE P_CATEGORY = :p_category
    AND S_REGION = :s_region
)) ORDER BY year, P_BRAND;
query_example: SELECT
  EXTRACT(YEAR FROM LO_ORDERDATE) AS year,
  P_BRAND,
  SUM(LO_REVENUE) AS sum_revenue
FROM lineorder_flat
WHERE P_CATEGORY = 'MFGR#12'
  AND S_REGION  = 'AMERICA'
GROUP BY year, P_BRAND
ORDER BY year, P_BRAND

split_query: SELECT year, P_BRAND, sum_revenue
FROM dbweaver((
  SELECT LO_ORDERDATE, P_BRAND, LO_REVENUE
  FROM lineorder_flat
  WHERE P_CATEGORY = 'MFGR#12'
    AND S_REGION = 'AMERICA'
)) ORDER BY year, P_BRAND;
*/
#define DUCKDB_EXTENSION_MAIN

#include "dbweaver_extension.hpp"
#include "duckdb.hpp"
#include "duckdb/function/table_function.hpp"
#include <atomic>
#include <limits>
#include <mutex>
#include <unordered_map>
#include <absl/container/flat_hash_map.h>
//TODO: Add more includes as needed


namespace duckdb {

//TODO: Define any helper structs or functions needed for binding/execution

struct FnBindData : public FunctionData {
    unique_ptr<FunctionData> Copy() const override { return make_uniq<FnBindData>(); }
    bool Equals(const FunctionData &) const override { return true; }
};

struct GroupKey {
    int32_t year;
    string_t brand;
    
    bool operator==(const GroupKey& other) const {
        return year == other.year && brand == other.brand;
    }
};

struct GroupKeyHash {
    size_t operator()(const GroupKey& k) const {
        size_t h = 0;
        h ^= std::hash<int32_t>{}(k.year) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= duckdb::Hash(k.brand.GetData(), k.brand.GetSize()) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

struct AggState {
    __int128 sum_revenue = 0;
};
struct FnGlobalState : public GlobalTableFunctionState {
    // TODO: Optional accumulators/counters (generator may append to this struct)
    absl::flat_hash_map<GroupKey, AggState, GroupKeyHash> agg_map;
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
    //TODO: initialize local state and other preparations
    absl::flat_hash_map<GroupKey, AggState, GroupKeyHash> agg_map;
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
    result.lower = static_cast<uint64_t>(acc);          // low 64 bits
    result.upper = static_cast<int64_t>(acc >> 64);     // high 64 bits (sign-extended)
    return result;
}

static unique_ptr<FunctionData> FnBind(ClientContext &, TableFunctionBindInput &,
                                            vector<LogicalType> &return_types,
                                            vector<string> &names) {
    //TODO: populate return_types and names
    return_types.push_back(LogicalType::BIGINT);  // year
    return_types.push_back(LogicalType::VARCHAR); // P_BRAND
    return_types.push_back(LogicalType::HUGEINT); // sum_revenue
    
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

    //TODO: process input chunk and produce output
    
    // Declare UnifiedVectorFormat handles for input columns
    UnifiedVectorFormat LO_ORDERDATE_uvf;
    UnifiedVectorFormat P_BRAND_uvf;
    UnifiedVectorFormat LO_REVENUE_uvf;

    // Load columns into UnifiedVectorFormat
    input.data[0].ToUnifiedFormat(input.size(), LO_ORDERDATE_uvf);
    input.data[1].ToUnifiedFormat(input.size(), P_BRAND_uvf);
    input.data[2].ToUnifiedFormat(input.size(), LO_REVENUE_uvf);

    // Create typed pointers to physical data
    date_t* LO_ORDERDATE_ptr = (date_t*)LO_ORDERDATE_uvf.data;
    string_t* P_BRAND_ptr = (string_t*)P_BRAND_uvf.data;
    uint32_t* LO_REVENUE_ptr = (uint32_t*)LO_REVENUE_uvf.data;

    // Validity bitmaps
    auto &valid_LO_ORDERDATE = LO_ORDERDATE_uvf.validity;
    auto &valid_P_BRAND = P_BRAND_uvf.validity;
    auto &valid_LO_REVENUE = LO_REVENUE_uvf.validity;

    const bool LO_ORDERDATE_all_valid = valid_LO_ORDERDATE.AllValid();
    const bool P_BRAND_all_valid = valid_P_BRAND.AllValid();
    const bool LO_REVENUE_all_valid = valid_LO_REVENUE.AllValid();

    // Process rows based on nullability
    if (LO_ORDERDATE_all_valid && P_BRAND_all_valid && LO_REVENUE_all_valid) {
        // Fast path: no per-row NULL checks
        for (idx_t row_idx = 0; row_idx < input.size(); ++row_idx) {
            idx_t i_LO_ORDERDATE = LO_ORDERDATE_uvf.sel->get_index(row_idx);
            idx_t i_P_BRAND = P_BRAND_uvf.sel->get_index(row_idx);
            idx_t i_LO_REVENUE = LO_REVENUE_uvf.sel->get_index(row_idx);

            date_t v_LO_ORDERDATE = LO_ORDERDATE_ptr[i_LO_ORDERDATE];
            string_t v_P_BRAND = P_BRAND_ptr[i_P_BRAND];
            uint32_t v_LO_REVENUE = LO_REVENUE_ptr[i_LO_REVENUE];

            // <<CORE_COMPUTE>>
            // Extract year from date
            int32_t year = Date::ExtractYear(v_LO_ORDERDATE);
            
            GroupKey key;
            key.year = year;
            key.brand = v_P_BRAND;
            
            auto &agg_state = l.agg_map[key];
            agg_state.sum_revenue += v_LO_REVENUE;
        }
    } else {
        // Slow path: at least one column may contain NULLs
        for (idx_t row_idx = 0; row_idx < input.size(); ++row_idx) {
            idx_t i_LO_ORDERDATE = LO_ORDERDATE_uvf.sel->get_index(row_idx);
            idx_t i_P_BRAND = P_BRAND_uvf.sel->get_index(row_idx);
            idx_t i_LO_REVENUE = LO_REVENUE_uvf.sel->get_index(row_idx);

            if (!LO_ORDERDATE_all_valid && !valid_LO_ORDERDATE.RowIsValid(i_LO_ORDERDATE)) {
                continue; // row is NULL in LO_ORDERDATE → skip
            }
            if (!P_BRAND_all_valid && !valid_P_BRAND.RowIsValid(i_P_BRAND)) {
                continue; // row is NULL in P_BRAND → skip
            }
            if (!LO_REVENUE_all_valid && !valid_LO_REVENUE.RowIsValid(i_LO_REVENUE)) {
                continue; // row is NULL in LO_REVENUE → skip
            }

            date_t v_LO_ORDERDATE = LO_ORDERDATE_ptr[i_LO_ORDERDATE];
            string_t v_P_BRAND = P_BRAND_ptr[i_P_BRAND];
            uint32_t v_LO_REVENUE = LO_REVENUE_ptr[i_LO_REVENUE];

            // <<CORE_COMPUTE>>
            // Extract year from date
            int32_t year = Date::ExtractYear(v_LO_ORDERDATE);
            
            GroupKey key;
            key.year = year;
            key.brand = v_P_BRAND;
            
            auto &agg_state = l.agg_map[key];
            agg_state.sum_revenue += v_LO_REVENUE;
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
            //TODO: merge local state with global state
            for (const auto &entry : l.agg_map) {
                const GroupKey &key = entry.first;
                const AggState &state = entry.second;
                g.agg_map[key].sum_revenue += state.sum_revenue;
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
            //TODO: get result from global state
            // Prepare output chunk for flat writing
            const idx_t max_rows = MinValue<idx_t>(STANDARD_VECTOR_SIZE, g.agg_map.size());
            if (max_rows == 0) {
                out.SetCardinality(0);
            } else {
                out.SetCardinality(max_rows);

                // Ensure vectors are flat and obtain direct data pointers
                auto year_data = FlatVector::GetData<int64_t>(out.data[0]);
                auto brand_data = FlatVector::GetData<string_t>(out.data[1]);
                auto revenue_data = FlatVector::GetData<hugeint_t>(out.data[2]);

                // All outputs are non-null
                FlatVector::Validity(out.data[0]).SetAllValid(max_rows);
                FlatVector::Validity(out.data[1]).SetAllValid(max_rows);
                FlatVector::Validity(out.data[2]).SetAllValid(max_rows);

                idx_t output_idx = 0;
                for (const auto &entry : g.agg_map) {
                    if (output_idx >= max_rows) {
                        break; // Output chunk is full
                    }

                    const GroupKey &key = entry.first;
                    const AggState &state = entry.second;

                    // year
                    year_data[output_idx] = static_cast<int64_t>(key.year);

                    // brand (string_t) - assume lifetime managed by DuckDB allocator upstream
                    brand_data[output_idx] = key.brand;

                    // sum_revenue as hugeint_t
                    revenue_data[output_idx] = ToHugeint(state.sum_revenue);

                    output_idx++;
                }

                out.SetCardinality(output_idx);
            }
        }
        //TODO: populate out chunk with final results

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