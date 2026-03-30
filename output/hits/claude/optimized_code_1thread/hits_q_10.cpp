#define DUCKDB_EXTENSION_MAIN

#include "dbweaver_extension.hpp"
#include "duckdb.hpp"
#include "duckdb/function/table_function.hpp"
#include <atomic>
#include <limits>
#include <mutex>
#include <vector>
#include <algorithm>
#include <absl/container/flat_hash_set.h>

#include <cstdint>

namespace duckdb {

struct FnBindData : public FunctionData {
    unique_ptr<FunctionData> Copy() const override { return make_uniq<FnBindData>(); }
    bool Equals(const FunctionData &) const override {
        return true;
    }
};

struct GroupKey {
    int32_t region_id;
};

struct ScalarAgg {
    int64_t sum_advengineid = 0;
    int64_t count_total = 0;
    int64_t sum_resolutionwidth = 0;
};
struct DistinctAgg {
    absl::flat_hash_set<int64_t> distinct_userids;
};


struct SortRow {
    GroupKey key;
    ScalarAgg scalar;
    int64_t distinct_count;
};

struct SortRowComparator {
    bool operator()(const SortRow &a, const SortRow &b) const {
        // DESC order for count_total ('c')
        if (a.scalar.count_total != b.scalar.count_total) {
            return a.scalar.count_total > b.scalar.count_total;
        }
        return false;
    }
};

struct SortState {
    std::vector<SortRow> rows;
    bool sorted = false;

    inline void AddRow(const GroupKey &key, const ScalarAgg &scalar, int64_t distinct_count) {
        rows.push_back(SortRow{key, scalar, distinct_count});
    }

    inline void SortNow() {
        if (!sorted) {
            std::sort(rows.begin(), rows.end(), SortRowComparator{});
            sorted = true;
        }
    }
};

struct FnGlobalState : public GlobalTableFunctionState {
    std::vector<uint32_t> region_to_id;
    std::vector<GroupKey> keys;
    std::vector<ScalarAgg> scalar_aggs;
    std::vector<DistinctAgg> distinct_aggs;
    std::mutex lock;
    std::atomic<idx_t> active_local_states {0};
    std::atomic<idx_t> merged_local_states {0};
    SortState sort_state;
    idx_t MaxThreads() const override {
        return std::numeric_limits<idx_t>::max();
    }
};
static unique_ptr<GlobalTableFunctionState> FnInit(ClientContext &, TableFunctionInitInput &) {
    auto gs = make_uniq<FnGlobalState>();
    gs->region_to_id.assign(150000, UINT32_MAX);
    return std::move(gs);
}


struct FnLocalState : public LocalTableFunctionState {
    std::vector<uint32_t> region_to_id;
    std::vector<GroupKey> keys;
    std::vector<ScalarAgg> scalar_aggs;
    std::vector<DistinctAgg> distinct_aggs;
    std::vector<uint32_t> row_ids;
    bool merged = false;
};
static unique_ptr<LocalTableFunctionState> FnInitLocal(ExecutionContext &, TableFunctionInitInput &,
                                                      GlobalTableFunctionState *global_state) {
    auto &g = global_state->Cast<FnGlobalState>();
    g.active_local_states.fetch_add(1, std::memory_order_relaxed);
    auto ls = make_uniq<FnLocalState>();
    ls->region_to_id.assign(150000, UINT32_MAX);
    return std::move(ls);
}


static unique_ptr<FunctionData> FnBind(ClientContext &, TableFunctionBindInput &,
                                      vector<LogicalType> &return_types,
                                      vector<string> &names) {
    return_types.emplace_back(LogicalType::INTEGER);  // RegionID
    return_types.emplace_back(LogicalType::HUGEINT); // sum_advengineid
    return_types.emplace_back(LogicalType::BIGINT);  // c
    return_types.emplace_back(LogicalType::DOUBLE);  // avg_resolutionwidth
    return_types.emplace_back(LogicalType::BIGINT);  // cnt_distinct_userid

    names.emplace_back("RegionID");
    names.emplace_back("sum_advengineid");
    names.emplace_back("c");
    names.emplace_back("avg_resolutionwidth");
    names.emplace_back("cnt_distinct_userid");

    return make_uniq<FnBindData>();
}

static OperatorResultType FnExecute(ExecutionContext &, TableFunctionInput &in,
                                    DataChunk &input, DataChunk &) {
    auto &l = in.local_state->Cast<FnLocalState>();

    if (input.size() == 0) {
        return OperatorResultType::NEED_MORE_INPUT;
    }

    UnifiedVectorFormat region_id_uvf;
    input.data[0].ToUnifiedFormat(input.size(), region_id_uvf);
    int32_t *region_id_ptr = (int32_t *)region_id_uvf.data;

    UnifiedVectorFormat adv_engine_id_uvf;
    input.data[1].ToUnifiedFormat(input.size(), adv_engine_id_uvf);
    int16_t *adv_engine_id_ptr = (int16_t *)adv_engine_id_uvf.data;

    UnifiedVectorFormat resolution_width_uvf;
    input.data[2].ToUnifiedFormat(input.size(), resolution_width_uvf);
    int16_t *resolution_width_ptr = (int16_t *)resolution_width_uvf.data;

    UnifiedVectorFormat user_id_uvf;
    input.data[3].ToUnifiedFormat(input.size(), user_id_uvf);
    int64_t *user_id_ptr = (int64_t *)user_id_uvf.data;

    auto &valid_region_id = region_id_uvf.validity;
    auto &valid_adv_engine_id = adv_engine_id_uvf.validity;
    auto &valid_resolution_width = resolution_width_uvf.validity;
    auto &valid_user_id = user_id_uvf.validity;
    const bool region_id_all_valid = valid_region_id.AllValid();
    const bool adv_engine_id_all_valid = valid_adv_engine_id.AllValid();
    const bool resolution_width_all_valid = valid_resolution_width.AllValid();
    const bool user_id_all_valid = valid_user_id.AllValid();

    const idx_t input_size = input.size();

    if (region_id_all_valid && adv_engine_id_all_valid && resolution_width_all_valid && user_id_all_valid) {
        l.row_ids.resize(input_size);
        for (idx_t row_idx = 0; row_idx < input_size; ++row_idx) {
            int32_t v1 = region_id_ptr[region_id_uvf.sel->get_index(row_idx)];
            if (v1 < 0) {
                l.row_ids[row_idx] = UINT32_MAX;
                continue;
            }
            if ((size_t)v1 >= l.region_to_id.size()) {
                l.region_to_id.resize(v1 + 4096, UINT32_MAX);
            }
            uint32_t id = l.region_to_id[v1];
            if (id == UINT32_MAX) {
                id = (uint32_t)l.scalar_aggs.size();
                l.region_to_id[v1] = id;
                l.scalar_aggs.emplace_back();
                l.distinct_aggs.emplace_back();
                l.keys.push_back(GroupKey{v1});
            }
            l.row_ids[row_idx] = id;
        }

        // Tight scalar loop for cache efficiency
        for (idx_t row_idx = 0; row_idx < input_size; ++row_idx) {
            uint32_t id = l.row_ids[row_idx];
            if (id == UINT32_MAX) continue;
            int16_t v2 = adv_engine_id_ptr[adv_engine_id_uvf.sel->get_index(row_idx)];
            int16_t v3 = resolution_width_ptr[resolution_width_uvf.sel->get_index(row_idx)];
            auto &s = l.scalar_aggs[id];
            s.sum_advengineid += v2;
            s.count_total++;
            s.sum_resolutionwidth += v3;
        }

        // Separate distinct counting loop to avoid polluting the scalar cache line with heap pointers
        for (idx_t row_idx = 0; row_idx < input_size; ++row_idx) {
            uint32_t id = l.row_ids[row_idx];
            if (id == UINT32_MAX) continue;
            int64_t v4 = user_id_ptr[user_id_uvf.sel->get_index(row_idx)];
            l.distinct_aggs[id].distinct_userids.insert(v4);
        }
    } else {
        for (idx_t row_idx = 0; row_idx < input_size; ++row_idx) {
            idx_t i_region_id = region_id_uvf.sel->get_index(row_idx);
            idx_t i_adv_engine_id = adv_engine_id_uvf.sel->get_index(row_idx);
            idx_t i_resolution_width = resolution_width_uvf.sel->get_index(row_idx);
            idx_t i_user_id = user_id_uvf.sel->get_index(row_idx);

            if ((!region_id_all_valid && !valid_region_id.RowIsValid(i_region_id)) ||
                (!adv_engine_id_all_valid && !valid_adv_engine_id.RowIsValid(i_adv_engine_id)) ||
                (!resolution_width_all_valid && !valid_resolution_width.RowIsValid(i_resolution_width)) ||
                (!user_id_all_valid && !valid_user_id.RowIsValid(i_user_id))) {
                continue;
            }

            int32_t v1 = region_id_ptr[i_region_id];
            if (v1 < 0) continue;
            int16_t v2 = adv_engine_id_ptr[i_adv_engine_id];
            int16_t v3 = resolution_width_ptr[i_resolution_width];
            int64_t v4 = user_id_ptr[i_user_id];
            
            if ((size_t)v1 >= l.region_to_id.size()) {
                l.region_to_id.resize(v1 + 4096, UINT32_MAX);
            }
            uint32_t id = l.region_to_id[v1];
            if (id == UINT32_MAX) {
                id = (uint32_t)l.scalar_aggs.size();
                l.region_to_id[v1] = id;
                l.scalar_aggs.emplace_back();
                l.distinct_aggs.emplace_back();
                l.keys.push_back(GroupKey{v1});
            }

            auto &s = l.scalar_aggs[id];
            s.sum_advengineid += v2;
            s.count_total++;
            s.sum_resolutionwidth += v3;
            l.distinct_aggs[id].distinct_userids.insert(v4);
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
            for (size_t i = 0; i < l.keys.size(); ++i) {
                const GroupKey &key = l.keys[i];
                int32_t v1 = key.region_id;
                if ((size_t)v1 >= g.region_to_id.size()) {
                    g.region_to_id.resize(v1 + 4096, UINT32_MAX);
                }
                uint32_t gid = g.region_to_id[v1];
                if (gid == UINT32_MAX) {
                    gid = (uint32_t)g.scalar_aggs.size();
                    g.region_to_id[v1] = gid;
                    g.scalar_aggs.push_back(l.scalar_aggs[i]);
                    g.distinct_aggs.push_back(std::move(l.distinct_aggs[i]));
                    g.keys.push_back(key);
                } else {
                    g.scalar_aggs[gid].sum_advengineid += l.scalar_aggs[i].sum_advengineid;
                    g.scalar_aggs[gid].count_total += l.scalar_aggs[i].count_total;
                    g.scalar_aggs[gid].sum_resolutionwidth += l.scalar_aggs[i].sum_resolutionwidth;
                    g.distinct_aggs[gid].distinct_userids.insert(l.distinct_aggs[i].distinct_userids.begin(),
                                                                 l.distinct_aggs[i].distinct_userids.end());
                }
            }
        }
        l.merged = true;
        g.merged_local_states.fetch_add(1, std::memory_order_relaxed);
    }

    const auto merged = g.merged_local_states.load(std::memory_order_relaxed);
    const auto active = g.active_local_states.load(std::memory_order_relaxed);

    idx_t output_row_idx = 0;
    if (active > 0 && merged == active) {
        {
            std::lock_guard<std::mutex> guard(g.lock);
            if (g.sort_state.rows.empty()) {
                for (size_t i = 0; i < g.keys.size(); ++i) {
                    g.sort_state.AddRow(g.keys[i], g.scalar_aggs[i], (int64_t)g.distinct_aggs[i].distinct_userids.size());
                }
                g.sort_state.SortNow();
            }

            output_row_idx = 0;
            size_t row_count = g.sort_state.rows.size();
            for (size_t i = 0; i < row_count && output_row_idx < 10; ++i) {
                const SortRow &row = g.sort_state.rows[i];
                double avg_res = row.scalar.count_total > 0 ? (double)row.scalar.sum_resolutionwidth / row.scalar.count_total : 0.0;

                out.SetValue(0, output_row_idx, Value::INTEGER(row.key.region_id));
                out.SetValue(1, output_row_idx, Value::HUGEINT(row.scalar.sum_advengineid));
                out.SetValue(2, output_row_idx, Value::BIGINT(row.scalar.count_total));
                out.SetValue(3, output_row_idx, Value::DOUBLE(avg_res));
                out.SetValue(4, output_row_idx, Value::BIGINT(row.distinct_count));
                output_row_idx++;
            }
        }
        out.SetCardinality(output_row_idx);
    } else {
        out.SetCardinality(0);
    }

    return OperatorFinalizeResultType::FINISHED;
}

static void LoadInternal(ExtensionLoader &loader) {
    TableFunction f("dbweaver", {LogicalType::TABLE}, nullptr, FnBind, FnInit, FnInitLocal);
    f.in_out_function = FnExecute;
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
