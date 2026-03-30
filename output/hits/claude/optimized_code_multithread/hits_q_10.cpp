#define DUCKDB_EXTENSION_MAIN

#include "dbweaver_extension.hpp"
#include "duckdb.hpp"
#include "duckdb/function/table_function.hpp"
#include <atomic>
#include <limits>
#include <mutex>
#include <vector>
#include <algorithm>
#include <absl/container/flat_hash_map.h>
#include <absl/container/flat_hash_set.h>

namespace duckdb {

struct DistinctUserID {
    static constexpr size_t VEC_THRESHOLD = 64;
    std::vector<int64_t> vec; // keeps it sorted
    absl::flat_hash_set<int64_t> set;
    bool is_set = false;

    void insert(int64_t v) {
        if (is_set) {
            set.insert(v);
            return;
        }
        auto it = std::lower_bound(vec.begin(), vec.end(), v);
        if (it != vec.end() && *it == v) return;
        vec.insert(it, v);
        if (vec.size() > VEC_THRESHOLD) {
            is_set = true;
            set.insert(vec.begin(), vec.end());
            vec.clear();
        }
    }

    void merge(DistinctUserID &&other) {
        if (other.is_set) {
            if (!is_set) {
                is_set = true;
                set = std::move(other.set);
                for (auto v : vec) set.insert(v);
                vec.clear();
            } else {
                if (set.size() < other.set.size()) set.swap(other.set);
                set.insert(other.set.begin(), other.set.end());
            }
        } else {
            if (is_set) {
                for (auto v : other.vec) set.insert(v);
            } else {
                if (vec.empty()) {
                    vec = std::move(other.vec);
                } else if (!other.vec.empty()) {
                    std::vector<int64_t> merged;
                    merged.reserve(vec.size() + other.vec.size());
                    std::set_union(vec.begin(), vec.end(), other.vec.begin(), other.vec.end(), std::back_inserter(merged));
                    vec = std::move(merged);
                    if (vec.size() > VEC_THRESHOLD) {
                        is_set = true;
                        set.insert(vec.begin(), vec.end());
                        vec.clear();
                    }
                }
            }
        }
    }

    int64_t size() const {
        return is_set ? (int64_t)set.size() : (int64_t)vec.size();
    }
};

struct FnBindData : public FunctionData {
    unique_ptr<FunctionData> Copy() const override { return make_uniq<FnBindData>(); }
    bool Equals(const FunctionData &) const override { return true; }
};

struct AggState {
    int64_t sum_advengineid = 0;
    int64_t count_total = 0;
    int64_t sum_resolutionwidth = 0;
    DistinctUserID distinct_userids;
};

struct SortRow {
    int32_t region_id;
    int64_t sum_advengineid;
    int64_t count_total;
    int64_t sum_resolutionwidth;
    int64_t cnt_distinct_userid;
};

struct SortRowComparator {
    bool operator()(const SortRow &a, const SortRow &b) const {
        if (a.count_total != b.count_total) return a.count_total > b.count_total;
        return a.region_id < b.region_id;
    }
};

struct FnGlobalState : public GlobalTableFunctionState {
    static constexpr size_t NUM_PARTITIONS = 64;
    absl::flat_hash_map<int32_t, AggState> partitions[NUM_PARTITIONS];
    std::mutex locks[NUM_PARTITIONS];
    
    std::atomic<idx_t> active_local_states {0};
    std::atomic<idx_t> merged_local_states {0};
    std::atomic<bool> result_emitted {false};
    
    idx_t MaxThreads() const override {
        return std::numeric_limits<idx_t>::max();
    }
};

static unique_ptr<GlobalTableFunctionState> FnInit(ClientContext &, TableFunctionInitInput &) {
    return make_uniq<FnGlobalState>();
}

struct FnLocalState : public LocalTableFunctionState {
    absl::flat_hash_map<int32_t, AggState> agg_map;
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
    return_types.emplace_back(LogicalType::INTEGER);  // RegionID
    return_types.emplace_back(LogicalType::HUGEINT);  // sum_advengineid
    return_types.emplace_back(LogicalType::BIGINT);   // c
    return_types.emplace_back(LogicalType::DOUBLE);   // avg_resolutionwidth
    return_types.emplace_back(LogicalType::BIGINT);   // cnt_distinct_userid
    
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
    if (input.size() == 0) return OperatorResultType::NEED_MORE_INPUT;

    UnifiedVectorFormat region_id_uvf, adv_engine_id_uvf, resolution_width_uvf, user_id_uvf;
    input.data[0].ToUnifiedFormat(input.size(), region_id_uvf);
    input.data[1].ToUnifiedFormat(input.size(), adv_engine_id_uvf);
    input.data[2].ToUnifiedFormat(input.size(), resolution_width_uvf);
    input.data[3].ToUnifiedFormat(input.size(), user_id_uvf);

    int32_t* region_id_ptr = (int32_t*)region_id_uvf.data;
    int16_t* adv_engine_id_ptr = (int16_t*)adv_engine_id_uvf.data;
    int16_t* resolution_width_ptr = (int16_t*)resolution_width_uvf.data;
    int64_t* user_id_ptr = (int64_t*)user_id_uvf.data;

    AggState* last_state = nullptr;
    int32_t last_region_id = -1;
    int64_t last_user_id = 0;
    bool has_last = false;
    bool has_last_user = false;

    for (idx_t row_idx = 0; row_idx < input.size(); ++row_idx) {
        idx_t i_reg = region_id_uvf.sel->get_index(row_idx);
        if (!region_id_uvf.validity.RowIsValid(i_reg)) continue;
        int32_t v1 = region_id_ptr[i_reg];

        idx_t i_adv = adv_engine_id_uvf.sel->get_index(row_idx);
        if (!adv_engine_id_uvf.validity.RowIsValid(i_adv)) continue;
        int16_t v2 = adv_engine_id_ptr[i_adv];

        idx_t i_res = resolution_width_uvf.sel->get_index(row_idx);
        if (!resolution_width_uvf.validity.RowIsValid(i_res)) continue;
        int16_t v3 = resolution_width_ptr[i_res];

        idx_t i_usr = user_id_uvf.sel->get_index(row_idx);
        if (!user_id_uvf.validity.RowIsValid(i_usr)) continue;
        int64_t v4 = user_id_ptr[i_usr];

        if (!has_last || v1 != last_region_id) {
            last_state = &l.agg_map[v1];
            last_region_id = v1;
            has_last = true;
            has_last_user = false;
        }

        last_state->sum_advengineid += v2;
        last_state->count_total++;
        last_state->sum_resolutionwidth += v3;
        
        // Skip redundant insertions using temporal locality
        if (!has_last_user || v4 != last_user_id) {
            last_state->distinct_userids.insert(v4);
            last_user_id = v4;
            has_last_user = true;
        }
    }

    return OperatorResultType::NEED_MORE_INPUT; 
}

static OperatorFinalizeResultType FnFinalize(ExecutionContext &, TableFunctionInput &in, 
                                                    DataChunk &out) {
    auto &g = in.global_state->Cast<FnGlobalState>();
    auto &l = in.local_state->Cast<FnLocalState>();

    if (!l.merged) {
        std::vector<int32_t> keys_by_partition[FnGlobalState::NUM_PARTITIONS];
        for (auto const& entry : l.agg_map) {
            int32_t region_id = entry.first;
            size_t p = static_cast<uint32_t>(region_id) % FnGlobalState::NUM_PARTITIONS;
            keys_by_partition[p].push_back(region_id);
        }

        for (size_t p = 0; p < FnGlobalState::NUM_PARTITIONS; ++p) {
            if (keys_by_partition[p].empty()) continue;
            std::lock_guard<std::mutex> guard(g.locks[p]);
            for (int32_t region_id : keys_by_partition[p]) {
                auto &src = l.agg_map[region_id];
                auto &dst = g.partitions[p][region_id];
                dst.sum_advengineid += src.sum_advengineid;
                dst.count_total += src.count_total;
                dst.sum_resolutionwidth += src.sum_resolutionwidth;
                dst.distinct_userids.merge(std::move(src.distinct_userids));
            }
        }
        l.agg_map.clear();
        l.merged = true;
        g.merged_local_states.fetch_add(1, std::memory_order_release);
    }

    const auto merged = g.merged_local_states.load(std::memory_order_acquire);
    const auto active = g.active_local_states.load(std::memory_order_acquire);

    if (active > 0 && merged == active) {
        if (!g.result_emitted.exchange(true)) {
            std::vector<SortRow> rows;
            for (size_t i = 0; i < FnGlobalState::NUM_PARTITIONS; ++i) {
                for (auto &entry : g.partitions[i]) {
                    rows.push_back({entry.first, entry.second.sum_advengineid, entry.second.count_total, 
                                   entry.second.sum_resolutionwidth, entry.second.distinct_userids.size()});
                }
                g.partitions[i].clear();
            }
            
            size_t n = std::min((size_t)10, rows.size());
            if (n > 0) {
                std::partial_sort(rows.begin(), rows.begin() + n, rows.end(), SortRowComparator{});

                auto region_ptr = FlatVector::GetData<int32_t>(out.data[0]);
                auto sum_adv_ptr = FlatVector::GetData<hugeint_t>(out.data[1]);
                auto count_ptr = FlatVector::GetData<int64_t>(out.data[2]);
                auto avg_res_ptr = FlatVector::GetData<double>(out.data[3]);
                auto dist_usr_ptr = FlatVector::GetData<int64_t>(out.data[4]);

                for (size_t i = 0; i < n; ++i) {
                    const auto &row = rows[i];
                    region_ptr[i] = row.region_id;
                    sum_adv_ptr[i] = hugeint_t(row.sum_advengineid);
                    count_ptr[i] = row.count_total;
                    avg_res_ptr[i] = row.count_total > 0 ? (double)row.sum_resolutionwidth / row.count_total : 0.0;
                    dist_usr_ptr[i] = row.cnt_distinct_userid;
                }
                out.SetCardinality(n);
            } else {
                out.SetCardinality(0);
            }
        } else {
            out.SetCardinality(0);
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
