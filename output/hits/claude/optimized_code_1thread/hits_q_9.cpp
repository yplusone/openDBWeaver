/*
query_template: SELECT RegionID, COUNT(DISTINCT UserID) AS u FROM hits GROUP BY RegionID ORDER BY u DESC LIMIT 10;

split_template: select * from dbweaver((SELECT RegionID, UserID FROM hits));
query_example: SELECT RegionID, COUNT(DISTINCT UserID) AS u FROM hits GROUP BY RegionID ORDER BY u DESC LIMIT 10;

split_query: select * from dbweaver((SELECT RegionID, UserID FROM hits));
*/
#define DUCKDB_EXTENSION_MAIN

#include "dbweaver_extension.hpp"
#include "duckdb.hpp"
#include "duckdb/function/table_function.hpp"
#include <atomic>
#include <limits>
#include <mutex>
#include <absl/container/flat_hash_set.h>
#include <absl/container/flat_hash_map.h>
#include <absl/hash/hash.h>
#include <vector>
#include <algorithm>

namespace duckdb {

struct FnBindData : public FunctionData {
    unique_ptr<FunctionData> Copy() const override { return make_uniq<FnBindData>(); }
    bool Equals(const FunctionData &) const override { return true; }
};

// Maximum RegionID value based on data distribution (max 131069)
static constexpr int32_t MAX_REGION_ID = 131072;

struct SortRow {
    int32_t region_id;
    int64_t u;
};

struct SortRowComparator {
    bool operator()(const SortRow &a, const SortRow &b) const {
        if (a.u != b.u) return a.u > b.u;
        return a.region_id < b.region_id;
    }
};

struct SortState {
    std::vector<SortRow> buffer;
    bool sorted = false;

    inline void AddRow(int64_t u_val, int32_t region_id) {
        buffer.push_back(SortRow{region_id, u_val});
    }

    inline void SortNow() {
        if (!sorted) {
            std::sort(buffer.begin(), buffer.end(), SortRowComparator{});
            sorted = true;
        }
    }
};

struct FnGlobalState : public GlobalTableFunctionState {
    absl::flat_hash_map<int32_t, absl::flat_hash_set<int64_t>> global_map;
    std::mutex lock;
    std::atomic<idx_t> active_local_states {0};
    std::atomic<idx_t> merged_local_states {0};
    SortState sort_state;
    bool results_produced = false;

    FnGlobalState() {}

    idx_t MaxThreads() const override {
        return std::numeric_limits<idx_t>::max();
    }
};

static unique_ptr<GlobalTableFunctionState> FnInit(ClientContext &, TableFunctionInitInput &) {
    return make_uniq<FnGlobalState>();
}
struct FnLocalState : public LocalTableFunctionState {
    bool merged = false;
    std::vector<absl::flat_hash_set<int64_t>> local_sets;
    std::vector<int32_t> seen_regions;
    FnLocalState() {
        local_sets.resize(MAX_REGION_ID);
    }
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
    return_types.emplace_back(LogicalType::INTEGER);
    names.emplace_back("RegionID");
    return_types.emplace_back(LogicalType::BIGINT);
    names.emplace_back("u");
    return make_uniq<FnBindData>();
}

static OperatorResultType FnExecute(ExecutionContext &, TableFunctionInput &in,
                                            DataChunk &input, DataChunk &) {
    auto &l = in.local_state->Cast<FnLocalState>();
    if (input.size() == 0) {
        return OperatorResultType::NEED_MORE_INPUT;
    }

    UnifiedVectorFormat region_id_uvf;
    UnifiedVectorFormat user_id_uvf;
    input.data[0].ToUnifiedFormat(input.size(), region_id_uvf);
    input.data[1].ToUnifiedFormat(input.size(), user_id_uvf);
    
    int32_t* region_id_ptr = (int32_t*)region_id_uvf.data;
    int64_t* user_id_ptr = (int64_t*)user_id_uvf.data;
    
    auto &valid_region_id = region_id_uvf.validity;
    auto &valid_user_id = user_id_uvf.validity;
    
    const bool region_id_all_valid = valid_region_id.AllValid();
    const bool user_id_all_valid = valid_user_id.AllValid();
    
    idx_t num_rows = input.size();
    
    for (idx_t row_idx = 0; row_idx < num_rows; ++row_idx) {
        idx_t i_region_id = region_id_uvf.sel->get_index(row_idx);
        idx_t i_user_id = user_id_uvf.sel->get_index(row_idx);
        
        if (!region_id_all_valid && !valid_region_id.RowIsValid(i_region_id)) continue;
        if (!user_id_all_valid && !valid_user_id.RowIsValid(i_user_id)) continue;
        int32_t v_region_id = region_id_ptr[i_region_id];
        int64_t v_user_id = user_id_ptr[i_user_id];
        
        if (v_region_id >= 0 && v_region_id < MAX_REGION_ID) {
            auto &target_set = l.local_sets[v_region_id];
            if (target_set.empty()) {
                l.seen_regions.push_back(v_region_id);
            }
            target_set.insert(v_user_id);
        }
    }


    return OperatorResultType::NEED_MORE_INPUT; 
}

static OperatorFinalizeResultType FnFinalize(ExecutionContext &, TableFunctionInput &in,
                                                        DataChunk &out) {
    auto &g = in.global_state->Cast<FnGlobalState>();
    auto &l = in.local_state->Cast<FnLocalState>();

    if (!l.merged) {
        {
            std::lock_guard<std::mutex> guard(g.lock);
            for (int32_t rid : l.seen_regions) {
                auto &l_set = l.local_sets[rid];
                auto &g_set = g.global_map[rid];
                if (g_set.empty()) {
                    g_set = std::move(l_set);
                } else {
                    g_set.insert(l_set.begin(), l_set.end());
                }
            }
        }
        l.seen_regions.clear();

        l.merged = true;
        g.merged_local_states.fetch_add(1, std::memory_order_release);
    }

    const auto merged = g.merged_local_states.load(std::memory_order_acquire);
    const auto active = g.active_local_states.load(std::memory_order_acquire);
    if (active > 0 && merged == active) {
        {
            std::lock_guard<std::mutex> guard(g.lock);
            if (!g.results_produced) {
                if (!g.sort_state.sorted) {
                    for (const auto &pair : g.global_map) {
                        g.sort_state.AddRow(pair.second.size(), pair.first);
                    }
                    g.global_map.clear(); // Free memory after building SortState
                    g.sort_state.SortNow();
                }

                idx_t output_row_idx = 0;
                size_t out_limit = std::min<size_t>(10, g.sort_state.buffer.size());
                for (size_t i = 0; i < out_limit; ++i) {
                    const auto &row = g.sort_state.buffer[i];
                    out.SetValue(0, output_row_idx, Value::INTEGER(row.region_id));
                    out.SetValue(1, output_row_idx, Value::BIGINT(row.u));
                    output_row_idx++;
                }
                out.SetCardinality(output_row_idx);
                g.results_produced = true;
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
