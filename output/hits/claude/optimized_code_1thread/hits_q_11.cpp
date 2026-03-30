/*
query_template: SELECT MobilePhoneModel, COUNT(DISTINCT UserID) AS u FROM hits WHERE MobilePhoneModel <> '' GROUP BY MobilePhoneModel ORDER BY u DESC LIMIT 10;

split_template: select * from dbweaver((SELECT MobilePhoneModel, UserID FROM hits WHERE (MobilePhoneModel!='')));
query_example: SELECT MobilePhoneModel, COUNT(DISTINCT UserID) AS u FROM hits WHERE MobilePhoneModel <> '' GROUP BY MobilePhoneModel ORDER BY u DESC LIMIT 10;

split_query: select * from dbweaver((SELECT MobilePhoneModel, UserID FROM hits WHERE (MobilePhoneModel!='')));
*/
#define DUCKDB_EXTENSION_MAIN

#include "dbweaver_extension.hpp"
#include "duckdb.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/common/types/string_heap.hpp"
#include "duckdb/common/types/hash.hpp"
#include <atomic>
#include <limits>
#include <mutex>
#include <absl/container/flat_hash_map.h>
#include <absl/container/flat_hash_set.h>


#include <algorithm>
#include <vector>
#include <numeric>

namespace duckdb {

// Helper structs for binding/execution

struct FnBindData : public FunctionData {
    unique_ptr<FunctionData> Copy() const override { return make_uniq<FnBindData>(); }
    bool Equals(const FunctionData &) const override { return true; }
};

struct GroupKey {
    string_t mobile_phone_model;
    
    bool operator==(const GroupKey& other) const {
        return mobile_phone_model == other.mobile_phone_model;
    }
};

struct GroupKeyHash {
    size_t operator()(const GroupKey& k) const {
        return duckdb::Hash<string_t>(k.mobile_phone_model);
    }
};
struct AggState {
    absl::flat_hash_set<int64_t> user_ids;
};


struct SortKeyView {
    int64_t u;
};

struct SortState {
    std::vector<SortKeyView> buffer;
    std::vector<GroupKey> keys;
    bool sorted = false;
    
    inline void AddRow(int64_t u_value, const GroupKey &key) {
        buffer.push_back(SortKeyView{u_value});
        keys.push_back(key);
    }
    inline void SortNow() {
        if (!sorted && !buffer.empty()) {
            size_t n = buffer.size();
            size_t k = std::min<size_t>(10, n);
            std::vector<size_t> indices(n);
            std::iota(indices.begin(), indices.end(), 0);
            std::partial_sort(indices.begin(), indices.begin() + k, indices.end(), [&](size_t a, size_t b) {
                return buffer[a].u > buffer[b].u;
            });
            // Reorder only the top k elements based on sorted indices
            std::vector<GroupKey> sorted_keys;
            std::vector<SortKeyView> sorted_buffer;
            sorted_keys.reserve(k);
            sorted_buffer.reserve(k);
            for (size_t i = 0; i < k; ++i) {
                size_t idx = indices[i];
                sorted_keys.push_back(keys[idx]);
                sorted_buffer.push_back(buffer[idx]);
            }
            keys = std::move(sorted_keys);
            buffer = std::move(sorted_buffer);
            sorted = true;
        }
    }

};

// Parallel/Global/Local aggregation states
struct FnGlobalState : public GlobalTableFunctionState {
    std::mutex lock;
    std::atomic<idx_t> active_local_states {0};
    std::atomic<idx_t> merged_local_states {0};
    absl::flat_hash_map<GroupKey, AggState, GroupKeyHash> agg_map;
    StringHeap heap;

    SortState sort_state;
    bool results_outputted = false;

    idx_t MaxThreads() const override {
        return std::numeric_limits<idx_t>::max();
    }
};

static unique_ptr<GlobalTableFunctionState> FnInit(ClientContext &, TableFunctionInitInput &) {
    return make_uniq<FnGlobalState>();
}
struct FnLocalState : public LocalTableFunctionState {
    absl::flat_hash_map<GroupKey, AggState, GroupKeyHash> agg_map;
    StringHeap heap;

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
    names.push_back("MobilePhoneModel");
    return_types.push_back(LogicalType::VARCHAR);

    names.push_back("u");
    return_types.push_back(LogicalType::BIGINT);

    return make_uniq<FnBindData>();
}

static OperatorResultType FnExecute(ExecutionContext &, TableFunctionInput &in,
                                                    DataChunk &input, DataChunk &) {
    auto &l = in.local_state->Cast<FnLocalState>();

    if (input.size() == 0) {
        return OperatorResultType::NEED_MORE_INPUT;
    }

    UnifiedVectorFormat mobile_phone_model_uvf;
    input.data[0].ToUnifiedFormat(input.size(), mobile_phone_model_uvf);
    string_t* mobile_phone_model_ptr = (string_t*)mobile_phone_model_uvf.data;

    UnifiedVectorFormat user_id_uvf;
    input.data[1].ToUnifiedFormat(input.size(), user_id_uvf);
    int64_t* user_id_ptr = (int64_t*)user_id_uvf.data;

    auto &valid_mobile_phone_model = mobile_phone_model_uvf.validity;
    auto &valid_user_id = user_id_uvf.validity;

    const bool mobile_phone_model_all_valid = valid_mobile_phone_model.AllValid();
    const bool user_id_all_valid = valid_user_id.AllValid();

    AggState* last_agg_state = nullptr;
    string_t last_model;
    bool has_last = false;

    for (idx_t row_idx = 0; row_idx < input.size(); ++row_idx) {
        idx_t i_mobile_phone_model = mobile_phone_model_uvf.sel->get_index(row_idx);
        idx_t i_user_id = user_id_uvf.sel->get_index(row_idx);

        if (!mobile_phone_model_all_valid && !valid_mobile_phone_model.RowIsValid(i_mobile_phone_model)) {
            continue;
        }
        if (!user_id_all_valid && !valid_user_id.RowIsValid(i_user_id)) {
            continue;
        }

        string_t v_mobile_phone_model = mobile_phone_model_ptr[i_mobile_phone_model];
        int64_t v_user_id = user_id_ptr[i_user_id];

        if (v_mobile_phone_model.GetSize() == 0) {
            continue;
        }
        if (has_last && v_mobile_phone_model == last_model) {
            last_agg_state->user_ids.insert(v_user_id);
        } else {
            GroupKey key;
            key.mobile_phone_model = v_mobile_phone_model;
            auto it = l.agg_map.find(key);
            if (it == l.agg_map.end()) {
                GroupKey new_key;
                new_key.mobile_phone_model = l.heap.AddString(v_mobile_phone_model);
                last_agg_state = &l.agg_map[new_key];
            } else {
                last_agg_state = &it->second;
            }
            last_model = v_mobile_phone_model;
            has_last = true;
            last_agg_state->user_ids.insert(v_user_id);
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
            for (auto &entry : l.agg_map) {
                auto it = g.agg_map.find(entry.first);
                if (it == g.agg_map.end()) {
                    GroupKey global_key;
                    global_key.mobile_phone_model = g.heap.AddString(entry.first.mobile_phone_model);
                    g.agg_map[global_key].user_ids = std::move(entry.second.user_ids);
                } else {
                    it->second.user_ids.insert(entry.second.user_ids.begin(), entry.second.user_ids.end());
                }
            }
        }
        l.merged = true;
        g.merged_local_states.fetch_add(1, std::memory_order_relaxed);
    }

    const auto merged = g.merged_local_states.load(std::memory_order_relaxed);
    const auto active = g.active_local_states.load(std::memory_order_relaxed);
    
    if (active > 0 && merged == active) {
        std::lock_guard<std::mutex> guard(g.lock);
        if (!g.results_outputted) {
            // Fill sort state from distinct sets
            for (auto &entry : g.agg_map) {
                g.sort_state.AddRow(entry.second.user_ids.size(), entry.first);
            }

            
            g.sort_state.SortNow();
            
            idx_t output_idx = 0;
            for (size_t i = 0; i < g.sort_state.keys.size() && output_idx < 10; ++i) {
                const GroupKey &key = g.sort_state.keys[i];
                const SortKeyView &sort_kv = g.sort_state.buffer[i];
                out.SetValue(0, output_idx, key.mobile_phone_model);
                out.SetValue(1, output_idx, Value::BIGINT(sort_kv.u));
                output_idx++;
            }
            out.SetCardinality(output_idx);
            g.results_outputted = true;
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
