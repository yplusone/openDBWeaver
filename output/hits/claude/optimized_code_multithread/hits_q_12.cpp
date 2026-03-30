/*
query_template: SELECT MobilePhone, MobilePhoneModel, COUNT(DISTINCT UserID) AS u FROM hits WHERE MobilePhoneModel <> '' GROUP BY MobilePhone, MobilePhoneModel ORDER BY u DESC LIMIT 10;

split_template: select * from dbweaver((SELECT MobilePhoneModel, MobilePhone, UserID FROM hits WHERE (MobilePhoneModel!='')));
query_example: SELECT MobilePhone, MobilePhoneModel, COUNT(DISTINCT UserID) AS u FROM hits WHERE MobilePhoneModel <> '' GROUP BY MobilePhone, MobilePhoneModel ORDER BY u DESC LIMIT 10;

split_query: select * from dbweaver((SELECT MobilePhoneModel, MobilePhone, UserID FROM hits WHERE (MobilePhoneModel!='')));
*/
#define DUCKDB_EXTENSION_MAIN

#include "dbweaver_extension.hpp"
#include "duckdb.hpp"
#include "duckdb/function/table_function.hpp"
#include <atomic>
#include <limits>
#include <mutex>
#include <algorithm>
#include <queue>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/hash/hash.h"


namespace duckdb {

struct FnBindData : public FunctionData {
    unique_ptr<FunctionData> Copy() const override { return make_uniq<FnBindData>(); }
    bool Equals(const FunctionData &) const override { return true; }
};

struct GroupKey {
    int16_t mobile_phone;
    std::string mobile_phone_model;
    
    bool operator==(const GroupKey& other) const {
        return mobile_phone == other.mobile_phone && mobile_phone_model == other.mobile_phone_model;
    }
};

struct GroupKeyHash {
    size_t operator()(const GroupKey& k) const {
        return absl::Hash<std::pair<int16_t, std::string>>{}(std::make_pair(k.mobile_phone, k.mobile_phone_model));
    }
};

struct AggState {
    absl::flat_hash_set<int64_t> distinct_user_ids;
};

struct SortItem {
    int64_t u;
    GroupKey key;
};

struct SortState {
    std::vector<SortItem> items;
    bool sorted = false;
    
    inline void AddRow(const GroupKey &key, int64_t u_value) {
        items.push_back({u_value, key});
    }
    
    inline void SortNow() {
        if (!sorted && !items.empty()) {
            std::sort(items.begin(), items.end(), [](const SortItem &a, const SortItem &b) {
                if (a.u != b.u) return a.u > b.u;
                return false;
            });
            sorted = true;
        }
    }
};

struct FnGlobalState : public GlobalTableFunctionState {
    static constexpr idx_t PARTITION_COUNT = 64;
    struct Partition {
        absl::flat_hash_map<GroupKey, AggState, GroupKeyHash> agg_map;
        std::mutex lock;
    };
    std::vector<unique_ptr<Partition>> partitions;

    SortState sort_state;
    std::atomic<idx_t> active_local_states {0};
    std::atomic<idx_t> merged_local_states {0};
    std::atomic<bool> finalized {false};

    FnGlobalState() {
        for (idx_t i = 0; i < PARTITION_COUNT; ++i) {
            partitions.push_back(make_uniq<Partition>());
        }
    }

    idx_t MaxThreads() const override {
        return std::numeric_limits<idx_t>::max();
    }
};

static unique_ptr<GlobalTableFunctionState> FnInit(ClientContext &, TableFunctionInitInput &) {
    return make_uniq<FnGlobalState>();
}

struct FnLocalState : public LocalTableFunctionState {
    struct LocalGroupKey {
        int16_t mobile_phone;
        uint32_t model_id;
        bool operator==(const LocalGroupKey& o) const { return mobile_phone == o.mobile_phone && model_id == o.model_id; }
    };
    struct LocalGroupKeyHash {
        size_t operator()(const LocalGroupKey& k) const {
            uint64_t packed = (uint64_t(static_cast<uint16_t>(k.mobile_phone)) << 32) | k.model_id;
            return absl::Hash<uint64_t>{}(packed);
        }
    };

    absl::flat_hash_map<std::string, uint32_t> string_to_id;
    std::vector<std::string> id_to_string;
    
    absl::flat_hash_map<LocalGroupKey, AggState, LocalGroupKeyHash> agg_map;

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
    names.push_back("MobilePhone");
    names.push_back("MobilePhoneModel");
    names.push_back("u");
    
    return_types.push_back(LogicalType::SMALLINT);
    return_types.push_back(LogicalType::VARCHAR);
    return_types.push_back(LogicalType::BIGINT);

    return make_uniq<FnBindData>();
}

static OperatorResultType FnExecute(ExecutionContext &, TableFunctionInput &in,
                                                    DataChunk &input, DataChunk &) {
    auto &l = in.local_state->Cast<FnLocalState>();

    if (input.size() == 0) {       
        return OperatorResultType::NEED_MORE_INPUT;
    }

    UnifiedVectorFormat MobilePhoneModel_uvf;
    input.data[0].ToUnifiedFormat(input.size(), MobilePhoneModel_uvf);
    string_t* MobilePhoneModel_ptr = (string_t*)MobilePhoneModel_uvf.data;
    
    UnifiedVectorFormat MobilePhone_uvf;
    input.data[1].ToUnifiedFormat(input.size(), MobilePhone_uvf);
    int16_t* MobilePhone_ptr = (int16_t*)MobilePhone_uvf.data;
    
    UnifiedVectorFormat UserID_uvf;
    input.data[2].ToUnifiedFormat(input.size(), UserID_uvf);
    int64_t* UserID_ptr = (int64_t*)UserID_uvf.data;
    
    auto &valid_MobilePhoneModel = MobilePhoneModel_uvf.validity;
    auto &valid_MobilePhone = MobilePhone_uvf.validity;
    auto &valid_UserID = UserID_uvf.validity;
    
    const bool MobilePhoneModel_all_valid = valid_MobilePhoneModel.AllValid();
    const bool MobilePhone_all_valid = valid_MobilePhone.AllValid();
    const bool UserID_all_valid = valid_UserID.AllValid();
    
    for (idx_t row_idx = 0; row_idx < input.size(); ++row_idx) {
        idx_t i_MobilePhoneModel = MobilePhoneModel_uvf.sel->get_index(row_idx);
        idx_t i_MobilePhone = MobilePhone_uvf.sel->get_index(row_idx);
        idx_t i_UserID = UserID_uvf.sel->get_index(row_idx);
        
        if (!MobilePhoneModel_all_valid && !valid_MobilePhoneModel.RowIsValid(i_MobilePhoneModel)) continue;
        if (!MobilePhone_all_valid && !valid_MobilePhone.RowIsValid(i_MobilePhone)) continue;
        if (!UserID_all_valid && !valid_UserID.RowIsValid(i_UserID)) continue;
        
        string_t v_MobilePhoneModel = MobilePhoneModel_ptr[i_MobilePhoneModel];
        int16_t v_MobilePhone = MobilePhone_ptr[i_MobilePhone];
        int64_t v_UserID = UserID_ptr[i_UserID];
        
        // 1. Dictionary lookup for MobilePhoneModel
        std::string v_str(v_MobilePhoneModel.GetData(), v_MobilePhoneModel.GetSize());
        auto it = l.string_to_id.find(v_str);
        uint32_t model_id;
        if (it == l.string_to_id.end()) {
            model_id = static_cast<uint32_t>(l.id_to_string.size());
            l.id_to_string.push_back(v_str);
            l.string_to_id[v_str] = model_id;
        } else {
            model_id = it->second;
        }

        // 2. Aggregate with local numeric key
        FnLocalState::LocalGroupKey lk {v_MobilePhone, model_id};
        auto &state = l.agg_map[lk];
        state.distinct_user_ids.insert(v_UserID);
    }
    
    return OperatorResultType::NEED_MORE_INPUT; 
}

static OperatorFinalizeResultType FnFinalize(ExecutionContext &, TableFunctionInput &in,
                                                            DataChunk &out) {
    auto &g = *reinterpret_cast<FnGlobalState *>(in.global_state.get());
    auto &l = in.local_state->Cast<FnLocalState>();

    if (!l.merged) {
        for (const auto &entry : l.agg_map) {
            const auto &local_key = entry.first;
            GroupKey global_key;
            global_key.mobile_phone = local_key.mobile_phone;
            global_key.mobile_phone_model = l.id_to_string[local_key.model_id];
            
            size_t h = GroupKeyHash{}(global_key);
            idx_t part_idx = h % FnGlobalState::PARTITION_COUNT;
            auto &partition = *g.partitions[part_idx];
            
            std::lock_guard<std::mutex> guard(partition.lock);
            auto &global_state = partition.agg_map[global_key];
            for (const auto &user_id : entry.second.distinct_user_ids) {
                global_state.distinct_user_ids.insert(user_id);
            }
        }
        l.merged = true;
        g.merged_local_states.fetch_add(1, std::memory_order_acq_rel);
    }

    const auto merged = g.merged_local_states.load(std::memory_order_acquire);
    const auto active = g.active_local_states.load(std::memory_order_relaxed);
    
    if (active > 0 && merged == active) {
        if (!g.finalized.exchange(true)) {
            for (idx_t i = 0; i < FnGlobalState::PARTITION_COUNT; ++i) {
                auto &partition = *g.partitions[i];
                for (const auto &entry : partition.agg_map) {
                    const GroupKey &key = entry.first;
                    const AggState &state = entry.second;
                    int64_t u_value = static_cast<int64_t>(state.distinct_user_ids.size());
                    g.sort_state.AddRow(key, u_value);
                }
            }
            
            g.sort_state.SortNow();
            
            idx_t output_idx = 0;
            for (size_t i = 0; i < g.sort_state.items.size() && output_idx < 10; ++i) {
                const auto &item = g.sort_state.items[i];
                
                out.SetValue(0, output_idx, Value::SMALLINT(item.key.mobile_phone));
                out.SetValue(1, output_idx, Value(item.key.mobile_phone_model));
                out.SetValue(2, output_idx, Value::BIGINT(item.u));
                
                output_idx++;
            }
            out.SetCardinality(output_idx);
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
