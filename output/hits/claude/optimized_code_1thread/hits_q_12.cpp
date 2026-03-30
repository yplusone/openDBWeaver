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
#include <cstdlib>
#include "absl/container/flat_hash_set.h"
#include "absl/container/flat_hash_map.h"
#include "absl/hash/hash.h"
#include "absl/strings/string_view.h"
#include <algorithm>
#include <string>
#include <functional>
#include <numeric>
#include <queue>
#include <vector>

namespace duckdb {

struct UserIDSet {
    int64_t *data = nullptr;
    uint32_t capacity = 0;
    uint32_t count = 0;
    static constexpr int64_t EMPTY = std::numeric_limits<int64_t>::min();

    UserIDSet() = default;
    ~UserIDSet() { if (data) std::free(data); }

    UserIDSet(UserIDSet&& other) noexcept : data(other.data), capacity(other.capacity), count(other.count) {
        other.data = nullptr;
        other.capacity = 0;
        other.count = 0;
    }

    UserIDSet& operator=(UserIDSet&& other) noexcept {
        if (this != &other) {
            if (data) std::free(data);
            data = other.data;
            capacity = other.capacity;
            count = other.count;
            other.data = nullptr;
            other.capacity = 0;
            other.count = 0;
        }
        return *this;
    }

    UserIDSet(const UserIDSet&) = delete;
    UserIDSet& operator=(const UserIDSet&) = delete;

    void insert(int64_t key) {
        if (capacity == 0 || count * 2 >= capacity) resize();
        uint32_t mask = capacity - 1;
        uint64_t h = uint64_t(key) * 0x9e3779b97f4a7c15ULL;
        uint32_t idx = (uint32_t)(h ^ (h >> 32)) & mask;
        while (data[idx] != EMPTY) {
            if (data[idx] == key) return;
            idx = (idx + 1) & mask;
        }
        data[idx] = key;
        count++;
    }

    void resize() {
        uint32_t old_cap = capacity;
        int64_t *old_data = data;
        capacity = (capacity == 0) ? 64 : capacity * 2;
        data = (int64_t*)std::malloc(sizeof(int64_t) * capacity);
        for (uint32_t i = 0; i < capacity; ++i) data[i] = EMPTY;
        uint32_t mask = capacity - 1;
        if (old_data) {
            for (uint32_t i = 0; i < old_cap; ++i) {
                int64_t key = old_data[i];
                if (key != EMPTY) {
                    uint64_t h = uint64_t(key) * 0x9e3779b97f4a7c15ULL;
                    uint32_t idx = (uint32_t)(h ^ (h >> 32)) & mask;
                    while (data[idx] != EMPTY) {
                        idx = (idx + 1) & mask;
                    }
                    data[idx] = key;
                }
            }
            std::free(old_data);
        }
    }

    size_t size() const { return count; }
};

struct FnBindData : public FunctionData {
    unique_ptr<FunctionData> Copy() const override { return make_uniq<FnBindData>(); }
    bool Equals(const FunctionData &) const override { return true; }
};

struct GroupKey {
    int16_t mobile_phone;
    uint32_t model_id;
    
    bool operator==(const GroupKey& other) const {
        return mobile_phone == other.mobile_phone && model_id == other.model_id;
    }
};

struct GroupKeyHash {
    size_t operator()(const GroupKey& k) const {
        uint64_t h = (static_cast<uint64_t>(static_cast<uint16_t>(k.mobile_phone)) << 32) | k.model_id;
        return absl::Hash<uint64_t>{}(h);
    }
};

struct StringHash {
    using is_transparent = void;
    size_t operator()(absl::string_view v) const {
        return absl::Hash<absl::string_view>{}(v);
    }
    size_t operator()(const std::string& v) const {
        return absl::Hash<std::string>{}(v);
    }
};

struct StringEq {
    using is_transparent = void;
    bool operator()(absl::string_view a, absl::string_view b) const {
        return a == b;
    }
};

struct AggState {
    UserIDSet distinct_user_ids;
};

struct SortKeyView {
    int64_t u;
};

struct SortRowComparator {
    bool operator()(const SortKeyView &a, const SortKeyView &b) const {
        if (a.u != b.u) return a.u > b.u;
        return false;
    }
};

struct SortState {
    std::vector<SortKeyView> buffer;
    std::vector<GroupKey> keys;
    std::vector<AggState> states;
    bool sorted = false;
    
    inline void AddRow(const GroupKey &key, AggState &&state, int64_t u_value) {
        buffer.push_back(SortKeyView{u_value});
        keys.push_back(key);
        states.push_back(std::move(state));
    }
    
    inline void SortNow() {
        if (!sorted && !buffer.empty()) {
            std::vector<size_t> indices(buffer.size());
            std::iota(indices.begin(), indices.end(), 0);
            
            std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
                return SortRowComparator{}(buffer[a], buffer[b]);
            });
            
            std::vector<GroupKey> sorted_keys;
            std::vector<AggState> sorted_states;
            sorted_keys.reserve(indices.size());
            sorted_states.reserve(indices.size());
            for (auto idx : indices) {
                sorted_keys.push_back(keys[idx]);
                sorted_states.push_back(std::move(states[idx]));
            }
            
            keys = std::move(sorted_keys);
            states = std::move(sorted_states);
            sorted = true;
        }
    }
};

struct FnGlobalState : public GlobalTableFunctionState {
    absl::flat_hash_map<GroupKey, AggState, GroupKeyHash> agg_map;
    absl::flat_hash_map<std::string, uint32_t, StringHash, StringEq> string_to_id;
    std::vector<std::string> id_to_string;
    SortState sort_state;
    std::mutex lock;
    std::atomic<idx_t> active_local_states {0};
    std::atomic<idx_t> merged_local_states {0};
};

static unique_ptr<GlobalTableFunctionState> FnInit(ClientContext &, TableFunctionInitInput &) {
    return make_uniq<FnGlobalState>();
}

struct FnLocalState : public LocalTableFunctionState {
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
    auto &g = in.global_state->Cast<FnGlobalState>();
    if (input.size() == 0) return OperatorResultType::NEED_MORE_INPUT;

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

    {
        std::lock_guard<std::mutex> guard(g.lock);

        const char* last_model_data = nullptr;
        uint32_t last_model_size = 0;
        uint32_t last_model_id = 0;
        bool has_last_model = false;

        GroupKey last_key = { -1, 0xFFFFFFFF };
        AggState* last_agg_ptr = nullptr;

        for (idx_t row_idx = 0; row_idx < input.size(); ++row_idx) {
            idx_t i_MobilePhoneModel = MobilePhoneModel_uvf.sel->get_index(row_idx);
            idx_t i_MobilePhone = MobilePhone_uvf.sel->get_index(row_idx);
            idx_t i_UserID = UserID_uvf.sel->get_index(row_idx);
            
            if (!MobilePhoneModel_all_valid && !valid_MobilePhoneModel.RowIsValid(i_MobilePhoneModel)) continue;
            if (!MobilePhone_all_valid && !valid_MobilePhone.RowIsValid(i_MobilePhone)) continue;
            if (!UserID_all_valid && !valid_UserID.RowIsValid(i_UserID)) continue;
            
            const string_t &v_MobilePhoneModel = MobilePhoneModel_ptr[i_MobilePhoneModel];
            int16_t v_MobilePhone = MobilePhone_ptr[i_MobilePhone];
            int64_t v_UserID = UserID_ptr[i_UserID];

            uint32_t model_id;
            const char* model_data = v_MobilePhoneModel.GetData();
            uint32_t model_size = v_MobilePhoneModel.GetSize();
            
            if (has_last_model && model_data == last_model_data && model_size == last_model_size) {
                model_id = last_model_id;
            } else {
                absl::string_view model_sv(model_data, model_size);
                auto it = g.string_to_id.find(model_sv);
                if (it == g.string_to_id.end()) {
                    std::string model_str(model_data, model_size);
                    model_id = (uint32_t)g.id_to_string.size();
                    g.string_to_id.emplace(model_str, model_id);
                    g.id_to_string.push_back(std::move(model_str));
                } else {
                    model_id = it->second;
                }
                last_model_data = model_data;
                last_model_size = model_size;
                last_model_id = model_id;
                has_last_model = true;
            }

            GroupKey key { v_MobilePhone, model_id };
            AggState* state_ptr;
            if (last_agg_ptr && key == last_key) {
                state_ptr = last_agg_ptr;
            } else {
                state_ptr = &g.agg_map[key];
                last_key = key;
                last_agg_ptr = state_ptr;
            }
            state_ptr->distinct_user_ids.insert(v_UserID);
        }
    }
    
    return OperatorResultType::NEED_MORE_INPUT; 
}

static OperatorFinalizeResultType FnFinalize(ExecutionContext &, TableFunctionInput &in, 
                                             DataChunk &out) {
    auto &g = in.global_state->Cast<FnGlobalState>();
    auto &l = in.local_state->Cast<FnLocalState>();

    if (!l.merged) {
        // Merge is a no-op because data was written directly to global state in FnExecute
        l.merged = true;
        g.merged_local_states.fetch_add(1, std::memory_order_relaxed);
    }

    const auto merged = g.merged_local_states.load(std::memory_order_relaxed);
    const auto active = g.active_local_states.load(std::memory_order_relaxed);
    if (active > 0 && merged == active) {
        std::lock_guard<std::mutex> guard(g.lock);
        if (!g.sort_state.sorted) {
            for (auto &entry : g.agg_map) {
                int64_t u_value = (int64_t)entry.second.distinct_user_ids.size();
                g.sort_state.AddRow(entry.first, std::move(entry.second), u_value);
            }
            g.sort_state.SortNow();
        }
        
        idx_t output_idx = 0;
        for (size_t i = 0; i < g.sort_state.keys.size() && output_idx < 10; ++i) {
            const GroupKey &key = g.sort_state.keys[i];
            const AggState &state = g.sort_state.states[i];
            
            out.SetValue(0, output_idx, Value::SMALLINT(key.mobile_phone));
            out.SetValue(1, output_idx, Value(g.id_to_string[key.model_id]));
            out.SetValue(2, output_idx, Value::BIGINT((int64_t)state.distinct_user_ids.size()));
            output_idx++;
        }
        out.SetCardinality(output_idx);
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
