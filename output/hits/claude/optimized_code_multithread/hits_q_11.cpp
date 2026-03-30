#define DUCKDB_EXTENSION_MAIN

#include "dbweaver_extension.hpp"
#include "duckdb.hpp"
#include "duckdb/function/table_function.hpp"
#include <atomic>
#include <limits>
#include <mutex>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>
#include <vector>
#include <numeric>
#include <deque>
#include <string_view>

namespace duckdb {

// Helper structs for binding/execution

struct FnBindData : public FunctionData {
    unique_ptr<FunctionData> Copy() const override { return make_uniq<FnBindData>(); }
    bool Equals(const FunctionData &) const override { return true; }
};

struct GroupKey {
    std::string mobile_phone_model;
    
    bool operator==(const GroupKey& other) const {
        return mobile_phone_model == other.mobile_phone_model;
    }
};
struct AggState {
    std::vector<int64_t> user_ids;
    void SortAndUnique() {
        if (user_ids.empty()) return;
        std::sort(user_ids.begin(), user_ids.end());
        user_ids.erase(std::unique(user_ids.begin(), user_ids.end()), user_ids.end());
    }
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
            std::vector<size_t> indices(buffer.size());
            std::iota(indices.begin(), indices.end(), 0);
            std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
                return buffer[a].u > buffer[b].u;
            });
            std::vector<GroupKey> sorted_keys;
            std::vector<SortKeyView> sorted_buffer;
            for (auto idx : indices) {
                sorted_keys.push_back(std::move(keys[idx]));
                sorted_buffer.push_back(buffer[idx]);
            }
            keys = std::move(sorted_keys);
            buffer = std::move(sorted_buffer);
            sorted = true;
        }
    }
};

struct FnGlobalState : public GlobalTableFunctionState {
    std::mutex lock;
    std::atomic<idx_t> active_local_states {0};
    std::atomic<idx_t> merged_local_states {0};
    std::unordered_map<std::string, AggState> agg_map;
    SortState sort_state;
    
    idx_t MaxThreads() const override {
        return std::numeric_limits<idx_t>::max();
    }
};

static unique_ptr<GlobalTableFunctionState> FnInit(ClientContext &, TableFunctionInitInput &) {
    return make_uniq<FnGlobalState>();
}

struct StringTHash {
    size_t operator()(const string_t& s) const {
        return duckdb::Hash(s.GetData(), s.GetSize());
    }
};

struct StringTEq {
    bool operator()(const string_t& lhs, const string_t& rhs) const {
        return lhs == rhs;
    }
};

struct FnLocalState : public LocalTableFunctionState {
    std::unordered_map<string_t, uint32_t, StringTHash, StringTEq> string_to_id;
    std::vector<AggState> agg_states;
    std::deque<std::string> id_to_string; // Deque keeps element pointers stable

    // Fast path cache for the last processed string
    string_t last_s;
    uint32_t last_id = 0;
    bool has_last = false;
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

    for (idx_t row_idx = 0; row_idx < input.size(); ++row_idx) {
        idx_t i_mobile_phone_model = mobile_phone_model_uvf.sel->get_index(row_idx);
        idx_t i_user_id = user_id_uvf.sel->get_index(row_idx);

        if (!mobile_phone_model_all_valid && !valid_mobile_phone_model.RowIsValid(i_mobile_phone_model)) continue;
        if (!user_id_all_valid && !valid_user_id.RowIsValid(i_user_id)) continue;

        string_t v_mobile_phone_model = mobile_phone_model_ptr[i_mobile_phone_model];
        int64_t v_user_id = user_id_ptr[i_user_id];

        if (v_mobile_phone_model.GetSize() == 0) continue;

        uint32_t id;
        if (l.has_last && v_mobile_phone_model == l.last_s) {
            id = l.last_id;
        } else {
            auto it = l.string_to_id.find(v_mobile_phone_model);
            if (it == l.string_to_id.end()) {
                id = (uint32_t)l.agg_states.size();
                l.id_to_string.emplace_back(v_mobile_phone_model.GetData(), v_mobile_phone_model.GetSize());
                const std::string& s = l.id_to_string.back();
                string_t st(s.data(), (uint32_t)s.size());
                l.string_to_id[st] = id;
                l.agg_states.emplace_back();
                l.last_s = st;
            } else {
                id = it->second;
                l.last_s = it->first;
            }
            l.last_id = id;
            l.has_last = true;
        }
        l.agg_states[id].user_ids.push_back(v_user_id);

    }

    return OperatorResultType::NEED_MORE_INPUT;
}

static OperatorFinalizeResultType FnFinalize(ExecutionContext &, TableFunctionInput &in,
                                                            DataChunk &out) {
    auto &g = *reinterpret_cast<FnGlobalState *>(in.global_state.get());
    auto &l = in.local_state->Cast<FnLocalState>();

    bool is_last = false;
    if (!l.merged) {
        for (auto &agg : l.agg_states) {
            agg.SortAndUnique();
        }
        {
            std::lock_guard<std::mutex> guard(g.lock);
            for (auto const& entry : l.string_to_id) {
                string_t st = entry.first;
                uint32_t id = entry.second;
                std::string key_str(st.GetData(), st.GetSize());
                auto &global_agg = g.agg_map[key_str];
                auto &local_agg = l.agg_states[id];
                
                if (global_agg.user_ids.empty()) {
                    global_agg.user_ids = std::move(local_agg.user_ids);
                } else {
                    global_agg.user_ids.reserve(global_agg.user_ids.size() + local_agg.user_ids.size());
                    global_agg.user_ids.insert(global_agg.user_ids.end(), local_agg.user_ids.begin(), local_agg.user_ids.end());
                }
            }
        }
        l.merged = true;
        if (g.merged_local_states.fetch_add(1) + 1 == g.active_local_states.load()) {
            is_last = true;
        }
    }

    if (is_last) {
        std::lock_guard<std::mutex> guard(g.lock);
        if (g.sort_state.keys.empty()) {
            for (auto &entry : g.agg_map) {
                entry.second.SortAndUnique();
                g.sort_state.AddRow((int64_t)entry.second.user_ids.size(), GroupKey{entry.first});
            }
            g.sort_state.SortNow();
        }


        idx_t count = std::min((idx_t)10, (idx_t)g.sort_state.keys.size());
        for (idx_t i = 0; i < count; ++i) {
            const GroupKey &key = g.sort_state.keys[i];
            const SortKeyView &sort_kv = g.sort_state.buffer[i];
            out.SetValue(0, i, Value(key.mobile_phone_model));
            out.SetValue(1, i, Value::BIGINT(sort_kv.u));
        }
        out.SetCardinality(count);
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