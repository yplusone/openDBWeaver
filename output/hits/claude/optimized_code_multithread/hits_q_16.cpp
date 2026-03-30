/*
query_template: SELECT UserID, COUNT(*) AS cnt FROM hits GROUP BY UserID ORDER BY COUNT(*) DESC LIMIT 10;

split_template: select * from dbweaver((SELECT UserID FROM hits));
query_example: SELECT UserID, COUNT(*) AS cnt FROM hits GROUP BY UserID ORDER BY COUNT(*) DESC LIMIT 10;

split_query: select * from dbweaver((SELECT UserID FROM hits));
*/
#define DUCKDB_EXTENSION_MAIN

#include "dbweaver_extension.hpp"
#include "duckdb.hpp"
#include "duckdb/function/table_function.hpp"
#include <atomic>
#include <limits>
#include <mutex>
#include "absl/container/flat_hash_map.h"
#include <vector>
#include <algorithm>
#include <queue>
#include <functional>

namespace duckdb {

struct FnBindData : public FunctionData {
    vector<LogicalType> return_types;

    unique_ptr<FunctionData> Copy() const override {
        auto copy = make_uniq<FnBindData>();
        copy->return_types = return_types;
        return std::move(copy);
    }
    bool Equals(const FunctionData &) const override { return true; }
};

struct GroupKey {
    int64_t UserID;
    
    bool operator==(const GroupKey& other) const {
        return UserID == other.UserID;
    }
};

struct GroupKeyHash {
    size_t operator()(const GroupKey& k) const {
        // Simple identity hash for the int64_t UserID for efficiency inside the map
        return static_cast<size_t>(k.UserID);
    }
};

struct AggState {
    int64_t cnt = 0;
};

struct SortRow {
    GroupKey key;
    int64_t cnt;

    // For Top-K heap and final sort: "Greater" means "Better/More Desirable"
    // i.e., larger count or smaller UserID for ties.
    bool operator>(const SortRow &other) const {
        if (cnt != other.cnt) {
            return cnt > other.cnt;
        }
        return key.UserID < other.key.UserID;
    }
};

struct FnGlobalState : public GlobalTableFunctionState {
    static constexpr idx_t NUM_PARTITIONS = 256;

    struct Partition {
        absl::flat_hash_map<GroupKey, AggState, GroupKeyHash> agg_map;
        std::mutex lock;
    };

    std::unique_ptr<Partition[]> partitions;
    
    std::atomic<idx_t> active_local_states {0};
    std::atomic<idx_t> merged_local_states {0};

    FnGlobalState() : partitions(new Partition[NUM_PARTITIONS]) {}

    idx_t MaxThreads() const override {
        return std::numeric_limits<idx_t>::max();
    }
};

static unique_ptr<GlobalTableFunctionState> FnInit(ClientContext &, TableFunctionInitInput &) {
    return make_uniq<FnGlobalState>();
}

struct FnLocalState : public LocalTableFunctionState {
    bool merged = false;
    absl::flat_hash_map<GroupKey, AggState, GroupKeyHash> agg_map;
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
    // Output schema: UserID (BIGINT), cnt (BIGINT)
    names.push_back("UserID");
    names.push_back("cnt");
    return_types.push_back(LogicalType::BIGINT);
    return_types.push_back(LogicalType::BIGINT);

    auto bind_data = make_uniq<FnBindData>();
    bind_data->return_types = return_types;
    return std::move(bind_data);
}

static OperatorResultType FnExecute(ExecutionContext &, TableFunctionInput &in,
                                    DataChunk &input, DataChunk &) {
    auto &l = in.local_state->Cast<FnLocalState>();

    if (input.size() == 0) {       
        return OperatorResultType::NEED_MORE_INPUT;
    }

    UnifiedVectorFormat UserID_uvf;
    input.data[0].ToUnifiedFormat(input.size(), UserID_uvf);
    int64_t* UserID_ptr = (int64_t*)UserID_uvf.data;

    auto &valid_UserID  = UserID_uvf.validity;
    const bool UserID_all_valid = valid_UserID.AllValid();

    if (UserID_all_valid) {
        for (idx_t row_idx = 0; row_idx < input.size(); ++row_idx) {
            idx_t i_UserID = UserID_uvf.sel->get_index(row_idx);
            int64_t v1 = UserID_ptr[i_UserID];
            GroupKey key;
            key.UserID = v1;
            auto &state = l.agg_map[key];
            state.cnt++;
        }
    } else {
        for (idx_t row_idx = 0; row_idx < input.size(); ++row_idx) {
            idx_t i_UserID = UserID_uvf.sel->get_index(row_idx);
            if (!valid_UserID.RowIsValid(i_UserID)) {
                continue; 
            }
            int64_t v1 = UserID_ptr[i_UserID];
            GroupKey key;
            key.UserID = v1;
            auto &state = l.agg_map[key];
            state.cnt++;
        }
    }

    return OperatorResultType::NEED_MORE_INPUT; 
}

static OperatorFinalizeResultType FnFinalize(ExecutionContext &context, TableFunctionInput &in,
                                            DataChunk &out) {
    auto &g = in.global_state->Cast<FnGlobalState>();
    auto &l = in.local_state->Cast<FnLocalState>();
    auto &bind = in.bind_data->Cast<FnBindData>();

    out.Initialize(context.client, bind.return_types, STANDARD_VECTOR_SIZE);

    bool is_last_thread = false;
    if (!l.merged) {
        // Partitioned merge to global partitions
        for (const auto &entry : l.agg_map) {
            const GroupKey &key = entry.first;
            
            // Apply a mixer to the identity hash for better partition distribution
            uint64_t x = static_cast<uint64_t>(key.UserID);
            x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
            x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
            x = x ^ (x >> 31);
            idx_t part_idx = x % FnGlobalState::NUM_PARTITIONS;

            auto &part = g.partitions[part_idx];
            std::lock_guard<std::mutex> guard(part.lock);
            part.agg_map[key].cnt += entry.second.cnt;
        }
        l.merged = true;
        
        // The last thread to finalize performs the global aggregation and result emission
        idx_t merged_count = g.merged_local_states.fetch_add(1, std::memory_order_acq_rel) + 1;
        if (merged_count == g.active_local_states.load(std::memory_order_acquire)) {
            is_last_thread = true;
        }
    }

    if (is_last_thread) {
        // Use a min-priority queue to find the Top-10 results efficiently (O(N log K))
        // In a min-heap using std::greater, the "smallest" element (least desirable) is at the top.
        std::priority_queue<SortRow, std::vector<SortRow>, std::greater<SortRow>> top_k;

        for (idx_t i = 0; i < FnGlobalState::NUM_PARTITIONS; ++i) {
            for (const auto &entry : g.partitions[i].agg_map) {
                SortRow row{entry.first, entry.second.cnt};
                if (top_k.size() < 10) {
                    top_k.push(row);
                } else if (row > top_k.top()) {
                    top_k.pop();
                    top_k.push(row);
                }
            }
        }

        // Extract results and sort them descending (best first)
        std::vector<SortRow> final_results;
        while (!top_k.empty()) {
            final_results.push_back(top_k.top());
            top_k.pop();
        }
        std::sort(final_results.begin(), final_results.end(), std::greater<SortRow>());
        
        idx_t result_size = final_results.size();
        out.SetCardinality(result_size);

        for (idx_t i = 0; i < result_size; ++i) {
            const SortRow &row = final_results[i];
            out.SetValue(0, i, Value::BIGINT(row.key.UserID));
            out.SetValue(1, i, Value::BIGINT(row.cnt));
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
