/*
query_template: SELECT AdvEngineID, COUNT(*) AS cnt FROM hits WHERE AdvEngineID <> 0 GROUP BY AdvEngineID ORDER BY COUNT(*) DESC;

split_template: select * from dbweaver((SELECT AdvEngineID FROM hits WHERE (AdvEngineID!=0)));
query_example: SELECT AdvEngineID, COUNT(*) AS cnt FROM hits WHERE AdvEngineID <> 0 GROUP BY AdvEngineID ORDER BY COUNT(*) DESC;

split_query: select * from dbweaver((SELECT AdvEngineID FROM hits WHERE (AdvEngineID!=0)));
*/
#define DUCKDB_EXTENSION_MAIN

#include "dbweaver_extension.hpp"
#include "duckdb.hpp"
#include "duckdb/function/table_function.hpp"
#include <atomic>
#include <limits>
#include <mutex>
#include <vector>
#include <algorithm>

namespace duckdb {

struct FnBindData : public FunctionData {
    unique_ptr<FunctionData> Copy() const override { return make_uniq<FnBindData>(); }
    bool Equals(const FunctionData &) const override { return true; }
};

struct GroupKey {
    int16_t AdvEngineID;
};

struct AggState {
    int64_t cnt = 0;
};

struct RowEntry {
    GroupKey key;
    AggState state;
};

struct RowEntryComparator {
    // DESC order of count
    bool operator()(const RowEntry &a, const RowEntry &b) const {
        if (a.state.cnt != b.state.cnt) {
            return a.state.cnt > b.state.cnt;
        }
        return a.key.AdvEngineID < b.key.AdvEngineID;
    }
};

struct SortState {
    std::vector<RowEntry> rows;
    bool sorted = false;

    inline void AddRow(const GroupKey &key, const AggState &state) {
        rows.push_back(RowEntry{key, state});
    }

    inline void SortNow() {
        if (!sorted) {
            std::sort(rows.begin(), rows.end(), RowEntryComparator{});
            sorted = true;
        }
    }
};

struct FnGlobalState : public GlobalTableFunctionState {
    std::vector<int64_t> global_counts;
    std::vector<uint32_t> global_dirty_indices;
    SortState sort_state;
    std::mutex lock;
    std::atomic<idx_t> active_local_states {0};
    std::atomic<idx_t> merged_local_states {0};
    idx_t output_offset = 0;

    FnGlobalState() : global_counts(65536, 0) {}

    idx_t MaxThreads() const override {
        return std::numeric_limits<idx_t>::max();
    }
};

static unique_ptr<GlobalTableFunctionState> FnInit(ClientContext &, TableFunctionInitInput &) {
    return make_uniq<FnGlobalState>();
}

struct FnLocalState : public LocalTableFunctionState {
    bool merged = false;
    bool is_last_thread = false;
    std::vector<int64_t> counts;
    std::vector<uint32_t> dirty_indices;
    FnLocalState() : counts(65536, 0) {}
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
    return_types.emplace_back(LogicalType::SMALLINT);  // AdvEngineID
    return_types.emplace_back(LogicalType::BIGINT);   // cnt
    names.emplace_back("AdvEngineID");
    names.emplace_back("cnt");

    return make_uniq<FnBindData>();
}

static OperatorResultType FnExecute(ExecutionContext &, TableFunctionInput &in,
                                            DataChunk &input, DataChunk &) {
    auto &l = in.local_state->Cast<FnLocalState>();
    if (input.size() == 0) {       
        return OperatorResultType::NEED_MORE_INPUT;
    }

    UnifiedVectorFormat AdvEngineID_uvf;
    input.data[0].ToUnifiedFormat(input.size(), AdvEngineID_uvf);
    int16_t* AdvEngineID_ptr = (int16_t*)AdvEngineID_uvf.data;
    auto &valid_AdvEngineID = AdvEngineID_uvf.validity;
    
    int64_t* counts_ptr = l.counts.data();
    auto &dirty = l.dirty_indices;
    
    for (idx_t row_idx = 0; row_idx < input.size(); ++row_idx) {
        idx_t i_AdvEngineID = AdvEngineID_uvf.sel->get_index(row_idx);
        if (valid_AdvEngineID.RowIsValid(i_AdvEngineID)) {
            int16_t v_AdvEngineID = AdvEngineID_ptr[i_AdvEngineID];
            // Map int16 range [-32768, 32767] to [0, 65535]
            uint32_t idx = static_cast<uint32_t>(static_cast<int32_t>(v_AdvEngineID) + 32768);
            if (counts_ptr[idx] == 0) {
                dirty.push_back(idx);
            }
            counts_ptr[idx]++;
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
            for (uint32_t idx : l.dirty_indices) {
                if (g.global_counts[idx] == 0) {
                    g.global_dirty_indices.push_back(idx);
                }
                g.global_counts[idx] += l.counts[idx];
            }
        }
        l.merged = true;
        // Atomic check to identify the single thread that will emit results
        if (g.merged_local_states.fetch_add(1, std::memory_order_relaxed) == g.active_local_states.load(std::memory_order_relaxed) - 1) {
            l.is_last_thread = true;
        }
    }

    if (l.is_last_thread) {
        std::lock_guard<std::mutex> guard(g.lock);
        if (!g.sort_state.sorted) {
            for (uint32_t idx : g.global_dirty_indices) {
                GroupKey key;
                key.AdvEngineID = static_cast<int16_t>(static_cast<int32_t>(idx) - 32768);
                AggState state;
                state.cnt = g.global_counts[idx];
                g.sort_state.AddRow(key, state);
            }
            g.sort_state.SortNow();
        }
        
        idx_t output_idx = 0;
        idx_t row_count = g.sort_state.rows.size();
        auto out_key_ptr = FlatVector::GetData<int16_t>(out.data[0]);
        auto out_cnt_ptr = FlatVector::GetData<int64_t>(out.data[1]);

        while (g.output_offset < row_count && output_idx < STANDARD_VECTOR_SIZE) {
            const RowEntry &row = g.sort_state.rows[g.output_offset];
            out_key_ptr[output_idx] = row.key.AdvEngineID;
            out_cnt_ptr[output_idx] = row.state.cnt;
            output_idx++;
            g.output_offset++;
        }
        out.SetCardinality(output_idx);
        
        if (g.output_offset < row_count) {
            return OperatorFinalizeResultType::HAVE_MORE_OUTPUT;
        }
        return OperatorFinalizeResultType::FINISHED;
    } else {
        out.SetCardinality(0);
        return OperatorFinalizeResultType::FINISHED;
    }
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
