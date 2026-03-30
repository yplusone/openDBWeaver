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

struct RowEntry {
    int16_t id;
    int64_t cnt;
};

struct RowEntryComparator {
    // DESC order of count
    bool operator()(const RowEntry &a, const RowEntry &b) const {
        if (a.cnt != b.cnt) return a.cnt > b.cnt;
        return a.id < b.id;
    }
};

struct SortState {
    std::vector<RowEntry> rows;
    bool sorted = false;

    inline void AddRow(int16_t id, int64_t cnt) {
        rows.push_back(RowEntry{id, cnt});
    }

    inline void SortNow() {
        if (!sorted) {
            std::sort(rows.begin(), rows.end(), RowEntryComparator{});
            sorted = true;
        }
    }
};

struct FnGlobalState : public GlobalTableFunctionState {
    int64_t counts[65536];
    SortState sort_state;
    std::mutex lock;
    std::atomic<idx_t> active_local_states {0};
    std::atomic<idx_t> merged_local_states {0};
    idx_t output_offset = 0;

    FnGlobalState() {
        std::fill_n(counts, 65536, 0);
    }

    idx_t MaxThreads() const override {
        return std::numeric_limits<idx_t>::max();
    }
};

static unique_ptr<GlobalTableFunctionState> FnInit(ClientContext &, TableFunctionInitInput &) {
    return make_uniq<FnGlobalState>();
}

struct FnLocalState : public LocalTableFunctionState {
    int64_t counts[65536];
    bool merged = false;
    bool is_emitting = false;

    FnLocalState() {
        std::fill_n(counts, 65536, 0);
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
    const bool AdvEngineID_all_valid = valid_AdvEngineID.AllValid();
    
    if (AdvEngineID_all_valid) {
        for (idx_t row_idx = 0; row_idx < input.size(); ++row_idx) {
            idx_t i_AdvEngineID = AdvEngineID_uvf.sel->get_index(row_idx);
            int16_t v_AdvEngineID = AdvEngineID_ptr[i_AdvEngineID];
            l.counts[(uint16_t)v_AdvEngineID]++;
        }
    } else {
        for (idx_t row_idx = 0; row_idx < input.size(); ++row_idx) {
            idx_t i_AdvEngineID = AdvEngineID_uvf.sel->get_index(row_idx);
            if (valid_AdvEngineID.RowIsValid(i_AdvEngineID)) {
                int16_t v_AdvEngineID = AdvEngineID_ptr[i_AdvEngineID];
                l.counts[(uint16_t)v_AdvEngineID]++;
            }
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
            for (int i = 0; i < 65536; ++i) {
                g.counts[i] += l.counts[i];
            }
            l.merged = true;
            if (g.merged_local_states.fetch_add(1) == g.active_local_states.load() - 1) {
                l.is_emitting = true;
            }
        }
    }

    if (l.is_emitting) {
        {
            std::lock_guard<std::mutex> guard(g.lock);
            if (!g.sort_state.sorted) {
                for (int i = 0; i < 65536; ++i) {
                    if (g.counts[i] > 0) {
                        g.sort_state.AddRow((int16_t)(uint16_t)i, g.counts[i]);
                    }
                }
                g.sort_state.SortNow();
            }
        }

        idx_t start = g.output_offset;
        idx_t end = std::min(start + (idx_t)STANDARD_VECTOR_SIZE, (idx_t)g.sort_state.rows.size());
        idx_t output_count = 0;
        
        auto res_id = FlatVector::GetData<int16_t>(out.data[0]);
        auto res_cnt = FlatVector::GetData<int64_t>(out.data[1]);
        
        for (idx_t i = start; i < end; ++i) {
            const auto &row = g.sort_state.rows[i];
            res_id[output_count] = row.id;
            res_cnt[output_count] = row.cnt;
            output_count++;
        }
        out.SetCardinality(output_count);
        g.output_offset = end;
        if (g.output_offset < g.sort_state.rows.size()) {
            return OperatorFinalizeResultType::HAVE_MORE_OUTPUT;
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
