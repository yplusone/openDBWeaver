/*

query_template: SELECT MIN(EventDate) AS min_eventdate, MAX(EventDate) AS max_eventdate FROM hits;


split_template: SELECT min_eventdate, max_eventdate
FROM dbweaver((
  SELECT EventDate
  FROM hits
));

query_example: SELECT MIN(EventDate) AS min_eventdate, MAX(EventDate) AS max_eventdate FROM hits;


split_query: SELECT min_eventdate, max_eventdate
FROM dbweaver((
  SELECT EventDate
  FROM hits
));

*/
#define DUCKDB_EXTENSION_MAIN

#include "dbweaver_extension.hpp"
#include "duckdb.hpp"
#include "duckdb/function/table_function.hpp"
#include <atomic>
#include <limits>
#include <mutex>

namespace duckdb {

struct FnBindData : public FunctionData {
    unique_ptr<FunctionData> Copy() const override { return make_uniq<FnBindData>(); }
    bool Equals(const FunctionData &) const override { return true; }
};

struct FnGlobalState : public GlobalTableFunctionState {
    // Aggregates for MIN/MAX of EventDate
    bool has_min_value = false;
    date_t min_value;
    bool has_max_value = false;
    date_t max_value;
    
    std::mutex lock;
    std::atomic<idx_t> active_local_states {0};
    std::atomic<idx_t> merged_local_states {0};
    idx_t MaxThreads() const override {
        return std::numeric_limits<idx_t>::max();
    }

    FnGlobalState() {
        min_value.days = std::numeric_limits<int32_t>::max();
        max_value.days = std::numeric_limits<int32_t>::min();
    }
};

static unique_ptr<GlobalTableFunctionState> FnInit(ClientContext &, TableFunctionInitInput &) {
    return make_uniq<FnGlobalState>();
}

struct FnLocalState : public LocalTableFunctionState {
    // Aggregates for MIN/MAX of EventDate
    bool has_min_value = false;
    date_t min_value;
    bool has_max_value = false;
    date_t max_value;
    
    bool merged = false;

    FnLocalState() {
        min_value.days = std::numeric_limits<int32_t>::max();
        max_value.days = std::numeric_limits<int32_t>::min();
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
    // Define output schema based on authoritative mapping
    names.emplace_back("min_eventdate");
    return_types.emplace_back(LogicalType::DATE);
    
    names.emplace_back("max_eventdate");
    return_types.emplace_back(LogicalType::DATE);

    return make_uniq<FnBindData>();
}

static OperatorResultType FnExecute(ExecutionContext &, TableFunctionInput &in, 
                                            DataChunk &input, DataChunk &) {
    auto &l = in.local_state->Cast<FnLocalState>();
    idx_t n = input.size();
    if (n == 0) {       
        return OperatorResultType::NEED_MORE_INPUT;
    }

    auto &vec = input.data[0];
    auto vtype = vec.GetVectorType();

    if (vtype == VectorType::CONSTANT_VECTOR) {
        // O(1) processing for ConstantVector
        if (!ConstantVector::IsNull(vec)) {
            date_t v = *ConstantVector::GetData<date_t>(vec);
            l.min_value = (v < l.min_value) ? v : l.min_value;
            l.max_value = (v > l.max_value) ? v : l.max_value;
            l.has_min_value = true;
            l.has_max_value = true;
        }
    } else if (vtype == VectorType::FLAT_VECTOR && FlatVector::Validity(vec).AllValid()) {
        // Tight loop over raw pointer for FlatVector with no NULLs
        // Uses branchless ternary logic to avoid mispredictions and enable SIMD
        date_t *data = FlatVector::GetData<date_t>(vec);
        date_t cur_min = l.min_value;
        date_t cur_max = l.max_value;

        for (idx_t i = 0; i < n; ++i) {
            date_t v = data[i];
            cur_min = (v < cur_min) ? v : cur_min;
            cur_max = (v > cur_max) ? v : cur_max;
        }

        l.min_value = cur_min;
        l.max_value = cur_max;
        l.has_min_value = true;
        l.has_max_value = true;
    } else {
        // Fallback: Use UnifiedVectorFormat for selection vectors or nulls
        UnifiedVectorFormat uvf;
        vec.ToUnifiedFormat(n, uvf);
        date_t* data_ptr = (date_t*)uvf.data;
        auto &validity = uvf.validity;

        date_t cur_min = l.min_value;
        date_t cur_max = l.max_value;
        bool has_val = l.has_min_value;

        if (validity.AllValid()) {
            for (idx_t row_idx = 0; row_idx < n; ++row_idx) {
                idx_t idx = uvf.sel->get_index(row_idx);
                date_t v = data_ptr[idx];
                cur_min = (v < cur_min) ? v : cur_min;
                cur_max = (v > cur_max) ? v : cur_max;
            }
            has_val = true;
        } else {
            for (idx_t row_idx = 0; row_idx < n; ++row_idx) {
                idx_t idx = uvf.sel->get_index(row_idx);
                if (validity.RowIsValid(idx)) {
                    date_t v = data_ptr[idx];
                    cur_min = (v < cur_min) ? v : cur_min;
                    cur_max = (v > cur_max) ? v : cur_max;
                    has_val = true;
                }
            }
        }
        l.min_value = cur_min;
        l.max_value = cur_max;
        l.has_min_value = has_val;
        l.has_max_value = has_val;
    }

    return OperatorResultType::NEED_MORE_INPUT; 
}

static OperatorFinalizeResultType FnFinalize(ExecutionContext &, TableFunctionInput &in, 
                                                            DataChunk &out) {
    auto &g = *reinterpret_cast<FnGlobalState *>(in.global_state.get());
    auto &l = in.local_state->Cast<FnLocalState>();
    // Merge the local state into the global state exactly once.
    if (!l.merged) {
        {
            std::lock_guard<std::mutex> guard(g.lock);
            // Merge local min/max using branchless ternary logic
            if (l.has_min_value) {
                g.min_value = (l.min_value < g.min_value) ? l.min_value : g.min_value;
                g.max_value = (l.max_value > g.max_value) ? l.max_value : g.max_value;
                g.has_min_value = true;
                g.has_max_value = true;
            }
        }
        l.merged = true;
        g.merged_local_states.fetch_add(1, std::memory_order_relaxed);
    }

    // Only the *last* local state to merge emits the final result.
    // All other threads return FINISHED with an empty chunk.
    const auto merged = g.merged_local_states.load(std::memory_order_relaxed);
    const auto active = g.active_local_states.load(std::memory_order_relaxed);
    if (active > 0 && merged == active) {
        date_t final_min_value;
        date_t final_max_value;
        bool has_min = false;
        bool has_max = false;
        {
            std::lock_guard<std::mutex> guard(g.lock);
            final_min_value = g.min_value;
            final_max_value = g.max_value;
            has_min = g.has_min_value;
            has_max = g.has_max_value;
        }
        out.SetCardinality(1);
        if (has_min) {
            out.SetValue(0, 0, Value::DATE(final_min_value));
        } else {
            out.SetValue(0, 0, Value()); // null value
        }
        if (has_max) {
            out.SetValue(1, 0, Value::DATE(final_max_value));
        } else {
            out.SetValue(1, 0, Value()); // null value
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
