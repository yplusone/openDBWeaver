/*
query_template: SELECT ClientIP, ClientIP - 1, ClientIP - 2, ClientIP - 3, COUNT(*) AS c FROM hits GROUP BY ClientIP, ClientIP - 1, ClientIP - 2, ClientIP - 3 ORDER BY c DESC LIMIT 10;

split_template: select * from dbweaver((SELECT ClientIP FROM hits));
query_example: SELECT ClientIP, ClientIP - 1, ClientIP - 2, ClientIP - 3, COUNT(*) AS c FROM hits GROUP BY ClientIP, ClientIP - 1, ClientIP - 2, ClientIP - 3 ORDER BY c DESC LIMIT 10;

split_query: select * from dbweaver((SELECT ClientIP FROM hits));
*/
#define DUCKDB_EXTENSION_MAIN

#include "dbweaver_extension.hpp"
#include "duckdb.hpp"
#include "duckdb/function/table_function.hpp"
#include <atomic>
#include <limits>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <algorithm>

namespace duckdb {

struct FnBindData : public FunctionData {
    unique_ptr<FunctionData> Copy() const override { return make_uniq<FnBindData>(); }
    bool Equals(const FunctionData &) const override { return true; }
};

struct GroupKey {
    int32_t client_ip;
    int32_t client_ip_minus_1;
    int32_t client_ip_minus_2;
    int32_t client_ip_minus_3;
    
    bool operator==(const GroupKey& other) const {
        return client_ip == other.client_ip && 
               client_ip_minus_1 == other.client_ip_minus_1 &&
               client_ip_minus_2 == other.client_ip_minus_2 &&
               client_ip_minus_3 == other.client_ip_minus_3;
    }
};

struct GroupKeyHash {
    size_t operator()(const GroupKey& k) const {
        size_t h = 0;
        h ^= std::hash<int32_t>{}(k.client_ip) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int32_t>{}(k.client_ip_minus_1) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int32_t>{}(k.client_ip_minus_2) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int32_t>{}(k.client_ip_minus_3) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

struct AggState {
    int64_t count_star = 0;
};

struct SortKeyView {
    int64_t c;
};

struct SortRowComparator {
    bool operator()(const SortKeyView &a, const SortKeyView &b) const {
        // Sort by 'c' column in descending order
        if (a.c != b.c) return a.c > b.c; // Descending order
        return false; // For stable sort we could add tie-breaker here
    }
};

struct SortStateEntry {
    GroupKey key;
    AggState state;
    SortKeyView sort_key;
};

struct SortState {
    std::vector<SortStateEntry> buffer;
    bool sorted = false;
    
    inline void AddRow(const GroupKey &group_key, const AggState &agg_state) {
        SortStateEntry entry;
        entry.key = group_key;
        entry.state = agg_state;
        entry.sort_key.c = agg_state.count_star;
        buffer.push_back(entry);
    }
    
    inline void SortNow() {
        if (!sorted) {
            std::sort(buffer.begin(), buffer.end(), [this](const SortStateEntry &a, const SortStateEntry &b) {
                return SortRowComparator{}(a.sort_key, b.sort_key);
            });
            sorted = true;
        }
    }
};

struct FnGlobalState : public GlobalTableFunctionState {
    // TODO: Optional accumulators/counters (generator may append to this struct)
    std::mutex lock;
    std::atomic<idx_t> active_local_states {0};
    std::atomic<idx_t> merged_local_states {0};
    std::unordered_map<GroupKey, AggState, GroupKeyHash> agg_map;
    SortState sort_state;
    idx_t MaxThreads() const override {
        return std::numeric_limits<idx_t>::max();
    }
};

static unique_ptr<GlobalTableFunctionState> FnInit(ClientContext &, TableFunctionInitInput &) {
    return make_uniq<FnGlobalState>();
}

struct FnLocalState : public LocalTableFunctionState {
    //TODO: initialize local state and other preparations
    bool merged = false;
    std::unordered_map<GroupKey, AggState, GroupKeyHash> agg_map;
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
    // Set up return types and names according to the specified schema
    return_types.emplace_back(LogicalType::INTEGER);
    return_types.emplace_back(LogicalType::INTEGER);
    return_types.emplace_back(LogicalType::INTEGER);
    return_types.emplace_back(LogicalType::INTEGER);
    return_types.emplace_back(LogicalType::BIGINT);
    
    names.emplace_back("ClientIP");
    names.emplace_back("(ClientIP - 1)");
    names.emplace_back("(ClientIP - 2)");
    names.emplace_back("(ClientIP - 3)");
    names.emplace_back("c");

    return make_uniq<FnBindData>();
}

static OperatorResultType FnExecute(ExecutionContext &, TableFunctionInput &in,
                                                DataChunk &input, DataChunk &) {
    auto &l = in.local_state->Cast<FnLocalState>();

    
    if (input.size() == 0) {       
        return OperatorResultType::NEED_MORE_INPUT;
    }

    // Process input columns using UnifiedVectorFormat
    UnifiedVectorFormat ClientIP_uvf;
    input.data[0].ToUnifiedFormat(input.size(), ClientIP_uvf);
    int32_t* ClientIP_ptr = (int32_t*)ClientIP_uvf.data;

    // validity bitmaps
    auto &valid_ClientIP = ClientIP_uvf.validity;
    const bool ClientIP_all_valid = valid_ClientIP.AllValid();

    // Fast branch: all relevant columns have no NULLs in this batch
    if (ClientIP_all_valid) {
        // --- Fast path: no per-row NULL checks ---
        for (idx_t row_idx = 0; row_idx < input.size(); ++row_idx) {
            // Directly load values without RowIsValid checks

            idx_t i_ClientIP = ClientIP_uvf.sel->get_index(row_idx);
            int32_t v1 = ClientIP_ptr[i_ClientIP];
            //  vN = <COL_N>_ptr[i_<COL_N>];

            // ======================================
            //  Core computation logic (no NULLs)
            //<<CORE_COMPUTE>>
            GroupKey key;
            key.client_ip = v1;
            key.client_ip_minus_1 = v1 - 1;
            key.client_ip_minus_2 = v1 - 2;
            key.client_ip_minus_3 = v1 - 3;
            
            auto &state = l.agg_map[key];
            state.count_star++;
            // ============================
        }
    } else {
        // --- Slow path: at least one column may contain NULLs ---
        for (idx_t row_idx = 0; row_idx < input.size(); ++row_idx) {
            // For each column that is not fully valid, check this row
            idx_t i_ClientIP = ClientIP_uvf.sel->get_index(row_idx);

            if (!ClientIP_all_valid && !valid_ClientIP.RowIsValid(i_ClientIP)) {
                continue; // row is NULL in column 1 → skip
            }
            // Repeat for additional columns

            // At this point, all required columns are valid for this row

            int32_t v1 = ClientIP_ptr[i_ClientIP];
            // auto vN = <COL_N>_ptr[i_<COL_N>];

            // ======================================
            //  Core computation logic (NULL-safe)
            //<<CORE_COMPUTE>>
            GroupKey key;
            key.client_ip = v1;
            key.client_ip_minus_1 = v1 - 1;
            key.client_ip_minus_2 = v1 - 2;
            key.client_ip_minus_3 = v1 - 3;
            
            auto &state = l.agg_map[key];
            state.count_star++;
            // ======================================
        }
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
            //TODO: merge local state with global state
            for (const auto &entry : l.agg_map) {
                const GroupKey &key = entry.first;
                const AggState &state = entry.second;
                g.agg_map[key].count_star += state.count_star;
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
        // Populate the sort buffer with data from global agg_map
        for (const auto &entry : g.agg_map) {
            const GroupKey &key = entry.first;
            const AggState &state = entry.second;
            g.sort_state.AddRow(key, state);
        }
        
        // Perform the sort operation
        g.sort_state.SortNow();
        
        {
            std::lock_guard<std::mutex> guard(g.lock);
            // Output the sorted results
            idx_t output_row = 0;
            for (size_t i = 0; i < g.sort_state.buffer.size(); ++i) {
                const auto &entry = g.sort_state.buffer[i];
                const GroupKey &key = entry.key;
                const AggState &state = entry.state;
                
                if(output_row >= STANDARD_VECTOR_SIZE) {
                    out.SetCardinality(output_row);
                    return OperatorFinalizeResultType::HAVE_RESULT;
                }
                
                out.SetValue(0, output_row, Value::INTEGER(key.client_ip));
                out.SetValue(1, output_row, Value::INTEGER(key.client_ip_minus_1));
                out.SetValue(2, output_row, Value::INTEGER(key.client_ip_minus_2));
                out.SetValue(3, output_row, Value::INTEGER(key.client_ip_minus_3));
                out.SetValue(4, output_row, Value::BIGINT(state.count_star));
                output_row++;
            }
            out.SetCardinality(output_row);
        }
        //TODO: populate out chunk with final results
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