/*
query_template: SELECT SearchPhrase, COUNT(*) AS c FROM hits WHERE SearchPhrase <> '' GROUP BY SearchPhrase ORDER BY c DESC LIMIT 10;

split_template: select * from dbweaver((SELECT SearchPhrase FROM hits WHERE (SearchPhrase!='')));
query_example: SELECT SearchPhrase, COUNT(*) AS c FROM hits WHERE SearchPhrase <> '' GROUP BY SearchPhrase ORDER BY c DESC LIMIT 10;

split_query: select * from dbweaver((SELECT SearchPhrase FROM hits WHERE (SearchPhrase!='')));
*/
#define DUCKDB_EXTENSION_MAIN

#include "dbweaver_extension.hpp"
#include "duckdb.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/common/types/string_heap.hpp"
#include <atomic>
#include <limits>
#include <mutex>
#include <absl/container/flat_hash_map.h>

#include <functional>
#include <vector>
#include <algorithm>

namespace duckdb {

struct FnBindData : public FunctionData {
    unique_ptr<FunctionData> Copy() const override { return make_uniq<FnBindData>(); }
    bool Equals(const FunctionData &) const override { return true; }
};

struct StringTHash {
    size_t operator()(const string_t &v) const {
        return duckdb::Hash<string_t>(v);
    }
};

struct StringTEqual {
    bool operator()(const string_t &a, const string_t &b) const {
        return a == b;
    }
};

struct AggState {
    int64_t count_val = 0;
};

struct SortKeyView {
    string_t search_phrase;
    int64_t c;
};

struct SortRowComparator {
    bool operator()(const SortKeyView &a, const SortKeyView &b) const {
        if (a.c != b.c) {
            return a.c > b.c; // DESC order
        }
        return a.search_phrase < b.search_phrase;
    }
};

struct SortState {
    std::vector<SortKeyView> buffer;
    bool sorted = false;
    const idx_t k_limit = 10; // Top-K limit

    inline void AddRow(string_t search_phrase, int64_t c) {
        buffer.push_back(SortKeyView{search_phrase, c});
    }

    inline void SortNow() {
        if (!sorted) {
            if (k_limit != 0 && buffer.size() > k_limit) {
                std::partial_sort(buffer.begin(), 
                                buffer.begin() + std::min((size_t)k_limit, buffer.size()), 
                                buffer.end(), 
                                SortRowComparator{});
                if (buffer.size() > k_limit) {
                    buffer.resize(k_limit);
                }
            } else {
                std::sort(buffer.begin(), buffer.end(), SortRowComparator{});
            }
            sorted = true;
        }
    }
};

typedef absl::flat_hash_map<string_t, AggState, StringTHash, StringTEqual> agg_map_t;

struct FnGlobalState : public GlobalTableFunctionState {
    agg_map_t agg_map;
    StringHeap string_heap;
    std::mutex lock;
    std::atomic<idx_t> active_local_states {0};
    std::atomic<idx_t> merged_local_states {0};
    SortState sort_state;

    idx_t MaxThreads() const override {
        return std::numeric_limits<idx_t>::max();
    }
};

static unique_ptr<GlobalTableFunctionState> FnInit(ClientContext &, TableFunctionInitInput &) {
    return make_uniq<FnGlobalState>();
}

struct FnLocalState : public LocalTableFunctionState {
    bool merged = false;
    agg_map_t agg_map;
    StringHeap string_heap;
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
    return_types.push_back(LogicalType::VARCHAR);
    return_types.push_back(LogicalType::BIGINT);
    names.push_back("SearchPhrase");
    names.push_back("c");
    return make_uniq<FnBindData>();
}

static OperatorResultType FnExecute(ExecutionContext &, TableFunctionInput &in,
                                                    DataChunk &input, DataChunk &) {
    auto &l = in.local_state->Cast<FnLocalState>();
    if (input.size() == 0) {       
        return OperatorResultType::NEED_MORE_INPUT;
    }

    UnifiedVectorFormat SearchPhrase_uvf;
    input.data[0].ToUnifiedFormat(input.size(), SearchPhrase_uvf);
    string_t* SearchPhrase_ptr = (string_t*)SearchPhrase_uvf.data;
    auto &valid_SearchPhrase = SearchPhrase_uvf.validity;
    const bool SearchPhrase_all_valid = valid_SearchPhrase.AllValid();

    for (idx_t row_idx = 0; row_idx < input.size(); ++row_idx) {
        idx_t i_SearchPhrase = SearchPhrase_uvf.sel->get_index(row_idx);
        if (!SearchPhrase_all_valid && !valid_SearchPhrase.RowIsValid(i_SearchPhrase)) {
            continue;
        }

        string_t v1 = SearchPhrase_ptr[i_SearchPhrase];
        if (v1.GetSize() == 0) {
            continue;
        }

        auto it = l.agg_map.find(v1);
        if (it == l.agg_map.end()) {
            string_t stored_v1 = l.string_heap.AddString(v1);
            l.agg_map[stored_v1].count_val = 1;
        } else {
            it->second.count_val++;
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
            for (const auto &entry : l.agg_map) {
                auto it = g.agg_map.find(entry.first);
                if (it == g.agg_map.end()) {
                    string_t global_key = g.string_heap.AddString(entry.first);
                    g.agg_map[global_key].count_val = entry.second.count_val;
                } else {
                    it->second.count_val += entry.second.count_val;
                }
            }
        }
        l.merged = true;
        g.merged_local_states.fetch_add(1, std::memory_order_relaxed);
    }

    const auto merged = g.merged_local_states.load(std::memory_order_relaxed);
    const auto active = g.active_local_states.load(std::memory_order_relaxed);
    if (active > 0 && merged == active) {
        {
            std::lock_guard<std::mutex> guard(g.lock);
            if (!g.sort_state.sorted) {
                for (const auto &entry : g.agg_map) {
                    g.sort_state.AddRow(entry.first, entry.second.count_val);
                }
                g.sort_state.SortNow();
            }
            
            idx_t output_idx = 0;
            for (const auto &item : g.sort_state.buffer) {
                out.SetValue(0, output_idx, Value(item.search_phrase));
                out.SetValue(1, output_idx, Value::BIGINT(item.c));
                output_idx++;
            }
            out.SetCardinality(output_idx);
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
