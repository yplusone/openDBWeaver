#define DUCKDB_EXTENSION_MAIN

#include "dbweaver_extension.hpp"
#include "duckdb.hpp"
#include "duckdb/function/table_function.hpp"
#include <atomic>
#include <limits>
#include <mutex>
#include <absl/container/flat_hash_map.h>
#include <absl/hash/hash.h>
#include <absl/strings/string_view.h>
#include <functional>
#include <vector>
#include <algorithm>
#include <string>
#include <cstring>
#include <memory>
#include <cstdint>

namespace duckdb {

struct FnBindData : public FunctionData {
    unique_ptr<FunctionData> Copy() const override { return make_uniq<FnBindData>(); }
    bool Equals(const FunctionData &) const override { return true; }
};

struct StringArena {
    std::vector<std::unique_ptr<char[]>> chunks;
    char *next_ptr = nullptr;
    size_t remaining = 0;
    static constexpr size_t CHUNK_SIZE = 1048576; // 1MB

    char* Allocate(size_t size) {
        if (size > remaining) {
            size_t alloc_size = std::max((size_t)CHUNK_SIZE, size);
            chunks.push_back(std::unique_ptr<char[]>(new char[alloc_size]));
            next_ptr = chunks.back().get();
            remaining = alloc_size;
        }
        char *dest = next_ptr;
        next_ptr += size;
        remaining -= size;
        return dest;
    }
};

struct GroupKey {
    string_t search_phrase;
};

struct GroupKeyHash {
    using is_transparent = void;
    size_t operator()(const GroupKey& k) const {
        return operator()(k.search_phrase);
    }
    size_t operator()(string_t v) const {
        return absl::Hash<absl::string_view>{}(absl::string_view(v.GetData(), v.GetSize()));
    }
};

struct GroupKeyEq {
    using is_transparent = void;
    bool operator()(const GroupKey& lhs, const GroupKey& rhs) const {
        return lhs.search_phrase == rhs.search_phrase;
    }
    bool operator()(const GroupKey& lhs, string_t rhs) const {
        return lhs.search_phrase == rhs;
    }
    bool operator()(string_t lhs, const GroupKey& rhs) const {
        return lhs == rhs.search_phrase;
    }
};

struct AggState {
    int64_t count_val = 0;
    AggState() : count_val(0) {}
    explicit AggState(int64_t v) : count_val(v) {}
};

typedef absl::flat_hash_map<GroupKey, AggState, GroupKeyHash, GroupKeyEq> AggMapType;

struct SortKeyView {
    absl::string_view search_phrase;
    int64_t c;
};

struct SortRowComparator {
    bool operator()(const SortKeyView &a, const SortKeyView &b) const {
        if (a.c != b.c) {
            return a.c > b.c; 
        }
        return a.search_phrase < b.search_phrase;
    }
};

struct FnGlobalState : public GlobalTableFunctionState {
    AggMapType agg_map;
    StringArena arena;

    std::mutex lock;
    std::atomic<idx_t> active_local_states {0};
    std::atomic<idx_t> merged_local_states {0};
    
    idx_t MaxThreads() const override {
        return std::numeric_limits<idx_t>::max();
    }
};

static unique_ptr<GlobalTableFunctionState> FnInit(ClientContext &, TableFunctionInitInput &) {
    return make_uniq<FnGlobalState>();
}

struct TableSlot {
    uint64_t hash;
    string_t key;
    int64_t count;
};

struct FnLocalState : public LocalTableFunctionState {
    bool merged = false;
    std::vector<TableSlot> slots;
    uint64_t capacity;
    uint64_t mask;
    uint64_t num_elements;
    StringArena arena;
    FnLocalState() : capacity(4096), num_elements(0) {
        mask = capacity - 1;
        slots.resize(capacity, {0, string_t(), 0});
    }


    void Add(string_t v1) {
        if (v1.GetSize() == 0) return;
        uint64_t h = absl::Hash<absl::string_view>{}(absl::string_view(v1.GetData(), v1.GetSize()));
        AddWithHash(v1, h);
    }

    void AddWithHash(string_t v1, uint64_t h) {
        uint64_t idx = h & mask;
        while (slots[idx].count > 0) {
            if (slots[idx].hash == h && slots[idx].key == v1) {
                slots[idx].count++;
                return;
            }
            idx = (idx + 1) & mask;
        }
        if (num_elements * 4 >= capacity * 3) {
            Resize();
            AddWithHash(v1, h);
            return;
        }


        slots[idx].hash = h;
        if (!v1.IsInlined()) {
            char* buf = arena.Allocate(v1.GetSize());
            memcpy(buf, v1.GetData(), v1.GetSize());
            slots[idx].key = string_t(buf, v1.GetSize());
        } else {
            slots[idx].key = v1;
        }
        slots[idx].count = 1;
        num_elements++;
    }

    void Resize() {
        uint64_t new_capacity = capacity * 2;
        uint64_t new_mask = new_capacity - 1;
        std::vector<TableSlot> new_slots(new_capacity, {0, string_t(), 0});
        for (uint64_t i = 0; i < capacity; ++i) {
            if (slots[i].count > 0) {
                uint64_t h = slots[i].hash;
                uint64_t idx = h & new_mask;
                while (new_slots[idx].count > 0) {
                    idx = (idx + 1) & new_mask;
                }
                new_slots[idx] = slots[i];
            }
        }
        slots = std::move(new_slots);
        capacity = new_capacity;
        mask = new_mask;
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
    const bool is_identity = !SearchPhrase_uvf.sel->IsSet();
    
    auto process_row = [&](idx_t i) {
        l.Add(SearchPhrase_ptr[i]);
    };

    if (SearchPhrase_all_valid && is_identity) {
        for (idx_t row_idx = 0; row_idx < input.size(); ++row_idx) {
            process_row(row_idx);
        }
    } else if (SearchPhrase_all_valid) {
        for (idx_t row_idx = 0; row_idx < input.size(); ++row_idx) {
            process_row(SearchPhrase_uvf.sel->get_index(row_idx));
        }
    } else if (is_identity) {
        for (idx_t row_idx = 0; row_idx < input.size(); ++row_idx) {
            if (valid_SearchPhrase.RowIsValid(row_idx)) process_row(row_idx);
        }
    } else {
        for (idx_t row_idx = 0; row_idx < input.size(); ++row_idx) {
            idx_t idx = SearchPhrase_uvf.sel->get_index(row_idx);
            if (valid_SearchPhrase.RowIsValid(idx)) process_row(idx);
        }
    }

    return OperatorResultType::NEED_MORE_INPUT; 
}

static OperatorFinalizeResultType FnFinalize(ExecutionContext &, TableFunctionInput &in,
                                                            DataChunk &out) {
    auto &g = in.global_state->Cast<FnGlobalState>();
    auto &l = in.local_state->Cast<FnLocalState>();
    
    bool is_last = false;
    if (!l.merged) {
        {
            std::lock_guard<std::mutex> guard(g.lock);
            for (uint64_t i = 0; i < l.capacity; ++i) {
                if (l.slots[i].count > 0) {
                    const auto &entry = l.slots[i];
                    auto it = g.agg_map.find(entry.key);
                    if (it != g.agg_map.end()) {
                        it->second.count_val += entry.count;
                    } else {
                        string_t st = entry.key;
                        if (!st.IsInlined()) {
                            char* buf = g.arena.Allocate(st.GetSize());
                            memcpy(buf, st.GetData(), st.GetSize());
                            st = string_t(buf, st.GetSize());
                        }
                        g.agg_map.emplace(GroupKey{st}, AggState(entry.count));
                    }
                }
            }
        }
        l.merged = true;
        if (g.merged_local_states.fetch_add(1) == g.active_local_states.load() - 1) {
            is_last = true;
        }
    }

    if (is_last) {
        std::lock_guard<std::mutex> guard(g.lock);
        if (g.agg_map.empty()) {
            out.SetCardinality(0);
            return OperatorFinalizeResultType::FINISHED;
        }

        std::vector<SortKeyView> sort_buffer;
        sort_buffer.reserve(g.agg_map.size());
        for (const auto &entry : g.agg_map) {
            sort_buffer.push_back({absl::string_view(entry.first.search_phrase.GetData(), entry.first.search_phrase.GetSize()), entry.second.count_val});
        }

        const idx_t k_limit = 10;
        idx_t k = std::min((idx_t)k_limit, (idx_t)sort_buffer.size());
        std::partial_sort(sort_buffer.begin(), sort_buffer.begin() + k, sort_buffer.end(), SortRowComparator{});
        
        auto search_phrase_data = FlatVector::GetData<string_t>(out.data[0]);
        auto c_data = FlatVector::GetData<int64_t>(out.data[1]);
        for (idx_t i = 0; i < k; ++i) {
            const auto &item = sort_buffer[i];
            search_phrase_data[i] = StringVector::AddString(out.data[0], item.search_phrase.data(), item.search_phrase.size());
            c_data[i] = item.c;
        }
        out.SetCardinality(k);
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
