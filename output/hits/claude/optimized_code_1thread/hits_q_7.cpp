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
#include <immintrin.h>
#include <cstring>

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

// Helper for identity selection SIMD path
__attribute__((target("avx2")))
static void ScanMinMaxIdentityAVX2(const date_t* EventDate_ptr, idx_t size, date_t& final_min, date_t& final_max) {
    idx_t row_idx = 0;
    if (size >= 8) {
        // Load 8 dates into AVX2 registers. date_t is 32-bit.
        __m256i min_simd = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(EventDate_ptr));
        __m256i max_simd = min_simd;
        row_idx = 8;
        for (; row_idx + 7 < size; row_idx += 8) {
            __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(EventDate_ptr + row_idx));
            min_simd = _mm256_min_epi32(min_simd, v);
            max_simd = _mm256_max_epi32(max_simd, v);
        }

        // Horizontal reduction for min
        __m128i m_low = _mm256_castsi256_si128(min_simd);
        __m128i m_high = _mm256_extracti128_si256(min_simd, 1);
        __m128i m128 = _mm_min_epi32(m_low, m_high);
        m128 = _mm_min_epi32(m128, _mm_shuffle_epi32(m128, _MM_SHUFFLE(1, 0, 3, 2)));
        m128 = _mm_min_epi32(m128, _mm_shuffle_epi32(m128, _MM_SHUFFLE(0, 0, 0, 1)));
        int32_t r_min = _mm_cvtsi128_si32(m128);
        final_min = date_t{r_min};

        // Horizontal reduction for max
        __m128i x_low = _mm256_castsi256_si128(max_simd);
        __m128i x_high = _mm256_extracti128_si256(max_simd, 1);
        __m128i x128 = _mm_max_epi32(x_low, x_high);
        x128 = _mm_max_epi32(x128, _mm_shuffle_epi32(x128, _MM_SHUFFLE(1, 0, 3, 2)));
        x128 = _mm_max_epi32(x128, _mm_shuffle_epi32(x128, _MM_SHUFFLE(0, 0, 0, 1)));
        int32_t r_max = _mm_cvtsi128_si32(x128);
        final_max = date_t{r_max};
    } else {
        final_min = EventDate_ptr[0];
        final_max = EventDate_ptr[0];
        row_idx = 1;
    }

    // Tail handling
    for (; row_idx < size; ++row_idx) {
        date_t v = EventDate_ptr[row_idx];
        if (v < final_min) final_min = v;
        if (v > final_max) final_max = v;
    }
}

static OperatorResultType FnExecute(ExecutionContext &, TableFunctionInput &in,
                                            DataChunk &input, DataChunk &) {
    auto &l = in.local_state->Cast<FnLocalState>();
    if (input.size() == 0) {
        return OperatorResultType::NEED_MORE_INPUT;
    }

    // Detect ConstantVector to skip per-row iteration entirely
    if (input.data[0].GetVectorType() == VectorType::CONSTANT_VECTOR) {
        if (!ConstantVector::IsNull(input.data[0])) {
            date_t v_EventDate = ConstantVector::GetData<date_t>(input.data[0])[0];
            if (!l.has_min_value || v_EventDate < l.min_value) {
                l.min_value = v_EventDate;
                l.has_min_value = true;
            }
            if (!l.has_max_value || v_EventDate > l.max_value) {
                l.max_value = v_EventDate;
                l.has_max_value = true;
            }
        }
        return OperatorResultType::NEED_MORE_INPUT;
    }

    // Unpack input columns using UnifiedVectorFormat
    UnifiedVectorFormat EventDate_uvf;

    input.data[0].ToUnifiedFormat(input.size(), EventDate_uvf);
    date_t* EventDate_ptr = (date_t*)EventDate_uvf.data;

    // validity bitmaps
    auto &valid_EventDate  = EventDate_uvf.validity;
    const bool EventDate_all_valid = valid_EventDate.AllValid();
    // Process rows in the input chunk
    if (EventDate_all_valid) {
        const idx_t size = input.size();
        const SelectionVector &sel = *EventDate_uvf.sel;
        date_t final_min, final_max;

        // Identity selection: apply AVX2 SIMD
        if (sel.data() == nullptr) {
            ScanMinMaxIdentityAVX2(EventDate_ptr, size, final_min, final_max);
        } else {
            // Non-identity selection: fallback to 4-way scalar unroll for ILP
            const date_t v_init = EventDate_ptr[sel.get_index(0)];
            date_t m0 = v_init, m1 = v_init, m2 = v_init, m3 = v_init;
            date_t x0 = v_init, x1 = v_init, x2 = v_init, x3 = v_init;
            idx_t row_idx = 0;
            for (; row_idx + 3 < size; row_idx += 4) {
                date_t v0 = EventDate_ptr[sel.get_index(row_idx)];
                date_t v1 = EventDate_ptr[sel.get_index(row_idx + 1)];
                date_t v2 = EventDate_ptr[sel.get_index(row_idx + 2)];
                date_t v3 = EventDate_ptr[sel.get_index(row_idx + 3)];
                m0 = (v0 < m0) ? v0 : m0; m1 = (v1 < m1) ? v1 : m1;
                m2 = (v2 < m2) ? v2 : m2; m3 = (v3 < m3) ? v3 : m3;
                x0 = (v0 > x0) ? v0 : x0; x1 = (v1 > x1) ? v1 : x1;
                x2 = (v2 > x2) ? v2 : x2; x3 = (v3 > x3) ? v3 : x3;
            }
            for (; row_idx < size; ++row_idx) {
                date_t v = EventDate_ptr[sel.get_index(row_idx)];
                m0 = (v < m0) ? v : m0; x0 = (v > x0) ? v : x0;
            }
            final_min = (m0 < m1) ? m0 : m1;
            date_t m23 = (m2 < m3) ? m2 : m3;
            final_min = (final_min < m23) ? final_min : m23;

            final_max = (x0 > x1) ? x0 : x1;
            date_t x23 = (x2 > x3) ? x2 : x3;
            final_max = (final_max > x23) ? final_max : x23;
        }

        if (!l.has_min_value || final_min < l.min_value) {
            l.min_value = final_min;
            l.has_min_value = true;
        }
        if (!l.has_max_value || final_max > l.max_value) {
            l.max_value = final_max;
            l.has_max_value = true;
        }
    } else {
        // --- Slow path: at least one column may contain NULLs ---
        for (idx_t row_idx = 0; row_idx < input.size(); ++row_idx) {
            idx_t i_EventDate = EventDate_uvf.sel->get_index(row_idx);

            if (!EventDate_all_valid && !valid_EventDate.RowIsValid(i_EventDate)) {
                continue; // row is NULL in column EventDate → skip
            }

            date_t v_EventDate = EventDate_ptr[i_EventDate];
            if (!l.has_min_value || v_EventDate < l.min_value) {
                l.min_value = v_EventDate;
                l.has_min_value = true;
            }
            if (!l.has_max_value || v_EventDate > l.max_value) {
                l.max_value = v_EventDate;
                l.has_max_value = true;
            }
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
            // Merge local min
            if (l.has_min_value) {
                if (!g.has_min_value || l.min_value < g.min_value) {
                    g.min_value = l.min_value;
                    g.has_min_value = true;
                }
            }
            // Merge local max
            if (l.has_max_value) {
                if (!g.has_max_value || l.max_value > g.max_value) {
                    g.max_value = l.max_value;
                    g.has_max_value = true;
                }
            }
        }
        l.merged = true;
        g.merged_local_states.fetch_add(1, std::memory_order_relaxed);
    }

    // Only the *last* local state to merge emits the final result.
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