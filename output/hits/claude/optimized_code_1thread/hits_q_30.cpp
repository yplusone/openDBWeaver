#define DUCKDB_EXTENSION_MAIN

#include "dbweaver_extension.hpp"
#include "duckdb.hpp"
#include "duckdb/function/table_function.hpp"
#include <atomic>
#include <limits>
#include <mutex>
#include <immintrin.h>

#if defined(__x86_64__) || defined(_M_X64)
#define DBWEAVER_HAS_AVX2
#endif

namespace duckdb {

struct FnBindData : public FunctionData {
    unique_ptr<FunctionData> Copy() const override { return make_uniq<FnBindData>(); }
    bool Equals(const FunctionData &) const override { return true; }
};

struct FnGlobalState : public GlobalTableFunctionState {
    int64_t sum_base = 0;
    int64_t row_count = 0;
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
    int64_t sum_base = 0;
    int64_t row_count = 0;
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
    for(int i = 0; i < 90; i++) {
        return_types.push_back(LogicalType::HUGEINT);
        names.push_back("s" + std::to_string(i));
    }
    return make_uniq<FnBindData>();
}

#ifdef DBWEAVER_HAS_AVX2
#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx2")))
#endif
static void SumInt16AVX2(const int16_t* data, idx_t size, int64_t& sum_base) {
    idx_t i = 0;
    __m256i acc = _mm256_setzero_si256();
    // Process 16 int16_t values per iteration
    for (; i + 15 < size; i += 16) {
        // Load 16 values (256 bits)
        __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(data + i));
        // Widen lower 8 int16 values to int32 (256 bits)
        __m256i i32_low = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(v));
        // Widen upper 8 int16 values to int32 (256 bits)
        __m256i i32_high = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(v, 1));
        // Accumulate into the vector sum
        acc = _mm256_add_epi32(acc, i32_low);
        acc = _mm256_add_epi32(acc, i32_high);
    }
    
    // Reduce the 8 lanes in the accumulator to scalar sum_base
    int32_t temp[8];
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(temp), acc);
    for (int j = 0; j < 8; ++j) {
        sum_base += (int64_t)temp[j];
    }
    
    // Process tail elements
    for (; i < size; ++i) {
        sum_base += (int64_t)data[i];
    }
}
#endif

static OperatorResultType FnExecute(ExecutionContext &, TableFunctionInput &in,
                                                    DataChunk &input, DataChunk &) {
    auto &l = in.local_state->Cast<FnLocalState>();

    const idx_t input_size = input.size();
    if (input_size == 0) {       
        return OperatorResultType::NEED_MORE_INPUT;
    }

    UnifiedVectorFormat ResolutionWidth_uvf;
    input.data[0].ToUnifiedFormat(input_size, ResolutionWidth_uvf);
    const int16_t* ResolutionWidth_ptr = (const int16_t*)ResolutionWidth_uvf.data;
    
    auto &valid_ResolutionWidth = ResolutionWidth_uvf.validity;
    const bool ResolutionWidth_all_valid = valid_ResolutionWidth.AllValid();
    
    if (ResolutionWidth_all_valid) {
        // Check if the vector is flat, which implies it's contiguous and selection is identity
        if (input.data[0].GetVectorType() == VectorType::FLAT_VECTOR) {
#ifdef DBWEAVER_HAS_AVX2
            SumInt16AVX2(ResolutionWidth_ptr, input_size, l.sum_base);
            l.row_count += input_size;
#else
            for (idx_t i = 0; i < input_size; ++i) {
                l.sum_base += (int64_t)ResolutionWidth_ptr[i];
                l.row_count++;
            }
#endif
        } else {
            // Fallback for non-flat or selected vectors
            for (idx_t row_idx = 0; row_idx < input_size; ++row_idx) {
                idx_t i_ResolutionWidth = ResolutionWidth_uvf.sel->get_index(row_idx);
                int16_t v_ResolutionWidth = ResolutionWidth_ptr[i_ResolutionWidth];
                l.sum_base += (int64_t)v_ResolutionWidth;
                l.row_count++;
            }
        }
    } else {
        // Handle NULLs
        for (idx_t row_idx = 0; row_idx < input_size; ++row_idx) {
            idx_t i_ResolutionWidth = ResolutionWidth_uvf.sel->get_index(row_idx);
            if (!valid_ResolutionWidth.RowIsValid(i_ResolutionWidth)) {
                continue;
            }
            int16_t v_ResolutionWidth = ResolutionWidth_ptr[i_ResolutionWidth];
            l.sum_base += (int64_t)v_ResolutionWidth;
            l.row_count++;
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
            g.sum_base += l.sum_base;
            g.row_count += l.row_count;
        }
        l.merged = true;
        g.merged_local_states.fetch_add(1, std::memory_order_relaxed);
    }

    const auto merged = g.merged_local_states.load(std::memory_order_relaxed);
    const auto active = g.active_local_states.load(std::memory_order_relaxed);
    if (active > 0 && merged == active) {
        out.SetCardinality(1);
        for (int i = 0; i < 90; i++) {
            // SUM(ResolutionWidth + i) = SUM(ResolutionWidth) + i * COUNT(*)
            hugeint_t res = g.sum_base;
            res += (hugeint_t)i * (hugeint_t)g.row_count;
            out.SetValue(i, 0, Value::HUGEINT(res));
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
