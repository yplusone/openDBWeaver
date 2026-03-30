#define DUCKDB_EXTENSION_MAIN

#include "dbweaver_extension.hpp"
#include "duckdb.hpp"
#include "duckdb/function/table_function.hpp"
#include <atomic>
#include <limits>
#include <mutex>

#if defined(__AVX2__) || defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace duckdb {

struct FnBindData : public FunctionData {
    unique_ptr<FunctionData> Copy() const override { return make_uniq<FnBindData>(); }
    bool Equals(const FunctionData &) const override { return true; }
};

struct FnGlobalState : public GlobalTableFunctionState {
    int64_t base_sum = 0;
    int64_t non_null_count = 0;
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
    int64_t base_sum = 0;
    int64_t non_null_count = 0;
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

static OperatorResultType FnExecute(ExecutionContext &, TableFunctionInput &in,
                                                    DataChunk &input, DataChunk &) {
    auto &l = in.local_state->Cast<FnLocalState>();

    if (input.size() == 0) {       
        return OperatorResultType::NEED_MORE_INPUT;
    }

    UnifiedVectorFormat ResolutionWidth_uvf;
    input.data[0].ToUnifiedFormat(input.size(), ResolutionWidth_uvf);
    int16_t* ResolutionWidth_ptr = (int16_t*)ResolutionWidth_uvf.data;
    auto &valid_ResolutionWidth = ResolutionWidth_uvf.validity;
    const bool ResolutionWidth_all_valid = valid_ResolutionWidth.AllValid();
    
    auto vector_type = input.data[0].GetVectorType();

    if (vector_type == VectorType::CONSTANT_VECTOR) {
        if (ResolutionWidth_all_valid || valid_ResolutionWidth.RowIsValid(0)) {
            l.base_sum += static_cast<int64_t>(ResolutionWidth_ptr[0]) * static_cast<int64_t>(input.size());
            l.non_null_count += input.size();
        }
    } else if (ResolutionWidth_all_valid && vector_type == VectorType::FLAT_VECTOR) {
        l.non_null_count += input.size();
        idx_t row_idx = 0;
        idx_t size = input.size();

#if defined(__AVX2__)
        __m256i sum_64 = _mm256_setzero_si256();
        __m256i ones = _mm256_set1_epi16(1);

        for (; row_idx + 15 < size; row_idx += 16) {
            // Load 16 int16_t values (256 bits)
            __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ResolutionWidth_ptr + row_idx));
            // _mm256_madd_epi16: pairs (v[i], v[i+1]) * (1, 1) -> 8 int32_t sums
            __m256i v32 = _mm256_madd_epi16(v, ones);
            // Convert 8 int32_t to 8 int64_t (two 256-bit vectors)
            __m256i v64_0 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(v32));
            __m256i v64_1 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(v32, 1));
            // Accumulate
            sum_64 = _mm256_add_epi64(sum_64, v64_0);
            sum_64 = _mm256_add_epi64(sum_64, v64_1);
        }

        alignas(32) int64_t sums[4];
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(sums), sum_64);
        l.base_sum += (sums[0] + sums[1] + sums[2] + sums[3]);
#endif
        // Remainder loop
        for (; row_idx < size; ++row_idx) {
            l.base_sum += static_cast<int64_t>(ResolutionWidth_ptr[row_idx]);
        }
    } else {
        // General path with selection vector and/or nulls
        for (idx_t row_idx = 0; row_idx < input.size(); ++row_idx) {
            idx_t i_ResolutionWidth = ResolutionWidth_uvf.sel->get_index(row_idx);
            if (ResolutionWidth_all_valid || valid_ResolutionWidth.RowIsValid(i_ResolutionWidth)) {
                l.base_sum += static_cast<int64_t>(ResolutionWidth_ptr[i_ResolutionWidth]);
                l.non_null_count++;
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
            g.base_sum += l.base_sum;
            g.non_null_count += l.non_null_count;
        }
        l.merged = true;
        g.merged_local_states.fetch_add(1, std::memory_order_relaxed);
    }

    const auto merged = g.merged_local_states.load(std::memory_order_relaxed);
    const auto active = g.active_local_states.load(std::memory_order_relaxed);
    if (active > 0 && merged == active) {
        out.SetCardinality(1);
        std::lock_guard<std::mutex> guard(g.lock);
        hugeint_t sum_val = g.base_sum;
        hugeint_t count_val = g.non_null_count;
        for (int i = 0; i < 90; i++) {
            hugeint_t result = sum_val + (count_val * i);
            auto data = FlatVector::GetData<hugeint_t>(out.data[i]);
            data[0] = result;
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
