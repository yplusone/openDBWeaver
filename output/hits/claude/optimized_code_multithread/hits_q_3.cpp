#define DUCKDB_EXTENSION_MAIN

#include "dbweaver_extension.hpp"
#include "duckdb.hpp"
#include "duckdb/function/table_function.hpp"
#include <atomic>
#include <limits>
#include <mutex>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace duckdb {

struct FnBindData : public FunctionData {
    unique_ptr<FunctionData> Copy() const override { return make_uniq<FnBindData>(); }
    bool Equals(const FunctionData &) const override { return true; }
};

struct FnGlobalState : public GlobalTableFunctionState {
    std::mutex lock;
    std::atomic<idx_t> active_local_states {0};
    std::atomic<idx_t> merged_local_states {0};
    idx_t MaxThreads() const override {
        return std::numeric_limits<idx_t>::max();
    }
    // Aggregation accumulators for single-group aggregate
    int64_t sum_advengineid = 0;
    idx_t cnt = 0;
    int64_t sum_resolutionwidth = 0;
};


static unique_ptr<GlobalTableFunctionState> FnInit(ClientContext &, TableFunctionInitInput &) {
    return make_uniq<FnGlobalState>();
}
struct FnLocalState : public LocalTableFunctionState {
    // Aggregation accumulators for single-group aggregate
    int64_t sum_advengineid = 0;
    idx_t cnt = 0;
    int64_t sum_resolutionwidth = 0;
    
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
    return_types.push_back(LogicalType::HUGEINT);  // sum_advengineid
    return_types.push_back(LogicalType::BIGINT); // cnt
    return_types.push_back(LogicalType::DOUBLE);  // avg_resolutionwidth
    
    names.push_back("sum_advengineid");
    names.push_back("cnt");
    names.push_back("avg_resolutionwidth");

    return make_uniq<FnBindData>();
}
#if defined(__x86_64__) || defined(_M_X64)
__attribute__((target("avx2")))
static void SumSIMD_AVX2(const int16_t* v1_ptr, const int16_t* v2_ptr, idx_t num_rows, idx_t &i, int64_t &sum1, int64_t &sum2) {
    __m256i sum_v1_a = _mm256_setzero_si256();
    __m256i sum_v1_b = _mm256_setzero_si256();
    __m256i sum_v1_c = _mm256_setzero_si256();
    __m256i sum_v1_d = _mm256_setzero_si256();

    __m256i sum_v2_a = _mm256_setzero_si256();
    __m256i sum_v2_b = _mm256_setzero_si256();
    __m256i sum_v2_c = _mm256_setzero_si256();
    __m256i sum_v2_d = _mm256_setzero_si256();

    // Unrolled 4x loop (64 items per iteration)
    for (; i + 63 < num_rows; i += 64) {
        __m256i v1_0 = _mm256_loadu_si256((const __m256i *)(v1_ptr + i));
        __m256i v1_1 = _mm256_loadu_si256((const __m256i *)(v1_ptr + i + 16));
        __m256i v1_2 = _mm256_loadu_si256((const __m256i *)(v1_ptr + i + 32));
        __m256i v1_3 = _mm256_loadu_si256((const __m256i *)(v1_ptr + i + 48));

        __m256i v2_0 = _mm256_loadu_si256((const __m256i *)(v2_ptr + i));
        __m256i v2_1 = _mm256_loadu_si256((const __m256i *)(v2_ptr + i + 16));
        __m256i v2_2 = _mm256_loadu_si256((const __m256i *)(v2_ptr + i + 32));
        __m256i v2_3 = _mm256_loadu_si256((const __m256i *)(v2_ptr + i + 48));

        sum_v1_a = _mm256_add_epi32(sum_v1_a, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(v1_0)));
        sum_v1_a = _mm256_add_epi32(sum_v1_a, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(v1_0, 1)));
        sum_v1_b = _mm256_add_epi32(sum_v1_b, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(v1_1)));
        sum_v1_b = _mm256_add_epi32(sum_v1_b, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(v1_1, 1)));
        sum_v1_c = _mm256_add_epi32(sum_v1_c, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(v1_2)));
        sum_v1_c = _mm256_add_epi32(sum_v1_c, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(v1_2, 1)));
        sum_v1_d = _mm256_add_epi32(sum_v1_d, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(v1_3)));
        sum_v1_d = _mm256_add_epi32(sum_v1_d, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(v1_3, 1)));

        sum_v2_a = _mm256_add_epi32(sum_v2_a, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(v2_0)));
        sum_v2_a = _mm256_add_epi32(sum_v2_a, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(v2_0, 1)));
        sum_v2_b = _mm256_add_epi32(sum_v2_b, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(v2_1)));
        sum_v2_b = _mm256_add_epi32(sum_v2_b, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(v2_1, 1)));
        sum_v2_c = _mm256_add_epi32(sum_v2_c, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(v2_2)));
        sum_v2_c = _mm256_add_epi32(sum_v2_c, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(v2_2, 1)));
        sum_v2_d = _mm256_add_epi32(sum_v2_d, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(v2_3)));
        sum_v2_d = _mm256_add_epi32(sum_v2_d, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(v2_3, 1)));
    }

    // Handle remainder blocks of 16
    for (; i + 15 < num_rows; i += 16) {
        __m256i v1_16 = _mm256_loadu_si256((const __m256i *)(v1_ptr + i));
        __m256i v2_16 = _mm256_loadu_si256((const __m256i *)(v2_ptr + i));
        sum_v1_a = _mm256_add_epi32(sum_v1_a, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(v1_16)));
        sum_v1_a = _mm256_add_epi32(sum_v1_a, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(v1_16, 1)));
        sum_v2_a = _mm256_add_epi32(sum_v2_a, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(v2_16)));
        sum_v2_a = _mm256_add_epi32(sum_v2_a, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(v2_16, 1)));
    }

    // Reduce accumulators
    sum_v1_a = _mm256_add_epi32(_mm256_add_epi32(sum_v1_a, sum_v1_b), _mm256_add_epi32(sum_v1_c, sum_v1_d));
    sum_v2_a = _mm256_add_epi32(_mm256_add_epi32(sum_v2_a, sum_v2_b), _mm256_add_epi32(sum_v2_c, sum_v2_d));

    int32_t buffer1[8];
    int32_t buffer2[8];
    _mm256_storeu_si256((__m256i *)buffer1, sum_v1_a);
    _mm256_storeu_si256((__m256i *)buffer2, sum_v2_a);

    for (int k = 0; k < 8; ++k) {
        sum1 += buffer1[k];
        sum2 += buffer2[k];
    }
}

#endif

static OperatorResultType FnExecute(ExecutionContext &, TableFunctionInput &in,
                                    DataChunk &input, DataChunk &) {
    auto &l = in.local_state->Cast<FnLocalState>();

    if (input.size() == 0) {       
        return OperatorResultType::NEED_MORE_INPUT;
    }

    UnifiedVectorFormat AdvEngineID_uvf;
    input.data[0].ToUnifiedFormat(input.size(), AdvEngineID_uvf);
    int16_t* AdvEngineID_ptr = (int16_t*)AdvEngineID_uvf.data;
    
    UnifiedVectorFormat ResolutionWidth_uvf;
    input.data[1].ToUnifiedFormat(input.size(), ResolutionWidth_uvf);
    int16_t* ResolutionWidth_ptr = (int16_t*)ResolutionWidth_uvf.data;
    
    auto &valid_AdvEngineID = AdvEngineID_uvf.validity;
    auto &valid_ResolutionWidth = ResolutionWidth_uvf.validity;
    
    const bool AdvEngineID_all_valid = valid_AdvEngineID.AllValid();
    const bool ResolutionWidth_all_valid = valid_ResolutionWidth.AllValid();
    
    idx_t num_rows = input.size();
    
    if (AdvEngineID_all_valid && ResolutionWidth_all_valid) {
        if (input.data[0].GetVectorType() == VectorType::FLAT_VECTOR && 
            input.data[1].GetVectorType() == VectorType::FLAT_VECTOR) {
            
            idx_t i = 0;
#if defined(__x86_64__) || defined(_M_X64)
            SumSIMD_AVX2(AdvEngineID_ptr, ResolutionWidth_ptr, num_rows, i, l.sum_advengineid, l.sum_resolutionwidth);
#endif
            l.cnt += i;

            for (; i < num_rows; ++i) {
                l.sum_advengineid += AdvEngineID_ptr[i];
                l.sum_resolutionwidth += ResolutionWidth_ptr[i];
                l.cnt++;
            }
        } else {
            for (idx_t row_idx = 0; row_idx < num_rows; ++row_idx) {
                idx_t i_AdvEngineID = AdvEngineID_uvf.sel->get_index(row_idx);
                idx_t i_ResolutionWidth = ResolutionWidth_uvf.sel->get_index(row_idx);
                
                l.sum_advengineid += AdvEngineID_ptr[i_AdvEngineID];
                l.sum_resolutionwidth += ResolutionWidth_ptr[i_ResolutionWidth];
                l.cnt++;
            }
        }
    } else {
        for (idx_t row_idx = 0; row_idx < num_rows; ++row_idx) {
            idx_t i_AdvEngineID = AdvEngineID_uvf.sel->get_index(row_idx);
            idx_t i_ResolutionWidth = ResolutionWidth_uvf.sel->get_index(row_idx);
            
            if (!AdvEngineID_all_valid && !valid_AdvEngineID.RowIsValid(i_AdvEngineID)) {
                continue;
            }
            if (!ResolutionWidth_all_valid && !valid_ResolutionWidth.RowIsValid(i_ResolutionWidth)) {
                continue;
            }
            
            l.sum_advengineid += AdvEngineID_ptr[i_AdvEngineID];
            l.sum_resolutionwidth += ResolutionWidth_ptr[i_ResolutionWidth];
            l.cnt++;
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
            g.sum_advengineid += l.sum_advengineid;
            g.cnt += l.cnt;
            g.sum_resolutionwidth += l.sum_resolutionwidth;
        }
        l.merged = true;
        g.merged_local_states.fetch_add(1, std::memory_order_relaxed);
    }

    const auto merged = g.merged_local_states.load(std::memory_order_relaxed);
    const auto active = g.active_local_states.load(std::memory_order_relaxed);
    if (active > 0 && merged == active) {
        out.SetCardinality(1);
        hugeint_t sum_advengineid_hugeint;
        sum_advengineid_hugeint.lower = static_cast<uint64_t>(g.sum_advengineid);
        sum_advengineid_hugeint.upper = 0;
        out.SetValue(0, 0, Value::HUGEINT(sum_advengineid_hugeint));
        out.SetValue(1, 0, Value::BIGINT(g.cnt));
        double avg_res_width = (g.cnt > 0) ? (static_cast<double>(g.sum_resolutionwidth) / static_cast<double>(g.cnt)) : 0.0;
        out.SetValue(2, 0, Value::DOUBLE(avg_res_width));

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
