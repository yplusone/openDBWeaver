/*
query_template: SELECT AVG(UserID) AS avg_userid FROM hits;

split_template: select * from dbweaver((SELECT UserID FROM hits));
query_example: SELECT AVG(UserID) AS avg_userid FROM hits;

split_query: select * from dbweaver((SELECT UserID FROM hits));
*/
#define DUCKDB_EXTENSION_MAIN

#include "dbweaver_extension.hpp"
#include "duckdb.hpp"
#include "duckdb/function/table_function.hpp"
#include <atomic>
#include <limits>
#include <mutex>
#include <immintrin.h>

namespace duckdb {

            struct FnBindData : public FunctionData {
                unique_ptr<FunctionData> Copy() const override { return make_uniq<FnBindData>(); }
                bool Equals(const FunctionData &) const override { return true; }
            };

            struct FnGlobalState : public GlobalTableFunctionState {
                __int128 sum = 0;
                idx_t count = 0;
                
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
                __int128 sum = 0;
                idx_t count = 0;
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
                return_types.push_back(LogicalType::DOUBLE);
                names.push_back("avg_userid");
                return make_uniq<FnBindData>();
            }

            // Helper implementing signed 128-bit summation using AVX-512.
            // Doubling throughput with 8x int64_t lanes per instruction and 4x unrolling (32 lanes total).
            __attribute__((target("avx512f")))
            static void SumAVX512(const int64_t* data, idx_t size, idx_t &row_idx, __int128 &total_sum) {
                const __m512i zero = _mm512_setzero_si512();
                const __m512i one = _mm512_set1_epi64(1);

                __m512i l0 = zero, l1 = zero, l2 = zero, l3 = zero;
                __m512i h0 = zero, h1 = zero, h2 = zero, h3 = zero;

                for (; row_idx + 32 <= size; row_idx += 32) {
                    __m512i v0 = _mm512_loadu_si512((const __m512i*)(data + row_idx + 0));
                    __m512i v1 = _mm512_loadu_si512((const __m512i*)(data + row_idx + 8));
                    __m512i v2 = _mm512_loadu_si512((const __m512i*)(data + row_idx + 16));
                    __m512i v3 = _mm512_loadu_si512((const __m512i*)(data + row_idx + 24));

                    // Accumulate 128-bit sums by tracking low 64-bit and high 64-bit components
                    // Sign extension: if v < 0, high = high - 1
                    h0 = _mm512_mask_sub_epi64(h0, _mm512_cmp_epi64_mask(v0, zero, _MM_CMPINT_LT), h0, one);
                    __m512i l0_new = _mm512_add_epi64(l0, v0);
                    // Carry detection: if l_new < v (unsigned), then high = high + 1
                    h0 = _mm512_mask_add_epi64(h0, _mm512_cmp_epu64_mask(l0_new, v0, _MM_CMPINT_LT), h0, one);
                    l0 = l0_new;

                    h1 = _mm512_mask_sub_epi64(h1, _mm512_cmp_epi64_mask(v1, zero, _MM_CMPINT_LT), h1, one);
                    __m512i l1_new = _mm512_add_epi64(l1, v1);
                    h1 = _mm512_mask_add_epi64(h1, _mm512_cmp_epu64_mask(l1_new, v1, _MM_CMPINT_LT), h1, one);
                    l1 = l1_new;

                    h2 = _mm512_mask_sub_epi64(h2, _mm512_cmp_epi64_mask(v2, zero, _MM_CMPINT_LT), h2, one);
                    __m512i l2_new = _mm512_add_epi64(l2, v2);
                    h2 = _mm512_mask_add_epi64(h2, _mm512_cmp_epu64_mask(l2_new, v2, _MM_CMPINT_LT), h2, one);
                    l2 = l2_new;

                    h3 = _mm512_mask_sub_epi64(h3, _mm512_cmp_epi64_mask(v3, zero, _MM_CMPINT_LT), h3, one);
                    __m512i l3_new = _mm512_add_epi64(l3, v3);
                    h3 = _mm512_mask_add_epi64(h3, _mm512_cmp_epu64_mask(l3_new, v3, _MM_CMPINT_LT), h3, one);
                    l3 = l3_new;
                }

                for (; row_idx + 8 <= size; row_idx += 8) {
                    __m512i v = _mm512_loadu_si512((const __m512i*)(data + row_idx));
                    h0 = _mm512_mask_sub_epi64(h0, _mm512_cmp_epi64_mask(v, zero, _MM_CMPINT_LT), h0, one);
                    __m512i l_new = _mm512_add_epi64(l0, v);
                    h0 = _mm512_mask_add_epi64(h0, _mm512_cmp_epu64_mask(l_new, v, _MM_CMPINT_LT), h0, one);
                    l0 = l_new;
                }

                alignas(64) int64_t lr[32];
                alignas(64) int64_t hr[32];
                _mm512_storeu_si512((__m512i*)(lr+0), l0); _mm512_storeu_si512((__m512i*)(hr+0), h0);
                _mm512_storeu_si512((__m512i*)(lr+8), l1); _mm512_storeu_si512((__m512i*)(hr+8), h1);
                _mm512_storeu_si512((__m512i*)(lr+16), l2); _mm512_storeu_si512((__m512i*)(hr+16), h2);
                _mm512_storeu_si512((__m512i*)(lr+24), l3); _mm512_storeu_si512((__m512i*)(hr+24), h3);

                for (int i = 0; i < 32; ++i) {
                    total_sum += ((__int128)hr[i] << 64) + (unsigned __int128)(uint64_t)lr[i];
                }
            }

            // Helper implementing signed 128-bit summation using AVX2.
            __attribute__((target("avx2")))
            static void SumAVX2(const int64_t* data, idx_t size, idx_t &row_idx, __int128 &total_sum) {
                const __m256i zero = _mm256_setzero_si256();
                const __m256i msb = _mm256_set1_epi64x((int64_t)0x8000000000000000ULL);
                
                __m256i l0 = zero, l1 = zero, l2 = zero, l3 = zero;
                __m256i h0 = zero, h1 = zero, h2 = zero, h3 = zero;

                for (; row_idx + 16 <= size; row_idx += 16) {
                    __m256i v0 = _mm256_loadu_si256((const __m256i*)(data + row_idx + 0));
                    __m256i v1 = _mm256_loadu_si256((const __m256i*)(data + row_idx + 4));
                    __m256i v2 = _mm256_loadu_si256((const __m256i*)(data + row_idx + 8));
                    __m256i v3 = _mm256_loadu_si256((const __m256i*)(data + row_idx + 12));

                    h0 = _mm256_add_epi64(h0, _mm256_cmpgt_epi64(zero, v0));
                    l0 = _mm256_add_epi64(l0, v0);
                    h0 = _mm256_sub_epi64(h0, _mm256_cmpgt_epi64(_mm256_xor_si256(v0, msb), _mm256_xor_si256(l0, msb)));

                    h1 = _mm256_add_epi64(h1, _mm256_cmpgt_epi64(zero, v1));
                    l1 = _mm256_add_epi64(l1, v1);
                    h1 = _mm256_sub_epi64(h1, _mm256_cmpgt_epi64(_mm256_xor_si256(v1, msb), _mm256_xor_si256(l1, msb)));

                    h2 = _mm256_add_epi64(h2, _mm256_cmpgt_epi64(zero, v2));
                    l2 = _mm256_add_epi64(l2, v2);
                    h2 = _mm256_sub_epi64(h2, _mm256_cmpgt_epi64(_mm256_xor_si256(v2, msb), _mm256_xor_si256(l2, msb)));

                    h3 = _mm256_add_epi64(h3, _mm256_cmpgt_epi64(zero, v3));
                    l3 = _mm256_add_epi64(l3, v3);
                    h3 = _mm256_sub_epi64(h3, _mm256_cmpgt_epi64(_mm256_xor_si256(v3, msb), _mm256_xor_si256(l3, msb)));
                }

                for (; row_idx + 4 <= size; row_idx += 4) {
                    __m256i v = _mm256_loadu_si256((const __m256i*)(data + row_idx));
                    h0 = _mm256_add_epi64(h0, _mm256_cmpgt_epi64(zero, v));
                    l0 = _mm256_add_epi64(l0, v);
                    h0 = _mm256_sub_epi64(h0, _mm256_cmpgt_epi64(_mm256_xor_si256(v, msb), _mm256_xor_si256(l0, msb)));
                }

                alignas(32) int64_t lr[16];
                alignas(32) int64_t hr[16];
                _mm256_storeu_si256((__m256i*)(lr+0), l0); _mm256_storeu_si256((__m256i*)(hr+0), h0);
                _mm256_storeu_si256((__m256i*)(lr+4), l1); _mm256_storeu_si256((__m256i*)(hr+4), h1);
                _mm256_storeu_si256((__m256i*)(lr+8), l2); _mm256_storeu_si256((__m256i*)(hr+8), h2);
                _mm256_storeu_si256((__m256i*)(lr+12), l3); _mm256_storeu_si256((__m256i*)(hr+12), h3);

                for (int i = 0; i < 16; ++i) {
                    total_sum += ((__int128)hr[i] << 64) + (unsigned __int128)(uint64_t)lr[i];
                }
            }

            static OperatorResultType FnExecute(ExecutionContext &, TableFunctionInput &in,
                                                DataChunk &input, DataChunk &) {
                auto &l = in.local_state->Cast<FnLocalState>();
                const idx_t size = input.size();
                if (size == 0) return OperatorResultType::NEED_MORE_INPUT;

                UnifiedVectorFormat UserID_uvf;
                input.data[0].ToUnifiedFormat(size, UserID_uvf);
                int64_t* UserID_ptr = (int64_t*)UserID_uvf.data;
                const bool all_valid = UserID_uvf.validity.AllValid();
                
                if (all_valid) {
                    idx_t row_idx = 0;
                    if (input.data[0].GetVectorType() == VectorType::FLAT_VECTOR) {
                        if (__builtin_cpu_supports("avx512f")) {
                            SumAVX512(UserID_ptr, size, row_idx, l.sum);
                        } else if (__builtin_cpu_supports("avx2")) {
                            SumAVX2(UserID_ptr, size, row_idx, l.sum);
                        }
                    }
                    // Scalar fallback/tail loop
                    for (; row_idx < size; ++row_idx) {
                        l.sum += UserID_ptr[UserID_uvf.sel->get_index(row_idx)];
                    }
                    l.count += size;
                } else {
                    for (idx_t row_idx = 0; row_idx < size; ++row_idx) {
                        idx_t i_UserID = UserID_uvf.sel->get_index(row_idx);
                        if (UserID_uvf.validity.RowIsValid(i_UserID)) {
                            l.sum += UserID_ptr[i_UserID];
                            l.count++;
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
                    std::lock_guard<std::mutex> guard(g.lock);
                    g.sum += l.sum;
                    g.count += l.count;
                    l.merged = true;
                    g.merged_local_states.fetch_add(1, std::memory_order_relaxed);
                }
                const auto merged = g.merged_local_states.load(std::memory_order_relaxed);
                const auto active = g.active_local_states.load(std::memory_order_relaxed);
                if (active > 0 && merged == active) {
                    double final_avg = 0.0;
                    std::lock_guard<std::mutex> guard(g.lock);
                    if (g.count > 0) final_avg = (double)g.sum / g.count;
                    out.SetCardinality(1);
                    out.SetValue(0, 0, Value::DOUBLE(final_avg));
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
