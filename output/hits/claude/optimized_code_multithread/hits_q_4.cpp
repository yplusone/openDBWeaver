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

namespace duckdb {

            //TODO: Define any helper structs or functions needed for binding/execution
            
            struct FnBindData : public FunctionData {
                unique_ptr<FunctionData> Copy() const override { return make_uniq<FnBindData>(); }
                bool Equals(const FunctionData &) const override { return true; }
            };

            struct FnGlobalState : public GlobalTableFunctionState {
                // Accumulators for AVG aggregation
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
                // Local accumulators for AVG aggregation
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
                    // Define the output schema: avg_userid as DOUBLE
                    return_types.push_back(LogicalType::DOUBLE);
                    names.push_back("avg_userid");

                    return make_uniq<FnBindData>();
                }
            

                static OperatorResultType FnExecute(ExecutionContext &, TableFunctionInput &in,
                                                    DataChunk &input, DataChunk &) {
                    auto &l = in.local_state->Cast<FnLocalState>();

                    
                    if (input.size() == 0) {       
                        return OperatorResultType::NEED_MORE_INPUT;
                    }
                    //TODO: process input chunk and produce output
                    
                    if (input.data[0].GetVectorType() == VectorType::FLAT_VECTOR) {
                        auto data_ptr = FlatVector::GetData<int64_t>(input.data[0]);
                        auto &validity = FlatVector::Validity(input.data[0]);
                        if (validity.AllValid()) {
                            idx_t count = input.size();
                            __int128 local_sum0 = 0;
                            __int128 local_sum1 = 0;
                            __int128 local_sum2 = 0;
                            __int128 local_sum3 = 0;
                            idx_t row_idx = 0;
                            for (; row_idx + 3 < count; row_idx += 4) {
                                local_sum0 += data_ptr[row_idx];
                                local_sum1 += data_ptr[row_idx + 1];
                                local_sum2 += data_ptr[row_idx + 2];
                                local_sum3 += data_ptr[row_idx + 3];
                            }
                            for (; row_idx < count; ++row_idx) {
                                local_sum0 += data_ptr[row_idx];
                            }
                            l.sum += local_sum0 + local_sum1 + local_sum2 + local_sum3;
                            l.count += count;
                        } else {

                            for (idx_t row_idx = 0; row_idx < input.size(); ++row_idx) {
                                bool valid = validity.RowIsValid(row_idx);
                                l.sum += valid ? data_ptr[row_idx] : 0;
                                l.count += valid;
                            }
                        }

                    } else {
                    // Declare UnifiedVectorFormat handles for input columns
                    UnifiedVectorFormat UserID_uvf;

                    
                    // Load columns into UnifiedVectorFormat
                    input.data[0].ToUnifiedFormat(input.size(), UserID_uvf);
                    
                    // Create typed pointers to physical data
                    int64_t* UserID_ptr = (int64_t*)UserID_uvf.data;
                    
                    // validity bitmaps
                    auto &valid_UserID  = UserID_uvf.validity;
                    const bool UserID_all_valid = valid_UserID.AllValid();
                    
                    // Process the input chunk row by row
                    if (UserID_all_valid) {
                        // --- Fast path: no per-row NULL checks ---
                        __int128 local_sum0 = 0;
                        __int128 local_sum1 = 0;
                        __int128 local_sum2 = 0;
                        __int128 local_sum3 = 0;
                        idx_t row_idx = 0;
                        for (; row_idx + 3 < input.size(); row_idx += 4) {
                            local_sum0 += UserID_ptr[UserID_uvf.sel->get_index(row_idx)];
                            local_sum1 += UserID_ptr[UserID_uvf.sel->get_index(row_idx + 1)];
                            local_sum2 += UserID_ptr[UserID_uvf.sel->get_index(row_idx + 2)];
                            local_sum3 += UserID_ptr[UserID_uvf.sel->get_index(row_idx + 3)];
                        }
                        for (; row_idx < input.size(); ++row_idx) {
                            local_sum0 += UserID_ptr[UserID_uvf.sel->get_index(row_idx)];
                        }
                        l.sum += local_sum0 + local_sum1 + local_sum2 + local_sum3;
                        l.count += input.size();
                    } else {

                        // --- Slow path: at least one column may contain NULLs ---
                        for (idx_t row_idx = 0; row_idx < input.size(); ++row_idx) {
                            // For each column that is not fully valid, check this row
                            idx_t i_UserID = UserID_uvf.sel->get_index(row_idx);
                            
                            bool valid = valid_UserID.RowIsValid(i_UserID);
                            int64_t v_UserID = UserID_ptr[i_UserID];
                            
                            // ======================================
                            //  Core computation logic (NULL-safe)
                            //<<CORE_COMPUTE>>
                            l.sum += valid ? v_UserID : 0;
                            l.count += valid;
                            // ======================================
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
                            g.sum += l.sum;
                            g.count += l.count;
                        }
                        l.merged = true;
                        g.merged_local_states.fetch_add(1, std::memory_order_relaxed);
                    }

                    // Only the *last* local state to merge emits the final result.
                    // All other threads return FINISHED with an empty chunk.
                    const auto merged = g.merged_local_states.load(std::memory_order_relaxed);
                    const auto active = g.active_local_states.load(std::memory_order_relaxed);
                    if (active > 0 && merged == active) {
                        double final_avg;
                        {
                            std::lock_guard<std::mutex> guard(g.lock);
                            if (g.count > 0) {
                                final_avg = (double)g.sum / g.count;
                            } else {
                                final_avg = 0.0;  // Handle case where there are no valid entries
                            }
                        }
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