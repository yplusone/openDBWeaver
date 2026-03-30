/*
query_template: SELECT SUM(AdvEngineID), COUNT(*), AVG(ResolutionWidth) FROM hits;

split_template: SELECT sum_advengineid, count_star, avg_resolutionwidth
FROM dbweaver((
  SELECT AdvEngineID, ResolutionWidth
  FROM hits
));
query_example: SELECT SUM(AdvEngineID), COUNT(*), AVG(ResolutionWidth) FROM hits;

split_query: SELECT sum_advengineid, count_star, avg_resolutionwidth
FROM dbweaver((
  SELECT AdvEngineID, ResolutionWidth
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

            //TODO: Define any helper structs or functions needed for binding/execution
            
            struct FnBindData : public FunctionData {
                unique_ptr<FunctionData> Copy() const override { return make_uniq<FnBindData>(); }
                bool Equals(const FunctionData &) const override { return true; }
            };

            struct FnGlobalState : public GlobalTableFunctionState {
                // Accumulators for aggregates
                __int128 sum_advengineid = 0;
                int64_t count_star = 0;
                double sum_resolutionwidth = 0.0;
                int64_t count_resolutionwidth = 0;
                
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
                // Local accumulators for aggregates
                __int128 sum_advengineid = 0;
                int64_t count_star = 0;
                double sum_resolutionwidth = 0.0;
                int64_t count_resolutionwidth = 0;
                
                bool merged = false;
            };

            static unique_ptr<LocalTableFunctionState> FnInitLocal(ExecutionContext &, TableFunctionInitInput &,
                                                                  GlobalTableFunctionState *global_state) {
                auto &g = global_state->Cast<FnGlobalState>();
                g.active_local_states.fetch_add(1, std::memory_order_relaxed);
                return make_uniq<FnLocalState>();
            }
            
            inline hugeint_t ToHugeint(__int128 acc) {
                hugeint_t result;
                result.lower = static_cast<uint64_t>(acc);          // low 64 bits
                result.upper = static_cast<int64_t>(acc >> 64);     // high 64 bits (sign-extended)
                return result;
            }
            

                static unique_ptr<FunctionData> FnBind(ClientContext &, TableFunctionBindInput &,
                                                    vector<LogicalType> &return_types,
                                                    vector<string> &names) {
                    // Populate return types and names for SUM(AdvEngineID), COUNT(*), AVG(ResolutionWidth)
                    return_types.push_back(LogicalType::HUGEINT);  // sum_advengineid
                    return_types.push_back(LogicalType::BIGINT);  // count_star
                    return_types.push_back(LogicalType::DOUBLE);  // avg_resolutionwidth
                    
                    names.push_back("sum_advengineid");
                    names.push_back("count_star");
                    names.push_back("avg_resolutionwidth");

                    return make_uniq<FnBindData>();
                }
            

                static OperatorResultType FnExecute(ExecutionContext &, TableFunctionInput &in,
                                                    DataChunk &input, DataChunk &) {
                    auto &l = in.local_state->Cast<FnLocalState>();

                    
                    if (input.size() == 0) {       
                        return OperatorResultType::NEED_MORE_INPUT;
                    }

                    // Set up UnifiedVectorFormat handles for input columns
                    UnifiedVectorFormat AdvEngineID_uvf;
                    UnifiedVectorFormat ResolutionWidth_uvf;
                    
                    input.data[0].ToUnifiedFormat(input.size(), AdvEngineID_uvf);
                    input.data[1].ToUnifiedFormat(input.size(), ResolutionWidth_uvf);
                    
                    int16_t* AdvEngineID_ptr = (int16_t*)AdvEngineID_uvf.data;
                    int16_t* ResolutionWidth_ptr = (int16_t*)ResolutionWidth_uvf.data;
                    
                    // Prepare validity bitmaps and all-valid flags
                    auto &valid_AdvEngineID = AdvEngineID_uvf.validity;
                    auto &valid_ResolutionWidth = ResolutionWidth_uvf.validity;
                    
                    const bool AdvEngineID_all_valid = valid_AdvEngineID.AllValid();
                    const bool ResolutionWidth_all_valid = valid_ResolutionWidth.AllValid();
                    
                    idx_t num_rows = input.size();
                    
                    // Fast path: when no NULLs exist in either column
                    if (AdvEngineID_all_valid && ResolutionWidth_all_valid) {
                        for (idx_t row_idx = 0; row_idx < num_rows; ++row_idx) {
                            idx_t i_AdvEngineID = AdvEngineID_uvf.sel->get_index(row_idx);
                            idx_t i_ResolutionWidth = ResolutionWidth_uvf.sel->get_index(row_idx);
                            
                            int16_t v_AdvEngineID = AdvEngineID_ptr[i_AdvEngineID];
                            int16_t v_ResolutionWidth = ResolutionWidth_ptr[i_ResolutionWidth];
                            
                            // Update local accumulators
                            l.sum_advengineid += v_AdvEngineID;
                            l.count_star++;
                            l.sum_resolutionwidth += v_ResolutionWidth;
                            l.count_resolutionwidth++;
                        }
                    } else {
                        // Slow path: handle potential NULLs in one or both columns
                        for (idx_t row_idx = 0; row_idx < num_rows; ++row_idx) {
                            idx_t i_AdvEngineID = AdvEngineID_uvf.sel->get_index(row_idx);
                            idx_t i_ResolutionWidth = ResolutionWidth_uvf.sel->get_index(row_idx);
                            
                            // Skip rows where either column has NULL value
                            if (!AdvEngineID_all_valid && !valid_AdvEngineID.RowIsValid(i_AdvEngineID)) {
                                continue;
                            }
                            if (!ResolutionWidth_all_valid && !valid_ResolutionWidth.RowIsValid(i_ResolutionWidth)) {
                                continue;
                            }
                            
                            int16_t v_AdvEngineID = AdvEngineID_ptr[i_AdvEngineID];
                            int16_t v_ResolutionWidth = ResolutionWidth_ptr[i_ResolutionWidth];
                            
                            // Update local accumulators
                            l.sum_advengineid += v_AdvEngineID;
                            l.count_star++;
                            l.sum_resolutionwidth += v_ResolutionWidth;
                            l.count_resolutionwidth++;
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
                            g.sum_advengineid += l.sum_advengineid;
                            g.count_star += l.count_star;
                            g.sum_resolutionwidth += l.sum_resolutionwidth;
                            g.count_resolutionwidth += l.count_resolutionwidth;
                        }
                        l.merged = true;
                        g.merged_local_states.fetch_add(1, std::memory_order_relaxed);
                    }

                    // Only the *last* local state to merge emits the final result.
                    // All other threads return FINISHED with an empty chunk.
                    const auto merged = g.merged_local_states.load(std::memory_order_relaxed);
                    const auto active = g.active_local_states.load(std::memory_order_relaxed);
                    if (active > 0 && merged == active) {
                        __int128 final_sum_advengineid;
                        int64_t final_count_star;
                        double final_avg_resolutionwidth;
                        {
                            std::lock_guard<std::mutex> guard(g.lock);
                            final_sum_advengineid = g.sum_advengineid;
                            final_count_star = g.count_star;
                            final_avg_resolutionwidth = g.count_resolutionwidth > 0 ? g.sum_resolutionwidth / g.count_resolutionwidth : 0.0;
                        }
                        out.SetCardinality(1);
                        hugeint_t hugeint_result = ToHugeint(final_sum_advengineid);
                        out.SetValue(0, 0, Value::HUGEINT(hugeint_result));
                        out.SetValue(1, 0, Value::BIGINT(final_count_star));
                        out.SetValue(2, 0, Value::DOUBLE(final_avg_resolutionwidth));
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