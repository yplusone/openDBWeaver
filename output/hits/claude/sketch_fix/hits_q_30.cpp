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
    // Accumulator fields for SUM aggregations
    int64_t s0_acc = 0;
    int64_t s1_acc = 0;
    int64_t s2_acc = 0;
    int64_t s3_acc = 0;
    int64_t s4_acc = 0;
    int64_t s5_acc = 0;
    int64_t s6_acc = 0;
    int64_t s7_acc = 0;
    int64_t s8_acc = 0;
    int64_t s9_acc = 0;
    int64_t s10_acc = 0;
    int64_t s11_acc = 0;
    int64_t s12_acc = 0;
    int64_t s13_acc = 0;
    int64_t s14_acc = 0;
    int64_t s15_acc = 0;
    int64_t s16_acc = 0;
    int64_t s17_acc = 0;
    int64_t s18_acc = 0;
    int64_t s19_acc = 0;
    int64_t s20_acc = 0;
    int64_t s21_acc = 0;
    int64_t s22_acc = 0;
    int64_t s23_acc = 0;
    int64_t s24_acc = 0;
    int64_t s25_acc = 0;
    int64_t s26_acc = 0;
    int64_t s27_acc = 0;
    int64_t s28_acc = 0;
    int64_t s29_acc = 0;
    int64_t s30_acc = 0;
    int64_t s31_acc = 0;
    int64_t s32_acc = 0;
    int64_t s33_acc = 0;
    int64_t s34_acc = 0;
    int64_t s35_acc = 0;
    int64_t s36_acc = 0;
    int64_t s37_acc = 0;
    int64_t s38_acc = 0;
    int64_t s39_acc = 0;
    int64_t s40_acc = 0;
    int64_t s41_acc = 0;
    int64_t s42_acc = 0;
    int64_t s43_acc = 0;
    int64_t s44_acc = 0;
    int64_t s45_acc = 0;
    int64_t s46_acc = 0;
    int64_t s47_acc = 0;
    int64_t s48_acc = 0;
    int64_t s49_acc = 0;
    int64_t s50_acc = 0;
    int64_t s51_acc = 0;
    int64_t s52_acc = 0;
    int64_t s53_acc = 0;
    int64_t s54_acc = 0;
    int64_t s55_acc = 0;
    int64_t s56_acc = 0;
    int64_t s57_acc = 0;
    int64_t s58_acc = 0;
    int64_t s59_acc = 0;
    int64_t s60_acc = 0;
    int64_t s61_acc = 0;
    int64_t s62_acc = 0;
    int64_t s63_acc = 0;
    int64_t s64_acc = 0;
    int64_t s65_acc = 0;
    int64_t s66_acc = 0;
    int64_t s67_acc = 0;
    int64_t s68_acc = 0;
    int64_t s69_acc = 0;
    int64_t s70_acc = 0;
    int64_t s71_acc = 0;
    int64_t s72_acc = 0;
    int64_t s73_acc = 0;
    int64_t s74_acc = 0;
    int64_t s75_acc = 0;
    int64_t s76_acc = 0;
    int64_t s77_acc = 0;
    int64_t s78_acc = 0;
    int64_t s79_acc = 0;
    int64_t s80_acc = 0;
    int64_t s81_acc = 0;
    int64_t s82_acc = 0;
    int64_t s83_acc = 0;
    int64_t s84_acc = 0;
    int64_t s85_acc = 0;
    int64_t s86_acc = 0;
    int64_t s87_acc = 0;
    int64_t s88_acc = 0;
    int64_t s89_acc = 0;
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
    // Local accumulator fields for SUM aggregations
    int64_t s0_acc = 0;
    int64_t s1_acc = 0;
    int64_t s2_acc = 0;
    int64_t s3_acc = 0;
    int64_t s4_acc = 0;
    int64_t s5_acc = 0;
    int64_t s6_acc = 0;
    int64_t s7_acc = 0;
    int64_t s8_acc = 0;
    int64_t s9_acc = 0;
    int64_t s10_acc = 0;
    int64_t s11_acc = 0;
    int64_t s12_acc = 0;
    int64_t s13_acc = 0;
    int64_t s14_acc = 0;
    int64_t s15_acc = 0;
    int64_t s16_acc = 0;
    int64_t s17_acc = 0;
    int64_t s18_acc = 0;
    int64_t s19_acc = 0;
    int64_t s20_acc = 0;
    int64_t s21_acc = 0;
    int64_t s22_acc = 0;
    int64_t s23_acc = 0;
    int64_t s24_acc = 0;
    int64_t s25_acc = 0;
    int64_t s26_acc = 0;
    int64_t s27_acc = 0;
    int64_t s28_acc = 0;
    int64_t s29_acc = 0;
    int64_t s30_acc = 0;
    int64_t s31_acc = 0;
    int64_t s32_acc = 0;
    int64_t s33_acc = 0;
    int64_t s34_acc = 0;
    int64_t s35_acc = 0;
    int64_t s36_acc = 0;
    int64_t s37_acc = 0;
    int64_t s38_acc = 0;
    int64_t s39_acc = 0;
    int64_t s40_acc = 0;
    int64_t s41_acc = 0;
    int64_t s42_acc = 0;
    int64_t s43_acc = 0;
    int64_t s44_acc = 0;
    int64_t s45_acc = 0;
    int64_t s46_acc = 0;
    int64_t s47_acc = 0;
    int64_t s48_acc = 0;
    int64_t s49_acc = 0;
    int64_t s50_acc = 0;
    int64_t s51_acc = 0;
    int64_t s52_acc = 0;
    int64_t s53_acc = 0;
    int64_t s54_acc = 0;
    int64_t s55_acc = 0;
    int64_t s56_acc = 0;
    int64_t s57_acc = 0;
    int64_t s58_acc = 0;
    int64_t s59_acc = 0;
    int64_t s60_acc = 0;
    int64_t s61_acc = 0;
    int64_t s62_acc = 0;
    int64_t s63_acc = 0;
    int64_t s64_acc = 0;
    int64_t s65_acc = 0;
    int64_t s66_acc = 0;
    int64_t s67_acc = 0;
    int64_t s68_acc = 0;
    int64_t s69_acc = 0;
    int64_t s70_acc = 0;
    int64_t s71_acc = 0;
    int64_t s72_acc = 0;
    int64_t s73_acc = 0;
    int64_t s74_acc = 0;
    int64_t s75_acc = 0;
    int64_t s76_acc = 0;
    int64_t s77_acc = 0;
    int64_t s78_acc = 0;
    int64_t s79_acc = 0;
    int64_t s80_acc = 0;
    int64_t s81_acc = 0;
    int64_t s82_acc = 0;
    int64_t s83_acc = 0;
    int64_t s84_acc = 0;
    int64_t s85_acc = 0;
    int64_t s86_acc = 0;
    int64_t s87_acc = 0;
    int64_t s88_acc = 0;
    int64_t s89_acc = 0;
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
    // Populate return types and names for 90 SUM aggregation outputs
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

    //TODO: process input chunk and produce output

    // Declare UnifiedVectorFormat handles for input columns
    UnifiedVectorFormat ResolutionWidth_uvf;
    
    // Load input columns into UnifiedVectorFormat
    input.data[0].ToUnifiedFormat(input.size(), ResolutionWidth_uvf);
    
    // Create typed pointers to physical data
    int16_t* ResolutionWidth_ptr = (int16_t*)ResolutionWidth_uvf.data;
    
    // Batch-level NULL summary (whether each column is fully non-NULL)
    // validity bitmaps
    auto &valid_ResolutionWidth = ResolutionWidth_uvf.validity;
    const bool ResolutionWidth_all_valid = valid_ResolutionWidth.AllValid();
    
    // FAST BRANCH: all relevant columns have no NULLs in this batch
    if (ResolutionWidth_all_valid) {
        // --- Fast path: no per-row NULL checks ---
        for (idx_t row_idx = 0; row_idx < input.size(); ++row_idx) {
            // Directly load values without RowIsValid checks
            
            idx_t i_ResolutionWidth = ResolutionWidth_uvf.sel->get_index(row_idx);
            int16_t v_ResolutionWidth = ResolutionWidth_ptr[i_ResolutionWidth];
            
            // ======================================
            //  Core computation logic (no NULLs)
            //<<CORE_COMPUTE>>
            l.s0_acc += static_cast<int64_t>(v_ResolutionWidth);
            l.s1_acc += static_cast<int64_t>(v_ResolutionWidth + 1);
            l.s2_acc += static_cast<int64_t>(v_ResolutionWidth + 2);
            l.s3_acc += static_cast<int64_t>(v_ResolutionWidth + 3);
            l.s4_acc += static_cast<int64_t>(v_ResolutionWidth + 4);
            l.s5_acc += static_cast<int64_t>(v_ResolutionWidth + 5);
            l.s6_acc += static_cast<int64_t>(v_ResolutionWidth + 6);
            l.s7_acc += static_cast<int64_t>(v_ResolutionWidth + 7);
            l.s8_acc += static_cast<int64_t>(v_ResolutionWidth + 8);
            l.s9_acc += static_cast<int64_t>(v_ResolutionWidth + 9);
            l.s10_acc += static_cast<int64_t>(v_ResolutionWidth + 10);
            l.s11_acc += static_cast<int64_t>(v_ResolutionWidth + 11);
            l.s12_acc += static_cast<int64_t>(v_ResolutionWidth + 12);
            l.s13_acc += static_cast<int64_t>(v_ResolutionWidth + 13);
            l.s14_acc += static_cast<int64_t>(v_ResolutionWidth + 14);
            l.s15_acc += static_cast<int64_t>(v_ResolutionWidth + 15);
            l.s16_acc += static_cast<int64_t>(v_ResolutionWidth + 16);
            l.s17_acc += static_cast<int64_t>(v_ResolutionWidth + 17);
            l.s18_acc += static_cast<int64_t>(v_ResolutionWidth + 18);
            l.s19_acc += static_cast<int64_t>(v_ResolutionWidth + 19);
            l.s20_acc += static_cast<int64_t>(v_ResolutionWidth + 20);
            l.s21_acc += static_cast<int64_t>(v_ResolutionWidth + 21);
            l.s22_acc += static_cast<int64_t>(v_ResolutionWidth + 22);
            l.s23_acc += static_cast<int64_t>(v_ResolutionWidth + 23);
            l.s24_acc += static_cast<int64_t>(v_ResolutionWidth + 24);
            l.s25_acc += static_cast<int64_t>(v_ResolutionWidth + 25);
            l.s26_acc += static_cast<int64_t>(v_ResolutionWidth + 26);
            l.s27_acc += static_cast<int64_t>(v_ResolutionWidth + 27);
            l.s28_acc += static_cast<int64_t>(v_ResolutionWidth + 28);
            l.s29_acc += static_cast<int64_t>(v_ResolutionWidth + 29);
            l.s30_acc += static_cast<int64_t>(v_ResolutionWidth + 30);
            l.s31_acc += static_cast<int64_t>(v_ResolutionWidth + 31);
            l.s32_acc += static_cast<int64_t>(v_ResolutionWidth + 32);
            l.s33_acc += static_cast<int64_t>(v_ResolutionWidth + 33);
            l.s34_acc += static_cast<int64_t>(v_ResolutionWidth + 34);
            l.s35_acc += static_cast<int64_t>(v_ResolutionWidth + 35);
            l.s36_acc += static_cast<int64_t>(v_ResolutionWidth + 36);
            l.s37_acc += static_cast<int64_t>(v_ResolutionWidth + 37);
            l.s38_acc += static_cast<int64_t>(v_ResolutionWidth + 38);
            l.s39_acc += static_cast<int64_t>(v_ResolutionWidth + 39);
            l.s40_acc += static_cast<int64_t>(v_ResolutionWidth + 40);
            l.s41_acc += static_cast<int64_t>(v_ResolutionWidth + 41);
            l.s42_acc += static_cast<int64_t>(v_ResolutionWidth + 42);
            l.s43_acc += static_cast<int64_t>(v_ResolutionWidth + 43);
            l.s44_acc += static_cast<int64_t>(v_ResolutionWidth + 44);
            l.s45_acc += static_cast<int64_t>(v_ResolutionWidth + 45);
            l.s46_acc += static_cast<int64_t>(v_ResolutionWidth + 46);
            l.s47_acc += static_cast<int64_t>(v_ResolutionWidth + 47);
            l.s48_acc += static_cast<int64_t>(v_ResolutionWidth + 48);
            l.s49_acc += static_cast<int64_t>(v_ResolutionWidth + 49);
            l.s50_acc += static_cast<int64_t>(v_ResolutionWidth + 50);
            l.s51_acc += static_cast<int64_t>(v_ResolutionWidth + 51);
            l.s52_acc += static_cast<int64_t>(v_ResolutionWidth + 52);
            l.s53_acc += static_cast<int64_t>(v_ResolutionWidth + 53);
            l.s54_acc += static_cast<int64_t>(v_ResolutionWidth + 54);
            l.s55_acc += static_cast<int64_t>(v_ResolutionWidth + 55);
            l.s56_acc += static_cast<int64_t>(v_ResolutionWidth + 56);
            l.s57_acc += static_cast<int64_t>(v_ResolutionWidth + 57);
            l.s58_acc += static_cast<int64_t>(v_ResolutionWidth + 58);
            l.s59_acc += static_cast<int64_t>(v_ResolutionWidth + 59);
            l.s60_acc += static_cast<int64_t>(v_ResolutionWidth + 60);
            l.s61_acc += static_cast<int64_t>(v_ResolutionWidth + 61);
            l.s62_acc += static_cast<int64_t>(v_ResolutionWidth + 62);
            l.s63_acc += static_cast<int64_t>(v_ResolutionWidth + 63);
            l.s64_acc += static_cast<int64_t>(v_ResolutionWidth + 64);
            l.s65_acc += static_cast<int64_t>(v_ResolutionWidth + 65);
            l.s66_acc += static_cast<int64_t>(v_ResolutionWidth + 66);
            l.s67_acc += static_cast<int64_t>(v_ResolutionWidth + 67);
            l.s68_acc += static_cast<int64_t>(v_ResolutionWidth + 68);
            l.s69_acc += static_cast<int64_t>(v_ResolutionWidth + 69);
            l.s70_acc += static_cast<int64_t>(v_ResolutionWidth + 70);
            l.s71_acc += static_cast<int64_t>(v_ResolutionWidth + 71);
            l.s72_acc += static_cast<int64_t>(v_ResolutionWidth + 72);
            l.s73_acc += static_cast<int64_t>(v_ResolutionWidth + 73);
            l.s74_acc += static_cast<int64_t>(v_ResolutionWidth + 74);
            l.s75_acc += static_cast<int64_t>(v_ResolutionWidth + 75);
            l.s76_acc += static_cast<int64_t>(v_ResolutionWidth + 76);
            l.s77_acc += static_cast<int64_t>(v_ResolutionWidth + 77);
            l.s78_acc += static_cast<int64_t>(v_ResolutionWidth + 78);
            l.s79_acc += static_cast<int64_t>(v_ResolutionWidth + 79);
            l.s80_acc += static_cast<int64_t>(v_ResolutionWidth + 80);
            l.s81_acc += static_cast<int64_t>(v_ResolutionWidth + 81);
            l.s82_acc += static_cast<int64_t>(v_ResolutionWidth + 82);
            l.s83_acc += static_cast<int64_t>(v_ResolutionWidth + 83);
            l.s84_acc += static_cast<int64_t>(v_ResolutionWidth + 84);
            l.s85_acc += static_cast<int64_t>(v_ResolutionWidth + 85);
            l.s86_acc += static_cast<int64_t>(v_ResolutionWidth + 86);
            l.s87_acc += static_cast<int64_t>(v_ResolutionWidth + 87);
            l.s88_acc += static_cast<int64_t>(v_ResolutionWidth + 88);
            l.s89_acc += static_cast<int64_t>(v_ResolutionWidth + 89);
            // ============================
        }
    } else {
        // --- Slow path: at least one column may contain NULLs ---
        for (idx_t row_idx = 0; row_idx < input.size(); ++row_idx) {
            // For each column that is not fully valid, check this row
            idx_t i_ResolutionWidth = ResolutionWidth_uvf.sel->get_index(row_idx);
            
            if (!ResolutionWidth_all_valid && !valid_ResolutionWidth.RowIsValid(i_ResolutionWidth)) {
                continue; // row is NULL in column ResolutionWidth → skip
            }
            
            // At this point, all required columns are valid for this row
            
            int16_t v_ResolutionWidth = ResolutionWidth_ptr[i_ResolutionWidth];
            
            // ======================================
            //  Core computation logic (NULL-safe)
            //<<CORE_COMPUTE>>
            l.s0_acc += static_cast<int64_t>(v_ResolutionWidth);
            l.s1_acc += static_cast<int64_t>(v_ResolutionWidth + 1);
            l.s2_acc += static_cast<int64_t>(v_ResolutionWidth + 2);
            l.s3_acc += static_cast<int64_t>(v_ResolutionWidth + 3);
            l.s4_acc += static_cast<int64_t>(v_ResolutionWidth + 4);
            l.s5_acc += static_cast<int64_t>(v_ResolutionWidth + 5);
            l.s6_acc += static_cast<int64_t>(v_ResolutionWidth + 6);
            l.s7_acc += static_cast<int64_t>(v_ResolutionWidth + 7);
            l.s8_acc += static_cast<int64_t>(v_ResolutionWidth + 8);
            l.s9_acc += static_cast<int64_t>(v_ResolutionWidth + 9);
            l.s10_acc += static_cast<int64_t>(v_ResolutionWidth + 10);
            l.s11_acc += static_cast<int64_t>(v_ResolutionWidth + 11);
            l.s12_acc += static_cast<int64_t>(v_ResolutionWidth + 12);
            l.s13_acc += static_cast<int64_t>(v_ResolutionWidth + 13);
            l.s14_acc += static_cast<int64_t>(v_ResolutionWidth + 14);
            l.s15_acc += static_cast<int64_t>(v_ResolutionWidth + 15);
            l.s16_acc += static_cast<int64_t>(v_ResolutionWidth + 16);
            l.s17_acc += static_cast<int64_t>(v_ResolutionWidth + 17);
            l.s18_acc += static_cast<int64_t>(v_ResolutionWidth + 18);
            l.s19_acc += static_cast<int64_t>(v_ResolutionWidth + 19);
            l.s20_acc += static_cast<int64_t>(v_ResolutionWidth + 20);
            l.s21_acc += static_cast<int64_t>(v_ResolutionWidth + 21);
            l.s22_acc += static_cast<int64_t>(v_ResolutionWidth + 22);
            l.s23_acc += static_cast<int64_t>(v_ResolutionWidth + 23);
            l.s24_acc += static_cast<int64_t>(v_ResolutionWidth + 24);
            l.s25_acc += static_cast<int64_t>(v_ResolutionWidth + 25);
            l.s26_acc += static_cast<int64_t>(v_ResolutionWidth + 26);
            l.s27_acc += static_cast<int64_t>(v_ResolutionWidth + 27);
            l.s28_acc += static_cast<int64_t>(v_ResolutionWidth + 28);
            l.s29_acc += static_cast<int64_t>(v_ResolutionWidth + 29);
            l.s30_acc += static_cast<int64_t>(v_ResolutionWidth + 30);
            l.s31_acc += static_cast<int64_t>(v_ResolutionWidth + 31);
            l.s32_acc += static_cast<int64_t>(v_ResolutionWidth + 32);
            l.s33_acc += static_cast<int64_t>(v_ResolutionWidth + 33);
            l.s34_acc += static_cast<int64_t>(v_ResolutionWidth + 34);
            l.s35_acc += static_cast<int64_t>(v_ResolutionWidth + 35);
            l.s36_acc += static_cast<int64_t>(v_ResolutionWidth + 36);
            l.s37_acc += static_cast<int64_t>(v_ResolutionWidth + 37);
            l.s38_acc += static_cast<int64_t>(v_ResolutionWidth + 38);
            l.s39_acc += static_cast<int64_t>(v_ResolutionWidth + 39);
            l.s40_acc += static_cast<int64_t>(v_ResolutionWidth + 40);
            l.s41_acc += static_cast<int64_t>(v_ResolutionWidth + 41);
            l.s42_acc += static_cast<int64_t>(v_ResolutionWidth + 42);
            l.s43_acc += static_cast<int64_t>(v_ResolutionWidth + 43);
            l.s44_acc += static_cast<int64_t>(v_ResolutionWidth + 44);
            l.s45_acc += static_cast<int64_t>(v_ResolutionWidth + 45);
            l.s46_acc += static_cast<int64_t>(v_ResolutionWidth + 46);
            l.s47_acc += static_cast<int64_t>(v_ResolutionWidth + 47);
            l.s48_acc += static_cast<int64_t>(v_ResolutionWidth + 48);
            l.s49_acc += static_cast<int64_t>(v_ResolutionWidth + 49);
            l.s50_acc += static_cast<int64_t>(v_ResolutionWidth + 50);
            l.s51_acc += static_cast<int64_t>(v_ResolutionWidth + 51);
            l.s52_acc += static_cast<int64_t>(v_ResolutionWidth + 52);
            l.s53_acc += static_cast<int64_t>(v_ResolutionWidth + 53);
            l.s54_acc += static_cast<int64_t>(v_ResolutionWidth + 54);
            l.s55_acc += static_cast<int64_t>(v_ResolutionWidth + 55);
            l.s56_acc += static_cast<int64_t>(v_ResolutionWidth + 56);
            l.s57_acc += static_cast<int64_t>(v_ResolutionWidth + 57);
            l.s58_acc += static_cast<int64_t>(v_ResolutionWidth + 58);
            l.s59_acc += static_cast<int64_t>(v_ResolutionWidth + 59);
            l.s60_acc += static_cast<int64_t>(v_ResolutionWidth + 60);
            l.s61_acc += static_cast<int64_t>(v_ResolutionWidth + 61);
            l.s62_acc += static_cast<int64_t>(v_ResolutionWidth + 62);
            l.s63_acc += static_cast<int64_t>(v_ResolutionWidth + 63);
            l.s64_acc += static_cast<int64_t>(v_ResolutionWidth + 64);
            l.s65_acc += static_cast<int64_t>(v_ResolutionWidth + 65);
            l.s66_acc += static_cast<int64_t>(v_ResolutionWidth + 66);
            l.s67_acc += static_cast<int64_t>(v_ResolutionWidth + 67);
            l.s68_acc += static_cast<int64_t>(v_ResolutionWidth + 68);
            l.s69_acc += static_cast<int64_t>(v_ResolutionWidth + 69);
            l.s70_acc += static_cast<int64_t>(v_ResolutionWidth + 70);
            l.s71_acc += static_cast<int64_t>(v_ResolutionWidth + 71);
            l.s72_acc += static_cast<int64_t>(v_ResolutionWidth + 72);
            l.s73_acc += static_cast<int64_t>(v_ResolutionWidth + 73);
            l.s74_acc += static_cast<int64_t>(v_ResolutionWidth + 74);
            l.s75_acc += static_cast<int64_t>(v_ResolutionWidth + 75);
            l.s76_acc += static_cast<int64_t>(v_ResolutionWidth + 76);
            l.s77_acc += static_cast<int64_t>(v_ResolutionWidth + 77);
            l.s78_acc += static_cast<int64_t>(v_ResolutionWidth + 78);
            l.s79_acc += static_cast<int64_t>(v_ResolutionWidth + 79);
            l.s80_acc += static_cast<int64_t>(v_ResolutionWidth + 80);
            l.s81_acc += static_cast<int64_t>(v_ResolutionWidth + 81);
            l.s82_acc += static_cast<int64_t>(v_ResolutionWidth + 82);
            l.s83_acc += static_cast<int64_t>(v_ResolutionWidth + 83);
            l.s84_acc += static_cast<int64_t>(v_ResolutionWidth + 84);
            l.s85_acc += static_cast<int64_t>(v_ResolutionWidth + 85);
            l.s86_acc += static_cast<int64_t>(v_ResolutionWidth + 86);
            l.s87_acc += static_cast<int64_t>(v_ResolutionWidth + 87);
            l.s88_acc += static_cast<int64_t>(v_ResolutionWidth + 88);
            l.s89_acc += static_cast<int64_t>(v_ResolutionWidth + 89);
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
            g.s0_acc += l.s0_acc;
            g.s1_acc += l.s1_acc;
            g.s2_acc += l.s2_acc;
            g.s3_acc += l.s3_acc;
            g.s4_acc += l.s4_acc;
            g.s5_acc += l.s5_acc;
            g.s6_acc += l.s6_acc;
            g.s7_acc += l.s7_acc;
            g.s8_acc += l.s8_acc;
            g.s9_acc += l.s9_acc;
            g.s10_acc += l.s10_acc;
            g.s11_acc += l.s11_acc;
            g.s12_acc += l.s12_acc;
            g.s13_acc += l.s13_acc;
            g.s14_acc += l.s14_acc;
            g.s15_acc += l.s15_acc;
            g.s16_acc += l.s16_acc;
            g.s17_acc += l.s17_acc;
            g.s18_acc += l.s18_acc;
            g.s19_acc += l.s19_acc;
            g.s20_acc += l.s20_acc;
            g.s21_acc += l.s21_acc;
            g.s22_acc += l.s22_acc;
            g.s23_acc += l.s23_acc;
            g.s24_acc += l.s24_acc;
            g.s25_acc += l.s25_acc;
            g.s26_acc += l.s26_acc;
            g.s27_acc += l.s27_acc;
            g.s28_acc += l.s28_acc;
            g.s29_acc += l.s29_acc;
            g.s30_acc += l.s30_acc;
            g.s31_acc += l.s31_acc;
            g.s32_acc += l.s32_acc;
            g.s33_acc += l.s33_acc;
            g.s34_acc += l.s34_acc;
            g.s35_acc += l.s35_acc;
            g.s36_acc += l.s36_acc;
            g.s37_acc += l.s37_acc;
            g.s38_acc += l.s38_acc;
            g.s39_acc += l.s39_acc;
            g.s40_acc += l.s40_acc;
            g.s41_acc += l.s41_acc;
            g.s42_acc += l.s42_acc;
            g.s43_acc += l.s43_acc;
            g.s44_acc += l.s44_acc;
            g.s45_acc += l.s45_acc;
            g.s46_acc += l.s46_acc;
            g.s47_acc += l.s47_acc;
            g.s48_acc += l.s48_acc;
            g.s49_acc += l.s49_acc;
            g.s50_acc += l.s50_acc;
            g.s51_acc += l.s51_acc;
            g.s52_acc += l.s52_acc;
            g.s53_acc += l.s53_acc;
            g.s54_acc += l.s54_acc;
            g.s55_acc += l.s55_acc;
            g.s56_acc += l.s56_acc;
            g.s57_acc += l.s57_acc;
            g.s58_acc += l.s58_acc;
            g.s59_acc += l.s59_acc;
            g.s60_acc += l.s60_acc;
            g.s61_acc += l.s61_acc;
            g.s62_acc += l.s62_acc;
            g.s63_acc += l.s63_acc;
            g.s64_acc += l.s64_acc;
            g.s65_acc += l.s65_acc;
            g.s66_acc += l.s66_acc;
            g.s67_acc += l.s67_acc;
            g.s68_acc += l.s68_acc;
            g.s69_acc += l.s69_acc;
            g.s70_acc += l.s70_acc;
            g.s71_acc += l.s71_acc;
            g.s72_acc += l.s72_acc;
            g.s73_acc += l.s73_acc;
            g.s74_acc += l.s74_acc;
            g.s75_acc += l.s75_acc;
            g.s76_acc += l.s76_acc;
            g.s77_acc += l.s77_acc;
            g.s78_acc += l.s78_acc;
            g.s79_acc += l.s79_acc;
            g.s80_acc += l.s80_acc;
            g.s81_acc += l.s81_acc;
            g.s82_acc += l.s82_acc;
            g.s83_acc += l.s83_acc;
            g.s84_acc += l.s84_acc;
            g.s85_acc += l.s85_acc;
            g.s86_acc += l.s86_acc;
            g.s87_acc += l.s87_acc;
            g.s88_acc += l.s88_acc;
            g.s89_acc += l.s89_acc;
        }
        l.merged = true;
        g.merged_local_states.fetch_add(1, std::memory_order_relaxed);
    }

    // Only the *last* local state to merge emits the final result.
    // All other threads return FINISHED with an empty chunk.
    const auto merged = g.merged_local_states.load(std::memory_order_relaxed);
    const auto active = g.active_local_states.load(std::memory_order_relaxed);
    if (active > 0 && merged == active) {
        {
            std::lock_guard<std::mutex> guard(g.lock);
            //TODO: get result from global state
        }
        //TODO: populate out chunk with final results
        out.SetCardinality(1);
        out.SetValue(0, 0, Value::HUGEINT(g.s0_acc));
        out.SetValue(1, 0, Value::HUGEINT(g.s1_acc));
        out.SetValue(2, 0, Value::HUGEINT(g.s2_acc));
        out.SetValue(3, 0, Value::HUGEINT(g.s3_acc));
        out.SetValue(4, 0, Value::HUGEINT(g.s4_acc));
        out.SetValue(5, 0, Value::HUGEINT(g.s5_acc));
        out.SetValue(6, 0, Value::HUGEINT(g.s6_acc));
        out.SetValue(7, 0, Value::HUGEINT(g.s7_acc));
        out.SetValue(8, 0, Value::HUGEINT(g.s8_acc));
        out.SetValue(9, 0, Value::HUGEINT(g.s9_acc));
        out.SetValue(10, 0, Value::HUGEINT(g.s10_acc));
        out.SetValue(11, 0, Value::HUGEINT(g.s11_acc));
        out.SetValue(12, 0, Value::HUGEINT(g.s12_acc));
        out.SetValue(13, 0, Value::HUGEINT(g.s13_acc));
        out.SetValue(14, 0, Value::HUGEINT(g.s14_acc));
        out.SetValue(15, 0, Value::HUGEINT(g.s15_acc));
        out.SetValue(16, 0, Value::HUGEINT(g.s16_acc));
        out.SetValue(17, 0, Value::HUGEINT(g.s17_acc));
        out.SetValue(18, 0, Value::HUGEINT(g.s18_acc));
        out.SetValue(19, 0, Value::HUGEINT(g.s19_acc));
        out.SetValue(20, 0, Value::HUGEINT(g.s20_acc));
        out.SetValue(21, 0, Value::HUGEINT(g.s21_acc));
        out.SetValue(22, 0, Value::HUGEINT(g.s22_acc));
        out.SetValue(23, 0, Value::HUGEINT(g.s23_acc));
        out.SetValue(24, 0, Value::HUGEINT(g.s24_acc));
        out.SetValue(25, 0, Value::HUGEINT(g.s25_acc));
        out.SetValue(26, 0, Value::HUGEINT(g.s26_acc));
        out.SetValue(27, 0, Value::HUGEINT(g.s27_acc));
        out.SetValue(28, 0, Value::HUGEINT(g.s28_acc));
        out.SetValue(29, 0, Value::HUGEINT(g.s29_acc));
        out.SetValue(30, 0, Value::HUGEINT(g.s30_acc));
        out.SetValue(31, 0, Value::HUGEINT(g.s31_acc));
        out.SetValue(32, 0, Value::HUGEINT(g.s32_acc));
        out.SetValue(33, 0, Value::HUGEINT(g.s33_acc));
        out.SetValue(34, 0, Value::HUGEINT(g.s34_acc));
        out.SetValue(35, 0, Value::HUGEINT(g.s35_acc));
        out.SetValue(36, 0, Value::HUGEINT(g.s36_acc));
        out.SetValue(37, 0, Value::HUGEINT(g.s37_acc));
        out.SetValue(38, 0, Value::HUGEINT(g.s38_acc));
        out.SetValue(39, 0, Value::HUGEINT(g.s39_acc));
        out.SetValue(40, 0, Value::HUGEINT(g.s40_acc));
        out.SetValue(41, 0, Value::HUGEINT(g.s41_acc));
        out.SetValue(42, 0, Value::HUGEINT(g.s42_acc));
        out.SetValue(43, 0, Value::HUGEINT(g.s43_acc));
        out.SetValue(44, 0, Value::HUGEINT(g.s44_acc));
        out.SetValue(45, 0, Value::HUGEINT(g.s45_acc));
        out.SetValue(46, 0, Value::HUGEINT(g.s46_acc));
        out.SetValue(47, 0, Value::HUGEINT(g.s47_acc));
        out.SetValue(48, 0, Value::HUGEINT(g.s48_acc));
        out.SetValue(49, 0, Value::HUGEINT(g.s49_acc));
        out.SetValue(50, 0, Value::HUGEINT(g.s50_acc));
        out.SetValue(51, 0, Value::HUGEINT(g.s51_acc));
        out.SetValue(52, 0, Value::HUGEINT(g.s52_acc));
        out.SetValue(53, 0, Value::HUGEINT(g.s53_acc));
        out.SetValue(54, 0, Value::HUGEINT(g.s54_acc));
        out.SetValue(55, 0, Value::HUGEINT(g.s55_acc));
        out.SetValue(56, 0, Value::HUGEINT(g.s56_acc));
        out.SetValue(57, 0, Value::HUGEINT(g.s57_acc));
        out.SetValue(58, 0, Value::HUGEINT(g.s58_acc));
        out.SetValue(59, 0, Value::HUGEINT(g.s59_acc));
        out.SetValue(60, 0, Value::HUGEINT(g.s60_acc));
        out.SetValue(61, 0, Value::HUGEINT(g.s61_acc));
        out.SetValue(62, 0, Value::HUGEINT(g.s62_acc));
        out.SetValue(63, 0, Value::HUGEINT(g.s63_acc));
        out.SetValue(64, 0, Value::HUGEINT(g.s64_acc));
        out.SetValue(65, 0, Value::HUGEINT(g.s65_acc));
        out.SetValue(66, 0, Value::HUGEINT(g.s66_acc));
        out.SetValue(67, 0, Value::HUGEINT(g.s67_acc));
        out.SetValue(68, 0, Value::HUGEINT(g.s68_acc));
        out.SetValue(69, 0, Value::HUGEINT(g.s69_acc));
        out.SetValue(70, 0, Value::HUGEINT(g.s70_acc));
        out.SetValue(71, 0, Value::HUGEINT(g.s71_acc));
        out.SetValue(72, 0, Value::HUGEINT(g.s72_acc));
        out.SetValue(73, 0, Value::HUGEINT(g.s73_acc));
        out.SetValue(74, 0, Value::HUGEINT(g.s74_acc));
        out.SetValue(75, 0, Value::HUGEINT(g.s75_acc));
        out.SetValue(76, 0, Value::HUGEINT(g.s76_acc));
        out.SetValue(77, 0, Value::HUGEINT(g.s77_acc));
        out.SetValue(78, 0, Value::HUGEINT(g.s78_acc));
        out.SetValue(79, 0, Value::HUGEINT(g.s79_acc));
        out.SetValue(80, 0, Value::HUGEINT(g.s80_acc));
        out.SetValue(81, 0, Value::HUGEINT(g.s81_acc));
        out.SetValue(82, 0, Value::HUGEINT(g.s82_acc));
        out.SetValue(83, 0, Value::HUGEINT(g.s83_acc));
        out.SetValue(84, 0, Value::HUGEINT(g.s84_acc));
        out.SetValue(85, 0, Value::HUGEINT(g.s85_acc));
        out.SetValue(86, 0, Value::HUGEINT(g.s86_acc));
        out.SetValue(87, 0, Value::HUGEINT(g.s87_acc));
        out.SetValue(88, 0, Value::HUGEINT(g.s88_acc));
        out.SetValue(89, 0, Value::HUGEINT(g.s89_acc));
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