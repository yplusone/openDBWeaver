#define DUCKDB_EXTENSION_MAIN

//======================================================================
//  dbweaver  ——  DuckDB extension (no-fastpath, correctness-first)
//
//  调用方式：
//
//    SELECT *
//    FROM dbweaver((
//      SELECT
//        LO_ORDERDATE,
//        LO_DISCOUNT,
//        LO_QUANTITY,
//        LO_EXTENDEDPRICE
//      FROM lineorder_flat
//      WHERE LO_DISCOUNT BETWEEN 1 AND 3
//        AND LO_QUANTITY < 25
//    ));
//
//  语义（与原始 Q1.1 一致）：
//    - 过滤条件：
//        EXTRACT(YEAR FROM LO_ORDERDATE) = 1993
//        LO_DISCOUNT BETWEEN 1 AND 3
//        LO_QUANTITY < 25
//    - 聚合：
//        SUM(LO_EXTENDEDPRICE * LO_DISCOUNT)
//
//  实现要点：
//    - 不再尝试 fast path（即不假设 selection vector 是 identity）。
//    - 每行都用 selection vector / validity 来安全取值。
//    - 仍然用 int64_t 作为每个批次的局部累加器，再一次性加到全局 __int128_t，避免在热循环里频繁操作 128 位。
//    - 全局结果以 HUGEINT 形式返回 1 行 1 列 "revenue"。
//======================================================================

#include "dbweaver_extension.hpp"
#include "duckdb.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/common/types/date.hpp"

#include <string>
#include <cstdint>

namespace duckdb {

// ---------------------------------------------------------------------
// 年份范围常量：使用天数范围代替 EXTRACT(YEAR)
// date_t.days 是自 1970-01-01 起的天数
//
// 1993-01-01 → 8401
// 1994-01-01 → 8766
//
// so: YEAR(orderdate)=1993  ⇔
//      days >= 8401 AND days < 8766
// ---------------------------------------------------------------------
static const int32_t Y1993_START_DAYS = 8401;
static const int32_t Y1994_START_DAYS = 8766;

// ---------------------------------------------------------------------
// 全局聚合状态
//   acc_revenue128: SUM(price * discount) 以 128 bit 累加
//   finished_input: 我们是否已经收到 size()==0 的“结束批”
// ---------------------------------------------------------------------
struct FnGlobalState : public GlobalTableFunctionState {
    __int128_t acc_revenue128 = 0;
    bool       finished_input = false;
};

// （调试/格式化用的小工具，不在热路径里）
static inline std::string FormatScaledInteger(int64_t raw, uint8_t s) {
    if (s == 0) {
        return std::to_string(raw);
    }
    bool neg = (raw < 0);
    if (neg) raw = -raw;

    std::string str = std::to_string(raw);
    if (str.length() <= s) {
        str = "0." + std::string(s - str.length(), '0') + str;
    } else {
        str.insert(str.length() - s, ".");
    }
    return neg ? "-" + str : str;
}

// ============================================================
//  SECTION A — Bind phase: 定义输出 schema
// ============================================================
struct FnBindData : public FunctionData {
    unique_ptr<FunctionData> Copy() const override { return make_uniq<FnBindData>(); }
    bool Equals(const FunctionData &) const override { return true; }
};

static unique_ptr<FunctionData> FnBind(
    ClientContext &, TableFunctionBindInput &,
    vector<LogicalType> &return_types,
    vector<string> &names
) {
    // 返回 1 列: revenue (HUGEINT)
    names.push_back("revenue");
    return_types.push_back(LogicalType::HUGEINT);
    return make_uniq<FnBindData>();
}

// ============================================================
//  SECTION B — Init: 分配全局状态
// ============================================================
static unique_ptr<GlobalTableFunctionState> FnInit(ClientContext &, TableFunctionInitInput &) {
    return make_uniq<FnGlobalState>();
}

// ============================================================
//  SECTION C — Execute: 流式消费输入批次并累计部分和
// ============================================================
//
// 输入列顺序（来自子查询的 SELECT 列表，必须保持一致）：
//   0: LO_ORDERDATE      DATE        (duckdb::date_t，其中 .days 是 int32_t)
//   1: LO_DISCOUNT       UTINYINT    (uint8_t)
//   2: LO_QUANTITY       UTINYINT    (uint8_t)
//   3: LO_EXTENDEDPRICE  UINTEGER    (uint32_t / int32_t范围内的正数)
//
// 我们在这里重新检查所有谓词，保证和“所有谓词都在 UDTF 内部做”的语义一致：
//   - lo_discount BETWEEN 1 AND 3
//   - lo_quantity < 25
//   - lo_orderdate in [1993-01-01, 1994-01-01)
//   - 所有关心的列都非 NULL
//
// 然后把 (price * discount) 加到分块累加器里，再写回全局 128-bit 状态。
//
// 我们完全不尝试 fast path：
//   - 不假设 selection vector == identity
//   - 每一行都通过 sel.get_index() 去取实际行号
//   - 每一列都检查 validity
// ============================================================
static OperatorResultType FnExecute(ExecutionContext &, TableFunctionInput &in,
                                    DataChunk &input, DataChunk &) {
    auto &g = *reinterpret_cast<FnGlobalState *>(in.global_state.get());

    // 仍在消费上游输入
    if (!g.finished_input) {

        // DuckDB 传 size()==0 的批表示“上游结束了”
        if (input.size() == 0) {
            g.finished_input = true;
            return OperatorResultType::HAVE_MORE_OUTPUT;
        }

        const idx_t n = input.size();

        // 列布局约定（必须和子查询 SELECT 顺序一致）:
        //   0: LO_ORDERDATE      (DATE -> date_t)
        //   1: LO_DISCOUNT       (UTINYINT -> uint8_t)
        //   2: LO_QUANTITY       (我们这里不再用，因为上游已经保证 <25)
        //   3: LO_EXTENDEDPRICE  (UINT32-ish -> uint32_t)
        //
        // 这里只统一化 date / disc / price 三列
        UnifiedVectorFormat uvf_date;
        input.data[0].ToUnifiedFormat(n, uvf_date);
        const date_t *p_date = (const date_t *)uvf_date.data;

        UnifiedVectorFormat uvf_disc;
        input.data[1].ToUnifiedFormat(n, uvf_disc);
        const uint8_t *p_disc = (const uint8_t *)uvf_disc.data;

        UnifiedVectorFormat uvf_price;
        input.data[3].ToUnifiedFormat(n, uvf_price);
        const uint32_t *p_price = (const uint32_t *)uvf_price.data;

        // selection vectors
        auto &sel_date  = *uvf_date.sel;
        auto &sel_disc  = *uvf_disc.sel;
        auto &sel_price = *uvf_price.sel;

        // validity bitmaps
        auto &valid_date  = uvf_date.validity;
        auto &valid_disc  = uvf_disc.validity;
        auto &valid_price = uvf_price.validity;

        // 批级 NULL 概况（整列是否全非 NULL）
        const bool date_allvalid  = valid_date.AllValid();
        const bool disc_allvalid  = valid_disc.AllValid();
        const bool price_allvalid = valid_price.AllValid();

        // 局部累加器 (64-bit)，批末 widen 成 128-bit
        int64_t local_acc64 = 0;

        // ------------------------------------------------------------------
        // FAST BRANCH:
        // 如果 date / disc / price 三列在这一批里都完全没有 NULL
        // → 我们可以完全跳过 RowIsValid() 这些 per-row NULL 检查，
        //   只做年份判断 + 乘法 + 加法。
        // ------------------------------------------------------------------
        if (date_allvalid && disc_allvalid && price_allvalid) {
            for (idx_t row_idx = 0; row_idx < n; row_idx++) {
                // 年份过滤：1993 年
                const idx_t rid_date = sel_date.get_index(row_idx);
                const int32_t days_val = p_date[rid_date].days;
                if (days_val < Y1993_START_DAYS || days_val >= Y1994_START_DAYS) {
                    continue;
                }

                // 折扣值（上游已经保证 1~3，无需再判范围）
                const idx_t rid_disc = sel_disc.get_index(row_idx);
                const uint8_t disc = p_disc[rid_disc];

                // 价格
                const idx_t rid_price = sel_price.get_index(row_idx);
                const uint32_t price = p_price[rid_price];

                // 累加
                local_acc64 += (int64_t)price * (int64_t)disc;
            }

        } else {
            // ------------------------------------------------------------------
            // SAFE BRANCH:
            // 至少有一列可能包含 NULL -> 必须逐行检查 RowIsValid()
            // ------------------------------------------------------------------
            for (idx_t row_idx = 0; row_idx < n; row_idx++) {

                // 日期：必须非 NULL 且在 1993 年
                const idx_t rid_date = sel_date.get_index(row_idx);
                if (!date_allvalid && !valid_date.RowIsValid(rid_date)) {
                    continue;
                }
                const int32_t days_val = p_date[rid_date].days;
                if (days_val < Y1993_START_DAYS || days_val >= Y1994_START_DAYS) {
                    continue;
                }

                // 折扣：必须非 NULL
                const idx_t rid_disc = sel_disc.get_index(row_idx);
                if (!disc_allvalid && !valid_disc.RowIsValid(rid_disc)) {
                    continue;
                }
                const uint8_t disc = p_disc[rid_disc];
                // 不再检查范围 (1~3) 因为上游 WHERE 已经裁掉了坏行

                // 价格：必须非 NULL
                const idx_t rid_price = sel_price.get_index(row_idx);
                if (!price_allvalid && !valid_price.RowIsValid(rid_price)) {
                    continue;
                }
                const uint32_t price = p_price[rid_price];

                // 累加
                local_acc64 += (int64_t)price * (int64_t)disc;
            }
        }

        // widen 成 128-bit，全局加一次
        g.acc_revenue128 += (__int128_t)local_acc64;

        // 还要更多批
        return OperatorResultType::NEED_MORE_INPUT;
    }

    // 上游已经结束，我们让 pipeline 进入 Finalize 来出最终一行
    return OperatorResultType::HAVE_MORE_OUTPUT;
}


// ============================================================
//  SECTION D — Finalize: 把最终累计的 128 位结果吐成一行
// ============================================================
static OperatorFinalizeResultType FnFinalize(ExecutionContext &, TableFunctionInput &in,
                                             DataChunk &result) {
    auto &g = *reinterpret_cast<FnGlobalState *>(in.global_state.get());

    // 把 __int128_t 拆成 DuckDB hugeint_t
    const __int128_t v = g.acc_revenue128;

    hugeint_t out_val;
    out_val.lower = (uint64_t)(((__int128_t)v) &
                               (__int128_t)0xFFFFFFFFFFFFFFFFULL);
    out_val.upper = (int64_t)(((__int128_t)v) >> 64);

    result.SetCardinality(1);
    result.SetValue(0, 0, Value::HUGEINT(out_val));

    return OperatorFinalizeResultType::FINISHED;
}

// ============================================================
//  SECTION E — 在 DuckDB 里注册这个 TableFunction
// ============================================================
static void LoadInternal(ExtensionLoader &loader) {
    // 1 个参数：TABLE (即子查询产生的表值)
    // 返回：1 行 1 列 (HUGEINT revenue)
    TableFunction f("dbweaver", {LogicalType::TABLE}, nullptr, FnBind, FnInit);
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

} // namespace duckdb

// ============================================================
//  SECTION F — C entry point
// ============================================================
extern "C" {
    DUCKDB_CPP_EXTENSION_ENTRY(dbweaver, loader) {
        duckdb::LoadInternal(loader);
    }
}
