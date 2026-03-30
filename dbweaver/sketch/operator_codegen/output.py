# coding: utf-8
import json
from .operator_gen import BaseOperator, register_operator, CodegenContext


@register_operator("output")
class OutputOperator(BaseOperator):
     def build_prompt(self, state, ctx: CodegenContext, opt_msg: str) -> str:
          prev = state.get("snippets", {}) or {}
          previous_json = {
               "headers": prev.get("headers", "") or "",
               "bind_code": prev.get("bind_code", "") or "",
               "support_code": prev.get("support_code", "") or "",
               "execute_code": prev.get("execute_code", "") or "",
               "finalize_code": prev.get("finalize_code", "") or "",
               "function_define": prev.get("function_define", "") or "",
          }
          previous_json_str = json.dumps(previous_json, ensure_ascii=False, indent=2)

          # output type metadata is expected in ctx.extras["output"]
          if isinstance(ctx, CodegenContext):
               output_datatype = ctx.extras.get("output", {})
          else:
               output_datatype = (ctx or {}).get("output", {})

          output_datatype_json = json.dumps(output_datatype, ensure_ascii=False, indent=2)
          hints = ""
          for k, v in output_datatype.items():
               if "HUGEINT" in v:
                    hints += """
     When converting a 128-bit accumulator (__int128) into DuckDB’s hugeint_t, 
     use a helper that splits the two’s-complement value into a signed upper 64 bits and an unsigned lower 64 bits, like this:
     ```cpp
     inline hugeint_t ToHugeint(__int128 acc) {{
          hugeint_t result;
          result.lower = static_cast<uint64_t>(acc);          // low 64 bits
          result.upper = static_cast<int64_t>(acc >> 64);     // high 64 bits (sign-extended)
          return result;
          }}
     ```
                    """
          return f"""
You are generating C++ code for the OUTPUT operator of a DuckDB table function.

The OUTPUT operator is responsible for:
- Defining the final output schema in FnBind (names + LogicalType).
- Converting internal accumulators / state into the final DataChunk `out` in FnFinalize,
  including DECIMAL scale handling and type-safe Value construction.

Current snippets (must be preserved and extended, not discarded):
{previous_json_str}

Authoritative output type mapping (column_name → LogicalType kind):
{output_datatype_json}

Hints / notes:
{hints}

Your job is to extend the existing snippets by updating the following modules:

1) "bind_code"  — define FnBind and the output schema. Don't trust input snippets, if you find incorrect names, types, you should correct them.
   - Produce a C++ function body for:
        static unique_ptr<FunctionData> FnBind(
            ClientContext &, TableFunctionBindInput &,
            vector<LogicalType> &return_types,
            vector<string> &names
        ) {{
            ...
            return make_uniq<FnBindData>();
        }}
   - For each output column, in the exact order implied by `output_datatype`:
        * push its name into `names`,
        * push its LogicalType into `return_types`.
   - Mapping from `output_datatype[name]` (string) to LogicalType:
        * INTEGER-family → LogicalType::INTEGER / BIGINT / etc.
        * DOUBLE        → LogicalType::DOUBLE
        * VARCHAR       → LogicalType::VARCHAR
        * DECIMAL(p,s)  → LogicalType::DECIMAL(p, s) (parse p and s from the string).
   - Do NOT change how FnBindData is allocated; always return make_uniq<FnBindData>().

2) "finalize_code"  — materialize final rows into `out`
   - Extend the existing FnFinalize body so that it:
        * determines the total number of output rows,
        * calls `out.SetCardinality(row_count);`,
        * sets each cell via `out.SetValue(col_idx, row_idx, Value(...));` using the
          correct Value factory for the output column type.
   - Type-safe Value selection:
        * For integer-like outputs, choose from:
          Value::BOOLEAN, Value::TINYINT, Value::SMALLINT, Value::INTEGER,
          Value::BIGINT, Value::HUGEINT, Value::UHUGEINT, Value::UTINYINT,
          Value::USMALLINT, Value::UINTEGER, Value::UBIGINT.
        * For DOUBLE outputs (e.g., AVG/division), use Value::DOUBLE.
        * For VARCHAR outputs, use Value(string).
   - Make sure each output column reads from the correct accumulator or state variable
     (the ones defined by previous operators such as aggregation).

3) DECIMAL-aware output handling (only if `output_datatype` contains DECIMAL)
   - All intermediate math for DECIMAL(p,s) has already been done in the integer domain
     using the underlying integer storage (scaled by 10^s).
   - At OUTPUT, you are allowed to:
        * either construct a DECIMAL(p,s) Value directly (preferred if possible),
        * or format the scaled integer into a precise string, then use Value(formatted_string).
   - If you cannot construct a DECIMAL Value directly, add a helper in "support_code" like:

        // FormatScaledInteger(raw, s): format a scaled integer `raw` with `s` decimal places.
        // Examples:
        //   raw = 12345, s = 2 -> "123.45"
        //   raw = -7,    s = 3 -> "-0.007"
        std::string FormatScaledInteger(/* underlying integer type */ raw, uint8_t scale);

     and call it from FnFinalize:
        std::string dec = FormatScaledInteger(raw_value, scale);
        out.SetValue(col_idx, row_idx, Value(dec));

   - Never convert scaled DECIMAL integers to double in the streaming/execute phase; if a
     column is DOUBLE by design (e.g., AVG or division), convert to double only at OUTPUT.

4) DECIMAL precision propagation (for choosing LogicalType)
   - When inferring DECIMAL(p,s) outputs from DECIMAL inputs, follow these rules:
        Expression            result precision p'       result scale s'
        ----------------------------------------------------------------
        a ± b (same scale)    max(p₁, p₂) + 1          s
        a × b                 min(p₁ + p₂, 38)         min(s₁ + s₂, 38)
        SUM(a)                38                        s₁
        AVG(a), division      DOUBLE                    (use LogicalType::DOUBLE)
     - COUNT(*) → BIGINT.
     - For DECIMAL outputs, add brief inline comments explaining chosen p' and s'.

5) "headers" and "support_code"
   - Only extend these if needed:
        * Add declarations for FnBind, FnFinalize, FnGlobalState, FnBindData, and helpers
          like FormatScaledInteger, if they are missing in the existing snippets.
        * Deduplicate includes and helper definitions; do not introduce multiple identical
          declarations.
   - Keep helpers small and directly related to OUTPUT duties.

How to modify the JSON snippets:
- Start from `previous_json` and construct a new JSON object with five keys:
    {{
      "headers": "...",
      "support_code": "...",
      "bind_code": "...",
      "execute_code": "...",
      "finalize_code": "...",
      "function_define": "...",
    }}
- Preserve valid logic from `previous_json` in all four existing fields.
- Only append or minimally adjust:
    * "headers" and "support_code" to define FnBind/FnFinalize and helpers,
    * "bind_code" to define the binding logic,
    * "finalize_code" to materialize the final result.
- Keep naming and control flow consistent with the existing snippets and with how previous
  operators (e.g., aggregation) produced their state.
""".strip()
