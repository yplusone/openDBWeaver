# input.py
# coding: utf-8
from .operator_gen import BaseOperator, register_operator, CodegenContext


@register_operator("input")
class InputOperator(BaseOperator):
    def build_prompt(self, state, ctx: CodegenContext, opt_msg: str) -> str:
        # Columns metadata is expected in ctx.extras["columns"] as
        # { column_name: internal_ctype_string }
        if isinstance(ctx, CodegenContext):
            columns_meta = {k.lower(): v for k, v in ctx.extras.get("columns", {}).items()}
        else:
            # Fallback if a plain dict is passed instead of CodegenContext
            columns_meta = (ctx or {}).get("columns", {})

        columns_spec = ""
        for idx, col in enumerate(self.step.get("input_cols", [])):
            col_type = columns_meta.get(col.lower(), "UNKNOWN")
            columns_spec += f"- {col}, index {idx}, ctype {col_type}\n"
        expression_compute = f"At this stage, some additional data computation is still required: the input is {self.step.get('input_cols', [])}, the output is {self.step.get('output_cols', [])}, and the computed expressions are {self.step.get('expressions', [])}."
        snippets = state.get("snippets", {})
        return f"""
You are generating C++ code for the INPUT stage of a DuckDB table function.
The existing snippets JSON represents the current implementation of the operator:

SNIPPETS (current state to extend, not replace):
{snippets}

Your task:
- Extend the existing snippets so that FnExecute reads the required input columns
  from the incoming DataChunk `input` using UnifiedVectorFormat.
- Focus primarily on the "execute_code" field; only touch "support_code" or "headers"
  if you truly need additional helpers (e.g., small inline utilities).
- Do NOT change "finalize_code" unless it is strictly necessary.

For each input column, in the given order, generate C++ code that:
1) Declares a UnifiedVectorFormat handle:
   UnifiedVectorFormat <col>_uvf;
2) Loads the column into UVF:
   input.data[<index>].ToUnifiedFormat(input.size(), <col>_uvf);
3) Creates a typed pointer to the physical data:
   <ctype>* <col>_ptr = (<ctype>*)<col>_uvf.data;

The <ctype> must match DuckDB's internal physical C++ type, for example:
- BOOLEAN        → bool
- TINYINT        → int8_t
- SMALLINT       → int16_t
- INTEGER        → int32_t
- BIGINT         → int64_t
- UTINYINT       → uint8_t
- USMALLINT      → uint16_t
- UINTEGER       → uint32_t
- UBIGINT        → uint64_t
- REAL/FLOAT4    → float
- DOUBLE/FLOAT8  → double
- VARCHAR/BLOB   → string_t
- DATE           → date_t
- TIME           → dtime_t
- TIMESTAMP      → timestamp_t
- TIMESTAMP_TZ   → timestamp_tz_t
- INTERVAL       → interval_t
- DECIMAL(p,s)   → int16_t / int32_t / int64_t / hugeint_t (chosen by precision p)
- ENUM           → uint16_t / uint32_t (depending on physical storage)

DECIMAL handling rules (if any DECIMAL columns exist):
- Choose the physical integer type based on precision p:
    p <= 4  → int16_t
    p <= 9  → int32_t
    p <= 18 → int64_t
    p > 18  → hugeint_t
- Keep all arithmetic in that integer domain (do not cast to float/double in the hot loop).
- Treat values as scaled integers ×10^s and add a brief comment about the scale.

Consider the NULL handling and emit code refer to the following pattern:
```

// 1) Batch-level NULL summary (whether each column is fully non-NULL)
// validity bitmaps
auto &valid_<col1>  = <col1>_uvf.validity;
auto &valid_<col2>  = <col2>_uvf.validity;
// Add more columns as needed
const bool col1_all_valid = valid_<col1>.AllValid();
const bool col2_all_valid = valid_<col2>.AllValid();
// Add more columns as needed:
// const bool colN_all_valid = valid_<colN>.AllValid();

// 2) FAST BRANCH: all relevant columns have no NULLs in this batch
if (col1_all_valid && col2_all_valid /* && colN_all_valid ... */) {{
    // --- Fast path: no per-row NULL checks ---
    for (<ROW_INDEX_T> row_idx = 0; row_idx < <NUM_ROWS>; ++row_idx) {{
        // Directly load values without RowIsValid checks

        idx_t i_<col1> = <col1>_uvf.sel->get_index(row_idx);
        date_t v1 = <col1>_ptr[i_<col1>];
        uint32_t v2 = <col2>_ptr[i_<col2>];
        //  vN = <COL_N>_ptr[i_<COL_N>];

        // ======================================
        //  Core computation logic (no NULLs)
        //<<CORE_COMPUTE>>
        // ============================
    }}
}} else {{
    // --- Slow path: at least one column may contain NULLs ---
    for (<ROW_INDEX_T> row_idx = 0; row_idx < <NUM_ROWS>; ++row_idx) {{
        // For each column that is not fully valid, check this row
        idx_t i_<col1> = <col1>_uvf.sel->get_index(row_idx);
        idx_t i_<col2> = <col2>_uvf.sel->get_index(row_idx);

        if (!col1_all_valid && !valid_<col1>.RowIsValid(i_<col1>)) {{
            continue; // row is NULL in column 1 → skip
        }}
        if (!col2_all_valid && !valid_<col2>.RowIsValid(i_<col2>)) {{
            continue;
        }}
        // Repeat for additional columns

        // At this point, all required columns are valid for this row

        date_t v1 = <col1>_ptr[i_<col1>];
        uint32_t v2 = <col2>_ptr[i_<col2>];
        // auto vN = <COL_N>_ptr[i_<COL_N>];

        // ======================================
        //  Core computation logic (NULL-safe)
        //<<CORE_COMPUTE>>
        // ======================================
    }}
}}
```

Requirements about how to modify the snippets:
- Append your new UnifiedVectorFormat setup and loop to the existing "execute_code"
  instead of replacing it.
- If you introduce helper functions or structs, place them in "support_code" and
  keep them minimal and self-contained.
- Keep variable names and control flow consistent with what is already in the snippets.

Columns (name, index, internal_ctype) in order:
{columns_spec}

{expression_compute}

For expressions, add comments of the original expression.
""".strip()
