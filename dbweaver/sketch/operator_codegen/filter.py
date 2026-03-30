# coding: utf-8
from .operator_gen import BaseOperator, register_operator, CodegenContext
import json


@register_operator("filter")
class FilterOperator(BaseOperator):
    def build_prompt(self, state, ctx: CodegenContext, opt_msg: str) -> str:
        # Columns metadata is expected in ctx.extras["columns"] as
        # { column_name: internal_ctype_string }
        if isinstance(ctx, CodegenContext):
            columns_meta = ctx.extras.get("columns", {})
            extras = ctx.extras
        else:
            # Fallback if a plain dict is passed instead of CodegenContext
            columns_meta = (ctx or {}).get("columns", {})
            extras = ctx or {}

        # Only surface the last refine_round_summary (if any) to the LLM,
        # so we don't flood the prompt with all ctx extras.
        refine_summary = extras.get("refine_round_summary", "").strip() if extras else ""
        extras_text = refine_summary or "(no previous refinement summary available)"

        s = self.step
        filters = s.get("filters", [])
        filter_str = ",".join(filters)
        filter_datatype = []
        columns_spec = ""
        for idx, col in enumerate(s.get("input_cols", [])):
            col_type = columns_meta.get(col, "UNKNOWN")
            columns_spec += f"- {col}, index {idx}, ctype {col_type}\n"
            if col in filter_str:
                filter_datatype.append(col_type)
        api_hints = ""
        for col_type in filter_datatype:
            if col_type == "DATE":
                with open("template/api_reference/date.txt", "r") as f:
                    api_hints += f.read()
        snippets = state.get("snippets", {})

        return f"""
You are generating C++ code for a FILTER operator inside a DuckDB table function.

Last round summary from previous refinement, if available, you can avoid repeating the same mistakes:
{extras_text}

Current snippets (to be extended, not replaced):
{snippets}

New operator to add (incremental input):
- operator type: filter
- inputs: {s.get('inputs')}
- filters: {filters}

Where to modify the JSON snippets:
1) "support_code":
   - Add small helper functions or structs only if they are required by the new filter logic
     (for example, a local date-to-days helper, or small predicate utilities).
   - Keep helpers minimal and self-contained; do not introduce large generic frameworks.

2) "execute_code":
   - Extend the existing FnExecute body so that it evaluates the filter predicates for each row
     of the incoming DataChunk `input`.
   - Use the columns described below (names, indices, and internal C++ types) and the UVF setup
     created by previous operators (e.g., the input operator).
   - For each row, compute the predicate result using the appropriate C++ operators for the
     column types (comparison, arithmetic, logical AND/OR, etc.) and skip rows that do not
     satisfy the filters.

   The filtering pattern should look conceptually like this (simplified sketch):

       for (idx_t row_idx = 0; row_idx < input.size(); row_idx++) {{
           // map logical row index to physical index for each column via its UnifiedVectorFormat
           idx_t i_col = col_uvf.sel->get_index(row_idx);
           if (!col_uvf.validity.RowIsValid(i_col)) {{
               // decide whether NULL should be considered passing or failing the predicate;
               // by default, treat NULL as failing and skip this row
               continue;
           }}
           <ctype> col_val = col_ptr[i_col];

           // evaluate filter predicates on col_val (and other columns as needed)
           // if the combined predicate is false, continue;  // row is filtered out

           // if the row passes, keep executing downstream logic (e.g., update aggregates, copy to output, etc.)
       }}

3) "finalize_code":
   - Normally, filters do not need to change FnFinalize. Only extend "finalize_code" if the
     filter logic requires additional state finalization (most cases do not).
   - If you leave "finalize_code" unchanged, simply propagate the existing content.

Columns (name, index, internal_ctype) in input order:
{columns_spec}

Filter semantics:
- Interpret the list in `filters` as a set of predicates that must all hold for a row to pass.
- For simple comparisons (e.g., col > constant, col = constant, col BETWEEN a AND b),
  generate tight, branch-friendly code that uses the correct physical type and avoids
  unnecessary casts.
- For complex expressions (combinations of AND/OR, arithmetic, or functions), structure
  the code so that cheap checks come first and short-circuit when possible.

API Reference:
Please reference to this updated api list from DuckDB, and avoid using outdated APIs:
{api_hints} 

Modification guidelines:
- Extend the existing "support_code" and "execute_code" fields in the snippets JSON;
  avoid rewriting or deleting existing logic.
- Reuse any existing helpers if they already implement part of the required predicate logic.
- Keep naming and control flow consistent with the existing code.
- The final JSON you return must follow the global output format from the system prompt
  (same four fields: headers, support_code, execute_code, finalize_code).
""".strip()
