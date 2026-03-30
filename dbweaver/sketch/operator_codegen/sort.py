# coding: utf-8
from .operator_gen import BaseOperator, register_operator, CodegenContext
import json


@register_operator("sort")
class SortOperator(BaseOperator):
    """
    Generate prompt(s) for a SORT (ORDER BY) operator UDTF that can produce or refine C++ snippets.

    Expected step schema (subset):

        step = {
            "op": "sort",
            "name": "...",                 # optional
            "inputs": [...],               # logical input column names / positions
            "outputs": [...],              # logical output columns (must match BaseOperator.validate)
            "sort_keys": [
                {
                    "key": "o_orderdate",
                    "order": "DESC",       # optional, default "ASC"
                    "nulls_first": True,   # optional, default = None (meaning "use default")
                    "type": "DATE"         # optional type hint: "DATE", "INTEGER", "VARCHAR", etc.
                },
                ...
            ],
            "output_restriction": 100,     # optional LIMIT K (Top-K)
            "stable": False,               # optional stable sort requirement
        }
    """

    # -------------------------------
    # Validation of step definition
    # -------------------------------
    def validate(self) -> None:
        # Base checks (e.g., outputs must be a list)
        super().validate()

        s = self.step
        sk = s.get("sort_keys")
        if not isinstance(sk, list) or len(sk) == 0:
            raise ValueError(f"{self.name}: 'sort_keys' must be a non-empty list")

        for i, spec in enumerate(sk):
            if not isinstance(spec, dict) or "key" not in spec:
                raise ValueError(
                    f"{self.name}: sort_keys[{i}] must be a dict with at least 'key'"
                )

            # Normalize order
            ordv = str(spec.get("order", "ASC")).upper()
            if ordv not in ("ASC", "DESC"):
                raise ValueError(
                    f"{self.name}: sort_keys[{i}].order must be 'ASC' or 'DESC', got {ordv}"
                )

            # Normalize nulls_first if present
            nf = spec.get("nulls_first")
            if nf is not None and not isinstance(nf, bool):
                raise ValueError(
                    f"{self.name}: sort_keys[{i}].nulls_first must be bool if provided"
                )

    # ---------------------------------------------------------
    # Main prompt to extend previously generated JSON snippets
    # ---------------------------------------------------------
    def build_prompt(self, state, ctx: CodegenContext, opt_msg: str) -> str:
        s = self.step
        snippets = state.get("snippets", {})

        # Only surface the last refine_round_summary (if any) to the LLM
        extras = ctx.extras if isinstance(ctx, CodegenContext) else {}
        refine_summary = extras.get("refine_round_summary", "").strip() if extras else ""
        extras_text = refine_summary or "(no previous refinement summary available)"

        # Flatten sort specification for display
        def fmt_sk(spec):
            return {
                "key": spec.get("key"),
                "order": str(spec.get("order", "ASC")).upper(),
                "nulls_first": spec.get("nulls_first"),
                "type": spec.get("type"),
            }

        sort_spec = [fmt_sk(x) for x in s.get("sort_keys", [])]
        limit_val = s.get("output_restriction")
        stable = bool(s.get("stable", False))

        inputs = s.get("input_cols")
        outputs = s.get("output_cols")

        sort_spec_json = json.dumps(sort_spec, ensure_ascii=False, indent=2)
        limit_repr = (
            str(limit_val) if limit_val is not None else "None (no row count limit)"
        )

        return f"""
You are implementing a SORT (ORDER BY) operator inside an existing UDTF.
Your task is to extend the previously generated C++ snippets (stored as a JSON object `snippets`)
to implement ORDER BY semantics for the operator described below.

Last round summary from previous refinement, if available, you can avoid repeating the same mistakes:
{extras_text}

Sort specification:
- sort_keys (in priority order, each with per-key policy):
{sort_spec_json}
- limit (interpreted as Top-K if not None): {limit_repr}
- stable: {stable}

Operator I/O:
- inputs: {inputs}
- outputs: {outputs}

Algorithm policy (KEEP IT SIMPLE AND CONSISTENT):
- If limit (K) is NOT None:
  - Implement an in-memory **bounded Top-K** strategy:
    - Maintain a heap of row references keyed by a lightweight KeyView structure.
    - The heap size must never exceed K.
    - The comparator must respect the multi-key ORDER BY, including ASC/DESC and NULLS FIRST/LAST.
- If limit is None:
  - Implement a full in-memory sort:
    - Buffer row references + key views for all incoming rows in FnGlobalState.
    - Perform a single full sort in FnFinalize using a comparator that respects the ORDER BY semantics.
- If `stable` is true:
  - Ensure the overall ordering is stable, either by using a stable sort algorithm (or library call),
    or by adding a monotonically increasing tie-breaker index to the key and respecting it in the comparator.

Support_code:
- Define new structures like:
struct SortKeyView {{
    int32_t x;
    string_t y;
}};

struct SortRowComparator {{
    bool operator()(const SortKeyView &a, const SortKeyView &b) const {{
        // Implement multi-key comparison respecting ASC/DESC and NULLS FIRST/LAST
        // for each sort key in order of priority.
        // Return true if 'a' should come before 'b'.
        // ...
        return false; // Placeholder
    }}
}};

For example, if the sort keys are (x,y), the comparator should be:
```cpp
struct SortRowComparator {{
bool operator()(const SortKeyView &a, const SortKeyView &b) const {{
        if (a.x != b.x) return a.x < b.x;
        return std::strcmp(a.y, b.y) < 0;
    }};
}};
```
struct SortState {{
    std::vector<SortKeyView> buffer;
    bool sorted = false;

    inline void AddRow(...) {{
        buffer.push_back(SortKeyView{{...}});
    }}

    inline void SortNow() {{
        if (!sorted) {{
            std::sort(buffer.begin(), buffer.end(), SortRowComparator{{}});
            sorted = true;
        }}
    }}
}};

- Add SortState in FnGlobalState

Step1: Store data in Sort State buffer:
- If the sort data is determined during the Execution Phase, store it in the SortState buffer within FnExecution.
- If the sort data is determined during the Finalize Phase, store it in the SortState buffer within FnFinalize.
- If some data is encoded, and the encoded order is consistent with the original data order, decoding can be deferred: perform sorting directly on the encoded data and apply decoding only during output. This approach can significantly improve sorting performance.
- If decoding logic already exists in the code, you may modify it to sort first, then decode.


Step2: Sort data in Finalize (FnFinalize):
- Extend the existing FnFinalize implementation to:
  - For limit is None:
    - Add code in inline void SortNow() to perform full sort like std::sort(buffer.begin(), buffer.end(), SortRowComparator{{}});
  - For limit is not None:
    - Add code in inline void SortNow() to build a heap of size K using std::make_heap and std::push_heap with SortRowComparator.

- After producing all rows, mark the operator as finished according to the existing protocol.

Previous JSON snippets (must be EXTENDED, not replaced):
{snippets}

""".strip()

    # -----------------------------------------------------------------
    # Optimization helper prompt — currently disabled (no extra hint)
    # -----------------------------------------------------------------
    def build_opt_prompt(self, state, ctx: CodegenContext) -> str:
        """
        For now we don't generate a separate optimization hint for SORT.
        Algorithm choice is driven directly by the main build_prompt rules.
        """
        return ""
