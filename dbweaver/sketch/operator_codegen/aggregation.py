# aggregation.py
# coding: utf-8
from .operator_gen import BaseOperator, register_operator, CodegenContext


@register_operator("agg")
class AggregationOperator(BaseOperator):
    def validate(self) -> None:
        super().validate()
        if not self.step.get("aggs"):
            raise ValueError(f"{self.name}: 'aggs' required")

    def _is_global_group(self, step):
        gk = step.get("group_keys", [])
        if not gk:
            return True
        if len(gk) == 1 and (gk[0] == 1 or gk[0] == "1" or gk[0] in ("__GLOBAL__", "_ALL_ROWS_")):
            return True
        return False

    def _build_prompt_global(self, state, ctx: CodegenContext) -> str:
        """
        Single implicit group: all rows belong to one global aggregate.
        No GROUP BY keys, no hash table, all accumulators in FnGlobalState.
        """
        s = self.step
        snippets = state.get("snippets", {})
        s['outputs'] = state.get("ctx", {}).get("output", [])

        # Only surface the last refine_round_summary (if any) to the LLM,
        # so we don't flood the prompt with all ctx extras.
        extras = ctx.extras if isinstance(ctx, CodegenContext) else {}
        refine_summary = extras.get("refine_round_summary", "").strip() if extras else ""
        extras_text = refine_summary or "(no previous refinement summary available)"

        return f"""
You are generating C++ code for a SINGLE-GROUP aggregate (no explicit GROUP BY keys).
All input rows belong to one implicit global group.

Last round summary from previous refinement, if available, you can avoid repeating the same mistakes:
{extras_text}

Current snippets (to be extended, not replaced):
{snippets}

Semantics:
- There is exactly ONE group.
- Do NOT introduce any hash table, key struct, map, dictionary, or group→state lookup.
- All aggregate accumulators must be scalar fields inside FnLocalState and FnGlobalState for parallel execution, for example:
    struct FnGlobalState : public GlobalTableFunctionState {{
        // add fields like:  acc_sum; acc_count; ...
        std::mutex lock;
        std::atomic<idx_t> active_local_states {{0}};
        std::atomic<idx_t> merged_local_states {{0}};
        idx_t MaxThreads() const override {{
            return std::numeric_limits<idx_t>::max();
        }}
    }};
    struct FnLocalState : public LocalTableFunctionState {{
        // add fields like:  acc_sum; acc_count; ...
        bool merged = false;
    }};

- FnExecute should update local state scalar accumulators directly for each input row.
- FnFinalize should merge the local state into the global state and emit exactly ONE output row, with columns in the same order as `outputs`.

Where to modify snippets:
- In "support_code":
  - Extend the existing FnGlobalState and FnLocalState struct with the accumulator fields required
    by the aggregates defined in this step (e.g., sum, count, min, max, avg state).
  - Add small helper functions if necessary (e.g., initialization helpers), but keep them minimal.
  - Based on the input data types and output data types, ensure correct accumulator types to avoid overflow/underflow. Specifically, if the output is hugeint, use  __int128 for sum accumulators.
- In "execute_code":
  - Extend the existing FnExecute body so that it:
    * For each input row, based on group keys, expressions and measure values, updates the scalar accumulators in `l`:

- In "finalize_code":
  - Extend the existing FnFinalize body so that it:
    * merges the local state into the global state,
    * computes final aggregate values from the global state,
    * writes exactly one row into `out`,
    * and assigns columns in the same order as `outputs`.
    for example:
        ```cpp
                static OperatorFinalizeResultType FnFinalize(ExecutionContext &, TableFunctionInput &in,
                                                            DataChunk &out) {{
                    auto &g = *reinterpret_cast<FnGlobalState *>(in.global_state.get());
                    auto &l = in.local_state->Cast<FnLocalState>();

                    // Merge the local partial aggregate into the global aggregate exactly once.
                    if (!l.merged) {{
                        {{
                            std::lock_guard<std::mutex> guard(g.lock);
                            g.sum += l.sum;
                        }}
                        l.merged = true;
                        g.merged_local_states.fetch_add(1, std::memory_order_relaxed);
                    }}

                    // Only the *last* local state to merge emits the final single-row result.
                    // All other threads return FINISHED with an empty chunk.
                    const auto merged = g.merged_local_states.load(std::memory_order_relaxed);
                    const auto active = g.active_local_states.load(std::memory_order_relaxed);
                    if (active > 0 && merged == active) {{
                        int64 final_sum;
                        {{
                            std::lock_guard<std::mutex> guard(g.lock);
                            final_sum = g.sum;
                        }}
                        out.SetCardinality(1);
                        out.SetValue(0, 0, Value::INTEGER(final_sum));
                    }} else {{
                        out.SetCardinality(0);
                    }}
                    return OperatorFinalizeResultType::FINISHED;
                }}
        ```

New operator descriptor (incremental input):
- operator type: agg
- inputs: {s.get('input_cols')}
- outputs: {s.get('output_cols')}
- group_keys: {s.get('group_keys')}
- aggs: {s.get('aggs')}

Modification guidelines:
- Extend the existing fields in the JSON snippets:
  * Add accumulator fields and helpers to "support_code" for both FnGlobalState and FnLocalState.
  * Append the per-row update logic to "execute_code".
  * Append comment about the logical code to decode the group keys if the are encoded to give the hints to next operators.
- Do not remove or rewrite existing logic unless it is clearly incompatible with
  having a single group aggregate.
- Reuse any existing accumulators if they already match the required aggregates,
  to avoid duplicates.
""".strip()

    def build_prompt(self, state, ctx: CodegenContext, opt_msg: str) -> str:
        s = self.step
        if self._is_global_group(s):
            return self._build_prompt_global(state, ctx)

        snippets = state.get("snippets", {})

        return f"""
You are generating C++ code for a HASH-BASED GROUP BY aggregate operator inside a DuckDB table function.


Current snippets (to be extended, not replaced):
{snippets}

Aggregation information:
{s}

Goal:
- Add or extend aggregation logic for the specified group keys and aggregate expressions.
- Use a hash map (std::unordered_map) keyed by the grouping columns, where each entry holds an aggregate state.
- Keep all persistent state in FnGlobalState and FnLocalState for parallel execution, accessed via:
    auto &g = *reinterpret_cast<FnGlobalState *>(in.global_state.get());
    auto &l = in.local_state->Cast<FnLocalState>();

Where to modify snippets:
1) "support_code":
   - Define a GroupKey struct that holds all grouping column values (use appropriate DuckDB types: int32_t, int64_t, string_t, date_t, etc.)
   - Define a GroupKeyHash functor for hashing GroupKey
   - Define an AggState struct that holds accumulator fields for all aggregates (e.g., sum, count, min, max, avg)
   - Note: GroupKeyEqual is not needed if GroupKey has operator== defined
   - Add a hash map field to FnLocalState:
        // add fields like:  acc_sum; acc_count; ...
        std::unordered_map<GroupKey, AggState, GroupKeyHash> agg_map;
   - Include small helper functions if needed (e.g., to initialize AggState).
2) "execute_code":
   - Extend the existing FnExecute body to:
     * read input values (as prepared by previous operators, such as the input operator),
     * some value after expression computation, do not remove the computation logical, just use the value defined in previous steps.
     * check any filter conditions if needed,
    - Important:
     * Do not assume every grouping key is a raw input column.
     * If a grouping key is not provided, for example, an expression, compute its value inside the execution loop using the current row’s input values, then store that evaluated result into GroupKey.
     * update the local state scalar accumulators in `l`:
        auto &l = in.local_state->Cast<FnLocalState>();
   - For string keys, use string_t directly (avoid std::string conversions in hot loop).

3) "finalize_code":
   - Extend the existing FnFinalize body so that it:
     * merges the local state into the global state,
     * computes final aggregate values for each group,
     * writes exactly one row into `out`,
     * and assigns columns in the same order as `outputs`.

Example structure (for reference):
```cpp
// In support_code:
struct GroupKey {{
    int32_t key1;  // or int64_t, string_t, date_t, etc. based on actual group keys
    string_t key2; // if multiple keys
    // ... add all grouping columns
    
    bool operator==(const GroupKey& other) const {{
        return key1 == other.key1 && key2 == other.key2; // compare all fields
    }}
}};

struct GroupKeyHash {{
    size_t operator()(const GroupKey& k) const {{
        size_t h = 0;
        h ^= std::hash<int32_t>{{}}(k.key1) + 0x9e3779b9 + (h << 6) + (h >> 2);
        // For string, use DuckDB's Hash function:
        h ^= duckdb::Hash(k.key2.GetData(),k.key2.GetSize()) + 0x9e3779b9 + (h << 6) + (h >> 2);
        // If there are more keys, add them to the hash:
        h ^= std::hash<int32_t>{{}}(k.key3) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }}
}};

struct AggState {{
    int64_t sum_val = 0;
    int64_t count_val = 0;
    int32_t min_val = std::numeric_limits<int32_t>::max();
    int32_t max_val = std::numeric_limits<int32_t>::min();
    // ... add accumulators for all aggregates
}};

struct FnLocalState : public LocalTableFunctionState {{
    std::unordered_map<GroupKey, AggState, GroupKeyHash> agg_map;
}};

struct FnGlobalState : public GlobalTableFunctionState {{
    std::unordered_map<GroupKey, AggState, GroupKeyHash> agg_map;
    std::mutex lock;
    std::atomic<idx_t> active_local_states {{0}};
    std::atomic<idx_t> merged_local_states {{0}};
    idx_t MaxThreads() const override {{
        return std::numeric_limits<idx_t>::max();
    }}
}};

// In execute_code:
auto &l = in.local_state->Cast<FnLocalState>();
GroupKey key;
key.key1 = /* read from input.data[0] */;
key.key2 = /* read from input.data[1] */;
auto &state = l.agg_map[key];  // insert or get existing
state.sum_val += /* measure value */;
state.count_val++;
// ... update other accumulators

// In finalize_code:
if(!l.merged) {{
    {{
        std::lock_guard<std::mutex> guard(g.lock);
        for (const auto &entry : l.agg_map) {{
            const GroupKey &key = entry.first;
            const AggState &state = entry.second;
            // merge local state with global state
            g.agg_map[key] = state;
        }}
    }}
    l.merged = true;
    g.merged_local_states.fetch_add(1, std::memory_order_relaxed);
}}
const auto merged = g.merged_local_states.load(std::memory_order_relaxed);
const auto active = g.active_local_states.load(std::memory_order_relaxed);
if (active > 0 && merged == active) {{
    {{
        std::lock_guard<std::mutex> guard(g.lock);
        for (const auto &entry : g.agg_map) {{
            const GroupKey &key = entry.first;
            const AggState &state = entry.second;
            // compute final aggregates
            double avg_val = state.count_val > 0 ? (double)state.sum_val / state.count_val : 0.0;
            // write to out
            out.SetValue(0, key.key1);
            out.SetValue(1, key.key2);
            out.SetValue(2, state.sum_val);
            out.SetValue(3, state.count_val);
            out.SetValue(4, state.min_val);
            out.SetValue(5, state.max_val);
            out.SetValue(6, avg_val);
            // ... set all output columns
        }}
    }}
}} else {{
    out.SetCardinality(0);
}}
return OperatorFinalizeResultType::FINISHED;
```

Modification guidelines:
- Extend the existing JSON snippets in-place:
  * Add new types and global state fields to "support_code".
  * Append per-row update logic to "execute_code".
  * Append per-group finalization and output emission to "finalize_code".
- Reuse existing helpers and state where possible; avoid duplicate definitions.
- Keep the control flow and naming consistent with the existing snippets.
- Based on the input data types and output data types, ensure correct accumulator types to avoid overflow/underflow. Specifically, if the output is hugeint, use __int128 for sum accumulators.
""".strip()
