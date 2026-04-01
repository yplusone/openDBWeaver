"""Candidate generation chains for LATS (code optimization)."""
import json
import re
from typing import List, Dict, Any, Optional

from langchain_core.prompt_values import ChatPromptValue
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig, chain as as_runnable
from langchain_core.tools import tool
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dbweaver.optimize.setup import advanced_llm,think_llm
from dbweaver.utils.llm_output_parse import extract_json
from dbweaver.optimize.optimize_agent import optimize_agent
from dbweaver.env.code_checker import CodeChecker
from config import THREADS

# Default optimization hints (can be customized)
DEFAULT_OPTIMIZATION_HINTS = [
    {
        "id": "flat_array_aggregation",
        "title": "Use flat array instead of hash map for aggregation",
        "description": "Call tool request_datadistribution to get the date range, we can map values to array indices. This eliminates hash computation and improves cache locality."
    },
    {
        "id": "reduce_unecessary_memcpy_or_string_construction",
        "title": "Reduce unnecessary memcpy or string construction",
        "description": "Avoid unnecessary memcpy or string construction in the hot path. For example, if there is a string concatenation in the hot path, you can avoid it by using pointer instead of copy value."
    }
]


# Base prompt template for generating optimization directions
prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a C++ code optimization strategist for DuckDB extensions. And we want to rewrite execution code for query to beat duckdb's performance. You are encouraged to think how to use runtime information, query-specific features to customize execution code that are more fast than duckdb's performance."
            "Your task is to propose high-level optimization directions/strategies, NOT to write code patches.\n\n"
            "You will be given:\n"
            "1. C++ code to optimize\n"
            "2. SQL queries (original and split)\n"
            "3. Predefined optimization hints (MUST CHECK THESE FIRST)\n"
            "4. History of already tried optimizations (AVOID these!)\n\n"
            "IMPORTANT - Selection Priority:\n"
            "1. **FIRST**, propose hints that have most potential to improve performance\n"
            "2. **SECOND**, if there are any functions or data structures with mediocre performance, consider replacing them with higher-performance alternatives. For example, unordered_map can be replaced with flat_hash_set/flat_hash_map (e.g., by #include <absl/container/flat_hash_set.h>)\n"
            "3. **AVOID** hints that are already in the history\n\n"
            "SYSTEM INFORMATION:\n"
            f"1. The system executes queries using {THREADS} threads, following the morsel-driven parallelism strategy as implemented in DuckDB. "
            "2. Each morsel consists of 1024 rows by default, and morsels are distributed dynamically to worker threads for efficient load balancing. "
            "3. This approach minimizes contention and maximizes CPU utilization, especially under analytical workloads. "
            "Guidelines:\n"
            "- The input of the dbweaver() function is determined, don't give any suggestion on it\n"
            "- Focus on ONE optimization per response\n"
            "- Be specific about WHAT to optimize and WHY\n"
            "- **IMPORTANT**: You can return 0 to N optimizations:\n"
            "  * If NO optimization opportunities exist → return empty array []\n"
            "  * If only 1-2 valid optimizations exist → return 1-2 (don't force N)\n"
            "  * Only return as many as you genuinely believe are promising\n"
            "  * DO NOT generate low-quality hints just to meet the count\n\n"
            "Output format (STRICT JSON ARRAY only, no markdown)\n"
            "[\n"
            "    {{\n"
            '        "id": "hint_id_if_from_predefined_else_new_id",\n'
            '        "title": "short description",\n'
            '        "description": "detailed case-by-case description"\n'
            "    }},\n"
            "    {{\n"
            '        "id": "hint_id_if_from_predefined_else_new_id",\n'
            '        "title": "short description",\n'
            '        "description": "detailed case-by-case description"\n'
            "    }},\n"
            "    ... (repeat for each diverse optimization direction)\n"
            "]\n\n"
            "Example outputs refer to predefined hints.\n"
        ),
        (
            "user",
            "Task: Propose {n} DIVERSE optimization directions for the code below.\n\n"
            "Goal: {goal}\n\n"
            "=== STEP 1: Review the code ===\n"
            "C++ code:\n```cpp\n{cpp_code}\n```\n\n"
            "Original query:\n```sql\n{query_template}\n```\n\n"
            "Split query (UDF):\n```sql\n{split_template}\n```\n\n"
            "=== STEP 2: Check already tried optimizations (AVOID these) ===\n"
            "{history_section}"
            "\n=== STEP 3: Review predefined hints (SELECT ONE if applicable) ===\n"
            "{hints_section}"
            "\n=== STEP 4: Generate UP TO {n} DIVERSE optimization directions ===\n"
        ),
        MessagesPlaceholder(variable_name="messages", optional=True),
    ]
)


def format_hints_section(state: dict[str, Any]) -> str:
    """Format optimization hints into a readable section."""
    optimization_hints = DEFAULT_OPTIMIZATION_HINTS
    column_is_constant = state.get("ctx", {}).get("is_constant_candidate", {})
    column_hint = ""
    for column_name, is_constant in column_is_constant.items():
        if is_constant:
            column_hint += f"{column_name},"
    if column_hint:
        optimization_hints.append({
        "id": "constant_vector_optimization",
        "title": "Detect and optimize ConstantVector inputs",
        "description": f"""{column_hint} are possibly be constant in one date chunk. If the input is constant vector, directly dealing it will be more efficient. You can check in the code like this:

            const bool x_is_constant = input.data[0].GetVectorType() == VectorType::CONSTANT_VECTOR;
            if (x_is_constant) {{
                <ctype> x_constant_value = ConstantVector::GetData<date_t>(input.data[0])[0];
            }} else {{
                //follow the normal way to load the column into UVF
                input.data[0].ToUnifiedFormat(input.size(), x_uvf);
            }}

        Only apply this on column set {column_hint}, check whether this will help to facilitate running the code. For 
        example, if the column is a filter column, you can only check the column once for whole batch. 
        
        For example, one filter predicate is x between a and b, you can check the column like this:
            bool x_passes_filter = false;
            if (x_is_constant) {{
                x_passes_filter = (x_constant_value >= a && b_constant_value < b);
                if (!x_passes_filter) {{
                    // If the constant x doesn't pass the filter, skip the entire batch
                    return OperatorResultType::NEED_MORE_INPUT;
                }}
            }}
       
        For other columns, don't check this since this will make the code complex."""
        })

    
    hints_text = "Available predefined optimization hints (you can choose one or propose your own):\n\n"
    for i, hint in enumerate(optimization_hints, 1):
        hints_text += f"{i}. **{hint['title']}**\n"
        hints_text += f"   {hint['description']}\n\n"
    
    return hints_text


def format_history_section(optimize_hint_history: Optional[List[str]] = None) -> str:
    """Format optimization history into a readable section."""
    if not optimize_hint_history or len(optimize_hint_history) == 0:
        return ""
    
    history_text = "⚠️ ALREADY TRIED (DO NOT repeat these):\n\n"
    for i, tried_opt in enumerate(optimize_hint_history, 1):
        history_text += f"{i}. {tried_opt}\n"
    
    history_text += "\nPropose something DIFFERENT from the above!\n\n"
    return history_text


def generate_candidates(
    state: Dict[str, Any],
    n: int = 3,
    optimize_hint_history: Optional[List[str]] = None,
) -> List[str]:
    """
    Generate N diverse optimization directions (NOT code patches).
    
    Args:
        state: State dictionary containing cpp_code, query info, etc.
        n: Number of optimization directions to generate
        optimize_hint_history: List of already tried optimization directions
    
    Returns:
        List of optimization direction strings
    """
    # Extract required info from state
    cpp_code = state.get("cpp_code", "")
    query_template = state.get("query_template", "")
    decomposed = state.get("decomposed", {})
    split_template = decomposed.get("split_template", "")
    
    # Format sections
    hints_section = format_hints_section(state)
    history_section = format_history_section(optimize_hint_history)
    goal = ""
    if state['reflection'].performance['processing_time']>state['original_processing_time']:
        goal = f"Original DuckDB processing time: {state['original_processing_time']}, current code processing time: {state['reflection'].performance['processing_time']}, we want to beat the original DuckDB performance."
    else:
        goal = f"Original DuckDB processing time: {state['original_processing_time']}, current code processing time: {state['reflection'].performance['processing_time']}, we want to improve the performance."
    # Build the prompt with n parameter
    prompt_input = {
        "n": n,
        "goal": goal,
        "hints_section": hints_section,
        "history_section": history_section,
        "query_template": query_template,
        "split_template": split_template,
        "cpp_code": cpp_code,
    }
    for _ in range(5):
        try:
            # Single generation that returns a JSON array
            response = think_llm.bind(tool_choice="none").invoke(
                prompt_template.format_messages(**prompt_input)
            )
            break
        except Exception as e:
            continue
    
    
    content = response.content.strip()
    if content == '':
        return []
    # Parse the JSON array
    try:
        optimization_directions = extract_json(content)
    except ValueError as e:
        print(f"[Generate Candidates] Error parsing JSON: {e}")
        return []
    

    print(f"[Generate Candidates] Generated {len(optimization_directions)} hints (requested up to {n})")
    
    if len(optimization_directions) == 0:
        print("[Generate Candidates] ⚠️ No optimization opportunities found - returning empty list")
    elif len(optimization_directions) < n:
        print(f"[Generate Candidates] ℹ️ Found {len(optimization_directions)} promising optimizations (less than {n} requested)")
    
    return optimization_directions


def generate_and_optimize_candidates(
    state: Dict[str, Any],
    n: int = 1,
    optimize_hint_history: Optional[List[str]] = None,
    max_iterations_per_hint: int = 10,
) -> List[Dict[str, Any]]:
    """
    
    Args:
        state: State dictionary containing cpp_code, query info, etc.
        n: Number of optimization directions to generate
        optimize_hint_history: List of already tried optimization directions
        max_iterations_per_hint: Maximum iterations for optimize_agent
    
    Returns:
        List of candidate results, each containing:
        - optimization_hint: The optimization direction
        - success: Whether optimization succeeded
        - optimized_code: The optimized code (if success)
        - performance: Performance metrics (scan_time, total_time)
        - message: Result message
        - tool_calls: Tool calls made during optimization
    """
    
    # Step 1: Generate up to N optimization directions
    print(f"\n[Candidate Chain] Generating up to {n} optimization directions...")
    optimization_hints = generate_candidates(state, n, optimize_hint_history)
    
    # Handle empty result
    if not optimization_hints or len(optimization_hints) == 0:
        print("[Candidate Chain] ⚠️ No optimization opportunities found. Returning empty candidate list.")
        return []
    
    print(f"[Candidate Chain] Generated {len(optimization_hints)} hints:")
    for i, hint in enumerate(optimization_hints, 1):
        print(f"  {i}. {hint}...")
    
    # Step 2: For each hint, call optimize_agent
    candidates = []
    checker = CodeChecker()
    
    for i, hint in enumerate(optimization_hints, 1):
        print(f"\n[Candidate Chain] Processing hint {i}/{len(optimization_hints)}...")
        
        # Call optimize_agent with this hint
        success, message, optimized_code, tool_calls = optimize_agent(
            cpp_code=state.get("cpp_code", ""),
            optimization_direction=hint,
            state=state,
            max_iterations=max_iterations_per_hint,
        )
        if success:
            print(f"[Candidate Chain] Optimization succeeded, testing performance...")
            split_execution_time = float(message.split("Split Execution Time:")[1].split(",")[0][:-1])
            split_processing_time = float(message.split("Split Processing Time:")[1].split(",")[0][:-1])
        # Step 3: Run the optimized code and get performance
            performance = {"total_time": split_execution_time, "processing_time": split_processing_time}
        else:
            performance = {"total_time": 99999999, "processing_time": 99999999}
        # Step 4: Package the result
        candidate_result = {
            "optimization_hint": hint,
            "success": success,
            "optimized_code": optimized_code,
            "performance": performance,
            "message": message,
            "tool_calls": tool_calls,
        }
        candidates.append(candidate_result)
    
    return candidates


def candidate_to_message(candidate: Dict[str, Any]) -> str:
    """
    将候选结果格式化为 message，用于添加到 hints_history
    
    Args:
        candidate: Candidate result from generate_and_optimize_candidates
    
    Returns:
        Formatted message string
    """
    hint = candidate["optimization_hint"]
    success = candidate["success"]
    performance = candidate["performance"]
    message = candidate["message"]
    
    # Format the message
    msg_parts = [f"[TRIED] {hint}"]
    
    if success:
        total_time = performance.get("total_time", 0.0)
        processing_time = performance.get("processing_time", 0.0)
        msg_parts.append(f"Result: SUCCESS")
        msg_parts.append(f"Performance: total={total_time:.3f}s, processing={processing_time:.3f}s")
    else:
        msg_parts.append(f"Result: FAILED - {message}")
    
    return " | ".join(msg_parts)

