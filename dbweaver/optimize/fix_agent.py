"""
fix_agent.py
------------
一个自动修复 C++ 代码的代理，循环调用 LLM 来解决问题，直到运行成功并且执行结果匹配。
"""

from __future__ import annotations

from typing import Any, Optional, Tuple
import sys
import os
import json
import time
import csv

_sys_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _sys_root)
from dbweaver.env.code_checker import CodeChecker
from config import (
    VALIDATION_FLAG_COMPILE_FAIL,
    VALIDATION_FLAG_RESULT_MISMATCH,
    VALIDATION_FLAG_PERF_NEED_OPT,
    BENCHMARK,
    SKETCH_DIR,
    SKETCH_FIX_DIR,
)
from dbweaver.sketch.code_combine import CodeOutput
from dbweaver.utils.apply_replacement import apply_replacements_from_list
from langchain_core.prompts import ChatPromptTemplate
from dbweaver.optimize.setup import advanced_llm
from utils.llm_output_parse import extract_json


def _build_query_context(
    original_query: Optional[str] = None,
    split_query: Optional[str] = None,
) -> str:
    """Build optional ORIGINAL QUERY / UDF QUERY blocks for fix prompts."""
    query_context = ""
    if original_query:
        query_context += f"ORIGINAL QUERY:\n```text\n{original_query}\n```\n\n"
    if split_query:
        query_context += f"UDF QUERY:\n```text\n{split_query}\n```\n\n"
    return query_context


# Shared by fix_node and fix_result_node: role, output contract (patch vs full JSON).
FIX_AGENT_SYSTEM_PROMPT = """You are an expert DuckDB extension engineer fixing a single-file C++ UDTF for a DuckDB extension.

**What you receive in the user message**
- A short paragraph describing this round's task (compile/link/runtime vs result mismatch, etc.).
- Optional SQL context: original query and/or UDF query fragment.
- Current C++ source, validation output (errors or mismatches), and optional prior attempt history.

**What you must do**
- Apply minimal edits to satisfy the user message task while preserving overall structure and intent.
- The final source must include dbweaver_extension.hpp.

**Output format — respond with JSON only (no markdown fences, no prose outside the JSON object)**

Use exactly one of two modes.

**Mode A — Patch mode** (small, localized changes):
{{
  "mode": "patch",
  "prefix": "one-line description of what was fixed",
  "patches": [{{"old_content": "...", "new_content": "..."}}]
}}
Rules: Each old_content must match a contiguous substring of the provided source exactly (full lines, verbatim). Include enough context to make each replacement unique.

**Mode B — Full code mode** (large or scattered changes):
{{
  "mode": "full",
  "prefix": "one-line description of what was fixed",
  "full_code": "...complete C++ source..."
}}
Rules: full_code must be the entire self-contained file. Do not omit sections with comments like "// ... rest unchanged ...".

**Choosing a mode**
- If the fix touches at most three small regions, prefer patch mode.
- If the fix is widespread or restructures the code significantly, use full code mode.
"""

FIX_COMPILE_USER_INSTRUCTIONS = """**Task for this round (compilation / link / runtime error):**
- Fix the problems reported in the error log below.
- Resolve missing symbols, includes, namespaces, and DuckDB API mismatches.
- Remove unresolved placeholders; deduplicate includes; keep UDTF signatures consistent with DuckDB conventions."""

FIX_UNKNOWN_USER_INSTRUCTIONS = """**Task for this round (unclassified validation failure):**
- The error was not classified; use the error log and optional SQL context to infer what went wrong.
- Prefer fixes for compilation, linking, or obvious runtime/API issues first; if the log indicates wrong results, align logic with the SQL below."""

FIX_RESULT_USER_INSTRUCTIONS = """**Task for this round (result mismatch):**
- Fix the C++ so outputs match the SQL semantics of the queries provided (filters, aggregations, joins, ordering, null handling, type conversions).
- Compare code line by line with expected behavior; check input column order vs data chunk order and reorder if misaligned.

**If the root cause is unclear, you may add temporary debugging:**
1. Compare with any query plan hints in the error text.
2. Add printf-style traces, e.g. printf("[DEBUG] var_name=%d\\n", var_name);
3. Limit output, e.g. if (debug_count++ > 100) break; or if (row_count > 100) break;
4. Use DuckDB types: int32_t, int64_t, double, string_t, date_t.
5. Prefer correctness over optimization; keep bind/init/execute/finalize structure."""


def reflect_iteration(
    cpp_code: str,
    error_message: str,
    message_history: list[str],
    iteration: int,
    llm: Optional[Any] = None,
) -> dict[str, Any]:
    """
    每一轮失败后做 reflection，输出结构化信息，并用于指导下一轮修复。
    """
    llm_instance = llm or advanced_llm

    reflect_system_prompt = """You are a senior DuckDB extension engineer conducting a short reflection after a failed validation.

You will receive:
- Current C++ code
- Current error log
- Prior history (errors + previous reflections)

Your goals:
1) Identify the most likely root cause. Summarize the root cause.
2) Propose the minimal next edit strategy to fix it
3) Provide prompt hints that the next fixing step should follow (e.g., check specific DuckDB APIs, ensure dbweaver_extension.hpp included, avoid placeholders)

Output STRICT JSON only:
{{
  "diagnosis": "one sentence",
  "root_cause": ["..."],
  "next_actions": ["..."],
  "prompt_hints": ["..."]
}}

Important:
- Return ONLY valid JSON, no markdown, no extra text.
"""

    history_text = "\n".join(message_history)
    reflect_user_prompt = (
        "ITERATION:\n{iteration}\n\n"
        "CURRENT C++:\n```cpp\n{cpp_code}\n```\n\n"
        "ERROR LOG:\n```text\n{error_message}\n```\n\n"
        "PRIOR HISTORY:\n```text\n{history}\n```\n"
    )

    prompt = ChatPromptTemplate.from_messages(
        [("system", reflect_system_prompt), ("user", reflect_user_prompt)]
    )
    chain = prompt | llm_instance

    payload = {
        "iteration": iteration,
        "cpp_code": cpp_code,
        "error_message": error_message,
        "history": history_text[-12000:],
    }

    response = chain.invoke(payload)
    response_text = response.content.strip() if hasattr(response, "content") else str(response)

    try:
        obj = extract_json(response_text)
        if not isinstance(obj, dict):
            raise ValueError("reflection json is not an object")
        return obj
    except Exception as e:
        return {
            "diagnosis": "Reflection parse failed; fallback reflection used.",
            "root_cause_tags": ["reflection_parse_error"],
            "next_actions": ["Proceed with best-effort minimal patching based on error log."],
            "prompt_hints": [f"Failed to parse reflection JSON: {e}"],
            "raw_reflection": response_text[:800],
        }

def _fix_code(response_text: str, cpp_code: str) -> CodeOutput:
    try:
        result = extract_json(response_text)
        prefix = result.get("prefix", "// FIXED (LLM patch from error log)")
        mode = result.get("mode", "patch")

        if mode == "full":
            full_code = result.get("full_code", "")
            print(f"[Fix Agent] Using Full code mode")
            if not full_code.strip():
                return CodeOutput(cpp_code=cpp_code, prefix=prefix)
            return CodeOutput(cpp_code=full_code.strip(), prefix=prefix)
        else:
            patches = result.get("patches", [])
            print(f"[Fix Agent] Using Patch code mode")
            if not patches:
                return CodeOutput(cpp_code=cpp_code, prefix=prefix)
            fixed_code = apply_replacements_from_list(cpp_code, patches)
            return CodeOutput(cpp_code=fixed_code, prefix=prefix)

    except (json.JSONDecodeError, KeyError, ValueError) as e:
        print(f"Warning: Failed to parse LLM response as JSON: {e}")
        print(f"Response text: {response_text[:500]}")
        return CodeOutput(cpp_code=cpp_code, prefix="// FIXED (parse error, returning original)")

def fix_node(
    cpp_code: str,
    error_message: str,
    original_query: Optional[str] = None,
    split_query: Optional[str] = None,
    message_history: Optional[list[str]] = None,
    user_fix_instructions: Optional[str] = None,
    llm: Optional[Any] = None,
) -> CodeOutput:
    """
    修复编译/链接/运行时等错误，返回修复后的代码。

    Args:
        cpp_code: 需要修复的 C++ 代码
        error_message: 错误信息
        original_query: 可选，原始查询（SQL 上下文）
        split_query: 可选，UDF 分片查询（SQL 上下文）
        message_history: 近期错误与反思历史（用于上下文）
        user_fix_instructions: 本回合 user 里的任务说明；默认编译类说明
        llm: 可选的 LLM 实例，默认使用 advanced_llm

    Returns:
        CodeOutput: 修复后的代码
    """
    llm_instance = llm or advanced_llm
    instructions = user_fix_instructions or FIX_COMPILE_USER_INSTRUCTIONS
    query_context = _build_query_context(original_query, split_query)

    history_text = ""
    if message_history:
        history_joined = "\n".join(message_history)
        history_text = f"\n\nHISTORY (recent errors + reflections):\n```text\n{history_joined[-12000:]}\n```\n"

    fix_user_prompt = (
        "{user_fix_instructions}\n\n"
        "{query_context}"
        "C++ SOURCE:\n```cpp\n{cpp_code}\n```\n\n"
        "ERROR LOG:\n```text\n{error_message}\n```\n"
        "{history_text}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [("system", FIX_AGENT_SYSTEM_PROMPT), ("user", fix_user_prompt)]
    )
    chain = prompt | llm_instance

    payload = {
        "user_fix_instructions": instructions,
        "query_context": query_context,
        "cpp_code": cpp_code,
        "error_message": error_message,
        "history_text": history_text,
    }
    for _ in range(3):
        try:
            response = chain.invoke(payload)
            break
        except Exception as e:
            print(f"Warning: Failed to invoke LLM: {e}")
            time.sleep(5)
    if response is None:
        return CodeOutput(cpp_code=cpp_code, prefix="// FIXED (LLM invoke failed)")
    response_text = response.content.strip() if hasattr(response, "content") else str(response)

    return _fix_code(response_text,cpp_code)

def fix_result_node(
    cpp_code: str,
    error_message: str,
    original_query: Optional[str] = None,
    split_query: Optional[str] = None,
    message_history: Optional[list[str]] = None,
    user_fix_instructions: Optional[str] = None,
    llm: Optional[Any] = None,
) -> CodeOutput:
    """
    修复结果不匹配，返回修复后的代码。

    Args:
        cpp_code: 需要修复的 C++ 代码
        error_message: 结果不匹配的错误信息
        original_query: 可选的原始查询（用于上下文）
        split_query: 可选的分割查询（用于上下文）
        message_history: 近期错误与反思历史（用于上下文）
        user_fix_instructions: 本回合 user 里的任务说明；默认结果语义类说明
        llm: 可选的 LLM 实例，默认使用 advanced_llm

    Returns:
        CodeOutput: 修复后的代码
    """
    llm_instance = llm or advanced_llm
    instructions = user_fix_instructions or FIX_RESULT_USER_INSTRUCTIONS
    query_context = _build_query_context(original_query, split_query)

    history_text = ""
    if message_history:
        history_joined = "\n".join(message_history)
        history_text = f"\n\nHISTORY (recent errors + reflections):\n```text\n{history_joined[-12000:]}\n```\n"

    fixresult_user_prompt = (
        "{user_fix_instructions}\n\n"
        "{query_context}"
        "C++ SOURCE:\n```cpp\n{cpp_code}\n```\n\n"
        "ERRORS:\n```text\n{error_message}\n```\n"
        "{history_text}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [("system", FIX_AGENT_SYSTEM_PROMPT), ("user", fixresult_user_prompt)]
    )
    chain = prompt | llm_instance

    payload = {
        "user_fix_instructions": instructions,
        "cpp_code": cpp_code,
        "error_message": error_message,
        "history_text": history_text,
        "query_context": query_context,
    }

    response = None
    for _ in range(3):
        try:
            response = chain.invoke(payload)
            break
        except Exception as e:
            print(f"Warning: Failed to invoke LLM: {e}")
            time.sleep(5)
    if response is None:
        return CodeOutput(cpp_code=cpp_code, prefix="// FIXED (LLM invoke failed)")

    response_text = response.content.strip() if hasattr(response, "content") else str(response)

    return _fix_code(response_text,cpp_code)

def fix_agent(
    cpp_code: str,
    state: dict[str, Any],
    max_iterations: int = 10,
) -> Tuple[bool, str, Optional[str], int]:
    """
    自动修复 C++ 代码的代理函数。

    Args:
        cpp_code: 需要修复的 C++ 代码
        state: 包含 validate_code 所需字段的字典，必须包含：
            - 'decomposed': {'split_query': str}
            - 'query_example': str
            可选字段：
            - 'query_template': str
            - 'plan': list[str] 或 str
            - 'query_id': int
        max_iterations: 最大迭代次数，如果超过此次数还没成功，返回错误

    Returns:
        Tuple[bool, str, Optional[str], int]:
            - success: 是否成功修复
            - message: 成功或失败的消息
            - fixed_code: 修复后的代码（如果成功），否则为 None
            - fix_iterations: 实际进行的修复迭代轮数（外层循环次数，成功或失败时均返回）
    """
    checker = CodeChecker()

    # 初始化 message_history / reflection_history
    if "message_history" not in state:
        state["message_history"] = []
    if "reflection_history" not in state:
        state["reflection_history"] = []

    # 确保 state 中有必要的字段
    if "decomposed" not in state:
        raise ValueError("state must contain 'decomposed' with 'split_query'")
    if "split_query" not in state["decomposed"]:
        raise ValueError("state['decomposed'] must contain 'split_query'")
    if "query_example" not in state:
        raise ValueError("state must contain 'query_example'")

    # 将 cpp_code 包装成 CodeOutput 对象
    current_code = CodeOutput(cpp_code=cpp_code, prefix="// INITIAL CODE")
    state["code"] = current_code


    iteration = 0
    while iteration < max_iterations:
        iteration += 1
        print(f"\n[Fix Agent] Iteration {iteration}/{max_iterations}")

        # 使用 code_checker 验证当前代码
        success, validation_message = checker.validate_code(state, current_code.cpp_code)

        if VALIDATION_FLAG_PERF_NEED_OPT.lower() in validation_message.lower():
            # 性能需要优化，但代码已经可以运行且结果正确
            print(f"[Fix Agent] Performance optimization needed, but code is correct.")
            return (
                True,
                f"Code is correct but performance needs optimization. {validation_message}",
                current_code.cpp_code,
                iteration,
            )

        # 解析错误类型
        error_message = str(validation_message)
        error_lower = error_message.lower()

        # 记录错误到 message_history
        # error_entry = f"[ERROR][Iter {iteration}] {error_message[:800]}"
        # state["message_history"].append(error_entry)
        state["error_message"] = error_message

        print(f"[Fix Agent] Validation failed: {error_message[:200]}...")

        # 每一轮失败后先 reflection，并把反思结果写入 message_history
        reflection_obj = reflect_iteration(
            cpp_code=current_code.cpp_code,
            error_message=error_message,
            message_history=state["message_history"],
            iteration=iteration,
            llm=advanced_llm,
        )
        state["reflection_history"].append(reflection_obj)

        reflection_line = (
            f"[REFLECTION][Iter {iteration}]"
            f"root_cause={reflection_obj.get('root_cause', [])}; "
            f"diagnosis={reflection_obj.get('diagnosis','')}; "
            f"next_actions={reflection_obj.get('next_actions', [])}; "
            f"hints={reflection_obj.get('prompt_hints', [])}"
        )
        state["message_history"].append(reflection_line)

        # 仅取最近 8 条历史喂给下一轮 fix（按你的要求）
        recent_history = state["message_history"][-8:]

        original_query = state.get("query_example") or ""
        split_query = state["decomposed"].get("split_query") or ""

        # 根据错误类型选择修复策略
        if VALIDATION_FLAG_COMPILE_FAIL.lower() in error_lower:
            # 编译错误，使用 fix_node
            print(f"[Fix Agent] Compilation error detected, using fix_node...")
            fixed_output = fix_node(
                current_code.cpp_code,
                error_message,
                original_query=original_query,
                split_query=split_query,
                message_history=recent_history,
                llm=advanced_llm,
            )
            current_code = fixed_output
            state["code"] = current_code
            print(f"[Fix Agent] Applied fix from fix_node")

        elif VALIDATION_FLAG_RESULT_MISMATCH.lower() in error_lower:
            # 结果不匹配，使用 fix_result_node
            print(f"[Fix Agent] Result mismatch detected, using fix_result_node...")
            fixed_output = fix_result_node(
                current_code.cpp_code,
                error_message,
                original_query=original_query,
                split_query=split_query,
                message_history=recent_history,
                llm=advanced_llm,
            )
            current_code = fixed_output
            state["code"] = current_code
            print(f"[Fix Agent] Applied fix from fix_result_node")

        elif VALIDATION_FLAG_PERF_NEED_OPT.lower() in error_lower:
            # 性能需要优化，但代码已经可以运行且结果正确
            print(f"[Fix Agent] Performance optimization needed, but code is correct.")
            return (
                True,
                f"Code is correct but performance needs optimization. {error_message}",
                current_code.cpp_code,
                iteration,
            )

        else:
            # 未知错误类型
            print(f"[Fix Agent] Unknown error type: {error_message[:200]}")
            if iteration >= max_iterations:
                return (
                    False,
                    f"Failed after {max_iterations} iterations with unknown error: {error_message}",
                    None,
                    iteration,
                )

            # 默认尝试使用 fix_node
            try:
                fixed_output = fix_node(
                    current_code.cpp_code,
                    error_message,
                    original_query=original_query,
                    split_query=split_query,
                    message_history=recent_history,
                    user_fix_instructions=FIX_UNKNOWN_USER_INSTRUCTIONS,
                    llm=advanced_llm,
                )
                current_code = fixed_output
                state["code"] = current_code
                print(f"[Fix Agent] Applied fix from fix_node (fallback)")
            except Exception as e:
                error_msg = f"fix_node (fallback) failed: {str(e)}"
                print(f"[Fix Agent] {error_msg}")
                state["message_history"].append(f"[EXCEPTION][Iter {iteration}] {error_msg}")
                if iteration >= max_iterations:
                    return (
                        False,
                        f"Failed after {max_iterations} iterations. Last error: {error_message}",
                        None,
                        iteration,
                    )
                continue

    # 超过最大迭代次数
    return (
        False,
        f"Failed to fix code after {max_iterations} iterations. Last error: {error_message}",
        None,
        iteration,
    )

