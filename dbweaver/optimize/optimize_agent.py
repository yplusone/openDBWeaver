"""
optimize_agent.py
-----------------
一个优化代理，根据优化方向生成优化后的代码，可以调用工具获取信息，
并自动处理编译错误，最多迭代5次。
"""

from __future__ import annotations

from typing import Any, Optional, Tuple
import sys
import os
import json
import time
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dbweaver.env.code_checker import CodeChecker
from config import VALIDATION_FLAG_COMPILE_FAIL, VALIDATION_FLAG_PERF_NEED_OPT
from dbweaver.sketch.code_combine import CodeOutput
from dbweaver.utils.apply_replacement import apply_replacements_from_list

from dbweaver.optimize.setup import advanced_llm
from dbweaver.optimize.optimize_tools import ToolContext, create_optimize_tools
from dbweaver.utils.llm_output_parse import extract_code,extract_json


def optimize_agent(
    cpp_code: str,
    optimization_direction: str,
    state: dict[str, Any],
    max_iterations: int = 10,
) -> Tuple[bool, str, Optional[str], list[dict[str, Any]]]:
    checker = CodeChecker()
    
    # 确保 state 中有必要的字段
    if "decomposed" not in state:
        return False, "state must contain 'decomposed' with 'split_query'", None, []
    if "query_example" not in state:
        return False, "state must contain 'query_example'", None, []
    
    split_query = state["decomposed"].get("split_query", "")
    split_template = state["decomposed"].get("split_template", "")
    query_example = state.get("query_example", "")
    query_template = state.get("query_template", "")
    
    # 工具调用历史
    tool_calls_history = []
    
    # 优化系统提示词
    optimize_system_prompt = """You are a C++ code optimization expert for DuckDB extensions.

You will receive:
1) Original C++ code
2) An optimization direction/strategy to follow
3) Optional error logs if previous attempts failed

Your task:
- Follow the optimization direction to improve performance
- You have access to tools:
  * request_datadistribution: Query database statistics
  * runtime_profiler: Inject print statements to observe runtime behavior
- Use tools to gather information if needed, then generate the optimized code.
- Preserve correctness while improving performance

You have TWO output modes. Choose the one that best fits your changes:

**Mode A – Patch mode** (for small, localized changes):
{{
    "mode": "patch",
    "prefix": "Brief description of the optimization",
    "patches": [
        {{"old_content": "...", "new_content": "..."}}
    ]
}}
Rules for patch mode:
- Each old_content MUST match a contiguous substring in the source exactly (full lines, verbatim)
- Include enough context lines in old_content to make it unique

**Mode B – Full code mode** (for large-scale rewrites or many scattered changes):
{{
    "mode": "full",
    "prefix": "Brief description of the optimization",
    "full_code": "...the complete C++ source code..."
}}
Rules for full code mode:
- full_code must be the entire, self-contained C++ source file
- Do NOT omit any part of the code with comments like "// ... rest unchanged ..."

How to choose:
- If your changes touch <= 3 small regions, prefer patch mode (more precise, less error-prone).
- If your changes are widespread or restructure the code significantly, prefer full code mode.

Important:
- Make sure the new code follows the optimization direction. If errors arise, try to fix the error but keep the optimization direction.
- Return ONLY valid JSON, no markdown, no explanation text
- If you need more information, use the available tools first"""
    
    iteration = 0
    current_code = cpp_code
    error_history = []
    

    # 创建共享上下文和工具
    ctx = ToolContext(current_code, split_query, checker, tool_calls_history)
    tools = create_optimize_tools(ctx)
    runtime_profiler_wrapper, request_datadistribution_wrapper = tools
    llm_with_tools = advanced_llm.bind_tools(tools)
    
    # 初始化消息历史
    messages = [
        SystemMessage(content=optimize_system_prompt),
        HumanMessage(content=(
            f"OPTIMIZATION DIRECTION:\n{optimization_direction}\n\n"
            f"CURRENT C++ CODE:\n```cpp\n{current_code}\n```\n\n"
            f"ORIGINAL QUERY:\n```sql\n{query_example}\n```\n\n"
            f"UDF QUERY:\n```sql\n{split_query}\n```"
        ))
    ]
    
    while iteration < max_iterations:
        iteration += 1
        print(f"\n[Optimize Agent] Iteration {iteration}/{max_iterations}")
        
        # 每次迭代只调用一次 LLM
        print(f"[Optimize Agent] Calling LLM...")
        for _ in range(5):
            try:
                response = llm_with_tools.invoke(messages)
                break
            except Exception as e:
                print(f"[Optimize Agent] Error in LLM: {e}")
                time.sleep(5)
        
        # 将响应添加到消息历史
        messages.append(response)
        
        # 检查是否有工具调用
        if hasattr(response, "tool_calls") and response.tool_calls:
            print(f"[Optimize Agent] LLM requested {len(response.tool_calls)} tool calls, executing...")
            
            # 执行所有工具调用
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_id = tool_call["id"]
                
                print(f"  - Calling {tool_name} with args: {str(tool_args)[:100]}...")
                
                # 执行工具
                if tool_name == "runtime_profiler_wrapper":
                    tool_result = runtime_profiler_wrapper.invoke(tool_args)
                elif tool_name == "request_datadistribution_wrapper":
                    tool_result = request_datadistribution_wrapper.invoke(tool_args)
                else:
                    tool_result = f"Unknown tool: {tool_name}"
                
                print(f"    Result (first 1000 chars): {tool_result[:1000]}...")
                
                # 添加工具结果到消息历史，作为下一次迭代的上下文
                messages.append(ToolMessage(content=tool_result, tool_call_id=tool_id))
            
            # 继续下一次迭代，LLM 将看到工具结果
            continue
        
        try:
            response_text = response.content.strip() if hasattr(response, "content") else str(response)
            result = extract_json(response_text)
        
            prefix = result.get("prefix", "// Optimization applied")
            mode = result.get("mode", "patch")
            
            if mode == "full":
                # Full code mode
                full_code = result.get("full_code", "")
                if not full_code.strip():
                    if iteration >= max_iterations:
                        return False, "Empty full_code after max iterations", None, tool_calls_history
                    error_msg = "full_code is empty, please provide the complete C++ source."
                    print(f"[Optimize Agent] {error_msg}")
                    messages.append(HumanMessage(content=error_msg))
                    continue
                print(f"[Optimize Agent] Using full code mode")
                optimized_code = full_code
            else:
                # Patch mode
                patches = result.get("patches", [])
                if not patches:
                    if iteration >= max_iterations:
                        return False, "No patches generated after max iterations", None, tool_calls_history
                    error_msg = "No patches generated, please provide valid patches in the next response."
                    print(f"[Optimize Agent] {error_msg}")
                    messages.append(HumanMessage(content=error_msg))
                    continue
                print(f"[Optimize Agent] Using patch mode, applying {len(patches)} patches...")
                optimized_code = apply_replacements_from_list(current_code, patches)
        
            # 验证编译
            print(f"[Optimize Agent] Validating optimized code...")
            temp_state = state.copy()
            temp_state["code"] = CodeOutput(cpp_code=optimized_code, prefix=prefix)
            
            success, validation_message = checker.validate_code(temp_state, optimized_code)
        
        except json.JSONDecodeError as e:
            error_msg = f"Failed to parse LLM response as JSON: {e}\n\nResponse received:\n{response_text[:500]}\n\nPlease provide ONLY valid JSON in the specified format."
            print(f"[Optimize Agent] {error_msg[:200]}...")
            if iteration >= max_iterations:
                return False, f"Failed after {max_iterations} iterations: JSON parse error", None, tool_calls_history
            messages.append(HumanMessage(content=error_msg))
            continue
            
        if VALIDATION_FLAG_PERF_NEED_OPT in validation_message:
            print(f"[Optimize Agent] Success! Optimization applied and validated.")
            return True, f"Optimization successful: {prefix},Check_result: {validation_message}", optimized_code, tool_calls_history
            
        # 编译失败，记录错误并作为下一次的上下文
        error_message = str(validation_message)
        print(f"[Optimize Agent] Validation failed: {error_message[:200]}...")
        
        # 更新 current_code 用于工具调用
        current_code = optimized_code
        ctx.current_code = current_code
        
        # 将错误信息添加到消息历史，作为下一次的上下文
        error_context = f"VALIDATION ERROR:\n{error_message[:2000]}\n\nPlease fix the error in the next patch."
        messages.append(HumanMessage(content=error_context))
            
    # 超过最大迭代次数
    final_message = f"Failed to optimize after {max_iterations} iterations"
    if error_history:
        final_message += f". Last error: {error_history[-1]}"
    
    return False, final_message, None, tool_calls_history
