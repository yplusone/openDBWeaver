"""Tool wrappers for the optimize agent with shared mutable context.

The two tools (runtime_profiler_wrapper, request_datadistribution_wrapper)
need access to mutable state that changes during the optimization loop
(e.g. current_code). A ToolContext object is used to share this state.
"""

from __future__ import annotations

import json
import re
import subprocess
from typing import Any

from langchain_core.tools import tool

from dbweaver.env.code_checker import CodeChecker
from config import DUCKDB_BINARY_PATH, DB_PATH, DEFAULT_SOURCE_DIR
from dbweaver.env.duckdb_connector import DuckDBConnector
from dbweaver.optimize.setup import advanced_llm
from dbweaver.utils.llm_output_parse import extract_code,extract_json
from dbweaver.utils.apply_replacement import apply_replacements_from_list


class ToolContext:
    """Shared mutable state between the optimize agent and its tools."""

    def __init__(
        self,
        current_code: str,
        split_query: str,
        checker: CodeChecker,
        tool_calls_history: list[dict[str, Any]],
    ):
        self.current_code = current_code
        self.split_query = split_query
        self.checker = checker
        self.tool_calls_history = tool_calls_history

_FIX_OUTPUT_MODES = """
You have TWO output modes. Choose the one that best fits your fix:

**Mode A – Patch mode** (for small, localized fixes):
{{
    "mode": "patch",
    "patches": [
        {{"old_content": "...", "new_content": "..."}}
    ]
}}
Rules for patch mode:
- Each old_content MUST match a contiguous substring in the source exactly (full lines, verbatim)
- Include enough context lines in old_content to make it unique

**Mode B – Full code mode** (for large-scale fixes or many scattered changes):
{{
    "mode": "full",
    "full_code": "...the complete C++ source code..."
}}
Rules for full code mode:
- full_code must be the entire, self-contained C++ source file
- Do NOT omit any part of the code with comments like "// ... rest unchanged ..."

How to choose:
- If your fix touches <= 3 small regions, prefer patch mode (more precise, less error-prone).
- If your fix is widespread or restructures the code significantly, prefer full code mode.

Return ONLY valid JSON, no markdown, no explanation text.
"""


_PROFILER_PROMPT_TEMPLATE = """
You are a C++ performance profiling expert for DuckDB extensions. Your task is to inject strategic print statements into the code to help profile, debug, and optimize performance based on the user's guide.

GUIDE (what to profile/debug):
{guide}

AVAILABLE PROFILING TECHNIQUES:

1. **Vector Type Detection**:
   ```cpp
   std::cerr << "[PROFILER] VectorType of x: " << (int)input.data[0].GetVectorType() << std::endl;
   // VectorType values: FLAT=0, CONSTANT=1, DICTIONARY=2, SEQUENCE=3
   ```

2. **Variable Values** (primitive types):
   ```cpp
   std::cerr << "[PROFILER] variable_name = " << variable_name << std::endl;
   ```

3. **Variable Types** (using typeid):
   ```cpp
   std::cerr << "[PROFILER] Type of var: " << typeid(var).name() << std::endl;
   ```

4. **Batch/Loop Information**:
   ```cpp
   std::cerr << "[PROFILER] Batch size: " << input.size() << std::endl;
   std::cerr << "[PROFILER] Loop iteration: " << i << " / " << total << std::endl;
   ```

5. **Branch/Path Tracking**:
   ```cpp
   if (condition) {{{{
       std::cerr << "[PROFILER] Entered branch A" << std::endl;
   }}}} else {{{{
       std::cerr << "[PROFILER] Entered branch B" << std::endl;
   }}}}
   ```

6. **Data Structure Sizes**:
   ```cpp
   std::cerr << "[PROFILER] Map size: " << my_map.size() << std::endl;
   std::cerr << "[PROFILER] Array capacity: " << my_array.capacity() << std::endl;
   ```

7. **Timing Measurements**:
   ```cpp
   auto start = std::chrono::high_resolution_clock::now();
   // ... code to measure ...
   auto end = std::chrono::high_resolution_clock::now();
   auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
   std::cerr << "[PROFILER] Operation took: " << duration.count() << " us" << std::endl;
   ```

8. **Aggregates/Statistics** (instead of printing in loops):
   ```cpp
   int count = 0;
   for (...) {{{{
       if (condition) count++;
   }}}}
   std::cerr << "[PROFILER] Condition was true " << count << " times" << std::endl;
   ```

9. **Function Call Tracking**:
   ```cpp
   void MyFunction(...) {{{{
       static int call_count = 0;
       std::cerr << "[PROFILER] MyFunction called (count=" << ++call_count << ")" << std::endl;
       // ... function body ...
   }}}}
   ```

CRITICAL RULES:
1. **Always use `std::cerr << "[PROFILER] " << ... << std::endl;`** (so output is easily filtered)
2. **DO NOT change the logic** of the code - only add observability
3. **In hot loops**: Print only first 10 iterations OR aggregate/count instead
4. **For types**: Cast enums to int for readability
5. **Be specific**: Include descriptive labels in output (e.g., "VectorType of x_column:")
6. **Include context**: Print batch number, function name, or location markers
7. **Return the FULL C++ code** with injected print statements

Based on the guide above, strategically inject print statements that will help answer the profiling question or debug the issue.

C++ CODE TO INSTRUMENT:
{current_code}

OUTPUT: Return the complete instrumented C++ code.
"""


def create_optimize_tools(ctx: ToolContext):
    """Create tool functions bound to the given ToolContext.

    Returns:
        list: [runtime_profiler_wrapper, request_datadistribution_wrapper]
    """

    @tool
    def runtime_profiler_wrapper(guide: str) -> str:
        """
        Inject print statements into C++ code based on a natural language guide to profile runtime behavior.
        Can print variable types, values, vector types, execution paths, loop iterations, and any debugging info.

        Args:
            guide: Natural language instructions describing what to profile/debug
                   Examples:
                   - "Print the VectorType of input column x"
                   - "Show the value of filter variables in the first batch"
                   - "Count how many times each branch is taken"
                   - "Print array sizes and allocation counts"
        """
        prompt = _PROFILER_PROMPT_TEMPLATE.format(
            guide=guide,
            current_code=ctx.current_code,
        )
        try:
            response = advanced_llm.invoke(prompt)
            injected_code = extract_code(response.content)

            max_profile_fix_retries = 3
            for profile_retry in range(max_profile_fix_retries):
                cpp_file_path = DEFAULT_SOURCE_DIR + "/src/dbweaver.cpp"
                with open(cpp_file_path, "w") as f:
                    f.write(injected_code)

                build_result = ctx.checker._run_duckdb_build("release")
                if build_result.returncode != 0:
                    error_msg = build_result.stderr.strip()
                    if profile_retry < max_profile_fix_retries - 1:
                        print(
                            f"[runtime_profiler] Compile failed (attempt {profile_retry + 1}/{max_profile_fix_retries}), "
                            f"asking LLM to fix..."
                        )
                        fix_prompt = (
                            f"The instrumented C++ code failed to compile. Fix the compilation errors "
                            f"while keeping the profiling print statements.\n\n"
                            f"COMPILATION ERROR:\n{error_msg[:2000]}\n\n"
                            f"CURRENT INSTRUMENTED CODE:\n```cpp\n{injected_code}\n```\n\n"
                            f"ORIGINAL CODE (for reference):\n```cpp\n{ctx.current_code}\n```\n\n"
                            + _FIX_OUTPUT_MODES
                        )
                        fix_response = advanced_llm.invoke(fix_prompt)
                        replacements = extract_json(fix_response.content)
                        injected_code = apply_replacements_from_list(injected_code, replacements)
                        continue
                    result = f"Compilation failed after {max_profile_fix_retries} attempts:\n{error_msg}"
                    ctx.tool_calls_history.append({"tool": "runtime_profiler", "guide": guide, "result": result[:500]})
                    return result

                cmd = [DUCKDB_BINARY_PATH + "/release/duckdb", "-csv", DB_PATH, "-c", ctx.split_query]
                try:
                    run_result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                except subprocess.TimeoutExpired:
                    result = "Query execution timed out after 120 seconds."
                    ctx.tool_calls_history.append({"tool": "runtime_profiler", "guide": guide, "result": result[:500]})
                    return result

                if run_result.returncode != 0 and profile_retry < max_profile_fix_retries - 1:
                    runtime_err = run_result.stderr.strip()
                    print(
                        f"[runtime_profiler] Runtime error (attempt {profile_retry + 1}/{max_profile_fix_retries}), "
                        f"asking LLM to fix..."
                    )
                    fix_prompt = (
                        f"The instrumented C++ code compiled but crashed at runtime. Fix the runtime error "
                        f"while keeping the profiling print statements.\n\n"
                        f"RUNTIME ERROR:\n{runtime_err[:2000]}\n\n"
                        f"CURRENT INSTRUMENTED CODE:\n```cpp\n{injected_code}\n```\n\n"
                        f"ORIGINAL CODE (for reference):\n```cpp\n{ctx.current_code}\n```\n\n"
                        + _FIX_OUTPUT_MODES
                    )
                    fix_response = advanced_llm.invoke(fix_prompt)
                    replacements = extract_json(fix_response.content)
                    injected_code = apply_replacements_from_list(injected_code, replacements)
                    continue

                profiler_outputs = []
                all_output = run_result.stdout + "\n" + run_result.stderr
                for line in all_output.split("\n"):
                    if "[PROFILER]" in line:
                        profiler_outputs.append(line)

                if not profiler_outputs:
                    result = (
                        f"Code executed but no [PROFILER] output was captured.\n"
                        f"Stderr: {run_result.stderr[:200]}\nStdout: {run_result.stdout[:200]}"
                    )
                else:
                    result = "\n".join(profiler_outputs[:50])

                ctx.tool_calls_history.append({"tool": "runtime_profiler", "guide": guide, "result": result[:500]})
                return result

            result = f"Profiler code failed after {max_profile_fix_retries} fix attempts"
            ctx.tool_calls_history.append({"tool": "runtime_profiler", "guide": guide, "result": result[:500]})
            return result

        except Exception as e:
            result = f"Error in runtime_profiler: {str(e)}"
            ctx.tool_calls_history.append({"tool": "runtime_profiler", "guide": guide, "result": result[:500]})
            return result

    @tool
    def request_datadistribution_wrapper(sql_query: str) -> str:
        """
        Execute a SQL query against the database to gather statistics.

        Args:
            sql_query: SQL query to execute
        """
        try:
            connector = DuckDBConnector()
            success, result = connector.execute_query(sql_query, build_type="release", output_format="text")
            if success:
                result = str(result)
            else:
                result = f"Error executing query: {result}"
        except Exception as e:
            result = f"Exception occurred: {str(e)}"

        ctx.tool_calls_history.append({
            "tool": "request_datadistribution",
            "query": sql_query,
            "result": result[:2000],
        })
        return result

    return [runtime_profiler_wrapper, request_datadistribution_wrapper]
