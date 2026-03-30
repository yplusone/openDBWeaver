# -*- coding: utf-8 -*-
"""
DuckDBConnector - A connector class for DuckDB operations.

This class provides methods to:
- Connect to DuckDB
- Execute queries
- Get query plans
- Refresh source code
- Build DuckDB
"""

import os
import subprocess
import json
import datetime
import re
from pathlib import Path
from typing import Any, Tuple, Literal, Optional
import time

from config import (
    DUCKDB_BINARY_PATH,
    DB_PATH,
    DEFAULT_SOURCE_DIR,
    DOCKER_IMAGE,
)

BUILD_TYPE = Literal["release", "debug"]


class DuckDBConnector:
    """A connector class for DuckDB operations."""

    # Default PRAGMA settings to execute before every user query
    DEFAULT_PRAGMAS = []

    def __init__(self, enable_debug: bool = False):
        """
        Initialize the DuckDB connector.

        Args:
            enable_debug: Whether to enable debug mode (default: False)
        """
        self.enable_debug = enable_debug
        self.env = os.environ.copy()
        if enable_debug:
            self.env["LD_PRELOAD"] = "/usr/lib/gcc/x86_64-linux-gnu/11/libasan.so"
            self.env["ASAN_OPTIONS"] = "detect_leaks=0:abort_on_error=1"
        
        # Verify source directory exists
        if not os.path.exists(DEFAULT_SOURCE_DIR):
            raise FileNotFoundError(
                f"Source directory '{DEFAULT_SOURCE_DIR}' does not exist – check the path."
            )

    def execute_query(
        self, 
        query: str, 
        build_type: BUILD_TYPE = "release",
        output_format: str = "text"
    ) -> Tuple[bool, Any]:
        """
        Execute a SQL query in DuckDB.

        Args:
            query: SQL query string
            build_type: Build type ("release" or "debug")
            output_format: Output format ("text" or "csv")

        Returns:
            Tuple of (success: bool, result: Any)
            - On success: (True, query_result)
            - On failure: (False, error_message)
        """
        # Ensure the CLI binary exists
        binary_path = os.path.join(DUCKDB_BINARY_PATH, build_type, "duckdb")
        if not os.path.isfile(binary_path):
            return False, (
                f"Compiled DuckDB binary not found at {binary_path}.\n"
                "Did the build step place it in a different directory?"
            )

        # Ensure each pragma ends with a semicolon
        pragma_sql = " ".join(
            p.strip() if p.strip().endswith(";") else f"{p.strip()};"
            for p in self.DEFAULT_PRAGMAS
        )
        combined_sql = f"{pragma_sql} {query}"

        # Build command
        cmd = [binary_path, DB_PATH]
        if output_format == "csv":
            cmd.append("-csv")
        cmd.extend(["-c", combined_sql])

        # Run the query
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=self.env
        )

        # Handle allowed messages in stderr
        allowed_msg = "-- Loading resources from /home/yjn/.duckdbrc"
        error_msg = None if result.stderr.strip() == allowed_msg else result.stderr.strip()
        
        query_ok = result.returncode == 0 and not error_msg
        payload = result.stdout.strip() if query_ok else (
            result.stderr.strip() or result.stdout.strip()
        )

        return query_ok, payload

    def get_query_physical_plan(
        self, 
        query: str, 
        build_type: BUILD_TYPE = "release",
        analyze: bool = True,
        format: str = "json"
    ) -> Tuple[bool, Any]:
        """
        Get the query execution plan.

        Args:
            query: SQL query string
            build_type: Build type ("release" or "debug")
            analyze: Whether to include execution statistics (EXPLAIN ANALYZE)
            format: Output format ("json" or "text")

        Returns:
            Tuple of (success: bool, plan: Any)
            - On success: (True, query_plan_json_or_text)
            - On failure: (False, error_message)
        """
        # Build EXPLAIN query
        explain_query = "EXPLAIN"
        if analyze:
            explain_query += " ANALYZE"
        if format == "json":
            explain_query += " (FORMAT json)"
        explain_query += f" {query}"

        return self.execute_query(explain_query, build_type=build_type, output_format="text")

    _SECTION_HEADER_RE = re.compile(
        r"┌─+┐\s*\n│┌─+┐│\s*\n││\s*(.+?)\s*││\s*\n│└─+┘│\s*\n└─+┘"
    )

    @staticmethod
    def _extract_json_from_text(text: str) -> Any:
        """Parse the first complete JSON array/object from *text*.

        Raises ValueError when no valid JSON can be extracted.
        """
        text = text.strip()
        first_brace = text.find("{")
        first_bracket = text.find("[")

        if first_brace == -1 and first_bracket == -1:
            raise ValueError("No JSON structure found (no [ or {)")

        if first_brace != -1 and first_bracket != -1:
            start_idx = min(first_brace, first_bracket)
        elif first_brace != -1:
            start_idx = first_brace
        else:
            start_idx = first_bracket

        try:
            return json.loads(text[start_idx:])
        except json.JSONDecodeError:
            pass

        bracket_count = 0
        brace_count = 0
        in_string = False
        escape_next = False

        for i, char in enumerate(text[start_idx:], start=start_idx):
            if escape_next:
                escape_next = False
                continue
            if char == '\\':
                escape_next = True
                continue
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
            if in_string:
                continue

            if char == '[':
                bracket_count += 1
            elif char == ']':
                bracket_count -= 1
            elif char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1

            if bracket_count == 0 and brace_count == 0 and i > start_idx:
                return json.loads(text[start_idx:i + 1])

        raise ValueError("Failed to extract balanced JSON from text")

    def get_query_logical_plan(
        self,
        query: str,
        build_type: BUILD_TYPE = "release",
        plan_type: Literal["optimized_logical_plan", "physical_plan"] = "optimized_logical_plan",
    ) -> Tuple[bool, Any]:
        """
        Get a specific plan from DuckDB ``EXPLAIN (FORMAT json)`` output.

        Args:
            query: SQL query string
            build_type: Build type ("release" or "debug")
            plan_type: Which plan to extract —
                ``"optimized_logical_plan"`` (default) or ``"physical_plan"``.

        Returns:
            Tuple of (success: bool, plan: Any)
            - On success: (True, parsed_plan_json)
            - On failure: (False, error_message or None)
        """
        if not query:
            return False, None

        full_sql = f"PRAGMA explain_output='all'; EXPLAIN (FORMAT json) {query}"

        success, payload = self.execute_query(full_sql, build_type=build_type, output_format="text")
        if not success:
            return False, payload

        output = payload.strip()
        if not output:
            return False, "Empty output from query"

        target_title = {
            "optimized_logical_plan": "Optimized Logical Plan",
            "physical_plan": "Physical Plan",
        }.get(plan_type)
        if target_title is None:
            return False, f"Unknown plan_type: {plan_type!r}"

        headers = list(self._SECTION_HEADER_RE.finditer(output))
        if not headers:
            return False, "No plan section headers found in output"

        for idx, m in enumerate(headers):
            if m.group(1).strip() == target_title:
                body_start = m.end()
                body_end = headers[idx + 1].start() if idx + 1 < len(headers) else len(output)
                section_text = output[body_start:body_end].strip()
                if not section_text:
                    return False, f"No content found after '{target_title}' header"
                try:
                    return True, self._extract_json_from_text(section_text)
                except ValueError as e:
                    return False, f"Failed to parse JSON for '{target_title}': {e}"

        return False, f"'{target_title}' section not found in output"


    def extract_scan_and_total(self, json_text: str) -> Tuple[float, float]:
        """
        Extract scan time and total time from DuckDB EXPLAIN/PROFILE JSON.

        Args:
            json_text: JSON string from EXPLAIN ANALYZE

        Returns:
            Tuple of (scan_time: float, total_time: float) in seconds
        """
        plan = json.loads(json_text)

        # 顶层可能是数组 [ {...} ]，也可能是单个对象 { ... }
        if isinstance(plan, list) and len(plan) == 1:
            plan = plan[0]

        # 递归遍历所有节点
        def iter_nodes(node):
            if not isinstance(node, dict):
                return
            yield node
            for child in node.get("children", []):
                yield from iter_nodes(child)

        # -------- 1) 解析总时间 --------
        total_time = None

        # 首选：root 的 latency（PRAGMA profiling 的 JSON）
        if isinstance(plan, dict) and "latency" in plan and plan["latency"] > 0:
            total_time = plan["latency"]

        # 其次：root 的 operator_timing（有些版本 / EXPLAIN ANALYZE）
        if total_time is None and "operator_timing" in plan and plan["operator_timing"] > 0:
            total_time = plan["operator_timing"]

        # 最兜底：把所有 operator_timing 加起来当作一个近似
        if total_time is None:
            total_time = 0.0
            for n in iter_nodes(plan):
                total_time += float(n.get("operator_timing", 0.0))

        # -------- 2) 解析所有 scan 节点时间 --------
        scan_time = 0.0

        for n in iter_nodes(plan):
            # DuckDB 里常见键：
            #   operator_type: e.g. "TABLE_SCAN", "SEQ_SCAN"
            #   operator_name: e.g. "SEQ_SCAN"
            #   name:          e.g. "SEQ_SCAN"（非 ANALYZE 版本）
            op_type = (n.get("operator_type") or n.get("name") or "").upper()
            op_name = (n.get("operator_name") or "").upper()
            label = op_type + " " + op_name

            # 粗暴一点，只要包含 SCAN 就当作 scan（TABLE_SCAN / SEQ_SCAN / INDEX_SCAN 等）
            if "SCAN" in label:
                scan_time += float(n.get("operator_timing", 0.0))

        return scan_time, total_time

    def test_query_runtime(
        self, 
        query: str, 
        build_type: BUILD_TYPE = "release",
        max_retries: int = 2
    ) -> Tuple[float, float]:
        """
        Test query runtime and extract scan time and total time.

        Args:
            query: SQL query string
            build_type: Build type ("release" or "debug")
            max_retries: Maximum number of retries on failure

        Returns:
            Tuple of (scan_time: float, total_time: float) in seconds
        """
        # Build EXPLAIN ANALYZE query
        pragma_sql = " ".join(
            p.strip() if p.strip().endswith(";") else f"{p.strip()};"
            for p in self.DEFAULT_PRAGMAS
        )
        combined_sql = f"{pragma_sql} EXPLAIN (ANALYZE, FORMAT json) {query}"

        allowed_msg = "-- Loading resources from /home/yjn/.duckdbrc"
        binary_path = os.path.join(DUCKDB_BINARY_PATH, build_type, "duckdb")

        for attempt in range(1, max_retries + 1):
            try:
                result = subprocess.run(
                    [binary_path, DB_PATH, "-c", combined_sql],
                    capture_output=True,
                    text=True,
                    env=self.env
                )
                error_msg = None if result.stderr.strip() == allowed_msg else result.stderr.strip()
                query_ok = result.returncode == 0 and not error_msg
                
                if query_ok:
                    payload = result.stdout.strip()
                    scan_t, total_t = self.extract_scan_and_total(payload)
                    return scan_t, total_t
                else:
                    if attempt < max_retries:
                        time.sleep(3)
                        continue
                    else:
                        return 0.0, 0.0
            except Exception as e:
                if attempt < max_retries:
                    time.sleep(3)
                    continue
                else:
                    return 0.0, 0.0

        return 0.0, 0.0

    def build_duckdb(self, build_type: BUILD_TYPE = "release") -> Tuple[bool, str]:
        """
        Build DuckDB extension in a Docker container.

        Args:
            build_type: Build type ("release" or "debug")

        Returns:
            Tuple of (success: bool, message: str)
        """
        cmd = [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{DEFAULT_SOURCE_DIR}:/app",
        ]

        # 通过环境变量传给 Dockerfile 里的 BUILD_TYPE
        if build_type == "debug":
            cmd += ["-e", "BUILD_TYPE=debug"]
        else:
            # 显式指定一下，和 Dockerfile 里的默认值一致
            cmd += ["-e", "BUILD_TYPE=release"]

        # 添加性能优化选项
        cmd += ["--network=host"]  # 使用主机网络，避免网络开销
        # 传递并行构建参数（如果 makefile 支持）
        cmd += ["-e", "MAKEFLAGS=-j"]  # 让 make 自动检测 CPU 核心数，避免子 make 警告
        cmd.append(DOCKER_IMAGE)

        # 打印执行的命令（用于调试）
        print(f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] Executing: {' '.join(cmd)}\n")

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0 or result.stderr.strip():
            error_msg = result.stderr.strip() or result.stdout.strip()
            return False, error_msg

        return True, "Build succeeded."

    def get_query_results(self, query: str) -> Tuple[bool, Any]:
        """
        Get query results (wrapper for execute_query with default settings).

        Args:
            query: SQL query string

        Returns:
            Tuple of (success: bool, result: Any)
        """
        return self.execute_query(query, build_type="release", output_format="text")

