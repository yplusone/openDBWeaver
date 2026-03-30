# -*- coding: utf-8 -*-
"""
DockerDuckDBRunner - Runs Docker-based DuckDB builds and executes queries.

Provides:
- Docker container-based extension builds (release / debug)
- DuckDB query execution with CSV output
- EXPLAIN ANALYZE parsing helpers (text and JSON formats)
- Multi-run query runtime benchmarking
"""

import os
import re
import json
import subprocess
import datetime
import time
from typing import Any, Tuple, Literal, Optional
from pathlib import Path

from config import (
    DB_PATH,
    DEFAULT_SOURCE_DIR,
    DOCKER_IMAGE,
    DUCKDB_BINARY_PATH,
    THREADS,
)
from dbweaver.utils.debug_segfault import SegfaultDebugger

BUILD_TYPE = Literal["release", "debug"]

_ALLOWED_STDERR = "-- Loading resources from /home/yjn/.duckdbrc"


class DockerDuckDBRunner:
    """Run Docker-based DuckDB builds and execute queries against a local DB."""

    def __init__(self, enable_debug: bool = False):
        """
        Args:
            enable_debug: When True, injects AddressSanitizer into the
                          subprocess environment for debug builds.
        """
        self.env = os.environ.copy()
        if enable_debug:
            self.env["LD_PRELOAD"] = "/usr/lib/gcc/x86_64-linux-gnu/11/libasan.so"
            self.env["ASAN_OPTIONS"] = "detect_leaks=0:abort_on_error=1"

        self.debugger = SegfaultDebugger()

    # ------------------------------------------------------------------
    # Build helpers
    # ------------------------------------------------------------------

    def run_duckdb_build(self, build_type: BUILD_TYPE) -> subprocess.CompletedProcess[str]:
        """Run the Docker image and build the DuckDB extension.

        Returns the *CompletedProcess* object so the caller can inspect
        ``stdout``, ``stderr`` and ``returncode``.
        """
        cmd = [
            "docker", "run", "--rm",
            "-v", f"{DEFAULT_SOURCE_DIR}:/app",
        ]

        cmd += ["-e", f"BUILD_TYPE={build_type}"]
        cmd += ["-w", "/app"]
        cmd += ["--network=host"]
        cmd += ["-e", "MAKEFLAGS=-j"]
        cmd.append(DOCKER_IMAGE)

        print(f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] Executing: {' '.join(cmd)}\n")
        return subprocess.run(cmd, capture_output=True, text=True)

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def run_duckdb_query(self, query: str, build_type: BUILD_TYPE) -> Tuple[bool, Any]:
        """Execute *query* in an in-memory DuckDB instance via the CLI binary.

        Returns ``(True, result)`` on success or ``(False, error_message)`` on
        failure.
        """
        binary = os.path.join(DUCKDB_BINARY_PATH, build_type, "duckdb")
        if not os.path.isfile(binary):
            return False, (
                f"Compiled DuckDB binary not found at {binary}.\n"
                "Did the build step place it in a different directory?"
            )

        pragma_sql = f"SET threads = {THREADS};"
        combined_sql = f"{pragma_sql} {query}"

        try:
            result = subprocess.run(
                [binary, "-csv", DB_PATH, "-c", combined_sql],
                capture_output=True,
                text=True,
                env=self.env,
                timeout=300,
            )
        except subprocess.TimeoutExpired:
            return False, (
                "Query execution timed out after 300 seconds. "
                "The query may be too complex or contain an infinite loop."
            )

        error_msg = None if result.stderr.strip() == _ALLOWED_STDERR else result.stderr.strip()
        query_ok = result.returncode == 0 and not error_msg
        payload = result.stdout.strip() if query_ok else (result.stderr.strip() or result.stdout.strip())

        if query_ok:
            explain_result = subprocess.run(
                [DUCKDB_BINARY_PATH + build_type + "/duckdb","-csv", DB_PATH,"-c", f"{pragma_sql} EXPLAIN (ANALYZE) {query}"],
                capture_output=True,
                text=True,
                env=self.env,
                timeout=300  # 10分钟超时
            )
            if explain_result.returncode != 0:
                return False, f"Explain query failed:\n{explain_result.stderr.strip()}"

        if result.returncode == -11:
            build_debug = self.run_duckdb_build("debug")
            if build_debug.returncode != 0 or build_debug.stderr.strip():
                return False, f"Compilation error while building debug binary:\n{build_debug.stderr.strip()}"

            print("\n[DEBUG] Segmentation fault detected. Running debugger...")
            debug_report = self.debugger.comprehensive_debug(query)
            payload = f"Segmentation fault (core dumped)\n\nBrief summary:\n{debug_report[:2000]}"

        return query_ok, payload

    def run_explain_analyze(
        self,
        query: str,
        binary: str,
        pragma_sql: str,
    ) -> Tuple[bool, str]:
        """Run ``EXPLAIN (ANALYZE)`` for *query* and return (ok, output)."""
        try:
            result = subprocess.run(
                [binary, "-csv", DB_PATH, "-c", f"{pragma_sql} EXPLAIN (ANALYZE) {query}"],
                capture_output=True,
                text=True,
                env=self.env,
                timeout=300,
            )
        except subprocess.TimeoutExpired:
            return False, (
                "EXPLAIN ANALYZE timed out after 300 seconds."
            )

        error_msg = None if result.stderr.strip() == _ALLOWED_STDERR else result.stderr.strip()
        ok = result.returncode == 0 and not error_msg
        payload = result.stdout.strip() if ok else (result.stderr.strip() or result.stdout.strip())
        return ok, payload

    # ------------------------------------------------------------------
    # Timing extraction helpers
    # ------------------------------------------------------------------

    def extract_total_time_seconds(self, text: str) -> Optional[float]:
        """Parse ``Total Time: <value><unit>`` from *text* and return seconds."""
        m = re.search(r"Total Time:\s*([0-9]*\.?[0-9]+)\s*([a-zA-Zµμ]*)", text)
        if not m:
            return None
        val = float(m.group(1))
        unit = m.group(2).lower() or "s"
        factor = {"s": 1, "ms": 1e-3, "us": 1e-6, "µs": 1e-6, "μs": 1e-6, "ns": 1e-9}.get(unit, 1)
        return val * factor

    def extract_scan_and_total_from_text(self, text: str) -> Tuple[float, float]:
        """Parse SCAN time and total time from a text-format EXPLAIN ANALYZE output.

        Returns:
            ``(scan_time, total_time)`` in seconds.
        """
        total_time = 0.0
        match = re.search(r"Total Time:\s*([\d.]+)s", text)
        if match:
            total_time = float(match.group(1))

        operator_blocks = re.findall(
            r"┌[─┴]+┐\s*\n"
            r"│\s*([A-Z_]+(?:\s+[A-Z_]+)?)\s*│\s*\n"
            r"(?:.*?\n)*?"
            r"│\s*\(([\d.]+)s\)\s*│",
            text,
            re.MULTILINE,
        )

        scan_time = 0.0
        processing_time = 0.0
        for op_name, time_str in operator_blocks:
            t = float(time_str)
            if "SCAN" in op_name.strip().upper():
                scan_time += t
            else:
                processing_time += t

        total_op = scan_time + processing_time
        if total_op > 0:
            scan = scan_time / total_op * total_time
        else:
            scan = 0.0
        return scan, total_time

    def extract_scan_and_total(self, json_text: str) -> Tuple[float, float]:
        """Parse scan time and total time from a DuckDB EXPLAIN/PROFILE JSON.

        Returns:
            ``(scan_time, total_time)`` in seconds.
        """
        plan = json.loads(json_text)
        if isinstance(plan, list) and len(plan) == 1:
            plan = plan[0]

        def iter_nodes(node):
            if not isinstance(node, dict):
                return
            yield node
            for child in node.get("children", []):
                yield from iter_nodes(child)

        total_time: Optional[float] = None
        if isinstance(plan, dict):
            if plan.get("latency", 0) > 0:
                total_time = plan["latency"]
            elif plan.get("operator_timing", 0) > 0:
                total_time = plan["operator_timing"]

        if total_time is None:
            total_time = sum(float(n.get("operator_timing", 0.0)) for n in iter_nodes(plan))

        scan_time = 0.0
        for n in iter_nodes(plan):
            op_type = (n.get("operator_type") or n.get("name") or "").upper()
            op_name = (n.get("operator_name") or "").upper()
            if "SCAN" in f"{op_type} {op_name}":
                scan_time += float(n.get("operator_timing", 0.0))

        return scan_time, total_time

    # ------------------------------------------------------------------
    # Runtime benchmarking
    # ------------------------------------------------------------------

    def test_query_runtime(
        self,
        query: str,
        build_type: BUILD_TYPE = "release",
        runs: int = 3,
        max_retries: int = 2,
    ) -> Tuple[float, float]:
        """Run *query* multiple times and return the average (scan, total) times.

        The worst run (highest total time) is discarded before averaging.

        Returns:
            ``(avg_scan_time, avg_total_time)`` in seconds.
        """
        pragma_sql = f"SET threads = {THREADS};"
        combined_sql = f"{pragma_sql} EXPLAIN (ANALYZE) {query}"
        binary = os.path.join(DUCKDB_BINARY_PATH, build_type, "duckdb")

        results: list[Tuple[float, float]] = []
        for _ in range(runs):

            if THREADS == 1:
                cmd = ["taskset", "-c", "0", binary, DB_PATH, "-c", combined_sql]
            else:
                cmd = [binary, DB_PATH, "-c", combined_sql]

            result = subprocess.run(cmd, capture_output=True, text=True)
            error_msg = result.stderr.strip() if result.stderr.strip() else None
            if result.returncode != 0 or error_msg:
                raise RuntimeError(f"Query failed: {error_msg}")

            scan_t, total_t = self.extract_scan_and_total_from_text(result.stdout.strip())
            results.append((scan_t, total_t))
            break


        results_sorted = sorted(results, key=lambda x: x[1])
        filtered = results_sorted[:-1] if len(results_sorted) > 1 else results_sorted
        avg_scan = sum(r[0] for r in filtered) / len(filtered)
        avg_total = sum(r[1] for r in filtered) / len(filtered)
        return avg_scan, avg_total

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------

    def fresh_source_code(self, template_path: Optional[str] = None) -> Tuple[bool, str]:
        """
        Refresh the extension source code from the template and build in release mode.

        Args:
            template_path: Path to template file. If None, uses "template/quack_extension.cpp"

        Returns:
            Tuple of (success: bool, message: str)
        """
        if template_path is None:
            template_path = "dbweaver/sketch/template/quack_extension.cpp"
        
        cpp_file_path = os.path.join(DEFAULT_SOURCE_DIR, "src", "dbweaver.cpp")
        template_file = Path(template_path)

        if not template_file.is_file():
            msg = f"Template file not found: {template_path}"
            print(msg)
            return False, msg

        try:
            template_code = template_file.read_text(encoding="utf-8")
        except Exception as e:
            msg = f"Failed to read template {template_path}: {e}"
            print(msg)
            return False, msg

        try:
            os.makedirs(os.path.dirname(cpp_file_path), exist_ok=True)
            with open(cpp_file_path, "w", encoding="utf-8") as f:
                f.write(template_code)
        except Exception as e:
            msg = f"Failed to write refreshed code to {cpp_file_path}: {e}"
            print(msg)
            return False, msg

        # Build release binary after refreshing the code
        print(f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] Refreshing code and building release binary\n")
        build_result = self.run_duckdb_build("release")
        if build_result.returncode != 0 or build_result.stderr.strip():
            return False, build_result.stderr.strip()

        msg = f"Successfully refreshed code from {template_path} and built release binary."
        print(f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}\n")
        return True, msg