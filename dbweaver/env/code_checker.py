
import os
import subprocess
from pathlib import Path
from typing import Any, Tuple, Literal
import re
import json
import datetime
import time


from config import (
    BENCHMARK,
    DB_PATH,
    DEFAULT_SOURCE_DIR,
    DOCKER_IMAGE,
    DUCKDB_BINARY_PATH,
    OPTIMIZED_CODE_DIR,
    VALIDATION_FLAG_COMPILE_FAIL,
    VALIDATION_FLAG_PERF_NEED_OPT,
    VALIDATION_FLAG_RESULT_MISMATCH,
    THREADS,
)

from dbweaver.utils.debug_segfault import SegfaultDebugger
from dbweaver.env.duckdb_connector import DuckDBConnector
from dbweaver.env.result_match import ResultComparator
from dbweaver.env.docker_build import DockerDuckDBRunner

env = os.environ.copy()
env["LD_PRELOAD"] = "/usr/lib/gcc/x86_64-linux-gnu/11/libasan.so"
env["ASAN_OPTIONS"] = "detect_leaks=0:abort_on_error=1"

BUILD_TYPE = Literal["release", "debug"]




class CodeChecker:
    """Compile the extension in a Docker container & run a DuckDB query."""

    # Default PRAGMA settings to execute before every user query.
    # DEFAULT_PRAGMAS = ["PRAGMA threads = 1;"]
    DEFAULT_PRAGMAS = [f"SET threads = {THREADS};"]

    def __init__(self, enable_debug: bool = True):
        self.original_query_result = None
        self.enable_debug = enable_debug
        self.debugger = SegfaultDebugger() if enable_debug else None
        if not os.path.exists(DEFAULT_SOURCE_DIR):
            raise FileNotFoundError(
                f"Source directory '{DEFAULT_SOURCE_DIR}' does not exist – check the path."
            )
        self.dbconnector = DuckDBConnector()
        self.docker_build = DockerDuckDBRunner()
        
    def get_query_results(self, query: str) -> Tuple[bool, Any]:
        # print("Original query results:")
        original_query_ok, original_query_payload = self.docker_build.run_duckdb_query(query,"release")
        if not original_query_ok:
            return False, f"Original query failed:\n{original_query_payload}"
        self.original_query_result = original_query_payload
        return True, original_query_payload
    # ---------------------------------------------------------------------
    # Docker helpers
    # ---------------------------------------------------------------------


    def validate_code(self, state, cpp_code: str) -> Tuple[bool, Any]:
        split_query = state['decomposed']['split_query']
        original_query = state['query_example']

        cpp_file_path = DEFAULT_SOURCE_DIR + "src/dbweaver.cpp"
        if len(cpp_code):
            with open(cpp_file_path, "w") as f:
                f.write(cpp_code)

        # 先用 release 模式编译
        build_release = self.docker_build.run_duckdb_build("release")
        if build_release.returncode != 0 or build_release.stderr.strip():
            return False, f"{VALIDATION_FLAG_COMPILE_FAIL}: Build failed or produced errors:\n{build_release.stderr.strip()}"
        print(f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] Successfully built source code (release mode)\n")

        # 用 release 再测一次 split_query 的输出
        ok, new_payload = self.docker_build.run_duckdb_query(split_query, 'release')
        if not ok:
            return False, f"{VALIDATION_FLAG_COMPILE_FAIL}: {new_payload}"
        if self.original_query_result is None:
            ok, original_result = self.get_query_results(original_query)
            if not ok:
                raise RuntimeError(original_result)
            self.original_query_result = original_result
        comparator = ResultComparator()
        result_ok, detail = comparator.check_two_results(original_query, new_payload, self.original_query_result)
        if not result_ok:
            return False, f"{VALIDATION_FLAG_RESULT_MISMATCH}: Query results do not match.\n{detail}"

        t_original_scan, t_original_total = self.docker_build.test_query_runtime(original_query)
        t_split_scan, t_split_total = self.docker_build.test_query_runtime(split_query)
        
        # Check if performance is good
        is_faster = t_split_total < t_original_total and (t_original_total - t_original_scan) >= (t_split_total - t_split_scan) * 1.05

        processing_accelarate = (t_original_total - t_original_scan) / (t_split_total - t_split_scan+0.0001)
            # Return False to continue optimization
        return False, f"{VALIDATION_FLAG_PERF_NEED_OPT}: Processing Speedup:{processing_accelarate:.6f}, Original Execution Time: {t_original_total:.6f}s, Split Execution Time: {t_split_total:.6f}s, Original Processing Time: {t_original_total - t_original_scan:.6f}s, Split Processing Time: {t_split_total - t_split_scan:.6f}s"




if __name__ == "__main__":
    
    checker = CodeChecker()


    from benchmark.ssb_benchmark import SSBQueryTemplates
    from config import BENCHMARK

    if BENCHMARK == "ssb":
        from benchmark.ssb_benchmark import SSBQueryTemplates
        query_templates = SSBQueryTemplates()
    elif BENCHMARK == "hits":
        from benchmark.click_benchmark import ClickBenchmarkQueries
        query_templates = ClickBenchmarkQueries()

    for query_id in [29]:
        query_template = query_templates.get_template(query_id)
        query_example = query_templates.get_query(query_id, seed=-1)
        # _, split_template, pipeline_operators = decompose_query(query_example, query_template,use_cache=True)
        # split_query = query_templates.get_query(query_id, seed=1, split_template=split_template)
        # query_example = query_templates.get_query(query_id, seed=1)

        src_path = f"{OPTIMIZED_CODE_DIR}/{BENCHMARK}_q_{query_id}.cpp"
        # src_path = f"{SKETCH_FIX_DIR}/{BENCHMARK}_q_{query_id}.cpp"
        with open(src_path, "r") as f:
            cpp_code = f.read()
            split_query = ""
            split_template = ""
            import re
            header_match = re.match(
                r"/\*\s*query_template:.*?split_template:\s*(.*?)\s*;\s*query_example:.*?split_query:\s*(.*?)\s*;\s*\*/",
                cpp_code,
                re.DOTALL,
            )
            if header_match:
                split_template = header_match.group(1).strip()
                split_query = header_match.group(2).strip()

        state = {
            "input": cpp_code,
            "cpp_code": cpp_code,
            "query_example": query_example,
            "query_template": query_template,
            "decomposed": {"split_query": split_query, "split_template": split_template},
            "problem_desc": "",
            "query_id": query_id,
            "ctx": {},
        }

        success, message = checker.validate_code(state, cpp_code)
        if success:
            print(f"[Code Checker] Validation successful: {message}")
        else:
            print(f"[Code Checker] Validation failed: {message}")
