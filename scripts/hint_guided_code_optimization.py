import os

# --- 2. 导入图 ---

import sys
import datetime
import time

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from benchmark.ssb_benchmark import SSBQueryTemplates
from dbweaver.optimize.graph import graph
from dbweaver.sketch.gather_context import GatherContext
from config import DUCKDB_BINARY_PATH, BENCHMARK, SKETCH_FIX_DIR, OPTIMIZED_CODE_DIR, SKETCH_DIR
from benchmark.ssb_benchmark import SSBQueryTemplates
from benchmark.click_benchmark import ClickBenchmarkQueries
from dbweaver.env.code_checker import CodeChecker
from dbweaver.utils.parse_query import parse_query_from_file

def gather_context_node(state):
    """
    收集数据库 schema、表字段信息、已存在的 helper 函数或其它业务配置，
    并存入 state.context，供 generate_template / generate 使用。
    """
    gather_context = GatherContext()
    # 1. 读取数据库元数据举例
    print(f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] Begin to gather context\n")

    state["ctx"]['columns'] =  gather_context.gather_columns_context(state)
    state["ctx"]['output'] = gather_context.get_output_datatype(state)
    state['ctx']['groupby_slots'] = gather_context.gather_groupby_stats(state)
    state["ctx"]['is_constant_candidate'] = gather_context.check_constant_vector_columns(state)
    return state

if __name__ == "__main__":
    # --- 3. 执行代码优化 ---
    # 提供 C++ 代码和对应 SQL 上下文（用于结果校验与性能评估）

    checker = CodeChecker()
    if not os.path.isfile(DUCKDB_BINARY_PATH+"release/duckdb"):
        checker.refresh_code()
    if BENCHMARK == "ssb":
        query_templates = SSBQueryTemplates()
    elif BENCHMARK == "hits":
        query_templates = ClickBenchmarkQueries()
    for query_id in range(16,44):
        if BENCHMARK == "hits" and query_id in [1,2,20,21,24,25,39,41,42]: 
            continue
        try:
            src_path = f"{SKETCH_FIX_DIR}/{BENCHMARK}_q_{query_id}.cpp"
            if not os.path.isfile(src_path):
                continue
            query_example, query_template, split_query, split_template = parse_query_from_file(f"{SKETCH_DIR}/{BENCHMARK}_q_{query_id}.cpp")
        
            with open(src_path, "r") as f:
                cpp_code = f.read()
                          
            state = {
                "input": cpp_code,
                "cpp_code": cpp_code,
                "query_example": query_example,
                "query_template": query_template,
                "decomposed": {"split_query": split_query, "split_template": split_template},
                "problem_desc": "",
                "query_id":query_id,
                "ctx": {},
                "expand_count": 0,
                "original_processing_time": 0,
                "processing_time_history": {},
            }
            
            state = gather_context_node(state)
            last_step = None
            for step in graph.stream(state):
                last_step = step
                step_name, step_state = next(iter(step.items()))
                print(step_name) 
                print("rolled out: ", step_state["root"].height)
                print("---")
        except Exception as e:
            print(f"Error processing query {query_id}: {e}")
            time.sleep(10)
            continue