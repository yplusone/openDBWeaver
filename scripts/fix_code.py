import os
import csv
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import BENCHMARK, SKETCH_DIR, SKETCH_FIX_DIR
from dbweaver.optimize.fix_agent import fix_agent


def fix_query(query_id):
    if BENCHMARK == "ssb":
        from benchmark.ssb_benchmark import SSBQueryTemplates
        query_benchmark = SSBQueryTemplates()
    elif BENCHMARK == "hits":
        from benchmark.click_benchmark import ClickBenchmarkQueries
        query_benchmark = ClickBenchmarkQueries()

    query_template = query_benchmark.get_template(query_id)
    query_example = query_benchmark.get_query(query_id, -1)
    split_query, split_template = query_benchmark.get_split_query(query_id)
    src_path = f"{SKETCH_DIR}/{BENCHMARK}_q_{query_id}.cpp" 
    with open(src_path, "r") as f:
        cpp_code = f.read()

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

    max_iter = 10
    success, message, fixed_code, fix_iterations = fix_agent(cpp_code, state, max_iterations=max_iter)
    if success and fixed_code is not None:
        out_path = f"{SKETCH_FIX_DIR}/{BENCHMARK}_q_{query_id}.cpp"
        with open(out_path, "w") as f:
            f.write(fixed_code)
        print(f"[Fix Agent] Wrote fixed code to: {out_path}")
    else:
        print(f"[Fix Agent] Failed: {message}")

    csv_path = f"{SKETCH_FIX_DIR}/fix_agent_iterations.csv"
    write_header = not os.path.isfile(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as cf:
        w = csv.writer(cf)
        if write_header:
            w.writerow(
                [
                    "benchmark",
                    "query_id",
                    "success",
                    "fix_iterations",
                    "max_iterations",
                    "message",
                ]
            )
        w.writerow([BENCHMARK, query_id, success, fix_iterations, max_iter, message])

if __name__ == "__main__":
    for query_id in [29]:
        fix_query(query_id)