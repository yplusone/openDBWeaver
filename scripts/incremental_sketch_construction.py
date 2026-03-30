import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import BENCHMARK, SKETCH_DIR
from benchmark.ssb_benchmark import SSBQueryTemplates
from benchmark.click_benchmark import ClickBenchmarkQueries
from dbweaver.sketch.sketch_generation_subgraph import SketchGenerationSubgraph
from dbweaver.env.docker_build import DockerDuckDBRunner
from fix_code import fix_query

def sketch_construction(query_id):

    # docker_build = DockerDuckDBRunner()
    # docker_build.fresh_source_code()
    sketch_generation_subgraph = SketchGenerationSubgraph()

    if BENCHMARK == "ssb":
        workload = SSBQueryTemplates()
    elif BENCHMARK == "hits":
        workload = ClickBenchmarkQueries()


        print(f"Generating sketch for query {query_id}")
        app = sketch_generation_subgraph.build()

        # ✅ 正确初始化 GraphState
        initial_state = {
            "query_id":query_id,
            "plan": None,  
            "snippets": {}, 
            "code": None,
            "query_example":workload.get_query(query_id, -1),
            "query_template": workload.get_template(query_id),
            "decomposed": {}, 
            "ctx": dict(),
        }

        config = {"recursion_limit": 100}
        result = app.invoke(initial_state, config=config)
    
        with open(f"{SKETCH_DIR}/{BENCHMARK}_q_{result['query_id']}.cpp", "w") as f:
            # write query_template, split_template, query_example, split_example as comments
            f.write("/*\n")
            f.write(f"query_template: {result.get('query_template', '')}\n")
            f.write(f"split_template: {result.get('decomposed', {}).get('split_template', '')}\n")
            f.write(f"query_example: {result.get('query_example', '')}\n")
            f.write(f"split_query: {result.get('decomposed', {}).get('split_query', '')}\n")
            f.write("*/\n")
            f.write(result.get("code", "").cpp_code)

    print("\n=== FINAL OUTPUT ===")
    print("Description:", result["code"].prefix)

if __name__ == "__main__":
    for id in [40]:
        # sketch_construction(id)
        fix_query(id)