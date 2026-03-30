from langgraph.graph import END, StateGraph, START
import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dbweaver.optimize.graph_nodes import expand, generate_initial_response
from dbweaver.optimize.state import TreeState
from config import BENCHMARK, OPTIMIZED_CODE_DIR, TRACE_DIR


def _dump_search_tree(root, state: TreeState) -> str:
    """Dump the whole search tree (hints, performance, etc.) to a JSONL file."""
    query_id = state.get("query_id", "unknown")
    os.makedirs(TRACE_DIR, exist_ok=True)
    output_path = os.path.join(TRACE_DIR, f"{BENCHMARK}_q_{query_id}_tree.jsonl")

    all_nodes = [root]
    if hasattr(root, "_get_all_children"):
        all_nodes.extend(root._get_all_children())

    # Map python object id -> sequential index for parent lookup
    index_map = {id(node): idx for idx, node in enumerate(all_nodes)}

    with open(output_path, "w", encoding="utf-8") as f:
        for idx, node in enumerate(all_nodes):
            reflection = getattr(node, "reflection", None)
            performance = getattr(reflection, "performance", {}) if reflection else {}
            hints = []
            for msg in getattr(node, "messages", []):
                content = getattr(msg, "content", None)
                if isinstance(content, str):
                    hints.append(content)

            parent = getattr(node, "parent", None)
            parent_index = index_map.get(id(parent)) if parent is not None else None

            node_record = {
                "index": idx,
                "parent_index": parent_index,
                "depth": getattr(node, "depth", None),
                "visits": getattr(node, "visits", None),
                "value": getattr(node, "value", None),
                "is_solved": getattr(node, "is_solved", None),
                "is_terminal": getattr(node, "is_terminal", None),
                "score": getattr(reflection, "score", None) if reflection else None,
                "found_solution": getattr(reflection, "found_solution", None)
                if reflection
                else None,
                "performance": performance,
                "hints": hints,
            }
            f.write(json.dumps(node_record, ensure_ascii=False) + "\n")

    return output_path


def should_loop(state: TreeState):
    """Determine whether to continue the tree search."""
    root = state["root"]
    expand_count = state.get("expand_count", 0)
    
    # Check termination conditions
    should_terminate = False
    termination_reason = ""
    
    if root.is_solved:
        should_terminate = True
        termination_reason = "Found solution with speedup >= 5x"
    elif expand_count >= 5:
        should_terminate = True
        termination_reason = f"Reached maximum expand iterations ({expand_count})"
    elif root.height > 5:
        should_terminate = True
        termination_reason = f"Tree height exceeded limit ({root.height})"
    
    if should_terminate:
        print(f"\n[LATS] Terminating: {termination_reason}")
        
        # Save the best solution to file
        best_solution_node = root.get_best_solution()
        if best_solution_node and best_solution_node.reflection and best_solution_node.reflection.new_cpp_code:
            query_id = state.get("query_id", "unknown")
            output_dir = OPTIMIZED_CODE_DIR
            os.makedirs(output_dir, exist_ok=True)
            output_path = f"{output_dir}/{BENCHMARK}_q_{query_id}.cpp"
            
            with open(output_path, "w") as f:
                f.write(best_solution_node.reflection.new_cpp_code)
            
            performance = best_solution_node.reflection.performance
            processing_time = performance.get("processing_time", 0.0)
            total_time = performance.get("total_time", 0.0)
            
            print(f"[LATS] Best solution saved to: {output_path}")
            print(f"[LATS] Performance: processing={processing_time:.6f}s, total={total_time:.6f}s")
            print(f"[LATS] Reflection: {best_solution_node.reflection.reflections}")

        else:
            print("[LATS] Warning: No valid solution found to save")

        # Dump the whole search tree (including hints and performance) to file
        tree_path = _dump_search_tree(root, state)
        print(f"[LATS] Full search tree saved to: {tree_path}")
        
        return END
    
    return "expand"


builder = StateGraph(TreeState)
builder.add_node("start", generate_initial_response)
builder.add_node("expand", expand)
builder.add_edge(START, "start")


builder.add_conditional_edges(
    "start",
    # Either expand/rollout or finish
    should_loop,
    ["expand", END],
)
builder.add_conditional_edges(
    "expand",
    # Either continue to rollout or finish
    should_loop,
    ["expand", END],
)

graph = builder.compile()