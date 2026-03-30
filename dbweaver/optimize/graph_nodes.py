"""Graph node functions for LATS."""
import json
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from collections import defaultdict
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from dbweaver.optimize.candidate_chain import generate_and_optimize_candidates, candidate_to_message
from dbweaver.optimize.state import Node, TreeState, Reflection
from dbweaver.optimize.setup import advanced_llm
from dbweaver.env.code_checker import CodeChecker
from config import BENCHMARK, MAX_ITERATIONS_PER_HINT
from dbweaver.optimize.score import create_reflection_from_candidate

def select(root: Node) -> Node:
    """Starting from the root node a child node is selected at each tree level until a leaf node is reached."""
    if not root.children:
        return root

    node = root
    while node.children:
        all_one = all(child.reflection.score == 1 for child in node.children)
        if all_one:
            max_child = min(node.children, key=lambda child: child.reflection.performance.get('processing_time', float('inf')))
        else:
            max_child = max(node.children, key=lambda child: child.upper_confidence_bound())
        node = max_child

    return node


def _prompt_vars(state: TreeState, node: Node = None) -> dict:
    state_dict = dict(state)
    decomposed = state_dict.get("decomposed") or {}
    
    # 默认使用全局初始状态
    cpp_code = state_dict.get("cpp_code") or state_dict.get("input", ""),
    problem_desc = state_dict.get("problem_desc", "")

    # 如果提供了特定节点，则优先使用该节点内部记录的代码和报错
    if node and node.reflection:
        if node.reflection.new_cpp_code:
            cpp_code = node.reflection.new_cpp_code
        if node.reflection.reflections:
            problem_desc = node.reflection.reflections

    return {
        "cpp_code": cpp_code,
        "query_template": state_dict.get("query_template", ""),
        "split_template": decomposed.get("split_template", ""),
        "problem_desc": problem_desc,
    }


def _patch_to_messages(patch: dict) -> list:
    content = json.dumps(patch, ensure_ascii=False)
    # Wrap in markdown code block to be helpful
    ai_msg = AIMessage(content=f"```json\n{content}\n```")
    return [ai_msg]


def generate_initial_response(state: TreeState) -> dict:
    """Generate the initial candidate response."""
    output_messages = []
    # initial_response = initial_optimize_node(state, advanced_llm)
    # patch = _parse_patch.invoke(initial_response)
    # current_node_state = dict(state)
    # reflection = score_candidate(current_node_state, "")
    code_checker = CodeChecker()
    success, reason = code_checker.validate_code(state, state["cpp_code"])
    print(f"[Candidate Chain] Optimization succeeded, testing performance...")
    original_processing_time = float(reason.split("Original Processing Time:")[1].split(",")[0][:-1])
    state["original_processing_time"] = original_processing_time
    split_execution_time = float(reason.split("Split Execution Time:")[1].split(",")[0][:-1])
    split_processing_time = float(reason.split("Split Processing Time:")[1].split(",")[0][:-1])
    performance = {"total_time": split_execution_time, "processing_time": split_processing_time}
    
    reflection = create_reflection_from_candidate(state,{"success": True, "optimized_code": state["cpp_code"], "performance": performance, "message": reason}, state["original_processing_time"], split_processing_time)
    root = Node(output_messages, reflection=reflection)
    return {
        **state,
        "root": root,
    }


def expand(state: TreeState, config: RunnableConfig) -> dict:
    """Starting from the "best" node in the tree, generate N candidates for the next step."""
    root = state["root"]
    best_candidate: Node = select(root)
    # best_candidate = root.get_best_solution()
    messages = best_candidate.get_trajectory()
    
    # Extract optimize_hint_history from trajectory
    optimize_hint_history = []
    for msg in messages:
        if isinstance(msg, AIMessage):
            content = msg.content
            if "[TRIED]" in content:
                optimize_hint_history.append(content)
    
    # Prepare state for optimization
    current_node_state = dict(state)
    
    # Use the code from the best candidate if available
    if best_candidate.reflection and best_candidate.reflection.new_cpp_code:
        current_node_state["cpp_code"] = best_candidate.reflection.new_cpp_code
        
    if best_candidate.reflection:
        current_node_state['reflection'] = best_candidate.reflection
    # Get N value from config
    n = config.get("configurable", {}).get("N", 3)
    
    # Generate N optimization directions and apply them using optimize_agent
    print(f"\n[Expand] Generating {n} optimization candidates...")
    candidates = generate_and_optimize_candidates(
        state=current_node_state,
        n=n,
        optimize_hint_history=optimize_hint_history,
        max_iterations_per_hint=MAX_ITERATIONS_PER_HINT,
    )
    
    # Get root node's processing time as baseline
    root_processing_time = root.reflection.performance['processing_time']
    
    print(f"[Expand] expand node processing time: {best_candidate.reflection.performance['processing_time']:.3f}s")
    # Create output messages and reflections for each candidate
    output_messages = []
    reflections = []
    
    for candidate in candidates:
        # Create message with optimization hint
        hint_msg = AIMessage(content=str(candidate["optimization_hint"]))
        
        # Create result message with performance info
        result_msg = AIMessage(content=candidate_to_message(candidate))
        
        # Store both messages
        output_messages.append([hint_msg, result_msg])
        
        # Create reflection based on processing time speedup
        reflection = create_reflection_from_candidate(state, candidate, state["original_processing_time"], best_candidate.reflection.performance['processing_time'])
        reflections.append(reflection)
    
    # Grow tree
    child_nodes = [
        Node(msgs, parent=best_candidate, reflection=reflection)
        for msgs, reflection in zip(output_messages, reflections)
    ]
    best_candidate.children.extend(child_nodes)
    
    # Increment expand count
    state["expand_count"] = state.get("expand_count", 0) + 1
    print(f"[Expand] Completed iteration {state['expand_count']}")
    
    return state
