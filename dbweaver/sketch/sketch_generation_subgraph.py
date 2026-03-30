# coding: utf-8
"""
subgraph/sketch_generation_subgraph.py
--------------------------------------
子图：代码草图生成（sketch generation subgraph）

这个子图包含：
- plan_generate: 生成计划
- code_generate: 生成代码片段
- code_combine: 合并代码片段
"""

from __future__ import annotations

import datetime
from typing import Optional, Any, List, Dict
from langgraph.graph import StateGraph, START, END

from dbweaver.sketch.code_combine import CodeCombiner
from dbweaver.sketch.plan import PlanGenerator
from dbweaver.sketch.operator_codegen.operator_gen import build_operator, CodegenContext
from config import CODEGEN_LLM_MODEL, CODEGEN_API_KEY, CODEGEN_BASE_URL, TEMPLATE_PATH,client
from langchain_openai import ChatOpenAI
from pathlib import Path
from dbweaver.sketch.sketch_generation_state import SketchGenerationState
from dbweaver.sketch.gather_context import GatherContext

class SketchGenerationSubgraph:
    """代码草图生成子图：plan_generate -> code_generate -> code_combine"""
    
    def __init__(
        self,
        llm: Optional[Any] = None,
        template: Optional[str] = None,
        plan_generator: Optional[PlanGenerator] = None,
        code_combiner: Optional[CodeCombiner] = None,
        draw_workflow: bool = True,
    ):  
        self.llm = llm or ChatOpenAI(
            model=CODEGEN_LLM_MODEL,
            api_key=CODEGEN_API_KEY,
            base_url=CODEGEN_BASE_URL,
            http_client=client,
            max_retries=4
        )
        
        # 读取模板
        template_path = Path(TEMPLATE_PATH)
        self.template = template or (template_path.read_text() if template_path.exists() else "")
        
        # 初始化组件
        self.plan_generator = plan_generator or PlanGenerator()
        self.code_combiner = code_combiner or CodeCombiner(self.llm)
        self.draw_workflow = draw_workflow
        self.gather_context = GatherContext()
    
    def plan_generate_node(self, state: SketchGenerationState) -> SketchGenerationState:
        """生成计划节点"""
        print(f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] [Sketch Subgraph] Begin to generate plan\n")
        
        # 将子图状态转换为主图状态格式
        
        plan = self.plan_generator.generate_plan(state)
        state["plan"] = plan
        
        return state

    def gather_context_node(self, state):
        """
        收集数据库 schema、表字段信息、已存在的 helper 函数或其它业务配置，
        并存入 state.context，供 generate_template / generate 使用。
        """
        # 1. 读取数据库元数据举例
        print(f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}][Sketch Subgraph] Begin to gather context\n")
        state["ctx"]['columns'] =  self.gather_context.gather_columns_context(state)
        state["ctx"]['output'] = self.gather_context.get_output_datatype(state)
        return state

    def _make_codegen_step_fn(self, step: Dict[str, Any]):
        """代码生成步骤函数工厂"""
        def _node(state: SketchGenerationState) -> SketchGenerationState:
            op = build_operator(step, llm=self.llm)
            # 将 SketchGenerationState 转换为 operator 期望的格式
            ctx = state.get("ctx") or {}
            # 确保每个 operator 都能拿到上一轮 refine 的 reflection 信息
            op_state: Dict[str, Any] = {
                "snippets": state.get("snippets", {}),
                "ctx": ctx,
            }
            codegen_ctx = CodegenContext(extras=ctx)
            bundle = op.run(op_state, codegen_ctx)
            state['snippets'] = bundle
            return state
        return _node
    
    def code_generate_node(self, state: SketchGenerationState) -> SketchGenerationState:
        """生成代码片段节点"""
        print(f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] [Sketch Subgraph] Begin to generate code\n")
        
        plan: List[Dict[str, Any]] = state.get("plan", []) or []
        if not plan:
            return state
        
        # 按 plan 串行拼接子图
        sub = StateGraph(SketchGenerationState)
        prev = START
        for step in plan:
            fn = self._make_codegen_step_fn(step)
            step_name = step.get('op', 'unknown')
            sub.add_node(step_name, fn)
            if prev == START:
                sub.add_edge(START, step_name)
            else:
                sub.add_edge(prev, step_name)
            prev = step_name
        sub.add_edge(prev, END)
        
        compiled_sub = sub.compile()
        if self.draw_workflow:
            try:
                png_bytes = compiled_sub.get_graph().draw_mermaid_png()
                with open("plan.png", "wb") as f:
                    f.write(png_bytes)
            except Exception:
                pass
        
        out_state = compiled_sub.invoke(state)
        return out_state
    
    def code_combine_node(self, state: SketchGenerationState) -> SketchGenerationState:
        """合并代码片段节点"""
        print(f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] [Sketch Subgraph] Begin to combine code snippets\n")
        
        snippets = state.get("snippets", {})
        solution = self.code_combiner.generate(self.template, snippets)
        
        state["code"] = solution
        state["iterations"] = state.get("iterations", 0) + 1
        return state
    
    def build(self) -> Any:
        """构建并编译子图"""
        subgraph = StateGraph(SketchGenerationState)
        
        # 添加节点
        subgraph.add_node("plan_generate", self.plan_generate_node)
        subgraph.add_node("gather_context", self.gather_context_node)
        subgraph.add_node("code_generate", self.code_generate_node)
        subgraph.add_node("code_combine", self.code_combine_node)
        
        # 添加边
        subgraph.add_edge(START, "plan_generate")
        subgraph.add_edge("plan_generate", "gather_context")
        subgraph.add_edge("gather_context", "code_generate")
        subgraph.add_edge("code_generate", "code_combine")
        subgraph.add_edge("code_combine", END)
        
        return subgraph.compile()
    
    def run(self, initial_state: Optional[SketchGenerationState] = None) -> SketchGenerationState:
        """运行子图"""
        app = self.build()
        state = initial_state or SketchGenerationState()  # type: ignore
        return app.invoke(state)






