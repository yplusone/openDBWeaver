# coding: utf-8
import subprocess
import re
import json
from typing import Any, Dict, List, Optional
from langchain_openai import ChatOpenAI
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.duckdb_connector import DuckDBConnector
from config import  EXPT_LLM_MODEL, API_KEY, BASE_URL
from utils.llm_output_parse import extract_json
from utils.sqlast_parameterizer import parameterize_pipeline_ops_ast

PLAN_GENERATION_PROMPT = """\
You are a database logical-plan extractor. Given a SQL query and its “Whole Plan” and/or an input-subquery plan, your goal is to determine which operators are still required when part of the computation (e.g., scan, filter) has already been completed in the input-subquery. 
The purpose of this prompt is to generate only the remaining logical operators needed to execute the query. Enumerate only the operators that are actually required to execute the query. For each operator, precisely list its input columns (from the immediate child’s outputs) and its output columns. 
Use only the operator types you truly need: input, filter, join, sort, agg, output. And I the input operator list determine the logical steps, you only need to fill the fields.

Given:
- A SQL query.
- An initial operator list (skeleton) where each operator already has an `op` type and a `name`, and may contain partially filled fields.
- Whole plan and the input plan. Here you are only required to generate the op list for the whole plan minus the input plan. Just follow the operator list provided, filling the fields.

Steps:
(1) The first step must be an input operator, and the last step must be an output operator, and input and output cannot appear in any intermediate steps.
(2) For each operator, complete the required fields so that the data flow is well-defined and consistent with the SQL query.
(3) Ensure that each operator's `input_cols` come from the **immediate previous operator's** `output_cols` (except for the first `input` operator).
(4) Ensure all column names are consistent with the SQL / plans and with previous operators' output_cols.

For some operators, they should include specific information:
    For input:
    - the input operator take the input plan's output columns as its 'input_cols'.
    - the output_cols should be the same as the input columns of the next operator.
    - expressions: if the ouput_cols are not same as input_cols, there should be expression to compute from input_cols to output_cols. For example: the input is x, and the output square_x is square(x), the expressions should be ["square_x = square(x)"].
    For agg:
    - group_keys: list of columns to group by. If the query does not have explicit grouping keys, use the constant 1 as a placeholder.
    - aggs: list each aggregate expression with alias, e.g. "expr": "SUM(income)", "alias": "sum_income"
    For sort:
    - sort_keys (sort only): [{{"key": "col", "order": "ASC|DESC"}}]
    For join:
    - join_type, join_keys.left, join_keys.right, join_filter (join only)
    For output:
    - Output restriction: If the *next* operator is a LIMIT, include the information here to indicate that the operator’s output should be restricted to the specified number of tuples. For example, "order by xxx limit xx" you need to add output_restriction in sort operator.
    For filter:
    - filters: Only include the predicates represented in Whole Plan but not in Input plan. 

Rules:
3) Put column aliases where they are produced (output or inside the agg outputs).
4) Put complex expressions inside aggs (for agg) or output.
5) If the query is wrapped like plugin_name((SELECT ...)), treat the subquery output as the input of a scan operator with exactly its visible columns.
6) Do not invent columns; reuse exact names/case from inputs.
7) Output exactly one JSON object, no extra text.

Input operator list (skeleton to refine):
{operators}

Output operators exactly same with the input operator list but with all required fields filled!!!!
Output JSON schema:
{{
  "operators": [
    {{
      "op": "input|filter|join|sort|agg|output",
      "name": "string",
      "input_cols": ["colA", "colB"],
      "output_cols": ["colX", "colY"],
      "expresions": ["expr1", "expr2"],
      "filters":["expr1","expr2"],
      "group_keys": ["..."],
      "aggs": [{{"expr": "AGG_EXPR", "alias": "out"}}],
      "sort_keys": [{{"key": "col", "order": "ASC|DESC"}}],
      "output_restriction":["limit ..."]
      "notes": "optional"
    }}
  ]
}}

SQL:
{query}

Whole Plan (if any):
{unoptimized_logical_plan}

Input-subquery Plan:
{input_table_schema}

Task:
Return exactly one JSON object following the schema above, listing only the required operators.
"""


class PlanGenerator:
    def __init__(self, chat_complete_fn=None):
        """
        chat_complete_fn: 保留参数以兼容旧代码。
        如果你使用 generate_plan_via_llm，则可以传入一个函数：
            fn(model: str, prompt: str) -> str
        用来统一调用自己的 LLM 堆栈。
        """
        self._chat_complete_fn = chat_complete_fn
        self._duckdb_connector = DuckDBConnector()


    def _map_node_to_op(self, node: dict) -> Optional[Dict[str, Any]]:
        """
        将一个 DuckDB plan node 映射成我们需要的简化运算符格式。
        注意：PROJECTION 算子直接丢弃（projection算子不需要保留）。
        """
        op_type = str(node.get("operator_type") or node.get("name") or "").strip()
        extra = node.get("extra_info") or {}

        # projection 不保留
        if "PROJECTION" in op_type.upper():
            op: Dict[str, Any] = {
                "op": "projection",
                "name": "projection",
                "input_cols": [],
                "output_cols": node.get("extra_info", {}).get("Projections"),
            }
            return op

        std_op: Optional[str] = None
        up = op_type.upper()

        if "SCAN" in up:
            std_op = "scan"
        elif "JOIN" in up:
            std_op = "join"
        elif "AGGREGATE" in up or "GROUP_BY" in up or "GROUP" in up:
            std_op = "agg"
        elif "ORDER_BY" in up or "SORT" in up or "TOP_N" in up:
            std_op = "sort"
        elif "FILTER" in up:
            std_op = "filter"
        else:
            # 其它算子先跳过（比如 LIMIT / WINDOW 等），后面需要的话再扩展
            return None

        op: Dict[str, Any] = {
            "op": std_op,
            "name": node.get("operator_name", op_type),
            "input_cols": [],
            "output_cols": [],
        }

        # 输出列：优先用 Projections
        projections = extra.get("Projections") or extra.get("Output") or extra.get("Columns") or []
        if isinstance(projections, list):
            op["output_cols"] = projections
        elif isinstance(projections, str):
            op["output_cols"] = [projections]

        # filters: Filter / Filters / Expression / Conditions 中的内容都合并
        filters: List[str] = []
        for key in ("Filter", "Filters", "Expression", "Conditions"):
            val = extra.get(key)
            if val:
                if isinstance(val, list):
                    filters.extend([str(v) for v in val])
                else:
                    filters.append(str(val))
        if filters:
            op["filters"] = filters

        # agg 算子：Groups / Aggregates
        if std_op == "agg":
            op["group_keys"] = []
            op["aggs"] = []

            # -------- group keys --------
            groups = extra.get("Groups") or []
            if isinstance(groups, str):
                groups = [groups]
            if not groups:
                # 没有显式 GROUP BY，用常量 1 占位
                groups = ["1"]
            op["group_keys"] = groups

            # -------- 从子节点的 PROJECTION 里收集 #idx → expr 映射 --------
            proj_map: Dict[str, str] = {}
            for child in node.get("children", []):
                c_type = str(child.get("operator_type") or child.get("name") or "").upper()
                if "PROJECTION" in c_type:
                    child_extra = child.get("extra_info") or {}
                    proj_list = (
                        child_extra.get("Projections")
                        or child_extra.get("Output")
                        or child_extra.get("Columns")
                        or []
                    )
                    if isinstance(proj_list, str):
                        proj_list = [proj_list]
                    if isinstance(proj_list, list):
                        for idx, p in enumerate(proj_list):
                            proj_map[f"#{idx}"] = str(p)

            # -------- 解析 Aggregates --------
            aggs_raw = extra.get("Aggregates")
            if aggs_raw is None:
                aggs_raw = []
            elif isinstance(aggs_raw, str):
                aggs_raw = [aggs_raw]
            elif not isinstance(aggs_raw, list):
                aggs_raw = [str(aggs_raw)]

            aggs: List[Dict[str, str]] = []

            for a in aggs_raw:
                # 支持字符串或 dict
                if isinstance(a, str):
                    expr_str = a.strip()
                elif isinstance(a, dict):
                    expr_str = str(
                        a.get("Expression")
                        or a.get("expr")
                        or a.get("value")
                        or ""
                    ).strip()
                else:
                    expr_str = str(a).strip()

                if not expr_str:
                    continue

                # -------- 把 #0/#1 替换成 projection 对应表达式 --------
                def _sub_proj(m):
                    key = f"#{m.group(1)}"
                    return proj_map.get(key, key)

                resolved_expr = re.sub(r"#(\d+)", _sub_proj, expr_str)
                aggs.append({"expr": resolved_expr})

            op["aggs"] = aggs

        # sort 算子：Order By / Sort Keys
        if std_op == "sort":
            op["sort_keys"] = []
            sort_cols = extra.get("Order By") or extra.get("Sort Keys") or []
            sort_keys: List[Dict[str, str]] = []
            if isinstance(sort_cols, list):
                for s in sort_cols:
                    if isinstance(s, str):
                        parts = s.strip().split()
                        if not parts:
                            continue
                        key = parts[0]
                        order = "ASC"
                        if len(parts) > 1 and parts[1].upper() in ("ASC", "DESC"):
                            order = parts[1].upper()
                        sort_keys.append({"key": key, "order": order})
                    elif isinstance(s, dict):
                        key = s.get("key")
                        order = str(s.get("order", "ASC")).upper()
                        sort_keys.append({"key": key, "order": order})
            op["sort_keys"] = sort_keys

        # 附带一下表名，放在 notes 里（可选）
        table_name = extra.get("Table")
        if table_name:
            op["notes"] = f"Table: {table_name}"
            op["table"] = table_name
        return op

    def _collect_ops(self, node: dict) -> List[Dict[str, Any]]:
        """
        深度优先遍历 plan 树，先遍历 children，再记录当前节点，
        这样更像“自底向上”的 pipeline（scan → filter → agg → sort ...）
        """
        ops: List[Dict[str, Any]] = []
        for child in node.get("children", []):
            ops.extend(self._collect_ops(child))
        mapped = self._map_node_to_op(node)
        if mapped:
            ops.append(mapped)
        return ops

    def _add_input_output(self, ops: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        在算子列表前后补上 input / output：
        - 去掉第一个 scan 算子（input 算子代替它）
        - input.outputs = 原 scan 的输出列
        - output.inputs/outputs = 最后一个算子的输出列
        同时按线性 pipeline 填充中间算子的 inputs。
        """
        if not ops:
            return []

        # merge projection into other operators

        if ops and ops[0].get("op") == "scan":
            if ops[1].get("op") == "projection":
                output_cols = ops[1].get("output_cols", [])
            else:
                output_cols = []
            input_op = {
                "op": "input",
                "name": "input",
                "input_cols": ops[0].get("output_cols", []),
                "output_cols": output_cols,
            }
            ops = ops[1:]
        # delete projection op in ops
        ops = [op for op in ops if op.get("op") != "projection"]
        
        output_op = {
            "op": "output",
            "name": "output_final",
            "input_cols": [],
            "output_cols": ops[-1].get("output_cols", []),
        }

        return [input_op] + ops + [output_op]

    def _build_scan_query_from_ops(self, ops: List[Dict[str, Any]]) -> str:
        """
        从运算符列表中找到第一个 scan 节点，构造一个简单的扫描 SQL：

            SELECT <col1, col2, ...>
            FROM <table>
            [WHERE <filter1> AND <filter2> ...]

        如果找不到合适的信息，则返回空字符串。
        """
        if not ops:
            return ""

        scan_op: Optional[Dict[str, Any]] = None
        for op in ops:
            if op.get("op") == "scan":
                scan_op = op
                break

        if not scan_op:
            return ""

        table = scan_op.get("table")
        if not table:
            # 兼容老数据：从 notes 里 parse 一下 Table: xxx
            notes = str(scan_op.get("notes") or "")
            m = re.search(r"Table:\s*([^\s]+)", notes)
            if m:
                table = m.group(1)

        if not table:
            return ""

        cols = scan_op.get("output_cols") or []
        # 如果 outputs 为空，就退化成 SELECT * FROM table
        if not cols:
            select_list = "*"
        else:
            select_list = ", ".join(cols)

        sql = f"SELECT {select_list} FROM {table}"

        filters = scan_op.get("filters") or []
        filters = [item.replace("optional:","") for item in filters]
        if filters:
            where_clause = " AND ".join(f"({f})" for f in filters)
            sql += f" WHERE {where_clause}"

        return sql

    # ---------------------------
    #  LLM 相关辅助：构建 prompt + 调用
    # ---------------------------
    def _build_llm_prompt(
        self,
        query: str,
        unoptimized_plan: str,
        input_schema: str,
        operators: List[str],
    ) -> str:
        """使用 PLAN_GENERATION_PROMPT 构造给大模型的提示。"""
        return PLAN_GENERATION_PROMPT.format(
            query=query or "",
            operators = operators or "",
            unoptimized_logical_plan=unoptimized_plan or "",
            input_table_schema=input_schema or ""
        )

    def _chat_complete_llm(self, prompt: str) -> str:
        """
        调用大模型，期望返回符合 PLAN_GENERATION_PROMPT 的 JSON 字符串。
        优先使用 self._chat_complete_fn，其次直接用 ChatOpenAI。
        """
        if self._chat_complete_fn:
            return self._chat_complete_fn(EXPT_LLM_MODEL, prompt)

        llm = ChatOpenAI(model=EXPT_LLM_MODEL, api_key=API_KEY, base_url=BASE_URL)
        resp = llm.invoke([{"role": "user", "content": prompt}])
        text = resp.content
        return text

    def generate_plan_via_llm(self, query: str, split_query: str, operators: List[str]) -> List[Dict[str, Any]]:
        """
        使用 LLM 生成逻辑执行计划（operators 列表），不依赖 EXPLAIN json。
        这是一个“可选路径”：不影响 generate_plan 的行为。

        期望 state 至少包含：
          - state['query']       : 原始 SQL
          - state['plugin_name'] : 外层插件名（可选）
          - state['split_query'] : wrapper 形式 SQL，如 plugin((SELECT ...))（可选）
        """
        query = query
        split_query = split_query

        # 原始 query 的 Unoptimized Logical Plan
        original_plan = self._duckdb_connector.get_query_logical_plan(query,'release')

        wrapper_input_plan = self._duckdb_connector.get_query_logical_plan(split_query,'release')

        prompt = self._build_llm_prompt(
            query=query,
            unoptimized_plan=original_plan,
            input_schema=wrapper_input_plan,
            operators=operators
        )

        raw = self._chat_complete_llm(prompt).strip()

        obj = extract_json(raw)
        
        plan: List[Dict[str, Any]] = obj.get("operators", [])

        if not plan:
            return []

        return plan

    # ---------------------------
    #  对外主入口：不再调用大模型，直接用 DuckDB EXPLAIN JSON 生成计划
    # ---------------------------
    def generate_plan(self, state: Dict[str, Any]):
        """
        直接对 state['query'] 做:
          - EXPLAIN (FORMAT json)
        然后把 JSON 解析成我们之前约定的 operators 列表格式。
        其中：
          - 第一条算子为 op = 'input'
          - 最后一条算子为 op = 'output'
          - PROJECTION 算子会被丢弃

        返回值： (plan, split_query)
          - plan        : operators 列表
          - split_query : 从 plan 中自动推导出来的简单 scan SQL（如果能推出来）
        """
        query = (state.get("query_example") or "").strip()
        if not query:
            return [], ""

        # 目前不再区分 plugin_name / split_query，直接对完整 query 做分析
        status, plan_json = self._duckdb_connector.get_query_logical_plan(query,'release','physical_plan')
        if not status or not plan_json:
            raise ValueError(f"Failed to generate plan for query: {query}")
        root = plan_json[0]
        ops = self._collect_ops(root)
        plan = self._add_input_output(ops)
        split_query = self._build_scan_query_from_ops(ops)
        plan = self.generate_plan_via_llm(query, split_query, plan)

        parameterized_plan = parameterize_pipeline_ops_ast(state, plan, split_query)
        unused_params = []
        for param in parameterized_plan['unused_params_for_split_query']:
            for item in parameterized_plan['bindings']:
                if item['param_name'] == param:
                    unused_params.append(item['literal_sql'])
        if len(unused_params) > 0:
            state['decomposed']['split_query'] = f"select * from dbweaver(({split_query}),{','.join(unused_params)});"
            state['decomposed']['split_template'] = f"select * from dbweaver(({parameterized_plan['parameterized_split_query']}),{','.join(parameterized_plan['unused_params_for_split_query'])});"
        else:
            state['decomposed']['split_query'] = f"select * from dbweaver(({split_query}));"
            state['decomposed']['split_template'] = f"select * from dbweaver(({parameterized_plan['parameterized_split_query']}));"
        
        state['ctx']['background_info'] = (
            f"This project studies how to derive the remaining logical operators needed to execute a SQL query"
            f"when an input subquery has already completed part of the work. You are asked to completing the DBWeaver function in the split query "
            f"The query under analysis is: {query}. "
            f"The split query is: {split_query}. "
            f"The associated plan information is: {plan}. "
            f"This serves as the overall project background; however, for this prompt, you only need to concentrate on the task specified next."
        )
        
        return plan


if __name__ == "__main__":
    from benchmark.ssb_benchmark import SSBQueryTemplates
    benchmark_templates = SSBQueryTemplates()
    query = SSBQueryTemplates().get_query(9, -1)
    print(query)
    plan_generator = PlanGenerator()
    plan, split_query = plan_generator.generate_plan({"query": query})
    print(plan)
    print(split_query)