from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple, Dict, Any, Set

import sqlglot
from sqlglot import exp


@dataclass(frozen=True)
class ParamBinding:
    context_sql: str
    literal_sql: str
    param_name: str


class SQLAstParameterizer:
    PARAM_PREFIX = "__DBW_PARAM__"

    def __init__(self, dialect: Optional[str] = None):
        self.dialect = dialect

    # ============================================================
    # Public API
    # ============================================================
    def parameterize_pipeline_and_queries(
        self,
        state: dict,
        ops: list,
        split_query: Optional[str] = None,
    ) -> dict:
        template_sql = state["query_template"]
        example_sql = state["query_example"]

        bindings = self.build_bindings(template_sql, example_sql)
        new_ops = self.apply_bindings_to_ops(ops, bindings)

        result = {
            "bindings": [asdict(b) for b in bindings],
            "ops": new_ops,
        }

        if split_query is not None:
            parameterized_split_query = self.parameterize_sql_text(split_query, bindings)
            all_template_params = self.collect_all_template_params(template_sql)
            used_params = self.collect_used_params_from_sql(parameterized_split_query)
            unused_params = sorted(all_template_params - used_params)

            result.update(
                {
                    "parameterized_split_query": parameterized_split_query,
                    "used_params_in_split_query": sorted(used_params),
                    "unused_params_for_split_query": unused_params,
                }
            )

        return result

    def parameterize_pipeline_ops(self, state: dict, ops: list) -> dict:
        template_sql = state["query_template"]
        example_sql = state["query_example"]

        bindings = self.build_bindings(template_sql, example_sql)
        new_ops = self.apply_bindings_to_ops(ops, bindings)

        return {
            "bindings": [asdict(b) for b in bindings],
            "ops": new_ops,
        }

    def parameterize_sql_text(self, sql_text: str, bindings: List[ParamBinding]) -> str:
        try:
            ast = self._parse_sql(sql_text)
            new_ast = self._replace_in_expr_ast(ast, bindings)
            return self._sql(new_ast)
        except Exception:
            return self._fallback_contextual_replace(sql_text, bindings)

    def collect_all_template_params(self, template_sql: str) -> Set[str]:
        return set(re.findall(r":[A-Za-z_][A-Za-z0-9_]*", template_sql))

    def collect_used_params_from_sql(self, sql_text: str) -> Set[str]:
        return set(re.findall(r":[A-Za-z_][A-Za-z0-9_]*", sql_text))

    def build_bindings(self, query_template: str, query_example: str) -> List[ParamBinding]:
        template_ast = self._parse_sql(query_template)
        example_ast = self._parse_sql(query_example)

        template_where = template_ast.args.get("where")
        example_where = example_ast.args.get("where")

        if template_where is None or example_where is None:
            return []

        template_preds = self._flatten_and(template_where.this)
        example_preds = self._flatten_and(example_where.this)

        bindings: List[ParamBinding] = []
        used_example_idx = set()

        for tpred in template_preds:
            idx, local_bindings = self._match_template_predicate_to_example(
                tpred, example_preds, used_example_idx
            )
            if idx is not None:
                used_example_idx.add(idx)
                bindings.extend(local_bindings)

        return self._deduplicate_bindings(bindings)

    def apply_bindings_to_ops(self, ops: list, bindings: List[ParamBinding]) -> list:
        new_ops = deepcopy(ops)

        for op in new_ops:
            if "filters" in op:
                op["filters"] = [self._parameterize_expr_text(x, bindings) for x in op["filters"]]

            if "group_keys" in op:
                op["group_keys"] = [self._parameterize_expr_text(x, bindings) for x in op["group_keys"]]

            if "input_cols" in op:
                op["input_cols"] = [self._parameterize_expr_text(x, bindings) for x in op["input_cols"]]

            if "output_cols" in op:
                op["output_cols"] = [self._parameterize_expr_text(x, bindings) for x in op["output_cols"]]

            if "aggs" in op:
                for agg in op["aggs"]:
                    if "expr" in agg and isinstance(agg["expr"], str):
                        agg["expr"] = self._parameterize_expr_text(agg["expr"], bindings)
                    if "alias" in agg and isinstance(agg["alias"], str):
                        agg["alias"] = self._parameterize_expr_text(agg["alias"], bindings)

            if "sort_keys" in op:
                for sk in op["sort_keys"]:
                    if "key" in sk and isinstance(sk["key"], str):
                        sk["key"] = self._parameterize_expr_text(sk["key"], bindings)

        return new_ops

    # ============================================================
    # Parsing helpers
    # ============================================================
    def _preprocess_params(self, sql: str) -> str:
        return re.sub(
            r":([A-Za-z_][A-Za-z0-9_]*)",
            rf"{self.PARAM_PREFIX}\1",
            sql,
        )

    def _restore_params(self, text: str) -> str:
        return re.sub(
            rf"{re.escape(self.PARAM_PREFIX)}([A-Za-z_][A-Za-z0-9_]*)",
            r":\1",
            text,
        )

    def _parse_sql(self, sql: str) -> exp.Expression:
        return sqlglot.parse_one(self._preprocess_params(sql), read=self.dialect)

    def _parse_expr(self, text: str) -> exp.Expression:
        return sqlglot.parse_one(self._preprocess_params(text), read=self.dialect)

    def _sql(self, node: exp.Expression) -> str:
        return self._restore_params(node.sql(dialect=self.dialect))

    def _norm_sql(self, node: exp.Expression) -> str:
        return re.sub(r"\s+", " ", self._sql(node).strip()).upper()

    def _same_expr(self, a: exp.Expression, b: exp.Expression) -> bool:
        return self._norm_sql(self._unwrap_paren(a)) == self._norm_sql(self._unwrap_paren(b))

    # ============================================================
    # AST normalization
    # ============================================================
    def _unwrap_paren(self, node: exp.Expression) -> exp.Expression:
        while isinstance(node, exp.Paren):
            node = node.this
        return node

    def _flatten_and(self, node: exp.Expression) -> List[exp.Expression]:
        node = self._unwrap_paren(node)
        if isinstance(node, exp.And):
            return self._flatten_and(node.left) + self._flatten_and(node.right)
        return [node]

    def _flatten_or(self, node: exp.Expression) -> List[exp.Expression]:
        node = self._unwrap_paren(node)
        if isinstance(node, exp.Or):
            return self._flatten_or(node.left) + self._flatten_or(node.right)
        return [node]

    # ============================================================
    # Binding construction
    # ============================================================
    def _match_template_predicate_to_example(
        self,
        template_pred: exp.Expression,
        example_preds: List[exp.Expression],
        used_example_idx: set,
    ) -> Tuple[Optional[int], List[ParamBinding]]:
        template_pred = self._unwrap_paren(template_pred)

        for i, epred in enumerate(example_preds):
            if i in used_example_idx:
                continue

            epred = self._unwrap_paren(epred)
            bindings = self._match_predicate_pair(template_pred, epred)
            if bindings is not None:
                return i, bindings

        return None, []

    def _match_predicate_pair(
        self,
        tpred: exp.Expression,
        epred: exp.Expression,
    ) -> Optional[List[ParamBinding]]:
        tpred = self._unwrap_paren(tpred)
        epred = self._unwrap_paren(epred)

        for matcher in (
            self._match_in_predicate,
            self._match_between_predicate,
            lambda t, e: self._match_binary_predicate(t, e, exp.EQ),
            lambda t, e: self._match_binary_predicate(t, e, exp.GTE),
            lambda t, e: self._match_binary_predicate(t, e, exp.GT),
            lambda t, e: self._match_binary_predicate(t, e, exp.LTE),
            lambda t, e: self._match_binary_predicate(t, e, exp.LT),
            lambda t, e: self._match_binary_predicate(t, e, exp.NEQ),
        ):
            bindings = matcher(tpred, epred)
            if bindings is not None:
                return bindings

        return None

    def _match_in_predicate(
        self,
        tpred: exp.Expression,
        epred: exp.Expression,
    ) -> Optional[List[ParamBinding]]:
        tpred = self._unwrap_paren(tpred)
        epred = self._unwrap_paren(epred)

        if not isinstance(tpred, exp.In):
            return None

        t_context = self._unwrap_paren(tpred.this)
        t_params = []
        for item in tpred.expressions:
            pname = self._extract_param_name(item)
            if pname is None:
                return None
            t_params.append(pname)

        if isinstance(epred, exp.In):
            e_context = self._unwrap_paren(epred.this)
            if not self._same_expr(t_context, e_context):
                return None

            e_literals = [self._unwrap_paren(x) for x in epred.expressions]
            if len(e_literals) != len(t_params):
                return None
            if not all(self._is_literal_like(x) for x in e_literals):
                return None

            return [
                ParamBinding(
                    context_sql=self._sql(t_context),
                    literal_sql=self._sql(lit),
                    param_name=param,
                )
                for lit, param in zip(e_literals, t_params)
            ]

        or_terms = self._flatten_or(epred)
        eq_literals = []

        for term in or_terms:
            term = self._unwrap_paren(term)
            if not isinstance(term, exp.EQ):
                return None

            left = self._unwrap_paren(term.left)
            right = self._unwrap_paren(term.right)

            if self._same_expr(left, t_context) and self._is_literal_like(right):
                eq_literals.append(right)
            elif self._same_expr(right, t_context) and self._is_literal_like(left):
                eq_literals.append(left)
            else:
                return None

        if len(eq_literals) != len(t_params):
            return None

        return [
            ParamBinding(
                context_sql=self._sql(t_context),
                literal_sql=self._sql(lit),
                param_name=param,
            )
            for lit, param in zip(eq_literals, t_params)
        ]

    def _match_between_predicate(
        self,
        tpred: exp.Expression,
        epred: exp.Expression,
    ) -> Optional[List[ParamBinding]]:
        tpred = self._unwrap_paren(tpred)
        epred = self._unwrap_paren(epred)

        if not isinstance(tpred, exp.Between) or not isinstance(epred, exp.Between):
            return None

        t_context = self._unwrap_paren(tpred.this)
        e_context = self._unwrap_paren(epred.this)
        if not self._same_expr(t_context, e_context):
            return None

        low_param = self._extract_param_name(tpred.args.get("low"))
        high_param = self._extract_param_name(tpred.args.get("high"))
        if low_param is None or high_param is None:
            return None

        low_lit = self._unwrap_paren(epred.args.get("low"))
        high_lit = self._unwrap_paren(epred.args.get("high"))
        if not self._is_literal_like(low_lit) or not self._is_literal_like(high_lit):
            return None

        ctx = self._sql(t_context)
        return [
            ParamBinding(context_sql=ctx, literal_sql=self._sql(low_lit), param_name=low_param),
            ParamBinding(context_sql=ctx, literal_sql=self._sql(high_lit), param_name=high_param),
        ]

    def _match_binary_predicate(
        self,
        tpred: exp.Expression,
        epred: exp.Expression,
        cls: type,
    ) -> Optional[List[ParamBinding]]:
        tpred = self._unwrap_paren(tpred)
        epred = self._unwrap_paren(epred)

        if not isinstance(tpred, cls) or not isinstance(epred, cls):
            return None

        t_left = self._unwrap_paren(tpred.left)
        t_right = self._unwrap_paren(tpred.right)
        e_left = self._unwrap_paren(epred.left)
        e_right = self._unwrap_paren(epred.right)

        if self._same_expr(t_left, e_left):
            pname = self._extract_param_name(t_right)
            if pname and self._is_literal_like(e_right):
                return [
                    ParamBinding(
                        context_sql=self._sql(t_left),
                        literal_sql=self._sql(e_right),
                        param_name=pname,
                    )
                ]

        if self._same_expr(t_right, e_right):
            pname = self._extract_param_name(t_left)
            if pname and self._is_literal_like(e_left):
                return [
                    ParamBinding(
                        context_sql=self._sql(t_right),
                        literal_sql=self._sql(e_left),
                        param_name=pname,
                    )
                ]

        return None

    # ============================================================
    # Expression rewriting
    # ============================================================
    def _parameterize_expr_text(self, text: str, bindings: List[ParamBinding]) -> str:
        try:
            expr_ast = self._parse_expr(text)
            replaced = self._replace_in_expr_ast(expr_ast, bindings)
            return self._sql(replaced)
        except Exception:
            return self._fallback_contextual_replace(text, bindings)

    def _replace_in_expr_ast(self, node: exp.Expression, bindings: List[ParamBinding]) -> exp.Expression:
        result = node.copy()

        parsed_bindings = []
        for b in bindings:
            try:
                context_ast = self._parse_expr(b.context_sql)
                literal_ast = self._parse_expr(b.literal_sql)
                param_ast = self._parse_expr(b.param_name)
                parsed_bindings.append((b, context_ast, literal_ast, param_ast))
            except Exception:
                continue

        def transform(curr: exp.Expression) -> exp.Expression:
            curr_unwrapped = self._unwrap_paren(curr)
            parent = getattr(curr, "parent", None)
            if parent is None:
                return curr

            parent_unwrapped = self._unwrap_paren(parent)

            for _, context_ast, literal_ast, param_ast in parsed_bindings:
                if not self._same_expr(curr_unwrapped, literal_ast):
                    continue

                if isinstance(parent_unwrapped, (exp.EQ, exp.GTE, exp.GT, exp.LTE, exp.LT, exp.NEQ)):
                    left = self._unwrap_paren(parent_unwrapped.left)
                    right = self._unwrap_paren(parent_unwrapped.right)

                    if curr is parent_unwrapped.right and self._same_expr(left, context_ast):
                        return param_ast.copy()
                    if curr is parent_unwrapped.left and self._same_expr(right, context_ast):
                        return param_ast.copy()

                elif isinstance(parent_unwrapped, exp.In):
                    if self._same_expr(self._unwrap_paren(parent_unwrapped.this), context_ast):
                        return param_ast.copy()

                elif isinstance(parent_unwrapped, exp.Between):
                    if self._same_expr(self._unwrap_paren(parent_unwrapped.this), context_ast):
                        if curr is parent_unwrapped.args.get("low") or curr is parent_unwrapped.args.get("high"):
                            return param_ast.copy()

            return curr

        return result.transform(transform)

    def _fallback_contextual_replace(self, text: str, bindings: List[ParamBinding]) -> str:
        out = text
        upper_text = text.upper()
        ordered = sorted(bindings, key=lambda b: len(b.literal_sql), reverse=True)

        for b in ordered:
            if b.context_sql.upper() not in upper_text:
                continue

            lit = b.literal_sql
            param = b.param_name

            if lit.startswith("'") and lit.endswith("'"):
                out = out.replace(lit, param)
            else:
                out = re.sub(rf"\b{re.escape(lit)}\b", param, out)

        return out

    # ============================================================
    # Utils
    # ============================================================
    def _extract_param_name(self, node: Optional[exp.Expression]) -> Optional[str]:
        if node is None:
            return None

        node = self._unwrap_paren(node)
        text = self._sql(node).strip()

        if re.fullmatch(r":[A-Za-z_][A-Za-z0-9_]*", text):
            return text

        raw = re.sub(r"\s+", "", text)
        m = re.fullmatch(rf"{re.escape(self.PARAM_PREFIX)}([A-Za-z_][A-Za-z0-9_]*)", raw)
        if m:
            return f":{m.group(1)}"

        return None

    def _is_literal_like(self, node: exp.Expression) -> bool:
        node = self._unwrap_paren(node)
        if isinstance(node, exp.Literal):
            return True
        if isinstance(node, exp.Cast):
            return self._is_literal_like(node.this)
        if isinstance(node, exp.Neg):
            return self._is_literal_like(node.this)
        if isinstance(node, exp.Date):
            return True
        if isinstance(node, exp.Timestamp):
            return True
        return False

    def _deduplicate_bindings(self, bindings: List[ParamBinding]) -> List[ParamBinding]:
        seen = set()
        out = []
        for b in bindings:
            key = (b.context_sql.upper(), b.literal_sql, b.param_name)
            if key not in seen:
                seen.add(key)
                out.append(b)
        return out


def parameterize_pipeline_ops_ast(
    state: dict,
    ops: list,
    split_query: Optional[str] = None,
    dialect: Optional[str] = None,
) -> dict:
    return SQLAstParameterizer(dialect=dialect).parameterize_pipeline_and_queries(
        state=state,
        ops=ops,
        split_query=split_query,
    )