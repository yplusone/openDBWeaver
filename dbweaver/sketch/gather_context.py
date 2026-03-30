

import re
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import sqlparse
from sqlparse.sql import IdentifierList, Identifier
from sqlparse.tokens import Keyword, DML
from dbweaver.utils.sqlparse_util import extract_tables
import csv
from io import StringIO
from dbweaver.utils.parse_query import extract_inner_sql
from dbweaver.env.duckdb_connector import DuckDBConnector

class GatherContext:
    DEFAULT_PRAGMAS = ["PRAGMA threads = 1;"]
    def __init__(self):
        self.duckdb_connector = DuckDBConnector()
    # ============================================================
    #  Schema / 类型收集
    # ============================================================
    def get_table_schema(self, table_name: str):
        """
        使用 DuckDB DESCRIBE 获取某个表的列名/类型。
        返回: {column_name: column_type}
        """
        ok, payload = self.duckdb_connector.execute_query(f"DESCRIBE {table_name}",output_format="csv")
        if not ok:
            raise RuntimeError(f"Failed to describe table {table_name}: {payload}")

        reader = csv.DictReader(StringIO(payload))
        schema_dict = {row["column_name"]: row["column_type"] for row in reader}
        return schema_dict

    def gather_columns_context(self, state):
        """
        收集数据库 schema、表字段信息，供 generate_template / generate 使用。
        返回: {column_name: column_type}
        """
        query = state["query_example"]
        ast = sqlparse.parse(query)[0]
        tables = list(extract_tables(ast))
        if not len(tables):
            raise ValueError("No tables found in the query.")
        table_information = {}
        for table in tables:
            info = self.get_table_schema(table)
            table_information.update(info)
        return table_information

    def get_output_datatype(self, state):
        """
        对于输入 SQL，获取其结果列的类型信息。
        返回: {column_name: column_type}
        """
        query = state["query_example"]
        ok, payload = self.duckdb_connector.execute_query(f"DESCRIBE {query}",output_format="csv")
        if not ok:
            raise RuntimeError(f"Failed to describe query output: {payload}")

        reader = csv.DictReader(StringIO(payload))
        res = {row["column_name"]: row["column_type"] for row in reader}
        return res

    # ============================================================
    #  SQL 解析辅助
    # ============================================================
    def _strip_trailing_semicolon(self, sql: str) -> str:
        return sql.strip().rstrip(";").rstrip()

    def _extract_groupby_columns(self, parsed_stmt):
        """
        返回 GROUP BY 里的原始字符串列表，例如 ["year", "P_BRAND"]。
        """
        tokens = [t for t in parsed_stmt.tokens if not t.is_whitespace]

        groupby_cols = []

        for idx, token in enumerate(tokens):
            # 注意：这里要用 “in Keyword”，不能用 “is Keyword”
            if token.ttype in Keyword and token.normalized.upper() == "GROUP BY":
                if idx + 1 < len(tokens):
                    gb_token = tokens[idx + 1]
                    if isinstance(gb_token, IdentifierList):
                        for ident in gb_token.get_identifiers():
                            groupby_cols.append(ident.value.strip())
                    elif isinstance(gb_token, Identifier):
                        groupby_cols.append(gb_token.value.strip())
                    else:
                        groupby_cols.append(gb_token.value.strip())
                break

        return groupby_cols

    def _extract_select_alias_map(self, parsed_stmt):
        """
        使用 sqlparse AST，解析 SELECT 列中的 "expr AS alias"。

        返回: {alias: expr_without_alias}
        """
        alias_map = {}

        def parse_alias_expr_sqlparse(tok):
            """
            sqlparse 的 get_identifiers() 可能返回非 Identifier（例如常量 Token），
            这里统一做一个安全解析：
            - 优先走 tok.get_alias()
            - 否则从 tok.value 里匹配 '... AS alias'（支持引号）
            返回 (alias, expr_without_alias)；若没有 alias 返回 (None, None)
            """
            # 1) Identifier/Function/Case 等通常有 get_alias
            if hasattr(tok, "get_alias"):
                alias = tok.get_alias()
                if not alias:
                    return None, None
                expr = tok.value.strip()
            else:
                # 2) 常量/裸 Token：尝试从字符串里抠出 AS alias
                expr_full = getattr(tok, "value", str(tok)).strip()
                # alias 支持 "a", `a`, [a] 以及裸标识符
                m = re.search(
                    r"""(?is)\s+AS\s+("([^"]+)"|`([^`]+)`|\[([^\]]+)\]|([A-Za-z_][\w$]*))\s*$""",
                    expr_full,
                )
                if not m:
                    return None, None
                alias = m.group(2) or m.group(3) or m.group(4) or m.group(5)
                expr = expr_full[: m.start()].strip()

            # 统一去掉尾部 alias（兼容 'expr alias' 形式）
            upper_expr = expr.upper()
            if " AS " in upper_expr:
                expr = expr[: upper_expr.rfind(" AS ")].strip()
            else:
                # token.value 可能是 "1 AS a"（已经被截断），也可能是 "1 a"
                # 这里尽量只在末尾完整匹配 alias 时才裁剪，避免误伤函数参数等
                # e.g. "col_a" 不应该把 "a" 裁掉
                tail_pat = re.compile(rf"""(?is)\s+("{re.escape(alias)}"|`{re.escape(alias)}`|\[{re.escape(alias)}\]|{re.escape(alias)})\s*$""")
                if tail_pat.search(expr):
                    expr = tail_pat.sub("", expr).strip()

            return alias, expr

        tokens = [t for t in parsed_stmt.tokens if not t.is_whitespace]
        seen_select = False

        for idx, token in enumerate(tokens):
            # SELECT 是 DML，不是 Keyword，要用 ttype in (DML, Keyword)
            if not seen_select:
                if (token.ttype in DML or token.ttype in Keyword) and \
                        token.normalized.upper() == "SELECT":
                    seen_select = True
                continue

            # 遇到 FROM 说明 SELECT 列结束
            if token.ttype in Keyword and token.normalized.upper() == "FROM":
                break

            # 收集 Identifier / IdentifierList
            if isinstance(token, IdentifierList):
                for ident in token.get_identifiers():
                    alias, expr = parse_alias_expr_sqlparse(ident)
                    if not alias:
                        continue
                    alias_map[alias] = expr
            elif isinstance(token, Identifier):
                alias, expr = parse_alias_expr_sqlparse(token)
                if not alias:
                    continue
                alias_map[alias] = expr

        return alias_map

    # ============================================================
    #  “是否是底表原始列”判断 & 底表列收集
    # ============================================================
    def _is_simple_identifier_of_base_col(self, expr: str, base_columns):
        """
        判断 expr 是否只是一个底表列名（或带表别名的列名），比如：
          - "P_BRAND"
          - "t.P_BRAND"
        而不是 EXTRACT(...) 这类生成表达式。
        """
        s = expr.strip()
        s = s.strip('"`')
        if "." in s:
            s = s.split(".")[-1]
        return s in base_columns

    def _get_base_columns_for_query(self, parsed_stmt):
        """
        收集这个查询中涉及的底表列名集合（所有表所有列的并集）。
        """
        tables = list(extract_tables(parsed_stmt))
        base_cols = set()
        for tbl in tables:
            try:
                schema = self.get_table_schema(tbl)
                base_cols.update(schema.keys())
            except Exception:
                continue
        return base_cols

    # ============================================================
    #  计算：哪些 GROUP BY 列是“生成列”
    # ============================================================
    def _extract_generated_groupby_items(self, original_sql: str):
        """
        返回本查询中需要 profile 的“生成 group-by 列”列表。

        返回:
            base_table: 用来做 profiling 的主表名（当前取第一个 table）
            items: List[(logical_name, expr_sql)]
                - logical_name: 用作返回 JSON key（优先 alias，例如 "year"）
                - expr_sql: 需要在 SELECT DISTINCT 里使用的表达式
        """
        parsed = sqlparse.parse(original_sql)[0]
        gb_cols = self._extract_groupby_columns(parsed)
        if not gb_cols:
            return None, []

        alias_map = self._extract_select_alias_map(parsed)

        tables = list(extract_tables(parsed))
        if not tables:
            return None, []
        base_table = tables[0]

        base_cols = self._get_base_columns_for_query(parsed)

        items = []

        for gb in gb_cols:
            key = gb.strip()

            # Case A: group by 的字符串本身就是简单的底表列 → 不 profile
            if self._is_simple_identifier_of_base_col(key, base_cols):
                continue

            # Case B: group by 用的是 alias
            if key in alias_map:
                expr = alias_map[key].strip()

                # alias 对应的表达式只是底表列 → 不 profile
                if self._is_simple_identifier_of_base_col(expr, base_cols):
                    continue

                # 真正的生成列
                logical_name = key           # 用 alias 做 JSON key & AS 名
                expr_sql = expr              # EXTRACT(...) / 函数表达式等
                items.append((logical_name, expr_sql))
                continue

            # Case C: group by 是非别名的复杂表达式，例如 "EXTRACT(YEAR FROM ...)"
            # 如果它不是简单底表列，就视为生成列
            logical_name = key
            expr_sql = key
            items.append((logical_name, expr_sql))

        return base_table, items

    # ============================================================
    #  新增：对每个生成列单独 profiling
    # ============================================================
    def gather_groupby_stats(self, state, limit: int = 256):
        """
        对 GROUP BY 中的“生成列”（非底表原始列）逐个做 DISTINCT profiling。

        返回格式：
        {
            "<logical_name_1>": {
                "profiling_sql": "...",
                "summary": {...},
                "raw_rows": [...],
            },
            "<logical_name_2>": {
                ...
            },
            ...
        }

        例如，对于：

        SELECT
          EXTRACT(YEAR FROM LO_ORDERDATE) AS year,
          P_BRAND,
          SUM(LO_REVENUE) AS sum_revenue
        FROM lineorder_flat
        WHERE ...
        GROUP BY year, P_BRAND;

        只会对 "year" 这一列做 profiling，生成类似：

        {
            "year": {
                "profiling_sql": "SELECT DISTINCT EXTRACT(YEAR FROM LO_ORDERDATE) AS year
                                  FROM lineorder_flat
                                  LIMIT 256;",
                "summary": {
                    "year": {
                        "distinct_values": ["1993", "1994"],
                        "approx_distinct_count": 2
                    }
                },
                "raw_rows": [
                    {"year": "1993"},
                    {"year": "1994"}
                ]
            }
        }
        """
        original_sql = state["query_example"]
        base_table, items = self._extract_generated_groupby_items(original_sql)
        # 没有任何生成列需要 profiling
        if not base_table or not items:
            return {}

        results = {}

        for logical_name, expr_sql in items:
            # 每个生成列单独一个 DISTINCT 查询
            profiling_sql = f"""
                SELECT
                    {logical_name},
                    MIN({logical_name}) OVER () AS min_val,
                    MAX({logical_name}) OVER () AS max_val
                FROM (
                    SELECT DISTINCT {expr_sql} AS {logical_name}
                    FROM {base_table}
                ) t
                ORDER BY {logical_name}
                LIMIT {int(limit)};
            """

            ok, payload = self.duckdb_connector.execute_query(profiling_sql,output_format="csv")
            if not ok:
                results[logical_name] = {
                    "profiling_sql": profiling_sql,
                    "summary": {},
                    "raw_rows": [],
                    "error": payload,
                }
                continue

            reader = csv.DictReader(StringIO(payload))
            rows = list(reader)

            distinct_values = sorted(
                {row.get(logical_name) for row in rows if row.get(logical_name) is not None}
            )

            # 2) 全局 min/max（窗口函数保证每行一样，随便拿一行即可）
            first = rows[0]
            min_val = first.get("min_val")
            max_val = first.get("max_val")

            summary = {
                logical_name: {
                    "distinct_values": distinct_values,
                    "approx_distinct_count": len(distinct_values),
                    "min": min_val,
                    "max": max_val,
                }
            }

            results[logical_name] = {
                "profiling_sql": profiling_sql,
                "summary": summary,
                "raw_rows": rows,
            }

        return results

    def check_constant_vector_columns(self, state, batch_size: int = 1024):
        """
        检查split_query输出列的每一列在指定batch大小下是否可能是constant vector。
        
        Args:
            state: 包含split_query的状态字典
            batch_size: batch大小，默认1024
        
        Returns:
            dict: {column_name: bool} - True表示该列可能是constant vector，False表示不是
        """
        # 直接使用split_query作为要检查的查询
        split_query = state.get("decomposed", {}).get("split_query", "").strip()
        inner_query = extract_inner_sql(split_query)
        
        # 获取inner_query的输出列
        try:
            temp_state = {"query_example": inner_query}
            output_schema = self.get_output_datatype(temp_state)
            column_names = list(output_schema.keys())
        except Exception as e:
            return {}
        
        if not column_names:
            return {}
        
        # 构建检查查询：直接执行inner_query，对每列检查前batch_size行的值是否都相同
        # 使用聚合函数：如果MIN(col) = MAX(col)，说明所有值相同，即constant vector
        check_query = f"""
            WITH sample AS (
                {inner_query}
                LIMIT {batch_size}
            )
            SELECT
                {', '.join([
                    f"MIN({col}) = MAX({col}) AS {col}_is_constant"
                    for col in column_names
                ])}
            FROM sample
        """
        
        ok, payload = self.duckdb_connector.execute_query(check_query, output_format="csv")
        if not ok:
            # 如果查询失败，返回所有列为False
            return {col: False for col in column_names}
        
        try:
            reader = csv.DictReader(StringIO(payload))
            row = next(reader, None)
            if not row:
                return {col: False for col in column_names}
            
            # 解析结果
            result = {}
            for col in column_names:
                key = f"{col}_is_constant"
                # DuckDB可能返回'true'/'false'字符串或1/0
                value = row.get(key, "false")
                if isinstance(value, str):
                    is_constant = value.lower() in ("true", "1", "t")
                else:
                    is_constant = bool(value)
                result[col] = is_constant
            
            return result
        except Exception:
            # 解析失败，返回所有列为False
            return {col: False for col in column_names}
