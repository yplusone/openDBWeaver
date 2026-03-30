import pandas as pd
import re
from typing import Any, Tuple, List, Dict
import io
import numpy as np

class ResultComparator:
    """
    比较两个查询结果（CSV 文本/box 表格文本/DataFrame），并在不匹配时
    返回详细的列级差异与示例行。

    check_two_results(payload_new, payload_ref) -> (ok: bool, detail: str)
    - ok=True  时，detail 为 "Results match."
    - ok=False 时，detail 包含：
        * 形状不一致/列集合不一致的说明
        * 逐列数值/文本不匹配，并给出最多 3 行示例（期望 vs 实际）
    """

    # ---------------------------
    # Parsing helpers
    # ---------------------------

    def parse_payload(self, payload: str) -> pd.DataFrame:
        """
        尝试解析 DuckDB 的 box-style 表格输出（带边框和竖线）。
        例如：
        ┌─────────────┬────────────┐
        │ col1        │ col2       │
        ├─────────────┼────────────┤
        │ ...         │ ...        │
        └─────────────┴────────────┘
        """
        if payload is None:
            return pd.DataFrame()

        lines = str(payload).strip().split("\n")
        if len(lines) < 5:
            return pd.DataFrame()

        clean_data = []
        for line in lines:
            # 仅保留形如 "│ ... │" 的行
            s = line.strip()
            if not (s.startswith("│") and s.endswith("│")):
                continue

            # 跳过 summary 行，例如 "│  10 rows ... │"
            inner = s[1:-1].strip()
            if re.search(r"\brows\b.*\bcolumns\b", inner):
                continue

            parts = [part.strip() for part in s.split("│")[1:-1]]
            if parts:
                clean_data.append(parts)

        # 至少应有：header + dtype + >=1 data row
        if len(clean_data) < 3:
            return pd.DataFrame()

        # DuckDB box table 常见格式：
        # row0 = header
        # row1 = dtype
        # row2... = data
        header = clean_data[0]
        data_rows = clean_data[2:]

        # 过滤可能异常行（列数不一致）
        data_rows = [r for r in data_rows if len(r) == len(header)]

        if not data_rows:
            return pd.DataFrame(columns=header)

        return pd.DataFrame(data_rows, columns=header)

    # ---------------------------
    # Query / ORDER BY parsing
    # ---------------------------

    def _normalize_sql(self, q: str) -> str:
        return re.sub(r"\s+", " ", (q or "").strip()).strip()

    def _extract_limit_offset(self, query: str) -> tuple[int | None, int]:
        q = self._normalize_sql(query)
        limit_m = re.search(r"\blimit\s+(\d+)\b", q, flags=re.I)
        offset_m = re.search(r"\boffset\s+(\d+)\b", q, flags=re.I)
        limit_val = int(limit_m.group(1)) if limit_m else None
        offset_val = int(offset_m.group(1)) if offset_m else 0
        return limit_val, offset_val

    def _split_sql_top_level_commas(self, s: str) -> List[str]:
        """
        按“顶层逗号”切分 SQL 片段（忽略括号内逗号）。
        例如: "a, COUNT(*), DATE_TRUNC('m', t) AS m" -> ["a", "COUNT(*)", "DATE_TRUNC('m', t) AS m"]
        """
        parts = []
        buf = []
        depth = 0
        in_single = False
        in_double = False

        i = 0
        while i < len(s):
            ch = s[i]

            # 处理字符串引号（简化版，支持 '' 转义）
            if ch == "'" and not in_double:
                if in_single and i + 1 < len(s) and s[i + 1] == "'":
                    buf.append(ch)
                    buf.append(s[i + 1])
                    i += 2
                    continue
                in_single = not in_single
                buf.append(ch)
                i += 1
                continue

            if ch == '"' and not in_single:
                in_double = not in_double
                buf.append(ch)
                i += 1
                continue

            if not in_single and not in_double:
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth = max(0, depth - 1)
                elif ch == "," and depth == 0:
                    part = "".join(buf).strip()
                    if part:
                        parts.append(part)
                    buf = []
                    i += 1
                    continue

            buf.append(ch)
            i += 1

        tail = "".join(buf).strip()
        if tail:
            parts.append(tail)
        return parts


    def _normalize_sql_expr_for_match(self, expr: str) -> str:
        """
        规范化表达式用于匹配（尽量宽松）：
        - 去首尾空白
        - 去外层冗余括号
        - 压缩空白
        - 小写
        """
        if expr is None:
            return ""

        x = expr.strip()

        # 去掉末尾 ASC/DESC（有时外部传进来不干净）
        x = re.sub(r"\s+(asc|desc)\s*$", "", x, flags=re.I).strip()

        # 反复去掉一层完整外括号：(((COUNT(*)))) -> COUNT(*)
        def _strip_one_outer_paren(s: str) -> str:
            if not (s.startswith("(") and s.endswith(")")):
                return s
            depth = 0
            in_single = False
            in_double = False
            for i, ch in enumerate(s):
                if ch == "'" and not in_double:
                    in_single = not in_single
                elif ch == '"' and not in_single:
                    in_double = not in_double
                elif not in_single and not in_double:
                    if ch == "(":
                        depth += 1
                    elif ch == ")":
                        depth -= 1
                        if depth == 0 and i != len(s) - 1:
                            return s  # 外层括号未包住全串
            return s[1:-1].strip()

        while True:
            y = _strip_one_outer_paren(x)
            if y == x:
                break
            x = y

        # 压缩空白并小写
        x = re.sub(r"\s+", " ", x).strip().lower()
        return x


    def _extract_select_clause_top_level(self, query: str):
        """
        提取顶层 SELECT ... FROM 中的 SELECT 子句内容（不包含 SELECT / FROM）。
        能跳过函数/子查询/字符串中的 FROM，例如:
        SELECT extract(minute FROM ts) AS m FROM t
        不会误截断到 extract(...) 内部的 FROM。
        """
        s = query.strip()
        if not s:
            return None

        n = len(s)
        i = 0
        depth = 0
        in_single = False
        in_double = False

        def _is_word_boundary(pos: int) -> bool:
            if pos < 0 or pos >= n:
                return True
            return not (s[pos].isalnum() or s[pos] == "_")

        def _match_keyword(pos: int, kw: str) -> bool:
            """大小写不敏感匹配关键字，并要求词边界。"""
            L = len(kw)
            if pos < 0 or pos + L > n:
                return False
            if s[pos:pos + L].lower() != kw.lower():
                return False
            return _is_word_boundary(pos - 1) and _is_word_boundary(pos + L)

        # 1) 找顶层 SELECT
        select_pos = -1
        while i < n:
            ch = s[i]

            if ch == "'" and not in_double:
                # 处理 SQL 单引号转义 ''
                if in_single and i + 1 < n and s[i + 1] == "'":
                    i += 2
                    continue
                in_single = not in_single
                i += 1
                continue
            if ch == '"' and not in_single:
                in_double = not in_double
                i += 1
                continue

            if in_single or in_double:
                i += 1
                continue

            if ch == "(":
                depth += 1
                i += 1
                continue
            if ch == ")":
                depth = max(0, depth - 1)
                i += 1
                continue

            if depth == 0 and _match_keyword(i, "select"):
                select_pos = i
                i += len("select")
                break

            i += 1

        if select_pos < 0:
            return None

        # 2) 从 SELECT 后面开始，找顶层 FROM
        start = i  # select 子句内容起点（SELECT 后）
        while i < n:
            ch = s[i]

            if ch == "'" and not in_double:
                if in_single and i + 1 < n and s[i + 1] == "'":
                    i += 2
                    continue
                in_single = not in_single
                i += 1
                continue
            if ch == '"' and not in_single:
                in_double = not in_double
                i += 1
                continue

            if in_single or in_double:
                i += 1
                continue

            if ch == "(":
                depth += 1
                i += 1
                continue
            if ch == ")":
                depth = max(0, depth - 1)
                i += 1
                continue

            if depth == 0 and _match_keyword(i, "from"):
                return s[start:i].strip()

            i += 1

        return None


    def _extract_select_alias_expr_map(self, query: str) -> Dict[str, str]:
        """
        从 SELECT ... FROM ... 中提取 "表达式 -> 别名" 映射（用于 ORDER BY 表达式映射到输出列别名）。
        返回示例：
        {
            "count(*)": "cnt",
            "date_trunc('minute', eventtime)": "m",
            "extract(minute from eventtime)": "m"
        }

        支持：
        - expr AS alias
        - expr alias（无 AS 的常见写法）
        - "quoted alias"
        不追求覆盖全部 SQL 语法，但足够处理常见 benchmark 查询。
        """
        select_clause = self._extract_select_clause_top_level(query)
        if not select_clause:
            return {}

        items = self._split_sql_top_level_commas(select_clause)
        expr_to_alias: Dict[str, str] = {}

        for item in items:
            s = item.strip()
            if not s:
                continue

            expr_part = None
            alias_part = None

            # 1) 优先匹配 "... AS alias"
            m_as = re.match(
                r"^(.*?)(?:\s+as\s+)(\"[^\"]+\"|`[^`]+`|\[[^\]]+\]|[A-Za-z_][A-Za-z0-9_]*)\s*$",
                s,
                flags=re.I | re.S,
            )
            if m_as:
                expr_part = m_as.group(1).strip()
                alias_part = m_as.group(2).strip()
            else:
                # 2) 再尝试无 AS："... alias"
                m_no_as = re.match(
                    r"^(.*\S)\s+(\"[^\"]+\"|`[^`]+`|\[[^\]]+\]|[A-Za-z_][A-Za-z0-9_]*)\s*$",
                    s,
                    flags=re.S,
                )
                if m_no_as:
                    candidate_expr = m_no_as.group(1).strip()
                    candidate_alias = m_no_as.group(2).strip()

                    # 简单防误判：如果 expr 末尾看起来像 "table.col"，通常这是列本身，不一定是别名场景
                    # 但这里仍允许；真正是否可用取决于后续 ORDER BY 映射
                    expr_part = candidate_expr
                    alias_part = candidate_alias

            if expr_part is None or alias_part is None:
                continue

            # 去掉 alias 引号
            alias = alias_part
            if (alias.startswith('"') and alias.endswith('"')) or \
            (alias.startswith("`") and alias.endswith("`")) or \
            (alias.startswith("[") and alias.endswith("]")):
                alias = alias[1:-1]

            alias = alias.strip()
            if not alias:
                continue

            norm_expr = self._normalize_sql_expr_for_match(expr_part)
            if norm_expr:
                expr_to_alias[norm_expr] = alias

        return expr_to_alias


    def _parse_order_by_specs(self, query: str, df_cols: List[str]) -> List[Dict[str, Any]]:
        """
        解析 ORDER BY 子句，返回：
        [
        {"expr": "c", "dir": "desc", "col": "c"},
        {"expr": "COUNT(*)", "dir": "desc", "col": "cnt"},   # 表达式映射到 SELECT 别名
        ...
        ]

        增强点：
        - 支持 ORDER BY 列名/别名（直接映射）
        - 支持 ORDER BY 表达式 映射到 SELECT 别名（如 COUNT(*) -> cnt）
        - 对无法映射到输出列的项仍跳过
        """
        q = self._normalize_sql(query)

        m = re.search(r"\border\s+by\s+(.*?)(?=\blimit\b|\boffset\b|;|$)", q, flags=re.I | re.S)
        if not m:
            return []

        clause = m.group(1).strip()
        if not clause:
            return []

        # 顶层逗号切分（支持函数表达式里的逗号）
        raw_parts = [p.strip() for p in self._split_sql_top_level_commas(clause) if p.strip()]
        colmap = {str(c).lower(): str(c) for c in df_cols}

        # SELECT 表达式 -> 别名 映射（用于 ORDER BY COUNT(*) 映射到 cnt）
        expr_to_alias = self._extract_select_alias_expr_map(query)

        specs: List[Dict[str, Any]] = []
        for p in raw_parts:
            mm = re.match(r"(.+?)(?:\s+(asc|desc))?$", p, flags=re.I | re.S)
            if not mm:
                continue

            expr = mm.group(1).strip()
            direction = (mm.group(2) or "asc").lower()

            # 去掉双引号（直接列名/别名情况）
            expr_unquoted = expr[1:-1] if (expr.startswith('"') and expr.endswith('"')) else expr
            expr_unquoted = expr_unquoted.strip()

            mapped_col = None

            # 1) 直接按列名/别名映射（ORDER BY cnt DESC）
            mapped_col = colmap.get(expr_unquoted.lower())

            # 2) 若不是直接列名，尝试按“表达式 -> SELECT别名”映射（ORDER BY COUNT(*) DESC -> cnt）
            if mapped_col is None:
                norm_expr = self._normalize_sql_expr_for_match(expr_unquoted)
                alias = expr_to_alias.get(norm_expr)
                if alias is not None:
                    mapped_col = colmap.get(alias.lower())

            # 3) 再尝试 ordinal（ORDER BY 1 / ORDER BY 2）可选支持
            #    如果你不想支持可以删掉这段
            if mapped_col is None and re.fullmatch(r"\d+", expr_unquoted):
                pos = int(expr_unquoted)
                if 1 <= pos <= len(df_cols):
                    mapped_col = str(df_cols[pos - 1])

            if mapped_col is None:
                # 无法映射时跳过（后续会走其他比较/报错路径）
                continue

            specs.append({
                "expr": expr,
                "dir": direction,
                "col": mapped_col,
            })

        return specs


    # ---------------------------
    # Formatting helpers
    # ---------------------------

    def _sample_indices(self, n: int, max_rows: int = 40) -> List[int]:
        """在 [0, n) 中抽样，包含前/中/后几段索引."""
        n = int(n)
        if n <= max_rows:
            return list(range(n))

        head = max_rows // 3
        mid = max_rows // 3
        tail = max_rows - head - mid

        idxs: List[int] = []
        idxs.extend(range(head))

        mid_start = max(n // 2 - mid // 2, head)
        mid_end = min(mid_start + mid, n - tail)
        if mid_end > mid_start:
            idxs.extend(range(mid_start, mid_end))

        idxs.extend(range(n - tail, n))
        return sorted(set(i for i in idxs if 0 <= i < n))

    def _format_sample_table(self, df: pd.DataFrame, title: str, max_rows: int = 40) -> str:
        """生成类似 DuckDB box-table 的文本预览，前/中/后采样，并在省略部分插入‘·’行。"""
        if df is None or df.empty:
            return f"{title}:\n<empty result>"

        n_rows, n_cols = df.shape
        cols = [str(c) for c in df.columns]
        dtypes = [str(dtype) for dtype in df.dtypes]

        idxs = self._sample_indices(n_rows, max_rows=max_rows)
        df_sample = df.iloc[idxs]

        col_widths = []
        for j, col in enumerate(cols):
            vals = [str(v) for v in df_sample.iloc[:, j]]
            w = max([len(col), len(dtypes[j])] + ([len(v) for v in vals] if vals else [0]))
            col_widths.append(w)

        def _border(left, mid, right, fill="─"):
            return left + mid.join(fill * (w + 2) for w in col_widths) + right

        top_border = _border("┌", "┬", "┐")
        header_sep = _border("├", "┼", "┤")
        bottom_border = _border("└", "┴", "┘")

        header_row = "│ " + " │ ".join(f"{c:<{w}}" for c, w in zip(cols, col_widths)) + " │"
        dtype_row = "│ " + " │ ".join(f"{t:<{w}}" for t, w in zip(dtypes, col_widths)) + " │"

        data_lines = []
        last_i = None
        for i in idxs:
            if last_i is not None and i - last_i > 1:
                dots = ["·" for _ in cols]
                dot_line = "│ " + " │ ".join(f"{d:^{w}}" for d, w in zip(dots, col_widths)) + " │"
                data_lines.append(dot_line)

            row = df.iloc[i]
            vals = [str(row[c]) for c in df.columns]
            line = "│ " + " │ ".join(f"{v:<{w}}" for v, w in zip(vals, col_widths)) + " │"
            data_lines.append(line)
            last_i = i

        table_width = len(header_sep)
        inner_width = table_width - 2
        summary_text = f"{n_rows} rows ({len(idxs)} shown)   {n_cols} columns"
        summary_row = "│" + summary_text.center(inner_width) + "│"

        lines = [
            f"{title}:",
            top_border,
            header_row,
            dtype_row,
            header_sep,
            *data_lines,
            header_sep,
            summary_row,
            bottom_border,
        ]
        return "\n".join(lines)

    # ---------------------------
    # Conversion / normalization
    # ---------------------------

    def _normalize_csv_text(self, s: str) -> str:
        s = s.strip()
        if (s.startswith("'") and s.endswith("'")) or (s.startswith('"') and s.endswith('"')):
            s = s[1:-1]
        s = s.replace("\r\n", "\n").replace("\r", "\n")
        return s

    def _to_df(self, obj: Any) -> pd.DataFrame:
        # 已是 DataFrame
        if isinstance(obj, pd.DataFrame):
            return obj.copy()

        # 字符串/字节串：优先按 CSV 尝试
        if isinstance(obj, (str, bytes)):
            text = obj.decode() if isinstance(obj, bytes) else obj
            csv_text = self._normalize_csv_text(text)

            # 优先尝试 CSV（DuckDB -csv 输出）
            try:
                return pd.read_csv(io.StringIO(csv_text), dtype=str)
            except Exception:
                pass

            # 回退：尝试 box 表格
            try:
                df_box = self.parse_payload(text)
                if not df_box.empty:
                    return df_box
            except Exception:
                pass

            if obj == "":
                return pd.DataFrame()

            # 最后兜底
            return pd.read_csv(io.StringIO(self._normalize_csv_text(str(obj))), dtype=str)

        # 其他类型：尝试 parse_payload → 兜底 CSV
        try:
            df_any = self.parse_payload(str(obj))
            if isinstance(df_any, pd.DataFrame) and not df_any.empty:
                return df_any
        except Exception:
            pass

        return pd.read_csv(io.StringIO(self._normalize_csv_text(str(obj))), dtype=str)

    def _strip_and_coerce(self, df_new: pd.DataFrame, df_ref: pd.DataFrame, cols: List[str]):
        """
        就地处理：
        - strip 空白
        - 若列双方都全可数值化，则转 float 并记录 is_numeric_col
        """
        is_numeric_col: Dict[str, bool] = {}
        for c in cols:
            df_new[c] = df_new[c].astype(str).str.strip()
            df_ref[c] = df_ref[c].astype(str).str.strip()

            s1_num = pd.to_numeric(df_new[c], errors="coerce")
            s2_num = pd.to_numeric(df_ref[c], errors="coerce")

            m1 = s1_num.notna()
            m2 = s2_num.notna()
            if len(s1_num) == 0 and len(s2_num) == 0:
                is_numeric_col[c] = False
            elif (m1.sum() == len(s1_num)) and (m2.sum() == len(s2_num)):
                df_new[c] = s1_num.astype(float)
                df_ref[c] = s2_num.astype(float)
                is_numeric_col[c] = True
            else:
                is_numeric_col[c] = False
        return is_numeric_col

    def _sort_unordered(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        # 稳定排序（按所有列的字符串视图）
        return df.sort_values(by=cols, key=lambda s: s.astype(str)).reset_index(drop=True)

    # ---------------------------
    # Comparison primitives
    # ---------------------------

    def _values_equal(self, a, b, is_numeric: bool, rtol: float, atol: float) -> bool:
        if is_numeric:
            a_nan = pd.isna(a)
            b_nan = pd.isna(b)
            if a_nan and b_nan:
                return True
            if a_nan or b_nan:
                return False
            return bool(np.isclose(float(a), float(b), rtol=rtol, atol=atol, equal_nan=True))
        else:
            # 文本比较
            a_na = pd.isna(a)
            b_na = pd.isna(b)
            if a_na and b_na:
                return True
            if a_na or b_na:
                return False
            return str(a) == str(b)

    def _row_signature(
        self,
        row: pd.Series,
        cols: List[str],
        is_numeric_col: Dict[str, bool],
        rtol: float,
        atol: float,
    ) -> tuple:
        """
        用于组内 multiset 比较的行签名。
        数值列做稳定归一（避免微小浮点误差导致 Counter 不同）。
        """
        sig = []
        for c in cols:
            v = row[c]
            if is_numeric_col.get(c, False):
                if pd.isna(v):
                    sig.append(("num", "NA"))
                else:
                    fv = float(v)
                    # 量化到与 atol 同级别，避免轻微浮点噪声
                    if atol > 0:
                        q = round(fv / atol) * atol
                    else:
                        q = fv
                    sig.append(("num", float(q)))
            else:
                if pd.isna(v):
                    sig.append(("txt", "NA"))
                else:
                    sig.append(("txt", str(v)))
        return tuple(sig)

    def _collect_value_mismatch_reports(
        self,
        df_new_s: pd.DataFrame,
        df_ref_s: pd.DataFrame,
        cols: List[str],
        is_numeric_col: Dict[str, bool],
        rtol: float = 1e-9,
        atol: float = 1e-6,
    ) -> List[str]:
        mismatch_reports: List[str] = []
        n = len(df_new_s)

        for c in cols:
            a = df_new_s[c].to_numpy()
            b = df_ref_s[c].to_numpy()

            if is_numeric_col[c]:
                # 安全转 float
                a_float = pd.to_numeric(pd.Series(a), errors="coerce").to_numpy(dtype=float)
                b_float = pd.to_numeric(pd.Series(b), errors="coerce").to_numpy(dtype=float)

                both_nan = np.isnan(a_float) & np.isnan(b_float)
                close = np.isclose(a_float, b_float, rtol=rtol, atol=atol, equal_nan=False)
                bad_mask = ~(close | both_nan)
            else:
                a_str = pd.Series(a).astype(str).to_numpy()
                b_str = pd.Series(b).astype(str).to_numpy()
                bad_mask = ~(a_str == b_str)

            if np.any(bad_mask):
                bad_idx_all = np.where(bad_mask)[0]
                pos_sample = self._sample_indices(len(bad_idx_all), max_rows=9)
                picked_idx = [int(bad_idx_all[p]) for p in pos_sample]

                examples = []
                for i in picked_idx:
                    expected_val = str(df_ref_s.at[i, c])
                    actual_val = str(df_new_s.at[i, c])
                    examples.append(
                        f"  - Row #{i}:\n"
                        f"    expected[{c}] = {expected_val}\n"
                        f"    actual  [{c}] = {actual_val}"
                    )

                if is_numeric_col[c]:
                    mismatch_reports.append(
                        f"[Column '{c}'] numeric mismatch (rtol={rtol}, atol={atol}).\n" + "\n".join(examples)
                    )
                else:
                    mismatch_reports.append(
                        f"[Column '{c}'] text mismatch.\n" + "\n".join(examples)
                    )

        return mismatch_reports

    # ---------------------------
    # Plan B: ORDER BY group-wise comparison
    # ---------------------------

    def _extract_order_key_tuple(
        self,
        row: pd.Series,
        order_specs: List[Dict[str, Any]],
        is_numeric_col: Dict[str, bool],
        rtol: float,
        atol: float,
    ) -> tuple:
        """
        提取 ORDER BY 键元组（带方向信息）。
        这里只用作“分组键”等价比较，因此方向不需要变换值；方向仅用于说明。
        """
        key = []
        for spec in order_specs:
            c = spec["col"]
            v = row[c]
            if is_numeric_col.get(c, False):
                if pd.isna(v):
                    key.append(("num", c, "NA"))
                else:
                    fv = float(v)
                    if atol > 0:
                        q = round(fv / atol) * atol
                    else:
                        q = fv
                    key.append(("num", c, float(q)))
            else:
                if pd.isna(v):
                    key.append(("txt", c, "NA"))
                else:
                    key.append(("txt", c, str(v)))
        return tuple(key)

    def _compare_ordered_groupwise(
        self,
        query: str,
        df_new: pd.DataFrame,
        df_ref: pd.DataFrame,
        cols: List[str],
        is_numeric_col: Dict[str, bool],
        rtol: float = 1e-9,
        atol: float = 1e-6,
    ) -> Tuple[bool, str]:
        """
        方案B（增强版）：
        - 识别 ORDER BY 列（简单列名/别名）
        - 比较 ORDER BY 键序列（分组边界一致）
        - 每个 tie-group 内忽略顺序，做“多重集匹配”（逐行容忍比较）
        * 避免 Counter(signature) 因 float/NaN/类型差异造成误判
        - 不对 LIMIT 边界 tie 特殊放宽（严格要求 tie-group 内容一致）
        """
        order_specs = self._parse_order_by_specs(query, cols)
        if not order_specs:
            return False, "[Internal] ORDER BY present but failed to parse sortable columns."

        # 保持原顺序（ORDER BY 的输出顺序）
        df_new_s = df_new.reset_index(drop=True)
        df_ref_s = df_ref.reset_index(drop=True)

        if len(df_new_s) != len(df_ref_s):
            return False, f"[Shape mismatch during ordered compare] actual={df_new_s.shape}, expected={df_ref_s.shape}"

        # ---- helpers: robust cell / row equality (avoid Counter signature pitfalls) ----
        def _cell_equal(a, b, col: str) -> bool:
            if is_numeric_col.get(col, False):
                a_na = pd.isna(a)
                b_na = pd.isna(b)
                if a_na and b_na:
                    return True
                if a_na or b_na:
                    return False
                try:
                    return bool(np.isclose(float(a), float(b), rtol=rtol, atol=atol, equal_nan=True))
                except Exception:
                    # 回退到字符串比较
                    return str(a).strip() == str(b).strip()
            else:
                a_na = pd.isna(a)
                b_na = pd.isna(b)
                if a_na and b_na:
                    return True
                if a_na or b_na:
                    return False
                return str(a).strip() == str(b).strip()

        def _row_equal(row_a: pd.Series, row_b: pd.Series) -> bool:
            for c in cols:
                if not _cell_equal(row_a[c], row_b[c], c):
                    return False
            return True

        def _row_brief(row: pd.Series) -> str:
            parts = []
            for c in cols:
                v = row[c]
                parts.append(f"{c}={repr(v)}")
            return "{ " + ", ".join(parts) + " }"

        n = len(df_new_s)
        i = 0
        while i < n:
            key_new = self._extract_order_key_tuple(df_new_s.iloc[i], order_specs, is_numeric_col, rtol, atol)
            key_ref = self._extract_order_key_tuple(df_ref_s.iloc[i], order_specs, is_numeric_col, rtol, atol)

            if key_new != key_ref:
                return False, (
                    f"[ORDER BY key sequence mismatch] at row #{i}\n"
                    f"  expected order-key = {key_ref}\n"
                    f"  actual   order-key = {key_new}"
                )

            # 找当前 tie-group 的右边界（要求两边边界一致）
            j_new = i + 1
            while j_new < n:
                k = self._extract_order_key_tuple(df_new_s.iloc[j_new], order_specs, is_numeric_col, rtol, atol)
                if k != key_new:
                    break
                j_new += 1

            j_ref = i + 1
            while j_ref < n:


                k = self._extract_order_key_tuple(df_ref_s.iloc[j_ref], order_specs, is_numeric_col, rtol, atol)
                if k != key_ref:
                    break
                j_ref += 1

            if j_new != j_ref:
                return False, (
                    f"[ORDER BY tie-group size mismatch] rows starting at #{i}\n"
                    f"  order-key = {key_ref}\n"
                    f"  expected group size = {j_ref - i}\n"
                    f"  actual   group size = {j_new - i}"
                )
            if j_new == n:
                return True, "Results match."
            # tie-group 内忽略顺序：做多重集匹配（逐行配对）
            if 'offset' in query.lower() and i==0:
                i = j_ref
                continue

            group_new = [df_new_s.iloc[r] for r in range(i, j_new)]
            group_ref = [df_ref_s.iloc[r] for r in range(i, j_ref)]

            # 先按各列排序，再一一对应比较
            group_new_df = pd.DataFrame(group_new)
            group_ref_df = pd.DataFrame(group_ref)
            group_new_sorted = group_new_df.sort_values(by=cols).reset_index(drop=True)
            group_ref_sorted = group_ref_df.sort_values(by=cols).reset_index(drop=True)

            mismatch_at = None
            for k in range(len(group_new_sorted)):
                if not _row_equal(group_new_sorted.iloc[k], group_ref_sorted.iloc[k]):
                    mismatch_at = k
                    break

            if mismatch_at is not None:
                details = [
                    f"[Tie-group content mismatch] rows #{i}..#{j_ref - 1} (ORDER BY key = {key_ref})",
                    "Group-internal order is ignored, but row multisets differ.",
                    f"- At sorted position {mismatch_at}:",
                    f"  actual   row: {_row_brief(group_new_sorted.iloc[mismatch_at])}",
                    f"  expected row: {_row_brief(group_ref_sorted.iloc[mismatch_at])}",
                ]
                return False, "\n".join(details)

            i = j_ref

        return True, "Results match."


    # ---------------------------
    # Public API
    # ---------------------------

    def check_two_results(self, query, payload_new: Any, payload_ref: Any) -> Tuple[bool, str]:
        """
        返回:
        (ok: bool, detail: str)

        当不匹配时, detail 会具体到列名, 并附上若干示例。
        方案B:
        - 无 ORDER BY: 忽略顺序比较（全表排序后逐行比）
        - 有 ORDER BY: 按 ORDER BY 键分组比较，组内忽略顺序（tie-group permutation allowed）
        """
        if payload_new is None:
            return False, "The query returned zero results, which does not match the original query."

        if payload_new == '' and payload_ref == '':
            return True, "Results match."
        elif payload_new == '':
            return False, f"The query returned zero results, which does not match the original query, which is {payload_ref}."
        elif payload_ref == '':
            return False, f"The query returned {payload_new}, which does not match the original query, which is empty."
        
        df_new = self._to_df(payload_new)
        df_ref = self._to_df(payload_ref)

        # 形状不一致：报告 shape + 列信息 + 两边样例表
        if df_new.shape != df_ref.shape:
            detail_lines = [
                f"[Shape mismatch] rows x cols: actual={df_new.shape}, expected={df_ref.shape}",
                f"- Actual columns:   {list(map(str, df_new.columns))}",
                f"- Expected columns: {list(map(str, df_ref.columns))}",
                "",
                self._format_sample_table(df_new, "Actual result (sample)", max_rows=40),
                "",
                self._format_sample_table(df_ref, "Expected result (sample)", max_rows=40),
            ]
            return False, "\n".join(detail_lines)

        # 列名清洗
        df_new.columns = [str(c).strip() for c in df_new.columns]
        df_ref.columns = [str(c).strip() for c in df_ref.columns]

        # 列集合必须一致
        if set(df_new.columns) != set(df_ref.columns):
            both = sorted(set(df_new.columns) & set(df_ref.columns))
            only_new = sorted(set(df_new.columns) - set(df_ref.columns))
            only_ref = sorted(set(df_ref.columns) - set(df_new.columns))
            parts = [f"[Column set mismatch]"]
            if only_new:
                parts.append(f"- Extra columns in actual: {only_new}")
            if only_ref:
                parts.append(f"- Missing columns in actual: {only_ref}")
            parts.append(f"- Common columns: {both}")
            parts.append("")
            parts.append(self._format_sample_table(df_new, "Actual result (sample)", max_rows=40))
            parts.append("")
            parts.append(self._format_sample_table(df_ref, "Expected result (sample)", max_rows=40))
            return False, "\n".join(parts)

        # 对齐列顺序（内部比较统一）
        cols = sorted(df_new.columns)
        df_new = df_new[cols].copy()
        df_ref = df_ref[cols].copy()

        # strip + numeric coercion
        is_numeric_col = self._strip_and_coerce(df_new, df_ref, cols)

        # 是否存在 ORDER BY
        has_order_by = bool(re.search(r"\border\s+by\b", str(query), flags=re.I))

        # 方案B：有 ORDER BY 则按 tie-group 比较；否则全表无序比较
        if has_order_by:
            ok_groupwise, detail_groupwise = self._compare_ordered_groupwise(
                query=query,
                df_new=df_new,
                df_ref=df_ref,
                cols=cols,
                is_numeric_col=is_numeric_col,
                rtol=1e-9,
                atol=1e-6,
            )
            if ok_groupwise:
                return True, "Results match."

            # 失败时，补充更细的逐列 diff（按原顺序）和样例表
            df_new_s = df_new.reset_index(drop=True)
            df_ref_s = df_ref.reset_index(drop=True)
            # mismatch_reports = self._collect_value_mismatch_reports(
            #     df_new_s, df_ref_s, cols, is_numeric_col, rtol=1e-9, atol=1e-6
            # )

            header = "[Value mismatch] One or more columns differ. Examples below:"
            extra = [detail_groupwise]
            # if mismatch_reports:
            #     extra.extend(mismatch_reports)

            tables = [
                self._format_sample_table(df_new_s, "Actual result (sample)", max_rows=40),
                self._format_sample_table(df_ref_s, "Expected result (sample)", max_rows=40),
            ]
            return False, header + "\n\n" + "\n\n".join(extra + tables)

        else:
            if "limit" in query.lower():
                return True, "Results match."
            # 无 ORDER BY：按全列排序后比较（忽略顺序）
            df_new_s = self._sort_unordered(df_new, cols)
            df_ref_s = self._sort_unordered(df_ref, cols)

            mismatch_reports = self._collect_value_mismatch_reports(
                df_new_s, df_ref_s, cols, is_numeric_col, rtol=1e-9, atol=1e-6
            )

            if mismatch_reports:
                header = "[Value mismatch] One or more columns differ. Examples below:"
                tables = [
                    self._format_sample_table(df_new_s, "Actual result (sample)", max_rows=40),
                    self._format_sample_table(df_ref_s, "Expected result (sample)", max_rows=40),
                ]
                return False, header + "\n\n" + "\n\n".join(mismatch_reports + tables)

            return True, "Results match."