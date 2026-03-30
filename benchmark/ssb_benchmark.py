"""
SSB (Star Schema Benchmark) 查询模板生成器
==========================================

本模块提供 SSBQueryTemplates 类，封装了 SSB 全部 13 条查询的参数化模板与实例化逻辑。

对外接口
--------
1. SSBQueryTemplates(strict=True)
   构造函数。strict=True 时参数生成严格遵循 SSB 原始论文约束（如 Q7 客户/供应商同区域），
   strict=False 则放宽约束以扩大查询多样性。

2. get_template(tid: int) -> str
   返回编号 tid (1-13) 对应的参数化 SQL 模板，占位符格式为 :param_name。
   用途：查看模板结构、自行拼装参数、或传给 get_query 的 split_template 参数。

3. get_query(tid: int, seed: int, split_template: str = None) -> str
   生成一条可直接执行的 SQL 查询。
   - seed == -1 : 返回论文原始的 canonical 查询（固定参数）。
   - seed >= 0  : 基于 (tid, seed) 确定性随机生成参数并实例化查询，
                   相同 (tid, seed) 在任意机器/运行中返回完全相同的 SQL。
   - split_template: 若提供一个自定义模板字符串（含 :param 占位符），
                     则用生成的参数填充该模板，而非内置模板。

4. quantity: int  (属性)
   模板总数，值为 13。

参数域常量（可按需修改以匹配实际 SSB 数据集）
----------------------------------------------
YEARS, REGIONS, NATIONS, CITIES_US, BRANDS, CATEGORIES, MFGRS

查询分组
--------
Q1-Q3   : 收入聚合（按年/月/周 + 折扣/数量过滤）
Q4-Q6   : 按品牌/品类的收入分析（供应商区域过滤）
Q7-Q10  : 客户-供应商维度的收入分析（区域/国家/城市过滤）
Q11-Q13 : 利润分析（区域 + 制造商/品类过滤）
"""

import random
from typing import Dict, List


class SSBQueryTemplates:
    """
    - get_template(tid): return the parameterized SQL template (with :params)
    - get_query(tid, seed): deterministically instantiate an SQL query instance

    Determinism:
      Same (tid, seed) => identical query string across runs/machines,
      assuming the same Python 3.x version and unchanged parameter domains.
    """

    # -----------------------------
    # Parameter domains (edit to match your SSB dictionaries if needed)
    # -----------------------------
    YEARS: List[int] = [1992, 1993, 1994, 1995, 1996, 1997, 1998]
    REGIONS: List[str] = ["AMERICA", "EUROPE", "ASIA"]
    NATIONS: List[str] = ["UNITED STATES", "CHINA", "JAPAN", "GERMANY", "FRANCE"]
    CITIES_US: List[str] = ["UNITED KI1", "UNITED KI5", "UNITED KI2", "UNITED KI3"]

    # Dataset-dependent spaces; keep consistent with your generated data.
    BRANDS: List[str] = [f"MFGR#{i:04d}" for i in range(2000, 3000)]   # MFGR#2000..MFGR#2999
    CATEGORIES: List[str] = [f"MFGR#{i:02d}" for i in range(10, 50)]   # MFGR#10..MFGR#49
    MFGRS: List[str] = ["MFGR#1", "MFGR#2", "MFGR#3", "MFGR#4"]

    def __init__(self, strict: bool = True) -> None:
        self.quantity = 13
        self.strict = strict
        self._canonical_params: Dict[int, Dict[str, any]] = {
    1: {"year": 1993, "discount_lo": 1, "discount_hi": 3, "qty_lt": 25},
    2: {"yyyymm": 199401, "discount_lo": 4, "discount_hi": 6, "qty_lo": 26, "qty_hi": 35},
    3: {"week": 6, "year": 1994, "discount_lo": 5, "discount_hi": 7, "qty_lo": 26, "qty_hi": 35},
    4: {"p_category": "MFGR#12", "s_region": "AMERICA"},
    5: {"brand_lo": "MFGR#2221", "brand_hi": "MFGR#2228", "s_region": "ASIA"},
    6: {"brand": "MFGR#2239", "s_region": "EUROPE"},
    7: {"c_region": "ASIA", "s_region": "ASIA", "year_lo": 1992, "year_hi": 1997},
    8: {"c_nation": "UNITED STATES", "s_nation": "UNITED STATES", "year_lo": 1992, "year_hi": 1997},
    9: {"c_city_1": "UNITED KI1", "c_city_2": "UNITED KI5",
        "s_city_1": "UNITED KI1", "s_city_2": "UNITED KI5",
        "year_lo": 1992, "year_hi": 1997},
    10: {"c_city_1": "UNITED KI1", "c_city_2": "UNITED KI5",
         "s_city_1": "UNITED KI1", "s_city_2": "UNITED KI5",
         "yyyymm": 199712},
    11: {"c_region": "AMERICA", "s_region": "AMERICA", "mfgr_1": "MFGR#1", "mfgr_2": "MFGR#2"},
    12: {"c_region": "AMERICA", "s_region": "AMERICA", "year_1": 1997, "year_2": 1998,
         "mfgr_1": "MFGR#1", "mfgr_2": "MFGR#2"},
    13: {"s_nation": "UNITED STATES", "year_1": 1997, "year_2": 1998, "p_category": "MFGR#14"},
}
        self._canonical = {
    1: """
SELECT SUM(LO_EXTENDEDPRICE * LO_DISCOUNT) AS revenue
FROM lineorder_flat
WHERE EXTRACT(YEAR FROM LO_ORDERDATE) = 1993
  AND LO_DISCOUNT BETWEEN 1 AND 3
  AND LO_QUANTITY < 25
""",
    2: """
SELECT SUM(LO_EXTENDEDPRICE * LO_DISCOUNT) AS revenue
FROM lineorder_flat
WHERE CAST(strftime('%Y%m', LO_ORDERDATE) AS INTEGER) = 199401
  AND LO_DISCOUNT BETWEEN 4 AND 6
  AND LO_QUANTITY BETWEEN 26 AND 35
""",
    3: """
SELECT SUM(LO_EXTENDEDPRICE * LO_DISCOUNT) AS revenue
FROM lineorder_flat
WHERE CAST(strftime('%V', LO_ORDERDATE) AS INTEGER) = 6
  AND EXTRACT(YEAR FROM LO_ORDERDATE) = 1994
  AND LO_DISCOUNT BETWEEN 5 AND 7
  AND LO_QUANTITY BETWEEN 26 AND 35
""",
    4: """
SELECT
  EXTRACT(YEAR FROM LO_ORDERDATE) AS year,
  P_BRAND,
  SUM(LO_REVENUE) AS sum_revenue
FROM lineorder_flat
WHERE P_CATEGORY = 'MFGR#12'
  AND S_REGION  = 'AMERICA'
GROUP BY year, P_BRAND
ORDER BY year, P_BRAND
""",
    5: """
SELECT
  EXTRACT(YEAR FROM LO_ORDERDATE) AS year,
  P_BRAND,
  SUM(LO_REVENUE) AS sum_revenue
FROM lineorder_flat
WHERE P_BRAND >= 'MFGR#2221' AND P_BRAND <= 'MFGR#2228'
  AND S_REGION = 'ASIA'
GROUP BY year, P_BRAND
ORDER BY year, P_BRAND
""",
    6: """
SELECT
  SUM(LO_REVENUE) AS sum_revenue,
  EXTRACT(YEAR FROM LO_ORDERDATE) AS year,
  P_BRAND
FROM lineorder_flat
WHERE P_BRAND = 'MFGR#2239' AND S_REGION = 'EUROPE'
GROUP BY year, P_BRAND
ORDER BY year, P_BRAND
""",
    7: """
SELECT
  C_NATION,
  S_NATION,
  EXTRACT(YEAR FROM LO_ORDERDATE) AS year,
  SUM(LO_REVENUE) AS revenue
FROM lineorder_flat
WHERE
  C_REGION = 'ASIA'
  AND S_REGION = 'ASIA'
  AND EXTRACT(YEAR FROM LO_ORDERDATE) BETWEEN 1992 AND 1997
GROUP BY
  C_NATION,
  S_NATION,
  EXTRACT(YEAR FROM LO_ORDERDATE)
ORDER BY
  year ASC,
  revenue DESC
""",
    8: """
SELECT
  C_CITY,
  S_CITY,
  EXTRACT(YEAR FROM LO_ORDERDATE) AS year,
  SUM(LO_REVENUE) AS revenue
FROM lineorder_flat
WHERE
  C_NATION = 'UNITED STATES'
  AND S_NATION = 'UNITED STATES'
  AND EXTRACT(YEAR FROM LO_ORDERDATE) BETWEEN 1992 AND 1997
GROUP BY
  C_CITY,
  S_CITY,
  EXTRACT(YEAR FROM LO_ORDERDATE)
ORDER BY
  year ASC,
  revenue DESC
""",
    9: """
SELECT
  C_CITY,
  S_CITY,
  EXTRACT(YEAR FROM LO_ORDERDATE) AS year,
  SUM(LO_REVENUE) AS revenue
FROM lineorder_flat
WHERE
  (C_CITY = 'UNITED KI1' OR C_CITY = 'UNITED KI5')
  AND (S_CITY = 'UNITED KI1' OR S_CITY = 'UNITED KI5')
  AND EXTRACT(YEAR FROM LO_ORDERDATE) BETWEEN 1992 AND 1997
GROUP BY
  C_CITY,
  S_CITY,
  EXTRACT(YEAR FROM LO_ORDERDATE)
ORDER BY
  year ASC,
  revenue DESC
""",
    10: """
SELECT
  C_CITY,
  S_CITY,
  EXTRACT(YEAR FROM LO_ORDERDATE) AS year,
  SUM(LO_REVENUE) AS revenue
FROM lineorder_flat
WHERE
  (C_CITY = 'UNITED KI1' OR C_CITY = 'UNITED KI5')
  AND (S_CITY = 'UNITED KI1' OR S_CITY = 'UNITED KI5')
  AND CAST(strftime('%Y%m', LO_ORDERDATE) AS INTEGER) = 199712
GROUP BY
  C_CITY,
  S_CITY,
  EXTRACT(YEAR FROM LO_ORDERDATE)
ORDER BY
  year ASC,
  revenue DESC
""",
    11: """
SELECT
  EXTRACT(YEAR FROM LO_ORDERDATE) AS year,
  C_NATION,
  SUM(LO_REVENUE - LO_SUPPLYCOST) AS profit
FROM lineorder_flat
WHERE
  C_REGION = 'AMERICA'
  AND S_REGION = 'AMERICA'
  AND (P_MFGR = 'MFGR#1' OR P_MFGR = 'MFGR#2')
GROUP BY
  EXTRACT(YEAR FROM LO_ORDERDATE),
  C_NATION
ORDER BY
  year ASC,
  C_NATION ASC
""",
    12: """
SELECT
  EXTRACT(YEAR FROM LO_ORDERDATE) AS year,
  S_NATION,
  P_CATEGORY,
  SUM(LO_REVENUE - LO_SUPPLYCOST) AS profit
FROM lineorder_flat
WHERE
  C_REGION = 'AMERICA'
  AND S_REGION = 'AMERICA'
  AND (EXTRACT(YEAR FROM LO_ORDERDATE) = 1997 OR EXTRACT(YEAR FROM LO_ORDERDATE) = 1998)
  AND (P_MFGR = 'MFGR#1' OR P_MFGR = 'MFGR#2')
GROUP BY
  EXTRACT(YEAR FROM LO_ORDERDATE),
  S_NATION,
  P_CATEGORY
ORDER BY
  year ASC,
  S_NATION ASC,
  P_CATEGORY ASC
""",
    13: """
SELECT
  EXTRACT(YEAR FROM LO_ORDERDATE) AS year,
  S_CITY,
  P_BRAND,
  SUM(LO_REVENUE - LO_SUPPLYCOST) AS profit
FROM lineorder_flat
WHERE
  S_NATION = 'UNITED STATES'
  AND (EXTRACT(YEAR FROM LO_ORDERDATE) = 1997 OR EXTRACT(YEAR FROM LO_ORDERDATE) = 1998)
  AND P_CATEGORY = 'MFGR#14'
GROUP BY
  EXTRACT(YEAR FROM LO_ORDERDATE),
  S_CITY,
  P_BRAND
ORDER BY
  year ASC,
  S_CITY ASC,
  P_BRAND ASC
"""
}

        self._templates: Dict[int, str] = {
            1: """
SELECT SUM(LO_EXTENDEDPRICE * LO_DISCOUNT) AS revenue
FROM lineorder_flat
WHERE EXTRACT(YEAR FROM LO_ORDERDATE) = :year
  AND LO_DISCOUNT BETWEEN :discount_lo AND :discount_hi
  AND LO_QUANTITY < :qty_lt
""",
            2: """
SELECT SUM(LO_EXTENDEDPRICE * LO_DISCOUNT) AS revenue
FROM lineorder_flat
WHERE CAST(strftime('%Y%m', LO_ORDERDATE) AS INTEGER) = :yyyymm
  AND LO_DISCOUNT BETWEEN :discount_lo AND :discount_hi
  AND LO_QUANTITY BETWEEN :qty_lo AND :qty_hi
""",
            3: """
SELECT SUM(LO_EXTENDEDPRICE * LO_DISCOUNT) AS revenue
FROM lineorder_flat
WHERE CAST(strftime('%V', LO_ORDERDATE) AS INTEGER) = :week
  AND EXTRACT(YEAR FROM LO_ORDERDATE) = :year
  AND LO_DISCOUNT BETWEEN :discount_lo AND :discount_hi
  AND LO_QUANTITY BETWEEN :qty_lo AND :qty_hi
""",
            4: """
SELECT
  EXTRACT(YEAR FROM LO_ORDERDATE) AS year,
  P_BRAND,
  SUM(LO_REVENUE) AS sum_revenue
FROM lineorder_flat
WHERE P_CATEGORY = :p_category
  AND S_REGION  = :s_region
GROUP BY year, P_BRAND
ORDER BY year, P_BRAND
""",
            5: """
SELECT
  EXTRACT(YEAR FROM LO_ORDERDATE) AS year,
  P_BRAND,
  SUM(LO_REVENUE) AS sum_revenue
FROM lineorder_flat
WHERE P_BRAND BETWEEN :brand_lo AND :brand_hi
  AND S_REGION = :s_region
GROUP BY year, P_BRAND
ORDER BY year, P_BRAND
""",
            6: """
SELECT
  SUM(LO_REVENUE) AS sum_revenue,
  EXTRACT(YEAR FROM LO_ORDERDATE) AS year,
  P_BRAND
FROM lineorder_flat
WHERE P_BRAND = :brand
  AND S_REGION = :s_region
GROUP BY year, P_BRAND
ORDER BY year, P_BRAND
""",
            7: """
SELECT
  C_NATION,
  S_NATION,
  EXTRACT(YEAR FROM LO_ORDERDATE) AS year,
  SUM(LO_REVENUE) AS revenue
FROM lineorder_flat
WHERE C_REGION = :c_region
  AND S_REGION = :s_region
  AND EXTRACT(YEAR FROM LO_ORDERDATE) BETWEEN :year_lo AND :year_hi
GROUP BY
  C_NATION,
  S_NATION,
  EXTRACT(YEAR FROM LO_ORDERDATE)
ORDER BY
  year ASC,
  revenue DESC
""",
            8: """
SELECT
  C_CITY,
  S_CITY,
  EXTRACT(YEAR FROM LO_ORDERDATE) AS year,
  SUM(LO_REVENUE) AS revenue
FROM lineorder_flat
WHERE C_NATION = :c_nation
  AND S_NATION = :s_nation
  AND EXTRACT(YEAR FROM LO_ORDERDATE) BETWEEN :year_lo AND :year_hi
GROUP BY
  C_CITY,
  S_CITY,
  EXTRACT(YEAR FROM LO_ORDERDATE)
ORDER BY
  year ASC,
  revenue DESC
""",
            9: """
SELECT
  C_CITY,
  S_CITY,
  EXTRACT(YEAR FROM LO_ORDERDATE) AS year,
  SUM(LO_REVENUE) AS revenue
FROM lineorder_flat
WHERE C_CITY IN (:c_city_1, :c_city_2)
  AND S_CITY IN (:s_city_1, :s_city_2)
  AND EXTRACT(YEAR FROM LO_ORDERDATE) BETWEEN :year_lo AND :year_hi
GROUP BY
  C_CITY,
  S_CITY,
  EXTRACT(YEAR FROM LO_ORDERDATE)
ORDER BY
  year ASC,
  revenue DESC
""",
            10: """
SELECT
  C_CITY,
  S_CITY,
  EXTRACT(YEAR FROM LO_ORDERDATE) AS year,
  SUM(LO_REVENUE) AS revenue
FROM lineorder_flat
WHERE C_CITY IN (:c_city_1, :c_city_2)
  AND S_CITY IN (:s_city_1, :s_city_2)
  AND CAST(strftime('%Y%m', LO_ORDERDATE) AS INTEGER) = :yyyymm
GROUP BY
  C_CITY,
  S_CITY,
  EXTRACT(YEAR FROM LO_ORDERDATE)
ORDER BY
  year ASC,
  revenue DESC
""",
            11: """
SELECT
  EXTRACT(YEAR FROM LO_ORDERDATE) AS year,
  C_NATION,
  SUM(LO_REVENUE - LO_SUPPLYCOST) AS profit
FROM lineorder_flat
WHERE C_REGION = :c_region
  AND S_REGION = :s_region
  AND P_MFGR IN (:mfgr_1, :mfgr_2)
GROUP BY
  EXTRACT(YEAR FROM LO_ORDERDATE),
  C_NATION
ORDER BY
  year ASC,
  C_NATION ASC
""",
            12: """
SELECT
  EXTRACT(YEAR FROM LO_ORDERDATE) AS year,
  S_NATION,
  P_CATEGORY,
  SUM(LO_REVENUE - LO_SUPPLYCOST) AS profit
FROM lineorder_flat
WHERE C_REGION = :c_region
  AND S_REGION = :s_region
  AND EXTRACT(YEAR FROM LO_ORDERDATE) IN (:year_1, :year_2)
  AND P_MFGR IN (:mfgr_1, :mfgr_2)
GROUP BY
  EXTRACT(YEAR FROM LO_ORDERDATE),
  S_NATION,
  P_CATEGORY
ORDER BY
  year ASC,
  S_NATION ASC,
  P_CATEGORY ASC
""",
            13: """
SELECT
  EXTRACT(YEAR FROM LO_ORDERDATE) AS year,
  S_CITY,
  P_BRAND,
  SUM(LO_REVENUE - LO_SUPPLYCOST) AS profit
FROM lineorder_flat
WHERE S_NATION = :s_nation
  AND EXTRACT(YEAR FROM LO_ORDERDATE) IN (:year_1, :year_2)
  AND P_CATEGORY = :p_category
GROUP BY
  EXTRACT(YEAR FROM LO_ORDERDATE),
  S_CITY,
  P_BRAND
ORDER BY
  year ASC,
  S_CITY ASC,
  P_BRAND ASC
""",
        }

    # -----------------------------
    # Public APIs
    # -----------------------------
    def get_template(self, tid: int) -> str:
        if tid not in self._templates:
            raise ValueError(f"Unknown template id={tid}. Must be in 1..13.")
        return self._normalize(self._templates[tid])

    def get_query(self, tid: int, seed: int, split_template: str = None) -> str:
        if tid not in self._templates:
            raise ValueError(f"Unknown template id={tid}")

        # Canonical mode (paper-aligned)
        if seed == -1:
            if split_template is not None:
                params = self._canonical_params.get(tid)
                if params is None:
                    raise ValueError(f"Missing canonical params for template id={tid}")
                return self._normalize(self._replace_params_in_template(split_template, params))
            return self._normalize(self._canonical[tid])

        # Random instance mode
        return getattr(self, f"_inst_{tid}")(seed, split_template=split_template)


    # -----------------------------
    # Internal helpers
    # -----------------------------
    @staticmethod
    def _normalize(sql: str) -> str:
        return "\n".join(line.rstrip() for line in sql.strip().splitlines()) + "\n"

    @staticmethod
    def _rng(seed: int, tid: int) -> random.Random:
        # Independent deterministic stream per template
        return random.Random(seed * 1315423911 + tid * 2654435761)

    @staticmethod
    def _pick(rng: random.Random, xs: List):
        return xs[rng.randrange(len(xs))]

    @staticmethod
    def _randint(rng: random.Random, lo: int, hi: int) -> int:
        return rng.randint(lo, hi)

    @classmethod
    def _pick2_distinct(cls, rng: random.Random, xs: List[str]) -> (str, str):
        a = cls._pick(rng, xs)
        b = cls._pick(rng, xs)
        while b == a:
            b = cls._pick(rng, xs)
        return a, b

    @staticmethod
    def _replace_params_in_template(template: str, params: Dict[str, any]) -> str:
        """
        在split_template中替换参数占位符
        
        Args:
            template: 包含参数占位符的split_template（如:year, :discount_lo等）
            params: 参数字典，键为参数名（不含冒号），值为实际值
        
        Returns:
            替换后的SQL查询
        """
        result = template
        for param_name, param_value in params.items():
            # 替换 :param_name 格式的占位符
            # 使用单词边界确保只替换完整的参数名
            import re
            pattern = r':\b' + re.escape(param_name) + r'\b'
            if isinstance(param_value, str):
                # 字符串值需要加引号
                replacement = f"'{param_value}'"
            else:
                replacement = str(param_value)
            result = re.sub(pattern, replacement, result)
        return result

    # -----------------------------
    # Per-template instantiators
    # -----------------------------
    def _inst_1(self, seed: int, split_template: str = None) -> str:
        rng = self._rng(seed, 1)
        year = self._pick(rng, self.YEARS)
        d1 = self._randint(rng, 1, 7)
        d2 = self._randint(rng, d1, min(10, d1 + 3))
        qty_lt = self._randint(rng, 10, 50)
        
        if split_template:
            params = {
                "year": year,
                "discount_lo": d1,
                "discount_hi": d2,
                "qty_lt": qty_lt
            }
            return self._normalize(self._replace_params_in_template(split_template, params))
        
        sql = f"""
SELECT SUM(LO_EXTENDEDPRICE * LO_DISCOUNT) AS revenue
FROM lineorder_flat
WHERE EXTRACT(YEAR FROM LO_ORDERDATE) = {year}
  AND LO_DISCOUNT BETWEEN {d1} AND {d2}
  AND LO_QUANTITY < {qty_lt}
"""
        return self._normalize(sql)

    def _inst_2(self, seed: int, split_template: str = None) -> str:
        rng = self._rng(seed, 2)
        year = self._pick(rng, self.YEARS)
        month = self._randint(rng, 1, 12)
        yyyymm = year * 100 + month
        d1 = self._randint(rng, 1, 7)
        d2 = self._randint(rng, d1, min(10, d1 + 3))
        q_lo = self._randint(rng, 1, 30)
        q_hi = self._randint(rng, q_lo + 1, min(99, q_lo + 20))
        
        if split_template:
            params = {
                "yyyymm": yyyymm,
                "discount_lo": d1,
                "discount_hi": d2,
                "qty_lo": q_lo,
                "qty_hi": q_hi
            }
            return self._normalize(self._replace_params_in_template(split_template, params))
        
        sql = f"""
SELECT SUM(LO_EXTENDEDPRICE * LO_DISCOUNT) AS revenue
FROM lineorder_flat
WHERE CAST(strftime('%Y%m', LO_ORDERDATE) AS INTEGER) = {yyyymm}
  AND LO_DISCOUNT BETWEEN {d1} AND {d2}
  AND LO_QUANTITY BETWEEN {q_lo} AND {q_hi}
"""
        return self._normalize(sql)

    def _inst_3(self, seed: int, split_template: str = None) -> str:
        rng = self._rng(seed, 3)
        year = self._pick(rng, self.YEARS)
        week = self._randint(rng, 1, 52)
        d1 = self._randint(rng, 1, 7)
        d2 = self._randint(rng, d1, min(10, d1 + 3))
        q_lo = self._randint(rng, 1, 30)
        q_hi = self._randint(rng, q_lo + 1, min(99, q_lo + 20))

        if split_template:
            params = {
                "week": week,
                "year": year,
                "discount_lo": d1,
                "discount_hi": d2,
                "qty_lo": q_lo,
                "qty_hi": q_hi
            }
            return self._normalize(self._replace_params_in_template(split_template, params))
        
        sql = f"""
SELECT SUM(LO_EXTENDEDPRICE * LO_DISCOUNT) AS revenue
FROM lineorder_flat
WHERE CAST(strftime('%V', LO_ORDERDATE) AS INTEGER) = {week}
  AND EXTRACT(YEAR FROM LO_ORDERDATE) = {year}
  AND LO_DISCOUNT BETWEEN {d1} AND {d2}
  AND LO_QUANTITY BETWEEN {q_lo} AND {q_hi}
"""
        return self._normalize(sql)

    def _inst_4(self, seed: int, split_template: str = None) -> str:
        rng = self._rng(seed, 4)
        p_category = self._pick(rng, self.CATEGORIES)
        s_region = self._pick(rng, self.REGIONS)

        if split_template:
            params = {
                "p_category": p_category,
                "s_region": s_region
            }
            return self._normalize(self._replace_params_in_template(split_template, params))
        
        sql = f"""
SELECT
  EXTRACT(YEAR FROM LO_ORDERDATE) AS year,
  P_BRAND,
  SUM(LO_REVENUE) AS sum_revenue
FROM lineorder_flat
WHERE P_CATEGORY = '{p_category}'
  AND S_REGION  = '{s_region}'
GROUP BY year, P_BRAND
ORDER BY year, P_BRAND
"""
        return self._normalize(sql)

    def _inst_5(self, seed: int, split_template: str = None) -> str:
        rng = self._rng(seed, 5)
        b1 = self._pick(rng, self.BRANDS)
        b2 = self._pick(rng, self.BRANDS)
        if b1 > b2:
            b1, b2 = b2, b1
        s_region = self._pick(rng, self.REGIONS)
        
        if split_template:
            params = {
                "brand_lo": b1,
                "brand_hi": b2,
                "s_region": s_region
            }
            return self._normalize(self._replace_params_in_template(split_template, params))
        
        sql = f"""
SELECT
  EXTRACT(YEAR FROM LO_ORDERDATE) AS year,
  P_BRAND,
  SUM(LO_REVENUE) AS sum_revenue
FROM lineorder_flat
WHERE P_BRAND BETWEEN '{b1}' AND '{b2}'
  AND S_REGION = '{s_region}'
GROUP BY year, P_BRAND
ORDER BY year, P_BRAND
"""
        return self._normalize(sql)

    def _inst_6(self, seed: int, split_template: str = None) -> str:
        rng = self._rng(seed, 6)

        choice = [
    ("MFGR#1431", "AFRICA"),
    ("MFGR#5416", "EUROPE"),
    ("MFGR#1237", "ASIA"),
    ("MFGR#3239", "AFRICA"),
    ("MFGR#337", "MIDDLE EAST"),
    ("MFGR#225", "EUROPE"),
    ("MFGR#337", "AMERICA"),
    ("MFGR#5439", "AMERICA"),
    ("MFGR#5135", "AMERICA"),
    ("MFGR#3220", "ASIA"),
    ("MFGR#1234", "ASIA"),
    ("MFGR#3431", "AFRICA"),
    ("MFGR#2321", "AMERICA"),
    ("MFGR#253", "AFRICA"),
    ("MFGR#3324", "ASIA"),
    ("MFGR#235", "AFRICA"),
    ("MFGR#1232", "EUROPE"),
    ("MFGR#135", "AMERICA"),
    ("MFGR#249", "ASIA"),
    ("MFGR#5114", "AMERICA"),
    # ... 你中间省略的部分 ...
    ("MFGR#5536", "EUROPE"),
    ("MFGR#2416", "EUROPE"),
    ("MFGR#5138", "MIDDLE EAST"),
    ("MFGR#3216", "MIDDLE EAST"),
    ("MFGR#2318", "AFRICA"),
    ("MFGR#112", "ASIA"),
    ("MFGR#1312", "MIDDLE EAST"),
    ("MFGR#3322", "EUROPE"),
    ("MFGR#4431", "MIDDLE EAST"),
    ("MFGR#1440", "ASIA"),
    ("MFGR#155", "EUROPE"),
    ("MFGR#2140", "AMERICA"),
    ("MFGR#5333", "EUROPE"),
    ("MFGR#1522", "ASIA"),
    ("MFGR#5525", "AMERICA"),
    ("MFGR#4127", "AFRICA"),
    ("MFGR#1239", "ASIA"),
    ("MFGR#543", "AFRICA"),
    ("MFGR#5427", "ASIA"),
    ("MFGR#2512", "EUROPE"),
      ]

        brand, s_region = choice[rng.randrange(len(choice))]

        if split_template:
            params = {
                "brand": brand,
                "s_region": s_region
            }
            return self._normalize(self._replace_params_in_template(split_template, params))
        
        sql = f"""
SELECT
  SUM(LO_REVENUE) AS sum_revenue,
  EXTRACT(YEAR FROM LO_ORDERDATE) AS year,
  P_BRAND
FROM lineorder_flat
WHERE P_BRAND = '{brand}'
  AND S_REGION = '{s_region}'
GROUP BY year, P_BRAND
ORDER BY year, P_BRAND
"""
        return self._normalize(sql)

    def _inst_7(self, seed: int, split_template: str = None) -> str:
        rng = self._rng(seed, 7)
        # strict: same region on customer & supplier sides
        region = self._pick(rng, self.REGIONS)
        if self.strict:
            c_region, s_region = region, region
            y_lo, y_hi = 1992, 1997
        else:
            c_region = self._pick(rng, self.REGIONS)
            s_region = self._pick(rng, self.REGIONS)
            y_lo = self._randint(rng, 1992, 1996)
            y_hi = self._randint(rng, y_lo, 1998)

        if split_template:
            params = {
                "c_region": c_region,
                "s_region": s_region,
                "year_lo": y_lo,
                "year_hi": y_hi
            }
            return self._normalize(self._replace_params_in_template(split_template, params))
        
        sql = f"""
    SELECT
    C_NATION,
    S_NATION,
    EXTRACT(YEAR FROM LO_ORDERDATE) AS year,
    SUM(LO_REVENUE) AS revenue
    FROM lineorder_flat
    WHERE C_REGION = '{c_region}'
    AND S_REGION = '{s_region}'
    AND EXTRACT(YEAR FROM LO_ORDERDATE) BETWEEN {y_lo} AND {y_hi}
    GROUP BY
    C_NATION,
    S_NATION,
    EXTRACT(YEAR FROM LO_ORDERDATE)
    ORDER BY
    year ASC,
    revenue DESC
    """
        return self._normalize(sql)

    def _inst_8(self, seed: int, split_template: str = None) -> str:
      
        rng = self._rng(seed, 8)
        if self.strict:
            nation = self._pick(rng, self.NATIONS)
            c_nation, s_nation = nation, nation
            y_lo, y_hi = 1992, 1997
        else:
            c_nation = self._pick(rng, self.NATIONS)
            s_nation = self._pick(rng, self.NATIONS)
            y_lo = self._randint(rng, 1992, 1996)
            y_hi = self._randint(rng, y_lo, 1998)

        if split_template:
            params = {
                "c_nation": c_nation,
                "s_nation": s_nation,
                "year_lo": y_lo,
                "year_hi": y_hi
            }
            return self._normalize(self._replace_params_in_template(split_template, params))
        
        sql = f"""
    SELECT
    C_CITY,
    S_CITY,
    EXTRACT(YEAR FROM LO_ORDERDATE) AS year,
    SUM(LO_REVENUE) AS revenue
    FROM lineorder_flat
    WHERE C_NATION = '{c_nation}'
    AND S_NATION = '{s_nation}'
    AND EXTRACT(YEAR FROM LO_ORDERDATE) BETWEEN {y_lo} AND {y_hi}
    GROUP BY
    C_CITY,
    S_CITY,
    EXTRACT(YEAR FROM LO_ORDERDATE)
    ORDER BY
    year ASC,
    revenue DESC
    """
        return self._normalize(sql)

    def _inst_9(self, seed: int, split_template: str = None) -> str:
        rng = self._rng(seed, 9)
        if self.strict:
            # same city set on both sides
            c1, c2 = self._pick2_distinct(rng, self.CITIES_US)
            s1, s2 = c1, c2
            y_lo, y_hi = 1992, 1997
        else:
            c1, c2 = self._pick2_distinct(rng, self.CITIES_US)
            s1, s2 = self._pick2_distinct(rng, self.CITIES_US)
            y_lo = self._randint(rng, 1992, 1996)
            y_hi = self._randint(rng, y_lo, 1998)

        if split_template:
            params = {
                "c_city_1": c1,
                "c_city_2": c2,
                "s_city_1": s1,
                "s_city_2": s2,
                "year_lo": y_lo,
                "year_hi": y_hi
            }
            return self._normalize(self._replace_params_in_template(split_template, params))
        
        sql = f"""
    SELECT
    C_CITY,
    S_CITY,
    EXTRACT(YEAR FROM LO_ORDERDATE) AS year,
    SUM(LO_REVENUE) AS revenue
    FROM lineorder_flat
    WHERE C_CITY IN ('{c1}', '{c2}')
    AND S_CITY IN ('{s1}', '{s2}')
    AND EXTRACT(YEAR FROM LO_ORDERDATE) BETWEEN {y_lo} AND {y_hi}
    GROUP BY
    C_CITY,
    S_CITY,
    EXTRACT(YEAR FROM LO_ORDERDATE)
    ORDER BY
    year ASC,
    revenue DESC
    """
        return self._normalize(sql)

    def _inst_10(self, seed: int, split_template: str = None) -> str:
        rng = self._rng(seed, 10)
        if self.strict:
            c1, c2 = self._pick2_distinct(rng, self.CITIES_US)
            s1, s2 = c1, c2
            # original query fixed to 199712; keep that in strict mode
            yyyymm = 199712
        else:
            c1, c2 = self._pick2_distinct(rng, self.CITIES_US)
            s1, s2 = self._pick2_distinct(rng, self.CITIES_US)
            year = self._pick(rng, self.YEARS)
            month = self._randint(rng, 1, 12)
            yyyymm = year * 100 + month

        if split_template:
            params = {
                "c_city_1": c1,
                "c_city_2": c2,
                "s_city_1": s1,
                "s_city_2": s2,
                "yyyymm": yyyymm
            }
            return self._normalize(self._replace_params_in_template(split_template, params))
        
        sql = f"""
    SELECT
    C_CITY,
    S_CITY,
    EXTRACT(YEAR FROM LO_ORDERDATE) AS year,
    SUM(LO_REVENUE) AS revenue
    FROM lineorder_flat
    WHERE C_CITY IN ('{c1}', '{c2}')
    AND S_CITY IN ('{s1}', '{s2}')
    AND CAST(strftime('%Y%m', LO_ORDERDATE) AS INTEGER) = {yyyymm}
    GROUP BY
    C_CITY,
    S_CITY,
    EXTRACT(YEAR FROM LO_ORDERDATE)
    ORDER BY
    year ASC,
    revenue DESC
    """
        return self._normalize(sql)

    def _inst_11(self, seed: int, split_template: str = None) -> str:
        rng = self._rng(seed, 11)
        if self.strict:
            region = "AMERICA"
            c_region, s_region = region, region
            m1, m2 = "MFGR#1", "MFGR#2"
        else:
            region = self._pick(rng, self.REGIONS)
            c_region = region
            s_region = self._pick(rng, self.REGIONS)
            m1, m2 = self._pick2_distinct(rng, self.MFGRS)

        if split_template:
            params = {
                "c_region": c_region,
                "s_region": s_region,
                "mfgr_1": m1,
                "mfgr_2": m2
            }
            return self._normalize(self._replace_params_in_template(split_template, params))
        
        sql = f"""
    SELECT
    EXTRACT(YEAR FROM LO_ORDERDATE) AS year,
    C_NATION,
    SUM(LO_REVENUE - LO_SUPPLYCOST) AS profit
    FROM lineorder_flat
    WHERE C_REGION = '{c_region}'
    AND S_REGION = '{s_region}'
    AND P_MFGR IN ('{m1}', '{m2}')
    GROUP BY
    EXTRACT(YEAR FROM LO_ORDERDATE),
    C_NATION
    ORDER BY
    year ASC,
    C_NATION ASC
    """
        return self._normalize(sql)

    def _inst_12(self, seed: int, split_template: str = None) -> str:
        rng = self._rng(seed, 12)
        if self.strict:
            c_region, s_region = "AMERICA", "AMERICA"
            y1, y2 = 1997, 1998
            m1, m2 = "MFGR#1", "MFGR#2"
        else:
            region = self._pick(rng, self.REGIONS)
            c_region = region
            s_region = self._pick(rng, self.REGIONS)
            y1, y2 = self._pick2_distinct(rng, self.YEARS)
            m1, m2 = self._pick2_distinct(rng, self.MFGRS)

        if split_template:
            params = {
                "c_region": c_region,
                "s_region": s_region,
                "year_1": y1,
                "year_2": y2,
                "mfgr_1": m1,
                "mfgr_2": m2
            }
            return self._normalize(self._replace_params_in_template(split_template, params))
        
        sql = f"""
    SELECT
    EXTRACT(YEAR FROM LO_ORDERDATE) AS year,
    S_NATION,
    P_CATEGORY,
    SUM(LO_REVENUE - LO_SUPPLYCOST) AS profit
    FROM lineorder_flat
    WHERE C_REGION = '{c_region}'
    AND S_REGION = '{s_region}'
    AND EXTRACT(YEAR FROM LO_ORDERDATE) IN ({y1}, {y2})
    AND P_MFGR IN ('{m1}', '{m2}')
    GROUP BY
    EXTRACT(YEAR FROM LO_ORDERDATE),
    S_NATION,
    P_CATEGORY
    ORDER BY
    year ASC,
    S_NATION ASC,
    P_CATEGORY ASC
    """
        return self._normalize(sql)

    def _inst_13(self, seed: int, split_template: str = None) -> str:
        rng = self._rng(seed, 13)
        if self.strict:
            s_nation = "UNITED STATES"
            y1, y2 = 1997, 1998
            p_category = "MFGR#14"
        else:
            s_nation = self._pick(rng, self.NATIONS)
            y1, y2 = self._pick2_distinct(rng, self.YEARS)
            p_category = self._pick(rng, self.CATEGORIES)

        if split_template:
            params = {
                "s_nation": s_nation,
                "year_1": y1,
                "year_2": y2,
                "p_category": p_category
            }
            return self._normalize(self._replace_params_in_template(split_template, params))
        
        sql = f"""
    SELECT
    EXTRACT(YEAR FROM LO_ORDERDATE) AS year,
    S_CITY,
    P_BRAND,
    SUM(LO_REVENUE - LO_SUPPLYCOST) AS profit
    FROM lineorder_flat
    WHERE S_NATION = '{s_nation}'
    AND EXTRACT(YEAR FROM LO_ORDERDATE) IN ({y1}, {y2})
    AND P_CATEGORY = '{p_category}'
    GROUP BY
    EXTRACT(YEAR FROM LO_ORDERDATE),
    S_CITY,
    P_BRAND
    ORDER BY
    year ASC,
    S_CITY ASC,
    P_BRAND ASC
    """
        return self._normalize(sql)

   

# -----------------------------
# Example usage + determinism check
# -----------------------------
if __name__ == "__main__":
    qt = SSBQueryTemplates()

    # Template fetch
    print("T9 template:\n", qt.get_template(9))

    # Deterministic instantiation check
    q1 = qt.get_query(9, seed=-1)
    q2 = qt.get_query(9, seed=-1)
    assert q1 == q2, "Non-deterministic output!"
    print("Determinism OK for (tid=9, seed=42).")

    # Print an instance
    print("T9 instance:\n", q1)
