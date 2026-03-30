"""
ClickHouse benchmark queries loader.
Reads SQL queries directly from queries/queries.sql file.
"""

import os
from typing import Dict, List
from config import SKETCH_DIR, BENCHMARK


class ClickBenchmarkQueries:
    """
    - get_template(tid): return the SQL query template (same as query since no parameterization)
    - get_query(tid, seed): return the SQL query instance
      - seed == -1: return original query
      - seed >= 0: return same query (no parameterization)
    """

    def __init__(self, queries_file: str = None) -> None:
        """
        Initialize with queries file path.
        
        Args:
            queries_file: Path to queries.sql file. Defaults to queries/queries.sql
        """
        if queries_file is None:
            # Get the directory of this file and construct path to queries.sql
            current_dir = os.path.dirname(os.path.abspath(__file__))
            queries_file = os.path.join(os.path.dirname(current_dir), "benchmark", "hits_queries.sql")
        
        self.queries_file = queries_file
        self._queries: Dict[int, str] = {}
        self._load_queries()

    def _load_queries(self) -> None:
        """Load all queries from the SQL file."""
        if not os.path.exists(self.queries_file):
            raise FileNotFoundError(f"Queries file not found: {self.queries_file}")
        
        with open(self.queries_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        query_id = 1
        for line in lines:
            line = line.strip()
            if line:  # Skip empty lines
                self._queries[query_id] = self._normalize(line)
                query_id += 1

    @staticmethod
    def _normalize(sql: str) -> str:
        """Normalize SQL query format."""
        return sql.strip() + "\n"

    def get_template(self, tid: int) -> str:
        """
        Get the SQL template for the given template ID.
        
        Args:
            tid: Template ID (1-indexed)
            
        Returns:
            SQL query template (same as query since no parameterization)
        """
        if tid not in self._queries:
            max_id = max(self._queries.keys()) if self._queries else 0
            raise ValueError(f"Unknown template id={tid}. Must be in 1..{max_id}.")
        return self._queries[tid]

    def get_query(self, tid: int, seed: int) -> str:
        """
        Get the SQL query instance.
        
        Args:
            tid: Template ID (1-indexed)
            seed: Seed value
                - seed == -1: return original query
                - seed >= 0: return same query (no parameterization)
                
        Returns:
            SQL query string
        """
        if tid not in self._queries:
            max_id = max(self._queries.keys()) if self._queries else 0
            raise ValueError(f"Unknown template id={tid}. Must be in 1..{max_id}.")
        
        # Since queries are not parameterized, return the same query regardless of seed
        return self._queries[tid]
    
    def get_split_query(self, query_id) -> str:
        src_path = f"{SKETCH_DIR}/{BENCHMARK}_q_{query_id}.cpp"
        with open(src_path, "r") as f:
            cpp_code = f.read()
            # 解析出 split_query 和 split_template（如果 cpp_code 以类似 /* ... */ 段落开始）
            split_query = ""
            split_template = ""
            import re
            header_match = re.match(
                r"/\*\s*query_template:.*?split_template:\s*(.*?)\s*;\s*query_example:.*?split_query:\s*(.*?)\s*;\s*\*/",
                cpp_code,
                re.DOTALL,
            )
            if header_match:
                split_template = header_match.group(1).strip()
                split_query = header_match.group(2).strip()
        return split_query, split_template

    @property
    def quantity(self) -> int:
        """Get the number of available queries."""
        return len(self._queries)


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    cb = ClickBenchmarkQueries()

    # Template fetch
    print("Query 1 template:\n", cb.get_template(1))

    # Query fetch
    q1 = cb.get_query(14, seed=-1)
    q2 = cb.get_query(14, seed=42)
    assert q1 == q2, "Queries should be identical"
    print("Query 1 instance:\n", q1)
    
    print(f"Total queries available: {cb.quantity}")

