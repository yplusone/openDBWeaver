import re

def parse_query_from_file(src_path: str) -> tuple[str, str]:
        with open(src_path, "r") as f:
            cpp_code = f.read()
            header_match = re.match(
                r"/\*\s*query_template:\s*(.*?)\s*;\s*split_template:\s*(.*?)\s*;\s*query_example:\s*(.*?)\s*;\s*split_query:\s*(.*?)\s*;\s*\*/",
                cpp_code,
                re.DOTALL,
            )
            if header_match:
                query_template = header_match.group(1).strip()
                split_template = header_match.group(2).strip()
                query_example = header_match.group(3).strip()
                split_query = header_match.group(4).strip()

        return query_example, query_template, split_query, split_template

def extract_inner_sql(split_query: str) -> str:
    """
    从 split_query 中提取内部 SQL（去掉外层包装）。
    例如：SELECT * FROM dbweaver((SELECT ... FROM ... WHERE ...)) 
    返回：SELECT ... FROM ... WHERE ...
    """
    if not split_query:
        return ""
    
    # 匹配 dbweaver((SELECT ...)) 或类似的模式
    # 找到第一个 (( 和对应的 ))
    start_idx = split_query.find("((")
    if start_idx != -1:
        # 找到对应的 ))
        depth = 0
        for i in range(start_idx + 2, len(split_query)):
            if split_query[i] == '(':
                depth += 1
            elif split_query[i] == ')':
                if depth == 0:
                    return split_query[start_idx + 2:i].strip()
                depth -= 1
    return split_query