import re
import time
from langchain_openai import ChatOpenAI
from config import EXPT_LLM_MODEL,API_KEY,BASE_URL

def apply_replacements_from_list(original_code: str, replacements: list[dict]) -> str:
    """
    Apply replacements from a list of replacement dictionaries.
    If any replacement cannot be matched, use LLM to complete ALL replacements at once.
    """
    code = original_code
    def normalize_text(t):
        # 去除所有空白字符
        return "".join(t.split())

    def find_span(raw, pattern):
        """Find the span in raw code that matches the normalized pattern."""
        for i in range(len(raw)):
            raw_window = normalize_text(raw[i:])
            if raw_window.startswith(pattern):
                acc = ""
                for j in range(i, len(raw)):
                    acc = normalize_text(raw[i:j+1])
                    if acc == pattern:
                        return (i, j+1)
                    if len(acc) > len(pattern):
                        break
        return None

    # 先尝试匹配所有的 replacements
    failed_replacements = []
    code = original_code
    
    for rep in replacements:
        old_content = rep.get("old_content", "")
        new_content = rep.get("new_content", "")
        if not old_content:
            raise ValueError("old_content is empty")
        if old_content == new_content:
            continue
        # 忽略空白和换行符（空格/制表/换行等）进行不严格匹配替换
        norm_code = normalize_text(code)
        norm_old = normalize_text(old_content)

        match = re.search(re.escape(norm_old), norm_code)
        if match:
            # 找到开始和结束在原code中的位置
            span = find_span(code, norm_old)
            if span:
                start, end = span
                code = code[:start] +"\n" + new_content + "\n" + code[end:]
                continue
        
        # 如果无法匹配，记录这个 replacement
        failed_replacements.append(rep)
    
    # 如果有任何 replacement 匹配失败，使用 LLM 一次性完成所有 replacements
    if failed_replacements:
        print(f"Warning: {len(failed_replacements)} replacement(s) could not be matched, using LLM to apply all replacements")
        try:
            # 构建包含所有 replacements 的 prompt
            replacements_text = ""
            for i, rep in enumerate(replacements, 1):
                old_content = rep.get("old_content", "")
                new_content = rep.get("new_content", "")
                replacements_text += f"\nReplacement {i}:\n"
                replacements_text += f"- old_content:\n```cpp\n{old_content}\n```\n\n"
                replacements_text += f"- new_content:\n```cpp\n{new_content}\n```\n\n"
            
            prompt_text = f"""You are a code editor. Apply ALL the following replacements to the code.

Original code:
```cpp
{original_code}
```

Replacements to apply (apply them in order):
{replacements_text}

Please apply all these replacements to the code. The old_content might not exactly match due to whitespace differences, so find the semantically equivalent sections and replace them with the corresponding new_content.

Return ONLY the complete modified code in a code block, no explanations."""
            
            # 直接使用 ChatOpenAI
            llm = ChatOpenAI(model=EXPT_LLM_MODEL, api_key=API_KEY, base_url=BASE_URL)
            from langchain_core.messages import HumanMessage
            for _ in range(5):
                try:
                    response = llm.invoke([HumanMessage(content=prompt_text)]).content
                    break
                except Exception as e:
                    print(f"Error calling LLM: {e}")
                    time.sleep(5)

            # 提取代码块
            code_match = re.search(r'```(?:cpp|c\+\+|c)?\s*\n(.*?)```', response, re.DOTALL)
            if code_match:
                code = code_match.group(1).strip()
            else:
                # 如果没有代码块，尝试直接使用响应
                code = response.strip()
            print(f"Successfully applied all replacements using LLM")
        except Exception as e:
            print(f"Error using LLM to apply replacements: {e}")
            raise ValueError(f"Some replacements could not be matched and LLM fallback failed: {e}")
    return code
