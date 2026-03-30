import json
import re

def _fix_json_format_errors(json_str: str) -> str:
    """
    Attempt to fix common JSON format errors:
    - Single quotes to double quotes (for strings, not in string content)
    - Unquoted property names
    - Trailing commas
    - Comments
    
    This is a best-effort fix and may not handle all edge cases.
    """
    fixed = json_str
    
    # Remove single-line comments (// ...)
    fixed = re.sub(r'//.*?$', '', fixed, flags=re.MULTILINE)
    
    # Remove multi-line comments (/* ... */)
    fixed = re.sub(r'/\*.*?\*/', '', fixed, flags=re.DOTALL)
    
    # Fix unquoted property names (property: -> "property":)
    # Match word characters, underscores, and dots that appear before a colon
    # but are not already quoted
    def fix_unquoted_keys(match):
        key = match.group(1)
        # Skip if it's already a valid JSON value (number, true, false, null)
        if key in ['true', 'false', 'null'] or re.match(r'^-?\d+\.?\d*$', key):
            return match.group(0)
        # Add quotes around the key
        return f'"{key}":'
    
    # Pattern: word characters, underscores, dots before colon, not already quoted
    fixed = re.sub(r'([a-zA-Z_][a-zA-Z0-9_.]*)\s*:', fix_unquoted_keys, fixed)
    
    # Fix single quotes to double quotes (but be careful with string content)
    # This is tricky - we need to replace single quotes that are used as string delimiters
    # but not those inside strings. We'll use a simple heuristic:
    # Replace '...' with "..." when it looks like a string value (not a key)
    # We'll do this more carefully by tracking context
    
    # First, let's handle the simpler case: single quotes around string values
    # Pattern: : '...' or , '...' or [ '...' or { '...'
    def fix_single_quotes(match):
        prefix = match.group(1)
        content = match.group(2)
        # Escape any double quotes in the content
        content_escaped = content.replace('"', '\\"')
        return f'{prefix}"{content_escaped}"'
    
    # Match single-quoted strings that appear after : or , or [ or {
    fixed = re.sub(r'([:,\[\{])\s*\'([^\']*)\'', fix_single_quotes, fixed)
    
    # Fix trailing commas before } or ]
    fixed = re.sub(r',\s*([}\]])', r'\1', fixed)
    
    return fixed
def _extract_complete_json(s: str) -> str:
    """
    从字符串中提取完整的 JSON（找到匹配的闭合括号/大括号）。
    
    Args:
        s: 可能包含 JSON 的字符串
        
    Returns:
        提取出的完整 JSON 字符串
    """
    if not s:
        return s
    
    # 找到第一个 { 或 [
    first_brace = s.find('{')
    first_bracket = s.find('[')
    
    if first_brace == -1 and first_bracket == -1:
        return s
    
    start_idx = min(i for i in (first_brace, first_bracket) if i != -1)
    
    # 计算括号/大括号的匹配
    bracket_count = 0
    brace_count = 0
    in_string = False
    escape_next = False
    
    for i, char in enumerate(s[start_idx:], start=start_idx):
        if escape_next:
            escape_next = False
            continue
        if char == '\\':
            escape_next = True
            continue
        if char == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        
        if char == '[':
            bracket_count += 1
        elif char == ']':
            bracket_count -= 1
        elif char == '{':
            brace_count += 1
        elif char == '}':
            brace_count -= 1
        
        if bracket_count == 0 and brace_count == 0 and i > start_idx:
            return s[start_idx:i + 1]
    
    # 如果没有找到匹配的闭合，返回从开始到结尾
    return s[start_idx:]
    
def extract_json(s):
    """
    Extract and parse JSON from a string that may contain markdown code blocks.
    Attempts to fix common JSON format errors if initial parsing fails.
    
    Args:
        s: String that may contain JSON wrapped in markdown code blocks (```json ... ```)
        
    Returns:
        Parsed JSON as a dictionary or list
        
    Raises:
        ValueError: If no valid JSON can be extracted after all attempts
    """
    if not s:
        raise ValueError("Empty input string")
    
    text = s.strip()
    
    # Try to extract JSON from markdown code blocks first
    # Match ```json ... ``` or ``` ... ```
    json_block_pattern = r'```(?:json)?\s*(.*?)```'
    matches = re.findall(json_block_pattern, text, flags=re.DOTALL | re.IGNORECASE)
    if matches:
        # Use the first match (usually there's only one JSON block)
        json_str = matches[0].strip()
    else:
        # No markdown block found, try to find JSON directly
        # Look for first { or [
        first_brace = text.find('{')
        first_bracket = text.find('[')
        
        if first_brace == -1 and first_bracket == -1:
            raise ValueError("No JSON object or array found in input")
        
        start_idx = min(i for i in (first_brace, first_bracket) if i != -1)
        json_str = text[start_idx:]
        
        # Try to find the matching closing bracket/brace
        json_str = _extract_complete_json(json_str)
    
    # Try to parse the JSON string directly first
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        # If direct parsing fails, try to fix common format errors
        try:
            fixed_json = _fix_json_format_errors(json_str)
            return json.loads(fixed_json)
        except json.JSONDecodeError as e:
            # If fixing didn't work, try a more aggressive approach
            # Use a JSON repair library approach: try to extract just the JSON structure
            try:
                # Try to find and extract just the first complete JSON object/array
                fixed_json = _extract_complete_json(json_str)
                return json.loads(fixed_json)
            except json.JSONDecodeError:
                # Last resort: raise with original error
                raise ValueError(f"Failed to parse JSON after attempting fixes. Original error: {e}")

def extract_code(text):
    """Remove markdown code fences from text."""
    text = text.strip()
    # Remove ```json or ```cpp or ``` at start
    if text.startswith("```"):
        lines = text.split("\n")
        if len(lines) > 1:
            text = "\n".join(lines[1:])
    # Remove ``` at end
    if text.endswith("```"):
        text = text[:-3].strip()
    return text
