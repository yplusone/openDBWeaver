# coding: utf-8
"""
operator_codegen/define.py

Define operator: 用于修改 function_definition 中的代码，以及绑定参数。
如果 dbweaver 的输入包含其他参数（如常量值、过滤条件等），在这里进行处理。
"""

import json
import re
from typing import Dict, Any
from .operator_gen import BaseOperator, register_operator, CodegenContext


@register_operator("define")
class DefineOperator(BaseOperator):
    """
    Define operator 负责：
    1. 修改 function_define 中的 LoadInternal 函数，注册表函数
    2. 在 FnBind 中绑定参数（如果 dbweaver 输入包含其他参数）
    3. 将参数存储到 FnBindData 中供后续使用
    """

    
    def build_prompt(self, state, ctx: CodegenContext, opt_msg: str) -> str:
        """
        构建 prompt 用于生成 define 相关的代码
        
        Args:
            state: 当前状态，包含 snippets
            ctx: 代码生成上下文
            opt_msg: 优化消息（如果有）
        """
        snippets = state.get("snippets", {})
        previous_json = {
            "headers": snippets.get("headers", "") or "",
            "support_code": snippets.get("support_code", "") or "",
            "bind_code": snippets.get("bind_code", "") or "",
            "execute_code": snippets.get("execute_code", "") or "",
            "finalize_code": snippets.get("finalize_code", "") or "",
            "function_define": snippets.get("function_define", "") or "",
        }
        previous_json_str = json.dumps(previous_json, ensure_ascii=False, indent=2)
        
        # 从 step 中获取参数信息（如果有）
        # 参数可能来自 decomposed query 中的常量值或其他绑定参数
        bind_params = self.step.get("parameters", [])  # 例如: [{"name": "year", "type": "INTEGER", "value": 1993}]
        
        # 从 context 中获取分解查询的信息（如果有）
        decomposed = state.get("decomposed", {})
        split_query = decomposed.get("split_query", "")
        split_template = decomposed.get("split_template", "")
        
        # 构建参数描述
        params_description = ""
        if bind_params:
            params_description = "Parameters to bind:\n"
            for param in bind_params:
                param_name = param.get("name", "unknown")
                params_description += f"  - {param_name}\n"
        else:
            params_description = "No additional parameters to bind (dbweaver only takes TABLE input).\n"
        
        return f"""
You are generating C++ code for the DEFINE operator of a DuckDB table function.
The DEFINE operator is responsible for:
1. Modifying the `function_define` field to register the table function with correct parameter signatures
2. Updating `FnBind` to extract and store any additional parameters from `TableFunctionBindInput`
3. Storing parameters in `FnBindData` for use in execution

Current snippets (must be preserved and extended, not discarded):
{previous_json_str}

Decomposed query information (for reference):
- Split query: {split_query}
- Split template: {split_template}

{params_description}
Decide the type of parameters from the split query.

Your task is to extend the existing snippets by updating the following modules:

1) "function_define" — register the table function with parameter types
   - The current `LoadInternal` function should register a table function like:
        TableFunction f("dbweaver", {{LogicalType::TABLE}}, nullptr, FnBind, FnInit, FnInitLocal);
   - If there are additional parameters (beyond the TABLE input), modify the constructor:
            TableFunction f("dbweaver", {{LogicalType::TABLE, LogicalType::INTEGER}}, nullptr, FnBind, FnInit, FnInitLocal);

   - The parameter types should match the `bind_params` list above

2) "bind_code" — extract parameters and store in FnBindData
    for example:
    static unique_ptr<FunctionData> FnBind(
        ClientContext &, TableFunctionBindInput &input,
        vector<LogicalType> &return_types,
        vector<string> &names
    ) {{
        int32_t a = input.inputs[1].GetValue<int32_t>();
        int32_t b = input.inputs[2].GetValue<int32_t>(); 
        double c = input.inputs[3].GetValue<double>();
            
        return make_uniq<FnBindData>(a,b,c);
    }}

3) "support_code" — update FnBindData struct
   - Extend the existing `FnBindData` struct to include fields for all parameters
    for example:
        struct FnBindData : public FunctionData {{
            int32_t year;  // add parameter field
            
            explicit FnBindData(int32_t year_p) : year(year_p) {{}}  // constructor
            
            unique_ptr<FunctionData> Copy() const override {{
                return make_uniq<FnBindData>(year);  // copy the parameter
            }}  
            
            bool Equals(const FunctionData &other_p) const override {{
                auto &other = other_p.Cast<FnBindData>();
                return year == other.year;  // compare the parameter
            }}
        }};

    - add comments in the existing `FnInit` function for future use
        //auto &bind_data = in.bind_data->Cast<FnBindData>();
        //int32_t a = bind_data.a;


4) "execute_code" and "finalize_code" — access parameters from FnBindData(write comments for future reference)
   - In `FnExecute` and `FnFinalize`, you can access bound parameters via:
         auto &bind_data = in.bind_data->Cast<FnBindData>();
         int32_t a = bind_data.a;
         int32_t b = bind_data.b;
   - Use these parameters in your computation logic (write comments for future reference)
   - Note: The current implementation may not need to modify these, but ensure
     the code can access parameters if needed


Type mapping for parameters:
- INTEGER → LogicalType::INTEGER, GetValue<int32_t>()
- BIGINT → LogicalType::BIGINT, GetValue<int64_t>()
- DOUBLE → LogicalType::DOUBLE, GetValue<double>()
- VARCHAR → LogicalType::VARCHAR, GetValue<string>()
- BOOLEAN → LogicalType::BOOLEAN, GetValue<bool>()
- DATE → LogicalType::DATE, GetValue<date_t>()
- TIMESTAMP → LogicalType::TIMESTAMP, GetValue<timestamp_t>()

Modification guidelines:
- Start from the existing `previous_json` and construct a new JSON object
- Preserve all valid logic from existing fields
- Only append or modify:
  * "function_define" to update the TableFunction registration
  * "bind_code" to extract and store parameters
  * "support_code" to extend FnBindData with parameter fields
- Keep naming and control flow consistent with existing snippets
- If no additional parameters are needed, keep the original implementation
""".strip()

