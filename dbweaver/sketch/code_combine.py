# coding: utf-8
"""
code_gen/code_generator.py
--------------------------
Utility that calls an LLM to stitch a C++ DuckDB–extension *template* with
snippet JSON into a single, self-contained plugin source file.

The LLM receives:
  1) A C++ skeleton template (may contain markers like
     //<<HEADERS>>, //<<SUPPORT_CODE>>, //<<FN_EXECUTE_BODY>>, //<<FN_FINALIZE_BODY>>)
  2) A JSON object with fields:
       {
         "headers": "...",
         "support_code": "...",
         "execute_code": "...",
         "finalize_code": "..."
       }

The model merges the snippets into the template and returns ONLY the final C++
file text (no Markdown). We also instruct the model to pay attention to code
quality (includes, unresolved placeholders, API consistency).

Returned object (`CodeOutput`) contains:
    • prefix   – short English tag for the pipeline stage
    • cpp_code – the final C++ source (single file, self-contained)
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional, Union, Dict, Any

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from config import EXPT_LLM_MODEL,API_KEY,BASE_URL


# ---------------------------------------------------------------------------
#  Structured output
# ---------------------------------------------------------------------------
class CodeOutput(BaseModel):
    """Final stitched C++ file."""
    prefix: str = Field(
        default="// STITCHED",
        description="Short tag about how the code was produced.",
    )
    cpp_code: str = Field(
        description="Complete, self-contained C++ source file. Do not wrap in Markdown."
    )


# ---------------------------------------------------------------------------
#  Code-merging wrapper
# ---------------------------------------------------------------------------
class CodeCombiner:
    """
    Expand a C++ skeleton into a concrete DuckDB extension via LLM by merging
    snippet JSON into the template. The LLM is explicitly instructed to ensure
    code quality (dedupe includes, no unresolved placeholders, API consistency).
    """

    def __init__(self, llm=None):
        if llm is None:
            self.llm = ChatOpenAI(model=EXPT_LLM_MODEL, api_key=API_KEY, base_url=BASE_URL)
        else:
            self.llm = llm

        # 带示例代码的 system prompt
        system_prompt = """
        You are an expert DuckDB extension engineer.

        You will receive:
        1) A C++ skeleton template (may contain markers like
        //<<HEADERS>>, //<<SUPPORT_CODE>>, //<<BIND_CODE>>, //<<EXECUTE_CODE>>, //<<FINALIZE_CODE>>, //<<FUNCTION_DEFINE>>).
        2) A JSON object with fields: headers, support_code, bind_code, execute_code, finalize_code, function_define.

        Template:
        ```cpp
        {template}
        ```
        
        Task:

        Merge the JSON snippets into the template to produce ONE final C++ source file.
        Replace markders like //<<HEADERS>>, //<<BIND_CODE>>, //<<EXECUTE_CODE>>, //<<FINALIZE_CODE>>; with the corresponding JSON content.
        Please don't change the structure of the template except for filling in the markers. 
        For <<SUPPORT_CODE>>, there have existed structures in template, please either extend them or add new structures as needed. For example, add new variables in struct FnGlobalState rather than creating a new struct.
        Please Check the order of decalations, avoid using variables or structs before they are declared!!!

        ⚠️ Quality rules you MUST follow:

        Organize the code to ensure clear and understandable structure without compilation errors.

        Deduplicate identical #include lines.

        Deduplicate redeclarations of variables, functions, classes, etc.

        Ensure that all used symbols have matching headers (e.g., string_t, hugeint_t,
        timestamp_tz_t, interval_t, list_entry_t).

        Ensure semantic consistency: no missing variables, no conflicting definitions,
        no dangling references, and correct namespace scoping.
        
        Check the headers, The DuckDB extension_util.hpp has been removed in recent versions, make sure not to use it.
        
        Check the type of the data used in the code snippets, make sure no overflow or underflow will happen. Check the output data types,
        make sure the intermediate temporary variables have correct types. For example, if the output is BIGINT, the sum variable should be at least BIGINT or larger type.

        The final output must compile as a single, self-contained C++ file.

        Return ONLY the final C++ source file as plain text.
        No Markdown fences, no explanations, no prose.
        """


        user_prompt = (
            "Snippets JSON:\n```json\n{snippets_json}\n```\n"
        )

        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("user", user_prompt),
            ]
        )

        # We want the raw text back (final C++), not a structured object
        self.chain = self.prompt | self.llm

    # ---------------------------------------------------------------------
    #  Public API
    # ---------------------------------------------------------------------
    def generate(
        self,
        template: str,
        snippets_json: Union[str, Dict[str, Any]],
    ) -> CodeOutput:
        """
        Call the LLM to stitch `snippets_json` into `template` and return final C++.
        - template: C++ skeleton string
        - snippets_json: JSON string or dict with keys: headers, support_code, bind_code, execute_code, finalize_code
        """
        payload = {
            "snippets_json": (
                json.dumps(snippets_json, ensure_ascii=False, indent=2)
                if isinstance(snippets_json, dict)
                else snippets_json.strip()
            ),
            "template": template.strip(),
        }
        resp = self.chain.invoke(payload).content

        # Hygiene: drop accidental markdown fences if the model added any
        cpp_code = self._strip_fences(resp).strip()
        return CodeOutput(prefix="// STITCHED (LLM merge + quality rules)", cpp_code=cpp_code)

    # ---------------------------------------------------------------------
    #  Helpers
    # ---------------------------------------------------------------------
    @staticmethod
    def _strip_fences(text: str) -> str:
        """Remove common Markdown fences if present."""
        if not isinstance(text, str):
            return ""
        return re.sub(
            r"^```(?:cpp|c\+\+|c|text)?\s*|\s*```$",
            "",
            text.strip(),
            flags=re.DOTALL | re.IGNORECASE,
        )
