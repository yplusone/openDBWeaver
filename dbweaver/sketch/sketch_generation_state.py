# coding: utf-8
"""
subgraph/sketch_generation_state.py
-----------------------------------
定义 sketch_generation_subgraph 的状态类型
"""

from __future__ import annotations

from typing import TypedDict, Optional, Any, List, Dict
from dbweaver.sketch.code_combine import CodeOutput


class SketchGenerationState(TypedDict, total=False):
    """子图自己的状态"""
    query_id: int
    plan: List[Dict[str, Any]]
    snippets: Dict[str, Any]
    code: CodeOutput
    query_example: str
    query_template: str
    decomposed: Dict[str, str]
    ctx: dict[str, Any]

