# coding: utf-8
"""
operator_codegen/operator_gen.py

Base class for operator codegen that directly uses a LangChain ChatOpenAI instance.
Each operator subclass:
- registers via @register_operator("op_name")
- implements build_prompt(self, state, ctx, opt_msg) -> str
- (optionally) overrides postprocess(self, raw_text) -> Dict[str, Any]
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Type
import json
import re
import time
import datetime
from langchain_openai import ChatOpenAI
from utils.llm_output_parse import extract_json
# ----------------------------- Registry ----------------------------- #

_REGISTRY: Dict[str, Type["BaseOperator"]] = {}


def register_operator(op_type: str):
    def _wrap(cls: Type["BaseOperator"]):
        key = (op_type or "").strip().lower()
        if not key:
            raise ValueError("op_type cannot be empty")
        if key in _REGISTRY:
            raise ValueError(f"Operator '{key}' already registered by {_REGISTRY[key].__name__}")
        _REGISTRY[key] = cls
        cls.OP_TYPE = key
        return cls

    return _wrap


def build_operator(step: Dict[str, Any], llm: ChatOpenAI) -> "BaseOperator":
    """Factory: create an operator instance from a plan step dict and a ChatOpenAI instance."""
    if not isinstance(step, dict) or "op" not in step:
        raise ValueError("Invalid step: must be a dict with an 'op' field")
    key = (step["op"] or "").strip().lower()
    cls = _REGISTRY.get(key)
    if not cls:
        raise KeyError(f"No operator registered for op='{key}'. Known: {sorted(_REGISTRY)}")
    return cls(step, llm=llm)


# ------------------------------ Context ----------------------------- #

@dataclass
class CodegenContext:
    indent: str = "    "
    newline: str = "\n"
    extras: Dict[str, Any] = field(default_factory=dict)


# ---------------------------- BaseOperator --------------------------- #

class BaseOperator:
    """Abstract base for all operator codegen steps that call an LLM."""
    OP_TYPE: str = "base"

    # Global system prompt: shared "hard rules" for all operators.
    SYSTEM_PROMPT: str = (
        "You are a DuckDB extension engineer. Generate additional C++ snippets for a new operator.\n"
        "You must extend the previously generated snippets (provided as a JSON object named `snippets`) "
        "and return a full JSON object representing the updated snippets.\n"
        "\n"
        "- Do NOT include or reference any DuckDB headers or namespaces (e.g., \"duckdb.hpp\",\n"
        "  \"duckdb/common/*\", \"duckdb/execution/*\", \"duckdb/planner/*\",\n"
        "  \"duckdb/optimizer/*\", etc.). These are all disallowed; assume includes/namespaces are handled elsewhere.\n"
        "- Pay attention to comments and add code in that place. Avoid modifying or deleting existing code unless "
        "absolutely required for correctness.\n"
        "- Preserve semantics, variable naming, and control flow; keep the generated code consistent with the "
        "original intent.\n"
        "\n"
        "**Output format**:\n"
        "- Return only one JSON object with these fields and nothing else:\n"
        "  {\n"
        '    "headers": "<merged headers, deduplicated>",\n'
        '    "support_code": "<Include the previous support_code plus your new helper functions. Ensure deduplication, maintain correct order, and place all declarations before their usage.>",\n'
        '    "bind_code": "<previous bind_code + your new code, deduplicated>",\n'
        '    "execute_code": "<previous execute_code extended with your new code>",\n'
        '    "finalize_code": "<previous finalize_code extended with your new code>"\n'
        '    "function_define": "<previous function_define extended with your new code>"\n'
        "  }\n"
        "- No prose outside JSON. No class boilerplate. Do NOT wrap the JSON in Markdown code fences.\n"
        "\n"
        "If a user message specifies a stricter or more specific output format, follow that user instruction exactly."
    )

    def __init__(self, step: Dict[str, Any], llm: ChatOpenAI) -> None:
        """
        step: the JSON dict from plan generation
        llm : a configured ChatOpenAI instance (e.g., Qwen/One-API/OpenAI-compatible)
        """
        
        self.step = step or {}
        print(f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] [OperatorCodegen] Begin to generate code for operator {self.step.get('op')}\n")
        if not isinstance(llm, ChatOpenAI):
            raise TypeError("llm must be an instance of langchain_openai.ChatOpenAI")
        self.llm = llm

        # Simplified optimize config
        raw = self.step.get("optimize")
        if raw is True:
            self.opt_cfg = {"enabled": True, "notes": ""}
        elif isinstance(raw, dict):
            self.opt_cfg = {"enabled": True, **raw}
        else:
            self.opt_cfg = {"enabled": False}

    # ----------------------------- main entry ----------------------------- #
    def run(self, state: Dict[str, Any], ctx: Optional[CodegenContext] = None) -> Dict[str, Any]:
        self.validate()
        ctx = ctx or CodegenContext()
        optimize_msg = ""
        if not len(state["snippets"]):
            state["snippets"] = self.build_initial_snippet()
        if self._has_overridden("build_opt_prompt"):
            prompt = self.build_opt_prompt(state, ctx)
            if len(prompt.strip()) > 0:
                optimize_msg = self._llm_generate(prompt)
            else:
                optimize_msg = ""

        # 1) Generate snippets
        bundle = self.emit_cpp_via_llm(state, ctx, optimize_msg)

        # 2) Subclasses may decide how to interpret optimization results
        return bundle

    # ------------------------- must-override hooks ------------------------ #
    def validate(self) -> None:
        outs = self.step.get("output_cols")
        if not isinstance(outs, list):
            raise ValueError(f"{self.name}: 'output_cols' must be a list")

    def build_prompt(self, state: Dict[str, Any], ctx: CodegenContext, opt_msg: str) -> str:
        """Return the full user prompt to ask the LLM. Must be implemented by subclass."""
        raise NotImplementedError

    def build_opt_prompt(self, state: Dict[str, Any], ctx: CodegenContext) -> str:
        """Optional: build an optimization-only prompt; subclasses may implement."""
        raise NotImplementedError

    # ------------------------ system prompt hook -------------------------- #
    def get_system_prompt(self) -> str:
        """
        Return the system prompt for this call.
        - If the step explicitly provides 'system_prompt', prefer that;
        - otherwise use BaseOperator.SYSTEM_PROMPT (or subclass override).
        """
        override = (self.step.get("system_prompt") or "").strip()
        if override:
            return override
        return self.SYSTEM_PROMPT

    # ------------------------ default LLM-driven flow --------------------- #
    def emit_cpp_via_llm(self, state: Dict[str, Any], ctx: CodegenContext, opt_msg: str) -> Dict[str, Any]:
        prompt = self.build_prompt(state, ctx, opt_msg)
        raw = self._llm_generate(prompt)
        return self.postprocess(raw)

    def _llm_generate(self, prompt: str, retries: int = 4, backoff: float = 0.5) -> str:
        last_err: Optional[Exception] = None
        for i in range(retries + 1):
            try:
                system_prompt = self.get_system_prompt()
                messages: List[Dict[str, str]] = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})
                msg = self.llm.invoke(messages)
                return msg.content
            except Exception as e:
                last_err = e
                print(e)
                time.sleep(10)
        raise last_err or RuntimeError("LLM generation failed")

    # ----------------------------- JSON fixing helper ---------------------- #


    # ----------------------------- postprocess ---------------------------- #
    def postprocess(self, raw_text: str) -> Dict[str, Any]:
        """
        Prefer JSON parsing with the standard fields:
            - headers
            - execute_code
            - finalize_code
            - support_code
            - bind_code   (optional; may be absent)
            - function_define (optional; may be absent)
        Fallback: if not valid JSON, extract C++ from fenced code, or use raw text as execute_code.
        """
        text = raw_text.strip()
        obj = extract_json(text)
        if isinstance(obj, dict) and ("execute_code" in obj or "finalize_code" in obj):
            return {
                "headers": obj.get("headers", ""),
                "execute_code": obj.get("execute_code", ""),
                "finalize_code": obj.get("finalize_code", ""),
                "support_code": obj.get("support_code", ""),
                "bind_code": obj.get("bind_code", ""),
                "function_define": obj.get("function_define", ""),
            }

    # ----------------------------- artifacts ------------------------------ #
    def _write_artifacts(self, state: Dict[str, Any], bundle: Dict[str, Any]) -> None:
        artifacts = state.setdefault("artifacts", {})
        ocg = artifacts.setdefault("operator_codegen", {})
        ocg.setdefault("snippets", []).append(bundle)
        used = ocg.setdefault("operators_used", [])
        if self.OP_TYPE not in used:
            used.append(self.OP_TYPE)

    def _has_overridden(self, method_name: str) -> bool:
        base_impl = getattr(BaseOperator, method_name, None)
        impl = getattr(self.__class__, method_name, None)
        return impl is not None and impl is not base_impl

    def _ensure_shape(self, base: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensure the optimized result has all standard fields; fallback to base values for missing ones.
        No semantic checks are performed here.
        """
        keys = ("headers", "support_code", "execute_code", "finalize_code", "bind_code")
        out = {}
        for k in keys:
            if isinstance(new, dict) and (k in new) and new[k] is not None:
                out[k] = new[k]
            else:
                out[k] = base.get(k, "")
        return out

    def build_initial_snippet(self) -> Dict[str, Any]:
        """Construct the initial snippet structure used as a starting point for optimization."""
        return {
            "headers": """
                #include "dbweaver_extension.hpp"
                #include "duckdb.hpp"
                #include "duckdb/function/table_function.hpp"
                #include <atomic>
                #include <limits>
                #include <mutex>
                //TODO: Add more includes as needed
            """,
            "support_code": """
            //TODO: Define any helper structs or functions needed for binding/execution
            
            struct FnBindData : public FunctionData {
                unique_ptr<FunctionData> Copy() const override { return make_uniq<FnBindData>(); }
                bool Equals(const FunctionData &) const override { return true; }
            };

            struct FnGlobalState : public GlobalTableFunctionState {
                // TODO: Optional accumulators/counters (generator may append to this struct)
                std::mutex lock;
                std::atomic<idx_t> active_local_states {0};
                std::atomic<idx_t> merged_local_states {0};
               	idx_t MaxThreads() const override {
                    return std::numeric_limits<idx_t>::max();
                }
            };

            static unique_ptr<GlobalTableFunctionState> FnInit(ClientContext &, TableFunctionInitInput &) {
                return make_uniq<FnGlobalState>();
            }
            
            struct FnLocalState : public LocalTableFunctionState {
                //TODO: initialize local state and other preparations
                bool merged = false;
            };

            static unique_ptr<LocalTableFunctionState> FnInitLocal(ExecutionContext &, TableFunctionInitInput &,
                                                                  GlobalTableFunctionState *global_state) {
                auto &g = global_state->Cast<FnGlobalState>();
                g.active_local_states.fetch_add(1, std::memory_order_relaxed);
                return make_uniq<FnLocalState>();
            }
            """,
            "bind_code": """
                static unique_ptr<FunctionData> FnBind(ClientContext &, TableFunctionBindInput &,
                                                    vector<LogicalType> &return_types,
                                                    vector<string> &names) {
                    //TODO: populate return_types and names

                    return make_uniq<FnBindData>();
                }
            """,
            "execute_code": """
                static OperatorResultType FnExecute(ExecutionContext &, TableFunctionInput &in,
                                                    DataChunk &input, DataChunk &) {
                    auto &l = in.local_state->Cast<FnLocalState>();

                    
                    if (input.size() == 0) {       
                        return OperatorResultType::NEED_MORE_INPUT;
                    }

                    //TODO: process input chunk and produce output

                    return OperatorResultType::NEED_MORE_INPUT; 
                }
            """,
            "finalize_code": """
                static OperatorFinalizeResultType FnFinalize(ExecutionContext &, TableFunctionInput &in,
                                                            DataChunk &out) {
                    auto &g = *reinterpret_cast<FnGlobalState *>(in.global_state.get());
                    auto &l = in.local_state->Cast<FnLocalState>();
                    // Merge the local state into the global state exactly once.
                    if (!l.merged) {
                        {
                            std::lock_guard<std::mutex> guard(g.lock);
                            //TODO: merge local state with global state
                        }
                        l.merged = true;
                        g.merged_local_states.fetch_add(1, std::memory_order_relaxed);
                    }

                    // Only the *last* local state to merge emits the final result.
                    // All other threads return FINISHED with an empty chunk.
                    const auto merged = g.merged_local_states.load(std::memory_order_relaxed);
                    const auto active = g.active_local_states.load(std::memory_order_relaxed);
                    if (active > 0 && merged == active) {
                        {
                            std::lock_guard<std::mutex> guard(g.lock);
                            //TODO: get result from global state
                        }
                        //TODO: populate out chunk with final results
                    } else {
                        out.SetCardinality(0);
                    }
                    
                    return OperatorFinalizeResultType::FINISHED;
                }
            """,
            "function_define":"""
                static void LoadInternal(ExtensionLoader &loader) {
                    TableFunction f("dbweaver", {LogicalType::TABLE}, nullptr, FnBind, FnInit, FnInitLocal);
                    f.in_out_function       = FnExecute;
                    f.in_out_function_final = FnFinalize;
                    loader.RegisterFunction(f);
                }
            """
        }
