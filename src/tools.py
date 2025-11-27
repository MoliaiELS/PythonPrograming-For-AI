# src/tools.py
import ast
import json
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple

import requests
from jsonschema import validate, ValidationError

# -----------------------------
# Tool 1: Safe calculator
# -----------------------------
_ALLOWED_AST_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Num, ast.Constant,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod,
    ast.USub, ast.UAdd, ast.FloorDiv,
    ast.Load, ast.Tuple
)

def _safe_eval_math(expr: str) -> float:
    """
    一个非常严格的数学表达式 eval。
    只允许数字和 + - * / ** % // ()，拒绝变量名、函数调用、属性访问等。
    """
    expr = expr.strip()
    node = ast.parse(expr, mode="eval")

    for n in ast.walk(node):
        if not isinstance(n, _ALLOWED_AST_NODES):
            raise ValueError(f"Disallowed expression part: {type(n).__name__}")

        # 禁止 Name（变量）与 Call（函数调用）
        if isinstance(n, ast.Name) or isinstance(n, ast.Call) or isinstance(n, ast.Attribute):
            raise ValueError("Names/calls/attributes are not allowed")

    val = eval(compile(node, "<math>", "eval"), {"__builtins__": {}}, {})
    if isinstance(val, (int, float)):
        return float(val)
    raise ValueError("Expression did not evaluate to a number")

def calc_tool(expression: str) -> Dict[str, Any]:
    """
    输入: expression
    输出: {"value": number}
    """
    return {"value": _safe_eval_math(expression)}

CALC_SCHEMA_IN = {
    "type": "object",
    "properties": {
        "expression": {"type": "string", "minLength": 1}
    },
    "required": ["expression"],
    "additionalProperties": False
}
CALC_SCHEMA_OUT = {
    "type": "object",
    "properties": {
        "value": {"type": "number"}
    },
    "required": ["value"],
    "additionalProperties": False
}

# -----------------------------
# Tool 2: Wikipedia search (simple)
# -----------------------------
def _strip_html(s: str) -> str:
    return re.sub(r"<.*?>", "", s)

def wiki_search_tool(query: str, top_k: int = 3) -> Dict[str, Any]:
    """
    用 MediaWiki API 做简单检索。
    输出给 agent 的内容必须当作“不可信数据”，仅用于事实支撑。
    """
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json",
        "srlimit": top_k,
        "utf8": 1
    }
    r = requests.get(url, params=params, timeout=8)
    r.raise_for_status()
    data = r.json()
    hits = []
    for item in data.get("query", {}).get("search", []):
        title = item.get("title", "")
        snippet = _strip_html(item.get("snippet", ""))
        pageid = item.get("pageid", None)
        hits.append({
            "title": title,
            "snippet": snippet,
            "pageid": pageid
        })
    return {"results": hits}

WIKI_SCHEMA_IN = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "minLength": 1},
        "top_k": {"type": "integer", "minimum": 1, "maximum": 5}
    },
    "required": ["query"],
    "additionalProperties": False
}
WIKI_SCHEMA_OUT = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "snippet": {"type": "string"},
                    "pageid": {"type": ["integer", "null"]}
                },
                "required": ["title", "snippet", "pageid"],
                "additionalProperties": False
            }
        }
    },
    "required": ["results"],
    "additionalProperties": False
}

# -----------------------------
# Tool registry + schema validation
# -----------------------------
@dataclass
class ToolSpec:
    name: str
    description: str
    schema_in: Dict[str, Any]
    schema_out: Dict[str, Any]
    fn: Callable[..., Dict[str, Any]]

class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, ToolSpec] = {}

    def register(self, tool: ToolSpec) -> None:
        self._tools[tool.name] = tool

    def list_for_prompt(self) -> str:
        """
        给 prompt 的工具说明文本：把 schema 清晰写出来。
        """
        blocks = []
        for t in self._tools.values():
            blocks.append(
                f"Tool: {t.name}\n"
                f"Description: {t.description}\n"
                f"Input JSON Schema: {json.dumps(t.schema_in, ensure_ascii=False)}\n"
                f"Output JSON Schema: {json.dumps(t.schema_out, ensure_ascii=False)}\n"
            )
        return "\n".join(blocks)

    def call(self, name: str, args: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
        """
        返回 (tool_output, error_message)
        err 为空字符串表示成功。
        """
        if name not in self._tools:
            return {}, f"Unknown tool: {name}"

        tool = self._tools[name]
        try:
            validate(instance=args, schema=tool.schema_in)
        except ValidationError as e:
            return {}, f"Tool input schema validation failed: {e.message}"

        try:
            out = tool.fn(**args)
        except TypeError as e:
            return {}, f"Tool argument error: {str(e)}"
        except Exception as e:
            return {}, f"Tool execution error: {str(e)}"

        # 输出也做 schema 校验，便于调试和防跑飞
        try:
            validate(instance=out, schema=tool.schema_out)
        except ValidationError as e:
            return {}, f"Tool output schema validation failed: {e.message}"

        return out, ""
