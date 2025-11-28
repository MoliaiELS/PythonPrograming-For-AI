import ast
import json
import re
from typing import Any, Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

from bs4 import BeautifulSoup
from urllib.parse import quote_plus
import requests


# =========================
# Tool 1: Safe calculator
# =========================
_ALLOWED_AST_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.FloorDiv,
    ast.UAdd, ast.USub,
)

def _safe_eval_math(expr: str) -> float:
    """
    Only allow numeric expressions with basic operators.
    Blocks variables, attributes, and function calls.
    """
    expr = expr.strip()
    node = ast.parse(expr, mode="eval")

    for n in ast.walk(node):
        if not isinstance(n, _ALLOWED_AST_NODES):
            raise ValueError(f"Disallowed AST node: {type(n).__name__}")
        if isinstance(n, ast.Name) or isinstance(n, ast.Call) or isinstance(n, ast.Attribute):
            raise ValueError("Names/calls/attributes are not allowed")

    val = eval(compile(node, "<safe-math>", "eval"), {"__builtins__": {}}, {})
    if not isinstance(val, (int, float)):
        raise ValueError("Expression did not evaluate to a number")
    return float(val)

def tool_calc(expression: str) -> Dict[str, Any]:
    return {"value": _safe_eval_math(expression)}


# =========================
# Tool 2: Wikipedia search
# =========================
def _strip_html(s: str) -> str:
    return re.sub(r"<.*?>", "", s or "")

def tool_wiki_search(query: str, top_k: int = 3) -> Dict[str, Any]:
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json",
        "srlimit": max(1, min(int(top_k), 5)),
        "utf8": 1,
    }
    headers = {
        # Wikipedia 推荐提供清晰 UA（含项目名/联系方式更好）
        "User-Agent": "pyforai-projectB/1.0 (contact: yguo704@connect.hkust-gz.edu.cn)",
        "Accept": "application/json",
        "Accept-Language": "en-US,en;q=0.9",
    }
    r = requests.get(url, params=params, headers=headers, timeout=8)
    r.raise_for_status()
    data = r.json()

    results: List[Dict[str, Any]] = []
    for item in data.get("query", {}).get("search", []):
        results.append({
            "title": item.get("title", ""),
            "snippet": _strip_html(item.get("snippet", "")),
            "pageid": item.get("pageid", None),
        })
    return {"results": results}

# =========================
# Tool 3: Baidu search
# =========================
def tool_baidu_search(query: str, top_k: int = 5) -> Dict[str, Any]:
    """
    Use Baidu search result page (HTML) and parse top results.
    Notes:
      - Baidu may change HTML structure; parsing is best-effort.
      - Returned 'url' may be a Baidu redirect link.
    """
    top_k = max(1, min(int(top_k), 10))
    q = (query or "").strip()
    if not q:
        return {"results": []}

    # Baidu search URL
    url = f"https://www.baidu.com/s?wd={quote_plus(q)}"

    headers = {
        # A realistic UA helps avoid 403 in some environments
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    }

    r = requests.get(url, headers=headers, timeout=10)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "lxml")

    # Baidu results are often in #content_left, each item can be .result or [data-log]
    container = soup.select_one("#content_left") or soup
    candidates = container.select("div.result, div.result-op, div[data-log]")

    results: List[Dict[str, Any]] = []
    for item in candidates:
        if len(results) >= top_k:
            break

        # Title/link
        a = item.select_one("h3 a") or item.select_one("a")
        if not a:
            continue
        title = a.get_text(" ", strip=True)
        href = a.get("href", "")

        # Snippet/abstract: Baidu often uses .c-abstract or .content-right_8Zs40 etc
        snippet_el = item.select_one(".c-abstract") or item.select_one(".content-right_8Zs40") or item.select_one("div.c-span-last")
        snippet = snippet_el.get_text(" ", strip=True) if snippet_el else ""

        # Filter out empty titles
        if not title:
            continue

        results.append({
            "title": title,
            "url": href,
            "snippet": snippet,
            "source": "baidu"
        })

    return {"results": results}

# =========================
# Tool 4: Planner (deterministic)
# =========================
def tool_planner(goal: str, constraints: str = "", tools_available: List[str] = None) -> Dict[str, Any]:
    """
    A lightweight non-LLM planner.
    It returns a plan the LLM can follow, with tool hints.
    """
    tools_available = tools_available or ["planner", "calc", "wiki_search"]
    g = (goal or "").strip()
    c = (constraints or "").strip()

    steps = []
    steps.append({
        "step": 1,
        "action": "Restate the goal in one sentence and identify what facts or computations are needed.",
        "tool_hint": None
    })

    # Heuristic tool hints
    needs_math = any(k in g.lower() for k in ["compute", "calculate", "math", "sum", "percentage", "times", "divide"])
    needs_search = any(k in g.lower() for k in ["who is", "what is", "capital", "definition", "wiki", "wikipedia", "history"])

    if needs_search and "wiki_search" in tools_available:
        steps.append({
            "step": len(steps) + 1,
            "action": "Fetch key facts via Wikipedia search.",
            "tool_hint": {"tool": "wiki_search", "args": {"query": "keywords from the question", "top_k": 3}}
        })

    if needs_math and "calc" in tools_available:
        steps.append({
            "step": len(steps) + 1,
            "action": "Perform required computations using the calculator.",
            "tool_hint": {"tool": "calc", "args": {"expression": "fill in expression"}}
        })

    steps.append({
        "step": len(steps) + 1,
        "action": "Draft the final answer with short justification. If tool results are uncertain, say so.",
        "tool_hint": None
    })

    return {
        "goal": g,
        "constraints": c,
        "tools_available": tools_available,
        "plan": steps,
        "stop_when": "You have enough info to answer or you hit the tool/step limits."
    }


# =========================
# Tool registry + execution with timeout
# =========================
class ToolRegistry:
    def __init__(self):
        self._fns = {
            "calc": tool_calc,
            "wiki_search": tool_wiki_search,
            "baidu_search": tool_baidu_search,
            "planner": tool_planner,
        }

    def allowed_tools(self) -> List[str]:
        return list(self._fns.keys())

    def call_with_timeout(self, name: str, args: Dict[str, Any], timeout_s: float) -> Tuple[Dict[str, Any], str]:
        if name not in self._fns:
            return {}, f"unknown_tool({name})"

        fn = self._fns[name]

        def run():
            return fn(**args)

        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(run)
            try:
                out = fut.result(timeout=timeout_s)
                if not isinstance(out, dict):
                    return {}, f"tool_output_not_dict({name})"
                return out, ""
            except FuturesTimeoutError:
                return {}, f"tool_timeout({name})"
            except TypeError as e:
                return {}, f"bad_args({name}): {str(e)}"
            except Exception as e:
                return {}, f"tool_failed({name}): {type(e).__name__}: {str(e)}"


def qwen_tools_schema() -> List[Dict[str, Any]]:
    """
    Tools schema for Qwen function calling (OpenAI-compatible tools format).
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "planner",
                "description": "Create a short step-by-step plan with tool hints before execution.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "goal": {"type": "string", "description": "User goal or question"},
                        "constraints": {"type": "string", "description": "Optional constraints"},
                        "tools_available": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Available tool names"
                        }
                    },
                    "required": ["goal"],
                    "additionalProperties": False
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "calc",
                "description": "Safely evaluate a basic arithmetic expression, returns numeric value.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string", "description": "Arithmetic expression, e.g. (12+8)/5"}
                    },
                    "required": ["expression"],
                    "additionalProperties": False
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "wiki_search",
                "description": "Search English Wikipedia and return top results with snippets.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "top_k": {"type": "integer", "minimum": 1, "maximum": 5}
                    },
                    "required": ["query"],
                    "additionalProperties": False
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "baidu_search",
                "description": "Search the web via Baidu and return top results with title/url/snippet.",
                "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 10}
                },
                "required": ["query"],
                "additionalProperties": False
                }
            }
        },
    ]
