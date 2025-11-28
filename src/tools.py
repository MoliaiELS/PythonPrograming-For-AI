import ast
import json
import os
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
        "User-Agent": "pyforai-projectB/1.0 (contact: yguo704@connect.hkust-g.edu.cn)",
        "Accept": "application/json",
        "Accept-Language": "en-US,en;q=0.9",
    }

    try:
        r = requests.get(url, params=params, headers=headers, timeout=8)
        r.raise_for_status()
        data = r.json()

        results: List[Dict[str, Any]] = []
        for item in data.get("query", {}).get("search", []):
            results.append({
                "title": item.get("title", ""),
                "snippet": _strip_html(item.get("snippet", "")),
                "pageid": item.get("pageid", None),
                "source": "wikipedia",
            })

        if results:
            return {"results": results}

        # 空结果也走 fallback
        raise RuntimeError("empty_results")

    except Exception as e:
        return {
            "results": [],
            "error": f"wiki_failed_{type(e).__name__}",
        }


# =========================
# Tool 3: Baidu search
# =========================
def tool_baidu_search(query: str, top_k: int = 5) -> Dict[str, Any]:
    top_k = max(1, min(int(top_k), 10))
    q = (query or "").strip()
    if not q:
        return {"results": []}

    url = f"https://www.baidu.com/s?wd={quote_plus(q)}"

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Referer": "https://www.baidu.com/",
    }

    r = requests.get(url, headers=headers, timeout=10, allow_redirects=True)

    # 把关键调试信息一起返回，避免“结果空但不知道为啥”
    debug = {
        "http_status": r.status_code,
        "final_url": r.url,
    }

    # 有时会返回验证码/安全校验页面
    text_head = (r.text or "")[:500]
    # if "百度安全验证" in text_head or "安全验证" in text_head or "请输入验证码" in text_head:
    #     return {"results": [], "debug": {**debug, "blocked": True}}
    if "百度安全验证" in text_head or "安全验证" in text_head or "请输入验证码" in text_head or "wappass.baidu.com/static/captcha" in r.url:
        return {"results": [], "error": "captcha_blocked", "debug": {**debug, "blocked": True}}


    soup = BeautifulSoup(r.text, "lxml")

    # 多种可能的结果容器
    container = soup.select_one("#content_left") or soup
    blocks = container.select("div.result, div.result-op, div[data-log]")

    results: List[Dict[str, Any]] = []
    for b in blocks:
        if len(results) >= top_k:
            break

        a = b.select_one("h3 a") or b.select_one("a")
        if not a:
            continue

        title = a.get_text(" ", strip=True)
        href = a.get("href", "").strip()
        if not title or not href:
            continue

        # 常见摘要 class
        snippet_el = (
            b.select_one(".c-abstract")
            or b.select_one(".content-right_8Zs40")
            or b.select_one("div.c-span-last")
            or b.select_one("span.content-right_8Zs40")
        )
        snippet = snippet_el.get_text(" ", strip=True) if snippet_el else ""
        snippet = re.sub(r"\s+", " ", snippet).strip()

        results.append({
            "title": title,
            "url": href,
            "snippet": snippet,
            "source": "baidu",
        })

    # 如果还是空，把页面 title 返回，方便判断是不是反爬页
    page_title = soup.title.get_text(strip=True) if soup.title else ""
    return {"results": results, "debug": {**debug, "page_title": page_title}}

# =========================
# Tool 4: Serper.dev Google search
# =========================
def tool_serper_search(query: str, top_k: int = 5) -> Dict[str, Any]:
    top_k = max(1, min(int(top_k), 10))
    q = (query or "").strip()
    if not q:
        return {"results": []}

    api_key = os.getenv("SERPER_API_KEY", "").strip()
    if not api_key:
        return {"results": [], "error": "missing_SERPER_API_KEY"}

    url = "https://google.serper.dev/search"
    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "pyforai-projectB/1.0",
    }
    payload = {"q": q, "num": top_k}

    r = requests.post(url, headers=headers, json=payload, timeout=12)

    if r.status_code != 200:
        return {
            "results": [],
            "error": f"serper_http_{r.status_code}",
            "debug": {
                "http_status": r.status_code,
                "body_head": (r.text or "")[:300],
            },
        }

    data = r.json()
    results: List[Dict[str, Any]] = []
    for item in (data.get("organic") or [])[:top_k]:
        results.append({
            "title": item.get("title", ""),
            "url": item.get("link", ""),
            "snippet": item.get("snippet", ""),
            "source": "serper",
        })
    return {"results": results}


# =========================
# Tool 5: Planner (deterministic)
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
            "serper_search": tool_serper_search,
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
        {
            "type": "function",
            "function": {
                "name": "serper_search",
                "description": "Web search via Serper (Google Search API). Returns title/url/snippet.",
                "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 10}
                },
                "required": ["query"],
                "additionalProperties": False
                }
            }
        },
    ]
