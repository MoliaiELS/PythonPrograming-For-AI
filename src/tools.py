import ast
import os
import re
import json
from typing import Any, Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from urllib.parse import quote_plus

import requests
from bs4 import BeautifulSoup


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
    expr = (expr or "").strip()
    node = ast.parse(expr, mode="eval")

    for n in ast.walk(node):
        if not isinstance(n, _ALLOWED_AST_NODES):
            raise ValueError(f"Disallowed AST node: {type(n).__name__}")
        if isinstance(n, (ast.Name, ast.Call, ast.Attribute)):
            raise ValueError("Names/calls/attributes are not allowed")

    val = eval(compile(node, "<safe-math>", "eval"), {"__builtins__": {}}, {})
    if not isinstance(val, (int, float)):
        raise ValueError("Expression did not evaluate to a number")
    return float(val)

def tool_calc(expression: str) -> Dict[str, Any]:
    return {"value": _safe_eval_math(expression)}


# =========================
# Shared helpers
# =========================
def _strip_html(s: str) -> str:
    return re.sub(r"<.*?>", "", s or "")

def _short(s: str, n: int = 300) -> str:
    s = s or ""
    return s[:n]


# =========================
# Tool 2: Wikipedia search (NO fallback here)
# =========================
def tool_wiki_search(query: str, top_k: int = 3) -> Dict[str, Any]:
    """
    Search English Wikipedia via official API.
    Returns structured error instead of falling back to other engines.
    """
    url = "https://en.wikipedia.org/w/api.php"
    top_k = max(1, min(int(top_k), 5))
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json",
        "srlimit": top_k,
        "utf8": 1,
    }
    headers = {
        "User-Agent": "pyforai-projectB/1.0 (contact: yguo704@connect.hkust-g.edu.cn)",
        "Accept": "application/json",
        "Accept-Language": "en-US,en;q=0.9",
    }

    try:
        # r = requests.get(url, params=params, headers=headers, timeout=10)
        r = requests.get(url, params=params, headers=headers, timeout=(3, 6))
        if r.status_code != 200:
            return {
                "results": [],
                "error": f"wiki_http_{r.status_code}",
                "debug": {"http_status": r.status_code, "body_head": _short(r.text)},
            }

        data = r.json()
        results: List[Dict[str, Any]] = []
        for item in data.get("query", {}).get("search", []):
            results.append({
                "title": item.get("title", ""),
                "snippet": _strip_html(item.get("snippet", "")),
                "pageid": item.get("pageid", None),
                "source": "wikipedia",
            })

        return {"results": results}

    except Exception as e:
        return {"results": [], "error": f"wiki_failed_{type(e).__name__}"}


# =========================
# Tool 3: Baidu search (HTML scrape, may captcha)
# =========================
def tool_baidu_search(query: str, top_k: int = 5) -> Dict[str, Any]:
    """
    Best-effort Baidu HTML search. Often blocked by captcha in some networks.
    When captcha is detected, returns error='captcha_blocked'.
    """
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

    try:
        # r = requests.get(url, headers=headers, timeout=12, allow_redirects=True)
        r = requests.get(url, headers=headers, timeout=(3, 6), allow_redirects=True)
    except Exception as e:
        return {"results": [], "error": f"baidu_failed_{type(e).__name__}"}

    debug = {"http_status": r.status_code, "final_url": r.url}
    text_head = (r.text or "")[:800]

    # Captcha/safety check detection (very common)
    if (
        "wappass.baidu.com/static/captcha" in (r.url or "")
        or "百度安全验证" in text_head
        or "安全验证" in text_head
        or "请输入验证码" in text_head
    ):
        return {"results": [], "error": "captcha_blocked", "debug": {**debug, "page_title": "baidu_captcha"}}

    # Non-200 is still useful to debug
    if r.status_code != 200:
        return {"results": [], "error": f"baidu_http_{r.status_code}", "debug": {**debug, "body_head": _short(r.text)}}

    soup = BeautifulSoup(r.text, "lxml")
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
        href = (a.get("href") or "").strip()
        if not title or not href:
            continue

        snippet_el = (
            b.select_one(".c-abstract")
            or b.select_one("div.c-span-last")
            or b.select_one(".content-right_8Zs40")
        )
        snippet = snippet_el.get_text(" ", strip=True) if snippet_el else ""
        snippet = re.sub(r"\s+", " ", snippet).strip()

        results.append({"title": title, "url": href, "snippet": snippet, "source": "baidu"})

    page_title = soup.title.get_text(strip=True) if soup.title else ""
    return {"results": results, "debug": {**debug, "page_title": page_title}}


# =========================
# Tool 4: Serper.dev (Google Search API)
# =========================
def tool_serpapi_search(query: str, top_k: int = 5) -> Dict[str, Any]:
    top_k = max(1, min(int(top_k), 10))
    q = (query or "").strip()
    if not q:
        return {"results": []}

    api_key = os.getenv("SERPAPI_API_KEY", "").strip()
    if not api_key:
        return {"results": [], "error": "missing_SERPAPI_API_KEY"}

    url = "https://serpapi.com/search.json"
    params = {
        "engine": "google",
        "q": q,
        "api_key": api_key,
        "num": top_k,
    }

    try:
        r = requests.get(url, params=params, timeout=(3, 8))
    except Exception as e:
        return {"results": [], "error": f"serpapi_failed_{type(e).__name__}"}

    if r.status_code != 200:
        return {
            "results": [],
            "error": f"serpapi_http_{r.status_code}",
            "debug": {"http_status": r.status_code, "body_head": (r.text or "")[:300]},
        }

    data = r.json()

    # SerpAPI 出错时常会返回 error 字段
    if "error" in data:
        return {"results": [], "error": "serpapi_error", "debug": {"message": data.get("error")}}

    results: List[Dict[str, Any]] = []
    for item in (data.get("organic_results") or [])[:top_k]:
        results.append({
            "title": item.get("title", ""),
            "url": item.get("link", ""),
            "snippet": item.get("snippet", ""),
            "source": "serpapi",
        })
    return {"results": results}



# =========================
# Tool 5: Aggregated web search (recommended for the agent)
# =========================
def tool_web_search(query: str, top_k: int = 5) -> Dict[str, Any]:
    """
    Combined search:
      1) Wikipedia
      2) Serper
      3) Baidu (best-effort; may captcha)

    Returns: results + source_used + debug on failure.
    """
    # 2) Serper
    s = tool_serpapi_search(query, top_k=top_k)
    if s.get("results"):
        s["source_used"] = "serper"
        return s

    # 3) Baidu
    b = tool_baidu_search(query, top_k=top_k)
    if b.get("results"):
        b["source_used"] = "baidu"
        return b
    # 1) Wikipedia
    w = tool_wiki_search(query, top_k=min(top_k, 5))
    if w.get("results"):
        w["source_used"] = "wikipedia"
        return w

    return {
        "results": [],
        "error": "no_search_source_available",
        "debug": {
            "serper": {k: s.get(k) for k in ["error", "debug"]},
            "baidu": {k: b.get(k) for k in ["error", "debug"]},
            "wiki": {k: w.get(k) for k in ["error", "debug"]},
        },
    }


# =========================
# Tool 6: Planner (deterministic)
# =========================
def tool_planner(goal: str, constraints: str = "", tools_available: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    A lightweight non-LLM planner.
    It returns a plan the LLM can follow, with tool hints.
    """
    # Default to FULL set (important)
    tools_available = tools_available or ["planner", "calc", "wiki_search", "baidu_search", "serper_search", "web_search"]

    g = (goal or "").strip()
    c = (constraints or "").strip()

    steps: List[Dict[str, Any]] = []
    steps.append({
        "step": 1,
        "action": "Restate the goal in one sentence and identify what facts or computations are needed.",
        "tool_hint": None
    })

    # Simple heuristics
    needs_math = any(k in g.lower() for k in ["compute", "calculate", "math", "sum", "percentage", "times", "divide"])
    needs_search = any(k in g.lower() for k in [
        "who is", "what is", "capital", "definition", "history", "cost", "price",
        "travel", "itinerary", "budget", "average", "per day", "tips"
    ])

    # Prefer web_search if available
    if needs_search and "web_search" in tools_available:
        steps.append({
            "step": len(steps) + 1,
            "action": "Gather key facts via web_search.",
            "tool_hint": {"tool": "web_search", "args": {"query": "keywords from the question", "top_k": 5}}
        })
    elif needs_search and "wiki_search" in tools_available:
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
            "serper_search": tool_serpapi_search,
            "web_search": tool_web_search,
            "planner": tool_planner,
        }

    def allowed_tools(self) -> List[str]:
        return list(self._fns.keys())

    # def call_with_timeout(self, name: str, args: Dict[str, Any], timeout_s: float) -> Tuple[Dict[str, Any], str]:
    #     if name not in self._fns:
    #         return {}, f"unknown_tool({name})"

    #     fn = self._fns[name]

    #     def run():
    #         return fn(**args)

    #     with ThreadPoolExecutor(max_workers=1) as ex:
    #         fut = ex.submit(run)
    #         try:
    #             out = fut.result(timeout=timeout_s)
    #             if not isinstance(out, dict):
    #                 return {}, f"tool_output_not_dict({name})"
    #             return out, ""
    #         except FuturesTimeoutError:
    #             return {}, f"tool_timeout({name})"
    #         except TypeError as e:
    #             return {}, f"bad_args({name}): {str(e)}"
    #         except Exception as e:
    #             return {}, f"tool_failed({name}): {type(e).__name__}: {str(e)}"
    def call_with_timeout(self, name: str, args: Dict[str, Any], timeout_s: float):
        if name not in self._fns:
            return {}, f"unknown_tool({name})"
        try:
            out = self._fns[name](**args)
            if not isinstance(out, dict):
                return {}, f"tool_output_not_dict({name})"
            return out, ""
        except TypeError as e:
            return {}, f"bad_args({name}): {str(e)}"
        except Exception as e:
            return {}, f"tool_failed({name}): {type(e).__name__}: {str(e)}"


# =========================
# Qwen tools schema (OpenAI-compatible function calling format)
# =========================
def qwen_tools_schema() -> List[Dict[str, Any]]:
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
                "description": "Search via Baidu (HTML). May be blocked by captcha. Returns title/url/snippet if available.",
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
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Aggregated search: tries Wikipedia, then Serper, then Baidu. Returns results + source_used.",
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
