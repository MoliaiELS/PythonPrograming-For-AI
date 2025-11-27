import json
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, List
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

"""
tool struct and registry
"""
@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: Dict[str, Any]
    fn: Callable[..., Any]
    timeout_s: float = 3.0


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, ToolSpec] = {}

    def register(self, tool: ToolSpec) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[ToolSpec]:
        return self._tools.get(name)

    def list_for_prompt(self) -> str:
        lines: List[str] = []
        for t in self._tools.values():
            lines.append(f"Tool: {t.name}\nDescription: {t.description}\nInput schema(JSON): {json.dumps(t.input_schema)}\n")
        return "\n".join(lines)


def safe_json_loads(text: str) -> Dict[str, Any]:
    """
    Strict JSON parser with a small recovery:
    if model wraps JSON with extra text, try extracting the outermost {...}.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        l = text.find("{")
        r = text.rfind("}")
        if l != -1 and r != -1 and r > l:
            return json.loads(text[l:r+1])
        raise


def run_tool_with_timeout(tool: ToolSpec, args: Dict[str, Any]) -> Any:
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(tool.fn, **args)
        return fut.result(timeout=tool.timeout_s)


def build_system_prompt(registry: ToolRegistry) -> str:
    # ReAct spec: thought -> action -> observation -> final
    return (
        "You are a tool-using agent using the ReAct pattern.\n"
        "You must respond ONLY with a single JSON object each turn.\n"
        "Choose exactly one:\n"
        "1) Call a tool: {\"thought\": \"...\", \"action\": {\"tool\": \"NAME\", \"args\": {...}}, \"final\": null}\n"
        "2) Finish: {\"thought\": \"...\", \"action\": null, \"final\": \"...\"}\n\n"
        "Available tools:\n"
        f"{registry.list_for_prompt()}\n\n"
        "Rules:\n"
        "If you call a tool, wait for the Observation next turn and then continue.\n"
        "If you receive an ERROR Observation, fix the tool name or args and try again.\n"
    )


def llm_generate(messages: List[Dict[str, str]]) -> str:
    """
    Replace this with your actual model call.
    Must return a string that is a JSON object as specified.
    """
    raise NotImplementedError


class ReactAgent:
    def __init__(self, registry: ToolRegistry, max_steps: int = 8):
        self.registry = registry
        self.max_steps = max_steps

    def run(self, user_query: str) -> Dict[str, Any]:
        system = build_system_prompt(self.registry)
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_query},
        ]

        trace: List[Dict[str, Any]] = []

        for step in range(self.max_steps):
            raw = llm_generate(messages)

            try:
                obj = safe_json_loads(raw)
            except Exception as e:
                obs = f"ERROR: bad_json ({type(e).__name__})"
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content": f"Observation: {obs}"})
                trace.append({"step": step, "raw": raw, "observation": obs})
                continue

            thought = obj.get("thought", "")
            action = obj.get("action")
            final = obj.get("final")

            trace.append({"step": step, "thought": thought, "action": action, "final": final})

            if final is not None and (action is None):
                return {"final": final, "trace": trace}

            if not isinstance(action, dict) or "tool" not in action or "args" not in action:
                obs = "ERROR: bad_action_format"
                messages.append({"role": "assistant", "content": json.dumps(obj)})
                messages.append({"role": "user", "content": f"Observation: {obs}"})
                trace.append({"step": step, "observation": obs})
                continue

            tool_name = action["tool"]
            args = action["args"]

            tool = self.registry.get(tool_name)
            if tool is None:
                obs = f"ERROR: unknown_tool ({tool_name})"
            elif not isinstance(args, dict):
                obs = "ERROR: bad_args (args must be JSON object)"
            else:
                try:
                    result = run_tool_with_timeout(tool, args)
                    obs = f"{result}"
                except FuturesTimeout:
                    obs = f"ERROR: tool_timeout ({tool_name})"
                except TypeError as e:
                    obs = f"ERROR: bad_args ({str(e)})"
                except Exception as e:
                    obs = f"ERROR: tool_failed ({type(e).__name__}: {str(e)})"

            messages.append({"role": "assistant", "content": json.dumps(obj)})
            messages.append({"role": "user", "content": f"Observation: {obs}"})
            trace.append({"step": step, "observation": obs})

        return {
            "final": f"Stopped: exceeded max_steps={self.max_steps}. See trace for details.",
            "trace": trace
        }


# Example tool: safe calculator for basic arithmetic only
import ast
import operator as op

_ALLOWED = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.USub: op.neg,
}

def _eval_expr(node):
    if isinstance(node, ast.Num):
        return node.n
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED:
        return _ALLOWED[type(node.op)](_eval_expr(node.left), _eval_expr(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED:
        return _ALLOWED[type(node.op)](_eval_expr(node.operand))
    raise ValueError("disallowed expression")

def calc(expression: str) -> float:
    tree = ast.parse(expression, mode="eval")
    return float(_eval_expr(tree.body))


def build_default_registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(ToolSpec(
        name="calculator",
        description="Evaluate basic arithmetic expression, for example 2*(3+4).",
        input_schema={"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]},
        fn=calc,
        timeout_s=1.0
    ))
    return reg
