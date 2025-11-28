from dataclasses import dataclass
import os

@dataclass
class AgentConfig:
    # Qwen OpenAI-compatible
    api_key: str = os.getenv("DASHSCOPE_API_KEY", "")
    base_url: str = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    model: str = os.getenv("QWEN_MODEL", "qwen-plus")

    temperature: float = 0.2

    # Guardrails
    recursion_limit: int = 20     # LangGraph max super-steps
    max_tool_calls: int = 6       # Hard limit on tool executions
    tool_timeout_s: float = 6.0   # Per-tool timeout
    max_repeat_same_call: int = 2 # Repeat same tool call signature limit

     # Self-consistency
    consistency_k: int = 3       # Number of traces to run (majority vote)

    # Safety
    enable_safety_filter: bool = True

    # Logging
    trace_path: str = "trace.jsonl"
