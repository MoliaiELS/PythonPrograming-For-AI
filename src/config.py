# src/config.py
from dataclasses import dataclass

@dataclass
class AgentConfig:
    model: str = "gpt-4.1"  # 也可换成 gpt-4o 等，看你额度和效果
    temperature: float = 0.2
    max_steps: int = 8                # 护栏 1: 防止无限循环
    tool_timeout_s: float = 8.0       # 护栏 2: tool 超时
    llm_timeout_s: float = 60.0       # API 请求超时（客户端层）
    trace_path: str = "trace.jsonl"   # 调试用 trace
