import os
from openai import OpenAI

from src.config import AgentConfig

def build_qwen_client(cfg: AgentConfig) -> OpenAI:
    """
    Qwen API in OpenAI-compatible mode.
    base_url:
      Singapore: https://dashscope-intl.aliyuncs.com/compatible-mode/v1
      Beijing:   https://dashscope.aliyuncs.com/compatible-mode/v1
    """
    if not cfg.api_key:
        raise RuntimeError("Missing DASHSCOPE_API_KEY in environment")

    return OpenAI(
        api_key=cfg.api_key,
        base_url=cfg.base_url,
    )
