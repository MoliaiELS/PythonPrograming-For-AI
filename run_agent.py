# run_agent.py
from dotenv import load_dotenv

def main():
    load_dotenv(override=True)  # 强制覆盖已有环境变量
    from src.config import AgentConfig
    from src.graph import build_app
    
    import os
    # print("DASHSCOPE_API_KEY exists?", bool(os.getenv("DASHSCOPE_API_KEY")))
    # print("DASHSCOPE_API_KEY head:", (os.getenv("DASHSCOPE_API_KEY") or "")[:6])

    cfg = AgentConfig()
    run = build_app(cfg)

    print("LangGraph + Qwen ReAct Agent. Ask about your holiday plan! Ctrl+C to exit.")
    while True:
        q = input("\nQuestion: ").strip()
        if not q:
            continue
        result = run(q)
        print("\nAnswer:\n", result["answer"])
        print("\nTrace written to:", cfg.trace_path)

if __name__ == "__main__":
    main()
