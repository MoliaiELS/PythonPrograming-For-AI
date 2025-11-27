# run_agent.py
from dotenv import load_dotenv

from src.config import AgentConfig
from src.tools import ToolRegistry, ToolSpec, calc_tool, wiki_search_tool, CALC_SCHEMA_IN, CALC_SCHEMA_OUT, WIKI_SCHEMA_IN, WIKI_SCHEMA_OUT
from src.react_agent import ReactAgent

def build_registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(ToolSpec(
        name="calc",
        description="Safely evaluate a basic arithmetic expression. Example: 19.99 * 3",
        schema_in=CALC_SCHEMA_IN,
        schema_out=CALC_SCHEMA_OUT,
        fn=calc_tool
    ))
    reg.register(ToolSpec(
        name="wiki_search",
        description="Search Wikipedia (English) and return top hits with snippets.",
        schema_in=WIKI_SCHEMA_IN,
        schema_out=WIKI_SCHEMA_OUT,
        fn=wiki_search_tool
    ))
    return reg

def main():
    load_dotenv()
    cfg = AgentConfig()
    tools = build_registry()
    agent = ReactAgent(cfg, tools)

    print("Type your question. Ctrl+C to exit.")
    while True:
        q = input("\n> ").strip()
        if not q:
            continue
        result = agent.run(q)
        print("\nAnswer:\n", result.answer)

if __name__ == "__main__":
    main()
