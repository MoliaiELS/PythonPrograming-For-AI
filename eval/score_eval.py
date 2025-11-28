import json
import re
from pathlib import Path
from dotenv import load_dotenv

from src.config import AgentConfig
from src.graph import build_app

def norm(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def main():
    load_dotenv()
    cfg = AgentConfig(trace_path="eval_trace.jsonl")
    run = build_app(cfg)

    items = [json.loads(x) for x in Path("eval/eval_set.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]

    ok = 0
    for ex in items:
        res = run(ex["question"])
        pred = norm(res["answer"])

        if ex["type"] == "calc":
            gold = norm(ex["gold"])
            passed = (pred == gold) or (gold in pred)
        else:
            keys = [norm(k) for k in ex["gold_contains"]]
            passed = all(k in pred for k in keys)

        ok += 1 if passed else 0
        print(ex["id"], "OK" if passed else "FAIL")
        print("pred:", res["answer"])
        print()

    print("Accuracy:", ok / max(1, len(items)))

if __name__ == "__main__":
    main()
