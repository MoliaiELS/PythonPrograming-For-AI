import json
import re
import time
import sys
from pathlib import Path
from dotenv import load_dotenv

# Try to import tqdm, if not available, use a dummy wrapper
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

# Add project root to sys.path so we can import src
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Load env vars BEFORE importing config so that default values are picked up
load_dotenv()

from src.config import AgentConfig
from src.graph import build_app
from src.qwen_client import build_qwen_client

def norm(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def llm_judge(question: str, gold: str, pred: str, client, model: str) -> bool:
    """
    Use LLM as a judge to determine if the predicted answer is correct based on the gold answer.
    """
    prompt = f"""
    You are an impartial judge.
    Question: {question}
    Gold Answer: {gold}
    Predicted Answer: {pred}

    Does the Predicted Answer convey the same meaning or answer the question correctly according to the Gold Answer? 
    Ignore minor formatting differences.
    Reply with only "YES" or "NO".
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        content = response.choices[0].message.content.strip().upper()
        return "YES" in content
    except Exception as e:
        print(f"Judge Error: {e}")
        # Fallback to simple inclusion check
        return norm(gold) in norm(pred)

def main():
    load_dotenv()
    
    # 1. Initialize
    # We use a separate trace file for evaluation to avoid cluttering the main logs
    cfg = AgentConfig(trace_path="eval_trace.jsonl")
    
    # Force consistency_k=1 for evaluation to speed it up
    cfg.consistency_k = 1

    # Build the agent runner
    run_agent = build_app(cfg) 
    
    # Build the judge client
    judge_client = build_qwen_client(cfg)

    # Read evaluation set
    eval_file = Path("eval/eval_set.jsonl")
    if not eval_file.exists():
        print(f"Error: {eval_file} not found.")
        return
        
    items = [json.loads(x) for x in eval_file.read_text(encoding="utf-8").splitlines() if x.strip()]
    
    results = []
    ok_count = 0

    print(f"Starting evaluation on {len(items)} items...")

    # 2. Loop through items
    for ex in tqdm(items):
        question = ex["question"]
        gold = ex.get("gold", "")
        q_type = ex.get("type", "general")
        
        start_time = time.time()
        pred = ""
        passed = False
        error_msg = None

        try:
            # --- Run Agent ---
            # The run_agent function returns a dict with "answer"
            res = run_agent(question)
            pred = res.get("answer", "")

        except Exception as e:
            error_msg = str(e)
            pred = "[ERROR]"
            print(f"Error running agent on id {ex.get('id')}: {e}")
        
        duration = time.time() - start_time

        # --- Scoring Logic ---
        if error_msg:
            passed = False
        elif q_type == "calc":
            # For calculation, we usually want exact match or the number to be present
            # Normalizing helps with spaces and case
            passed = norm(str(gold)) in norm(str(pred))
        else:
            # For text/search questions
            if "gold_contains" in ex:
                # If explicit keywords are provided, check them all
                keys = [norm(k) for k in ex["gold_contains"]]
                passed = all(k in norm(pred) for k in keys)
            else:
                # Otherwise use LLM Judge
                # Use "qwen-turbo" as a lightweight evaluator to speed up judging
                passed = llm_judge(question, gold, pred, judge_client, model="qwen-turbo")

        if passed:
            ok_count += 1

        # Record result
        results.append({
            "id": ex.get("id"),
            "type": q_type,
            "question": question,
            "gold": gold if gold else ex.get("gold_contains"),
            "pred": pred,
            "passed": passed,
            "duration": round(duration, 2),
            "error": error_msg
        })

    # 3. Output Statistics
    accuracy = ok_count / max(1, len(items))
    print(f"\n════════════════════════════════")
    print(f"Evaluation Complete")
    print(f"Accuracy: {accuracy:.2%} ({ok_count}/{len(items)})")
    print(f"════════════════════════════════")

    # Print detailed results to console instead of saving to file
    print("\nDetailed Results:")
    for res in results:
        status = "PASS" if res["passed"] else "FAIL"
        print(f"[{status}] ID: {res['id']}")
        print(f"  Q: {res['question']}")
        print(f"  Gold: {res['gold']}")
        print(f"  Pred: {res['pred']}")
        if res["error"]:
            print(f"  Error: {res['error']}")
        print("-" * 40)

if __name__ == "__main__":
    main()
