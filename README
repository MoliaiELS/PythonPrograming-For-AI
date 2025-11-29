# AIAA2290 (L02) Final Project (B)

#### Project Overview

This project implements a LangGraph based ReAct style agent using Qwen as the LLM. The agent supports a tool use loop, guardrails, and structured logging, and is designed to meet the core requirements of Project B (excluding advanced direction and evaluation parts, which are handled by other teammates).

---

### Team Members and Responsibilities

**Guo Ye (郭烨)**
Responsible for the ReAct main framework, guardrails, and logging system.

**Luo Xueyin (罗雪尹)**
Responsible for evaluation testing and the final report.

**Xie Yanzhe (谢彦哲)**
Responsible for the advanced direction implementation.

---

### How to Use

#### Environment Setup

1. Create a Python environment (recommended: conda)

   * Python version: 3.10 or 3.11 recommended

2. Install dependencies

   1. Make sure you are in the project root directory
   2. Install from `requirements.txt`

   ```bash
   pip install -r requirements.txt
   ```

#### 2.2 API Key Setup (.env)

This project requires API keys for LLM and optional search tools. 

1. Paste the relevant api key in `.env` 

   ```bash
    SERPAPI_API_KEY= "your_api_key"

    # DashScope API key (Qwen)
    DASHSCOPE_API_KEY="sk-your_api_key"

    # Singapore region base_url 
    DASHSCOPE_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"

    # Model suggestion: qwen-plus supports tools/function calling
    QWEN_MODEL="qwen-plus"
   ```

2. For `SERPAPI_API_KEY`, you can get a free key from [SerpAPI](https://serpapi.com/manage-api-key).
3. For `DASHSCOPE_API_KEY` and `DASHSCOPE_API_KEY`, you can get a free key from [DashScope](https://bailian.console.aliyun.com/?tab=api#/api/?type=model&url=2712195).

#### 2.3 Running the Agent

Run from the project root:

```bash
python run_agent.py
```

After launching, you will see an interactive CLI. Type your question after `Question >`.

```bash
════════════════════════════════════════════════════════════════════════
  LangGraph + Qwen ReAct Agent
════════════════════════════════════════════════════════════════════════
Ask a question. Commands: /help  /exit  /debug on|off  /clear
Logs are saved automatically.
Log dir: log\20251128_194618
Trace:   log\20251128_194618\trace.jsonl
────────────────────────────────────────────────────────────────────────

Question > 
```

Useful commands:

* `/help`
* `/exit`
* `/debug on` or `/debug off`
* `/clear`

Logs are saved automatically under the `log/` directory, including a `trace.jsonl` file for each run session.

#### 2.4 Run evaluation script

Use 'Qwen-turbo' to do automatic evaluation:

```bash
#under project root
python eval/score_eval.py
```
---

#### 2.4 Modifying Configuration

Configuration is defined in `config.py`. Common settings include:

```python
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

    # Logging
    trace_path: str = "trace.jsonl"
```

---

### 3. Project File Structure

A typical structure looks like:

```
.
├── run_agent.py                  # CLI entry point

├── .env                          # API keys (user must create)
├── requirements.txt              # Dependencies
├── src/
│   ├── graph.py                  # LangGraph agent graph and routing logic
│   ├── qwen_client.py            # Qwen client wrapper
│   ├── tools.py                  # Tool registry + tool implementations
│   └── config.py                 # Agent configuration
├── log/
│   └── YYYYMMDD_HHMMSS/
│       └── trace.jsonl           # Structured trace logs
├── eval/
│   ├── eval_set.jsonl            # Evaluation question set
│   └── score_eval.py             # Evaluation script
└── README.md                     # This file
```


---

### 4. Implementation Details

#### 4.1 ReAct Agent Loop (Core)

* Built using LangGraph.
* The agent follows a loop:

  1. Model generates Thought + Action + Action Input
  2. Router detects tool usage
  3. Tool node runs the tool and returns Observation
  4. Observation is appended back to messages
  5. Loop continues until Final answer or guardrail stop

Supports both:

* Native `tool_calls` (function calling)
* Text-based `Action:` `Action Input:` parsing fallback

#### 4.2 Tool System

* Tools are registered in a `ToolRegistry`.
* Each tool has:

  * Name
  * Schema / argument JSON
  * Python implementation returning a JSON dict

#### 4.3 Guardrails

Implemented guardrails include:

* Maximum recursion limit (prevents infinite loop)
* Maximum number of tool calls
* Tool timeout (prevents a tool from stalling the agent)
* Unknown tool handling
* Bad arguments handling
* Malformed JSON handling

#### 4.4 Logging and Tracing

Each run creates a timestamped log folder:

* `trace.jsonl` records events in order
* Event types typically include:

  * `model`: stores the model output, parsed thought, action, tool call info
  * `tool`: stores the tool name, args, and observation output/error

In debug mode, the CLI can print tool observations to the console.

---

### 5. To Be Completed (Placeholders for Teammates)

#### 5.1 Evaluation (Luo Xueyin)

* [ ] Evaluation script location:
* [ ] Test set description:
* [ ] Metrics:
* [ ] Results summary:
* [ ] Reproducibility notes:

#### 5.2 Advanced Direction (Xie Yanzhe)

* [ ] Advanced feature name:
* [ ] Design motivation:
* [ ] Implementation details:
* [ ] How to run advanced mode:
* [ ] Known issues / limitations:

#### 5.3 Final Report

* [ ] Report file name:
* [ ] Key findings:
* [ ] Screenshots / example conversations:
* [ ] Self reflection / limitations:

---

### 6. Notes and Troubleshooting

#### 6.1 No Tool Observation Appears in Console

* Turn on debug:

  ```text
  /debug on
  ```
* Observations are always written to `trace.jsonl` even if not printed.

#### 6.2 Agent Stops After Printing Action

This usually happens when:

* The model outputs text-based Action but tool routing is not enabled
* Or tool call parsing fails due to malformed JSON

Check `trace.jsonl` to see the final router decision and any parsing error.

#### 6.3 API Errors

If you see auth errors:

* Verify `.env` keys exist and are correct
* Ensure you restarted the terminal after editing `.env`

---

### 7. License

(Leave blank or fill based on course requirement.)
