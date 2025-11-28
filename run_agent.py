# run_agent.py
from __future__ import annotations

import os
import sys
import time
import json
import argparse
import logging
import threading
from typing import Dict
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv


# -------------------------
# Small CLI helpers
# -------------------------
def _supports_color() -> bool:
    if sys.platform == "win32":
        return bool(os.getenv("WT_SESSION")) or ("TERM" in os.environ)
    return sys.stdout.isatty()

USE_COLOR = _supports_color()

def _c(text: str, code: str) -> str:
    if not USE_COLOR:
        return text
    return f"\033[{code}m{text}\033[0m"

def hr(char: str = "─", n: int = 72) -> str:
    return char * n

def banner(title: str) -> str:
    return "\n".join([
        _c(hr("═"), "36"),
        _c(f"  {title}", "1;36"),
        _c(hr("═"), "36"),
    ])

def tip(text: str) -> str:
    return _c(text, "2")

def ok(text: str) -> str:
    return _c(text, "32")

def warn(text: str) -> str:
    return _c(text, "33")

def err(text: str) -> str:
    return _c(text, "31")

def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


class Spinner:
    def __init__(self, message: str = "Waiting", interval: float = 0.1):
        self.message = message
        self.interval = interval
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        if not sys.stdout.isatty():
            print(f"{self.message}...", flush=True)
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if sys.stdout.isatty():
            sys.stdout.write("\r" + " " * 80 + "\r")
            sys.stdout.flush()

    def _run(self):
        chars = "|/-\\"
        i = 0
        while not self._stop.is_set():
            ch = chars[i % len(chars)]
            sys.stdout.write("\r" + _c(f"{self.message} {ch}", "2"))
            sys.stdout.flush()
            time.sleep(self.interval)
            i += 1


# -------------------------
# Logging helpers
# -------------------------
def make_session_dir(base: str = "log") -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    session_dir = Path(base) / ts
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir

def setup_logger(session_dir: Path, debug: bool) -> logging.Logger:
    logger = logging.getLogger("agent_cli")
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(session_dir / "run.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG if debug else logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # console: keep it quiet unless debug
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG if debug else logging.WARNING)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    logger.info("Session started. debug=%s", debug)
    return logger

def append_chat_md(path: Path, turn: Dict[str, object]) -> None:
    t = turn.get("time", "")
    q = turn.get("question", "")
    a = turn.get("answer", "")
    elapsed = turn.get("elapsed_s", None)
    trace_path = turn.get("trace_path", "")

    with path.open("a", encoding="utf-8") as f:
        f.write(f"\n## {t}\n\n")
        f.write("### Question\n\n")
        f.write(str(q).strip() + "\n\n")
        f.write("### Answer\n\n")
        f.write(str(a).strip() + "\n\n")
        if elapsed is not None:
            f.write(f"### Meta\n\n")
            f.write(f"- elapsed_s: {elapsed}\n")
            f.write(f"- trace_path: {trace_path}\n")

def append_jsonl(path: Path, obj: Dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# -------------------------
# UI state
# -------------------------
@dataclass
class UiState:
    debug: bool = False
    last_trace_path: str = ""
    session_dir: str = ""


HELP_TEXT = """
Commands:
  /help            Show help
  /exit            Exit
  /clear           Clear screen
  /debug on|off     Toggle debug output
  /trace           Print trace file path
  /log             Print current log directory
Tips:
  - Exit: /exit or Ctrl+C
""".strip()


def main():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI colors")
    parser.add_argument("--config", type=str, default="", help="Path to a JSON config file (optional)")
    args = parser.parse_args()

    global USE_COLOR
    if args.no_color:
        USE_COLOR = False

    load_dotenv(override=True)

    # Create per-run log directory FIRST
    session_dir = make_session_dir("log")
    logger = setup_logger(session_dir, debug=args.debug)

    # Lazy imports after dotenv
    from src.config import AgentConfig
    from src.graph import build_app

    cfg = AgentConfig()

    # Optional: load settings from a JSON config file
    # Keys match AgentConfig field names, for example:
    # {"model":"qwen-plus","temperature":0.2,"max_tool_calls":6,"tool_timeout_s":6.0}
    if args.config:
        try:
            with open(args.config, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                for k, v in data.items():
                    if hasattr(cfg, k):
                        setattr(cfg, k, v)
            else:
                logger.warning("Config file is not a JSON object: %s", args.config)
        except Exception as e:
            logger.warning("Failed to load config file %s: %s", args.config, e)


    # Force trace path into this run directory
    trace_path = session_dir / "trace.jsonl"
    if hasattr(cfg, "trace_path"):
        setattr(cfg, "trace_path", str(trace_path))
    else:
        # fallback: still keep a record, even if graph doesn't use it
        logger.warning("AgentConfig has no trace_path attribute. Trace may not be written.")

    run = build_app(cfg)

    ui = UiState(
        debug=args.debug,
        last_trace_path=str(trace_path),
        session_dir=str(session_dir),
    )

    chat_md = session_dir / "chat.md"
    chat_jsonl = session_dir / "chat.jsonl"

    clear_screen()
    print(banner("LangGraph + Qwen ReAct Agent"))
    print(tip("Ask a question. Commands: /help  /exit  /debug on|off  /clear"))
    print(tip("Logs are saved automatically."))
    print(_c(f"Log dir: {ui.session_dir}", "2"))
    print(_c(f"Trace:   {ui.last_trace_path}", "2"))
    print(_c(hr(), "36"))

    while True:
        try:
            q = input(_c("\nQuestion > ", "1")).strip()
        except EOFError:
            print("\n" + tip("EOF received. Bye!"))
            logger.info("EOF exit.")
            break
        except KeyboardInterrupt:
            print("\n" + tip("Interrupted. Bye!"))
            logger.info("KeyboardInterrupt exit.")
            break

        if not q:
            continue

        if q.startswith("/"):
            parts = q.split()
            cmd = parts[0].lower()

            if cmd in ("/exit", "/quit", "/q"):
                print(tip("Bye!"))
                logger.info("Command exit.")
                break

            if cmd == "/help":
                print(_c(HELP_TEXT, "2"))
                continue

            if cmd == "/clear":
                clear_screen()
                print(banner("LangGraph + Qwen ReAct Agent"))
                print(tip("Ask a question. Commands: /help  /exit  /debug on|off  /clear"))
                print(_c(f"Log dir: {ui.session_dir}", "2"))
                print(_c(f"Trace:   {ui.last_trace_path}", "2"))
                print(_c(hr(), "36"))
                continue

            if cmd == "/trace":
                print(_c(f"Trace: {ui.last_trace_path}", "2"))
                continue

            if cmd == "/log":
                print(_c(f"Log dir: {ui.session_dir}", "2"))
                continue

            if cmd == "/debug":
                if len(parts) == 1:
                    print(_c(f"Debug is {'ON' if ui.debug else 'OFF'}. Use /debug on|off", "2"))
                else:
                    val = parts[1].lower()
                    if val in ("on", "1", "true", "yes"):
                        ui.debug = True
                        print(ok("Debug: ON"))
                        logger.info("Debug turned ON (note: run.log level fixed at startup).")
                    elif val in ("off", "0", "false", "no"):
                        ui.debug = False
                        print(ok("Debug: OFF"))
                        logger.info("Debug turned OFF.")
                    else:
                        print(warn("Usage: /debug on|off"))
                continue

            print(warn("Unknown command. Try /help"))
            continue

        spinner = Spinner(message="Waiting")
        started = time.time()
        try:
            logger.info("Question: %s", q)
            spinner.start()
            result = run(q)
        except KeyboardInterrupt:
            spinner.stop()
            print("\n" + warn("Cancelled."))
            logger.warning("Cancelled by user during run().")
            continue
        except Exception as e:
            spinner.stop()
            print("\n" + err("Agent error: ") + str(e))
            logger.exception("Agent error")
            if ui.debug:
                import traceback
                print(_c(traceback.format_exc(), "2"))
            continue
        finally:
            spinner.stop()

        elapsed = round(time.time() - started, 3)
        answer = (result or {}).get("answer", "")

        # trace path may be updated by config; keep it in UI
        ui.last_trace_path = getattr(cfg, "trace_path", ui.last_trace_path)

        # Pretty output
        print("\n" + _c(hr("─"), "36"))
        print(_c("Answer", "1;36") + _c(f"  ({elapsed:.2f}s)", "2"))
        print(_c(hr("─"), "36"))
        print(answer.strip() if answer else warn("(empty answer)"))
        print(_c(hr("─"), "36"))
        print(_c(f"Log dir: {ui.session_dir}", "2"))
        print(_c(f"Trace:   {ui.last_trace_path}", "2"))

        # Optional: show the latest tool observation when debug is on
        if ui.debug:
            try:
                trace = (result or {}).get("trace", []) or []
                last_tool = None
                for ev in reversed(trace):
                    if isinstance(ev, dict) and ev.get("type") == "tool":
                        last_tool = ev
                        break
                if last_tool:
                    print(_c("\n[debug] latest observation", "2"))
                    print(_c(f"tool: {last_tool.get('tool', last_tool.get('action',''))}", "2"))
                    print(_c(str(last_tool.get('observation',''))[:800], "2"))
            except Exception:
                logger.exception("Failed to print latest observation")

        # Write local log for this turn
        turn = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_s": elapsed,
            "question": q,
            "answer": answer,
            "trace_path": ui.last_trace_path,
        }

        try:
            append_chat_md(chat_md, turn)
            append_jsonl(chat_jsonl, turn)
            logger.info("Turn saved. elapsed_s=%s", elapsed)
        except Exception:
            logger.exception("Failed to write chat logs")

        # Optional debug dump
        if ui.debug:
            try:
                keys = sorted((result or {}).keys())
                logger.debug("Result keys: %s", keys)
                # print short info to console
                print(_c("\n[debug] result keys: " + ", ".join(keys), "2"))
            except Exception:
                logger.exception("Debug print failed")

    logger.info("Session ended.")
    print(_c(hr("═"), "36"))


if __name__ == "__main__":
    main()
