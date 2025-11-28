# run_agent.py
from __future__ import annotations

import os
import sys
import time
import threading
import argparse
import json
from dataclasses import dataclass
from dotenv import load_dotenv


# -------------------------
# Small CLI helpers
# -------------------------
def _supports_color() -> bool:
    if sys.platform == "win32":
        # Windows Terminal / new consoles usually support ANSI; cmd sometimes not.
        return bool(os.getenv("WT_SESSION")) or "TERM" in os.environ
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

def warn(text: str) -> str:
    return _c(text, "33")

def ok(text: str) -> str:
    return _c(text, "32")

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
            # clear spinner line
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


@dataclass
class UiState:
    debug: bool = False
    last_trace_path: str = ""


HELP_TEXT = """
Commands:
  /help            Show this help
  /exit            Exit the program
  /clear           Clear the screen
  /debug on|off     Toggle debug output
  /trace           Print trace file path
Tips:
  - You can also press Ctrl+C to exit, Ctrl+D to quit (Unix/macOS).
""".strip()


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI colors")
    args, _unknown = parser.parse_known_args()

    global USE_COLOR
    if args.no_color:
        USE_COLOR = False

    load_dotenv(override=True)

    # Lazy imports (after dotenv)
    from src.config import AgentConfig
    from src.graph import build_app

    cfg = AgentConfig()
    run = build_app(cfg)

    ui = UiState(debug=args.debug, last_trace_path=getattr(cfg, "trace_path", ""))

    clear_screen()
    print(banner("LangGraph + Qwen ReAct Agent"))
    print(tip("Type your question. Commands: /help  /exit  /debug on|off  /clear"))
    print(tip("Exit: /exit or Ctrl+C"))
    print(_c(f"Trace file: {ui.last_trace_path}", "2"))
    print(_c(hr(), "36"))

    while True:
        try:
            q = input(_c("\nQuestion > ", "1")).strip()
        except EOFError:
            print("\n" + tip("EOF received. Bye!"))
            break
        except KeyboardInterrupt:
            print("\n" + tip("Interrupted. Bye!"))
            break

        if not q:
            continue

        if q.startswith("/"):
            parts = q.split()
            cmd = parts[0].lower()

            if cmd in ("/exit", "/quit", "/q"):
                print(tip("Bye!"))
                break

            if cmd == "/help":
                print(_c(HELP_TEXT, "2"))
                continue

            if cmd == "/clear":
                clear_screen()
                print(banner("LangGraph + Qwen ReAct Agent"))
                print(tip("Type your question. Commands: /help  /exit  /debug on|off  /clear"))
                print(_c(f"Trace file: {ui.last_trace_path}", "2"))
                print(_c(hr(), "36"))
                continue

            if cmd == "/trace":
                print(_c(f"Trace file: {ui.last_trace_path}", "2"))
                continue

            if cmd == "/debug":
                if len(parts) == 1:
                    print(_c(f"Debug is {'ON' if ui.debug else 'OFF'}. Use /debug on|off", "2"))
                else:
                    val = parts[1].lower()
                    if val in ("on", "1", "true", "yes"):
                        ui.debug = True
                        print(ok("Debug: ON"))
                    elif val in ("off", "0", "false", "no"):
                        ui.debug = False
                        print(ok("Debug: OFF"))
                    else:
                        print(warn("Usage: /debug on|off"))
                continue

            print(warn("Unknown command. Try /help"))
            continue

        spinner = Spinner(message="Waiting")
        try:
            spinner.start()
            t0 = time.time()
            result = run(q)
            dt = time.time() - t0
        except KeyboardInterrupt:
            spinner.stop()
            print("\n" + warn("Cancelled."))
            continue
        except Exception as e:
            spinner.stop()
            print("\n" + err("Agent error: ") + str(e))
            if ui.debug:
                import traceback
                print(_c(traceback.format_exc(), "2"))
            continue
        finally:
            spinner.stop()

        answer = (result or {}).get("answer", "")
        ui.last_trace_path = getattr(cfg, "trace_path", ui.last_trace_path)

        # Pretty output
        print("\n" + _c(hr("─"), "36"))
        print(_c("Answer", "1;36") + _c(f"  ({dt:.2f}s)", "2"))
        print(_c(hr("─"), "36"))
        print(answer.strip() if answer else warn("(empty answer)"))
        print(_c(hr("─"), "36"))
        print(_c(f"Trace written to: {ui.last_trace_path}", "2"))

        # Optional debug dump (safe and short)
        if ui.debug:
            print(_c("\n[debug] raw result keys: " + ", ".join(sorted((result or {}).keys())), "2"))
            # If you store tool/debug info in result, print a short view:
            for k in ("debug", "tool_errors", "used_tools"):
                if k in (result or {}):
                    v = result.get(k)
                    s = json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else v
                    print(_c(f"[debug] {k}: {s[:800]}", "2"))

    print(_c(hr("═"), "36"))


if __name__ == "__main__":
    main()
