"""Read Johnny's logs in a desktop window; no robot or camera access.

Run: .venv/bin/python scripts/johnny_monitor.py
"""

from __future__ import annotations

import argparse
import json
import re
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/johnny"


def running_johnny() -> tuple[int, Path | None] | None:
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            args = (proc / "cmdline").read_bytes().split(b"\0")
            if b"-m" not in args or b"dlb.johnny" not in args or (proc / "cwd").resolve() != ROOT:
                continue
            for fd in ("1", "2"):
                target = (proc / "fd" / fd).resolve()
                if target.is_file():
                    return int(proc.name), target
            return int(proc.name), None
        except OSError:
            continue
    return None


def newest_log() -> Path | None:
    """Johnny's own log (results/johnny/johnny_<date>.log), for a Johnny whose output goes to a terminal."""
    logs = sorted(OUT.glob("johnny_*.log"))
    return logs[-1] if logs else None


class Tail:
    """Follow complete UTF-8 lines, including file replacement and truncation."""

    def __init__(self):
        self.identity = None
        self.offset = 0
        self.pending = b""

    def read(self, path: Path) -> tuple[bool, list[str]]:
        with path.open("rb") as stream:
            import os

            stat = os.fstat(stream.fileno())
            identity = (str(path), stat.st_dev, stat.st_ino)
            changed = identity != self.identity or stat.st_size < self.offset
            if changed:
                self.identity, self.pending = identity, b""
                self.offset = max(0, stat.st_size - 256 * 1024)
                stream.seek(self.offset)
                if self.offset:
                    stream.readline()  # discard an incomplete first line
                    self.offset = stream.tell()
            stream.seek(self.offset)
            data = self.pending + stream.read(256 * 1024)
            self.offset = stream.tell()
        *lines, self.pending = data.split(b"\n")
        return changed, [line.decode("utf-8", errors="replace") for line in lines]


def console_event(line: str) -> tuple[str, str] | None:
    line = line.strip()
    if line.startswith("[heard "):
        return "heard", "認識した音声：" + line.split("] ", 1)[-1]
    if line.startswith("Johnny:"):
        return "reply", "Johnny：" + line.removeprefix("Johnny:").strip()
    if line.startswith("listening..."):
        return "info", "音声待ち受けを開始しました。"
    match = re.match(r"(\d\d:\d\d:\d\d) (.*)", line)
    if not match:
        return None
    stamp, message = match.groups()
    for prefix, kind, label in (
        ("command (", "command", "依頼を認識"),
        ("ignored (", "ignored", "受付せず"),
        ("wake word (", "command", "呼びかけ"),
        ("stop word:", "command", "停止依頼"),
        ("skill ", "skill", "実行"),
        ("request failed:", "error", "エラー"),
        ("started ", "info", "起動"),
        ("stopped ", "info", "停止"),
        ("shutdown ", "info", "終了処理"),
    ):
        if message.startswith(prefix):
            if prefix == "skill " and " -> " in message:
                kind = "error" if "ok=False" in message else "success"
                label = "実行失敗" if kind == "error" else "実行結果"
            return kind, f"{stamp}  {label}：{message}"
    return None


def history_event(line: str) -> tuple[str, str] | None:
    try:
        record = json.loads(line)
        item = record["item"]
        stamp = datetime.fromtimestamp(record["t"]).strftime("%m/%d %H:%M:%S")
        role = item.get("role")
        if role in ("user", "assistant"):
            text = "\n".join(part.get("text", "") for part in item.get("content", []))
            return (
                "command" if role == "user" else "reply"
            ), f"{stamp}  {'あなた' if role == 'user' else 'Johnny'}：{text}"
        if item.get("type") == "function_call":
            return "skill", f"{stamp}  実行：{item['name']} {item.get('arguments', '')}"
        if item.get("type") == "function_call_output":
            text = str(item.get("output", ""))
            return ("error" if text.startswith("[failed]") else "info"), f"{stamp}  結果：{text}"
    except (ValueError, KeyError, TypeError, AttributeError, OverflowError, OSError):
        return None
    return None


class Monitor:
    def __init__(self, root: tk.Tk, log: Path | None = None):
        self.root, self.explicit_log = root, log
        self.console_tail, self.history_tail = Tail(), Tail()
        self.log = log
        root.title("Johnny — 会話ログ")
        root.geometry("1000x680")
        self.status = tk.StringVar(value="起動状態を確認中…")
        self.last = tk.StringVar(value="直近の実行結果：記録待ち")
        ttk.Label(root, textvariable=self.status, font=("", 14, "bold")).pack(
            anchor="w", padx=16, pady=(12, 6)
        )
        ttk.Label(root, textvariable=self.last, wraplength=960).pack(anchor="w", padx=16, pady=(0, 8))
        self.follow = tk.BooleanVar(value=True)
        ttk.Checkbutton(root, text="新着へ自動スクロール", variable=self.follow).pack(anchor="w", padx=16)
        tabs = ttk.Notebook(root)
        tabs.pack(fill="both", expand=True, padx=12, pady=8)
        self.console = self.make_text(tabs, "現在の音声・実行")
        self.history = self.make_text(tabs, "会話履歴")
        self.source = tk.StringVar()
        ttk.Label(root, textvariable=self.source, wraplength=960).pack(anchor="w", padx=16)
        ttk.Label(root, text="表示はログに基づきます。アームの現在状態は計測していません。").pack(
            anchor="w", padx=16, pady=(4, 10)
        )
        self.poll()

    @staticmethod
    def make_text(tabs: ttk.Notebook, name: str) -> ScrolledText:
        frame = ttk.Frame(tabs)
        tabs.add(frame, text=name)
        text = ScrolledText(frame, wrap="word", font=("", 12), padx=12, pady=10, state="disabled")
        text.pack(fill="both", expand=True)
        for tag, color in {
            "heard": "#536375",
            "command": "#125dbe",
            "reply": "#176943",
            "ignored": "#777777",
            "skill": "#805000",
            "success": "#176943",
            "error": "#be2830",
            "info": "#536375",
        }.items():
            text.tag_config(tag, foreground=color)
        return text

    def append(self, widget: ScrolledText, kind: str, text: str):
        widget.configure(state="normal")
        widget.insert("end", text + "\n\n", kind)
        if int(widget.index("end-1c").split(".")[0]) > 3000:
            widget.delete("1.0", "500.0")
        widget.configure(state="disabled")
        if self.follow.get():
            widget.see("end")

    def poll(self):
        try:
            running = running_johnny()
            self.status.set(f"Johnny 起動中 · PID {running[0]}" if running else "Johnny は停止しています")
            if not self.explicit_log and running:
                self.log = running[1] or newest_log()
            if self.log and self.log.exists():
                changed, lines = self.console_tail.read(self.log)
                if changed:
                    self.append(self.console, "info", "── " + self.log.name + " ──")
                    self.last.set("直近の実行結果：記録待ち")
                self.source.set(str(self.log))
                for line in lines:
                    item = console_event(line)
                    if item:
                        self.append(self.console, *item)
                        if item[0] in ("skill", "success", "error"):
                            self.last.set("ログ上の直近の実行：" + item[1])
            else:
                self.source.set("音声の詳細ログは未接続です。「会話履歴」タブを確認してください。")
            history = OUT / "history.jsonl"
            if history.exists():
                _, lines = self.history_tail.read(history)
                for line in lines:
                    item = history_event(line)
                    if item:
                        self.append(self.history, *item)
        except OSError as exc:
            self.source.set("ログの読み取りを再試行します：" + str(exc))
        finally:
            self.root.after(500, self.poll)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, help="指定した実行ログを表示")
    args = parser.parse_args()
    root = tk.Tk()
    Monitor(root, args.log)
    root.mainloop()


if __name__ == "__main__":
    main()
