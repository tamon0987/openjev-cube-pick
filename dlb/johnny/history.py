"""The chat history Johnny sends with every request: user utterances, assistant replies, tool calls and their
results, as Responses API input items. Kept in memory with a cap (whole user turns, oldest first) and appended
to a jsonl file so a restart continues the conversation.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any


def user_item(text: str) -> dict[str, Any]:
    return {"role": "user", "content": [{"type": "input_text", "text": text}]}


def assistant_item(text: str) -> dict[str, Any]:
    return {"role": "assistant", "content": [{"type": "output_text", "text": text}]}


def call_item(call_id: str, name: str, arguments: str) -> dict[str, Any]:
    # no "id": a function_call sent back by id must come with its reasoning item; without one it is just text
    return {"type": "function_call", "call_id": call_id, "name": name, "arguments": arguments}


def output_item(call_id: str, output: str) -> dict[str, Any]:
    return {"type": "function_call_output", "call_id": call_id, "output": output}


def item_chars(item: dict[str, Any]) -> int:
    return len(json.dumps(item, ensure_ascii=False))


class History:
    """``items`` is what goes to the model. ``path`` (jsonl, one ``{"t", "item"}`` per line) is appended on
    every ``add`` and read back by ``load``. Trimming drops the oldest user turns (a turn = a user item and
    everything up to the next one), so the input never starts in the middle of a tool call."""

    def __init__(self, path: str | Path | None = None, max_turns: int = 40, max_chars: int = 60_000):
        self.path = Path(path) if path else None
        self.max_turns, self.max_chars = max_turns, max_chars
        self.items: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    @classmethod
    def load(cls, path: str | Path, **kw: Any) -> History:
        h = cls(path, **kw)
        p = Path(path)
        for line in p.read_text().splitlines() if p.exists() else []:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(rec, dict) and isinstance(rec.get("item"), dict):
                h.items.append(rec["item"])
        h._trim()
        return h

    def add(self, item: dict[str, Any]) -> None:
        with self._lock:
            self.items.append(item)
            if self.path is not None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a") as f:
                    f.write(json.dumps({"t": time.time(), "item": item}, ensure_ascii=False) + "\n")
            self._trim()

    def user(self, text: str) -> None:
        self.add(user_item(text))

    def assistant(self, text: str) -> None:
        self.add(assistant_item(text))

    def call(self, call_id: str, name: str, arguments: str) -> None:
        self.add(call_item(call_id, name, arguments))

    def output(self, call_id: str, output: str) -> None:
        self.add(output_item(call_id, output))

    def snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self.items)

    def turns(self) -> int:
        return sum(1 for it in self.items if it.get("role") == "user")

    def _trim(self) -> None:
        starts = [i for i, it in enumerate(self.items) if it.get("role") == "user"]
        if not starts:
            return
        self.items = self.items[starts[0] :]  # never start with a tool call or a reply without its question
        starts = [i - starts[0] for i in starts]
        while len(starts) > 1 and (
            len(starts) > self.max_turns or sum(map(item_chars, self.items)) > self.max_chars
        ):
            self.items = self.items[starts[1] :]
            starts = [i - starts[1] for i in starts[1:]]
