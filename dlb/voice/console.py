"""Typed instructions -> the same intents as voice, to test interruption without audio.

    python -m dlb.voice.console --holding "red cube"
    > それをその場に置いて青いキューブをビンに入れて
    {"tasks": [...], "replace_queue": true, "clarify": null, "latency_s": ..., "utterance": ...}

Each line is parsed against the current state; the queue is then rebuilt or extended as the intent says, so the
next line sees it. ``:hold <name>`` / ``:hold`` (empty) / ``:queue`` / ``:clear`` change or show the state by hand
(the runner does that when it is connected). Intents go to stdout as JSON lines, everything else to stderr.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Iterable

from dlb.voice.intent import Intent, IntentParser, RobotState, apply, check_sequence


def add_state_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--holding", default=None, help="object in the gripper at the start, e.g. 'red cube'")
    ap.add_argument("--objects", default="red cube,blue cube,bin", help="comma-separated known objects")
    ap.add_argument("--queue", default="[]", help="current task queue as JSON")
    ap.add_argument("--intent-model", default="gpt-5.5")
    ap.add_argument("--effort", default="low", help="reasoning effort ('' for the model default)")


class Session:
    """Holds the state between utterances and prints each intent."""

    def __init__(self, parser: IntentParser, state: RobotState, emit: Callable[[Intent], None] | None = None):
        self.parser, self.state = parser, state
        self.emit = emit or (lambda i: print(json.dumps(i.to_json(), ensure_ascii=False), flush=True))

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> Session:
        objects = [o.strip() for o in args.objects.split(",") if o.strip()]
        state = RobotState(args.holding or None, objects, json.loads(args.queue))
        return cls(IntentParser(args.intent_model, args.effort or None, env_file=args.env_file), state)

    def handle(self, text: str) -> Intent | None:
        text = text.strip()
        if not text:
            return None
        if text.startswith(":"):
            cmd, _, arg = text[1:].partition(" ")
            if cmd == "hold":
                self.state.held = arg.strip() or None
            elif cmd == "clear":
                self.state.queue = []
            elif cmd != "queue":
                print(f"unknown command :{cmd} (:hold [name], :queue, :clear)", file=sys.stderr)
            print(
                f"  held={self.state.held} queue={json.dumps(self.state.queue, ensure_ascii=False)}",
                file=sys.stderr,
            )
            return None
        try:
            intent = self.parser.parse(text, self.state)
        except Exception as e:  # noqa: BLE001 - one failed request should not end the session
            print(f"  intent failed: {e}", file=sys.stderr)
            return None
        for p in check_sequence(intent, self.state):
            print(f"  warning: {p}", file=sys.stderr)
        self.emit(intent)
        self.state = apply(intent, self.state)
        if intent.clarify:
            print(f"  ? {intent.clarify}", file=sys.stderr)
        return intent

    def run(self, lines: Iterable[str]) -> None:
        for line in lines:
            self.handle(line)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dlb.voice.console", description=__doc__.split("\n")[0])
    add_state_args(ap)
    ap.add_argument("--env-file", default=".env")
    args = ap.parse_args(argv)
    session = Session.from_args(args)
    if sys.stdin.isatty():
        print(
            f"held={session.state.held} objects={session.state.objects}; type an instruction (Ctrl-D ends)",
            file=sys.stderr,
        )
    try:
        session.run(sys.stdin)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
