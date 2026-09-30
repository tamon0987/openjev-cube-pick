"""Live check of the instruction parser (gpt-5.5) on the examples in docs/voice.md. Three API calls.

uv run python scripts/voice_intent_check.py [--effort low] [--env-file .env]
"""

from __future__ import annotations

import argparse
import json

from dlb.voice.intent import IntentParser, RobotState, check_sequence

CASES = [
    # holding red on the way to the bin
    (
        "それをその場に置いて青いキューブをビンに入れて",
        RobotState("red cube", queue=[{"op": "place", "where": "bin"}]),
        [
            {"op": "place", "where": "here"},
            {"op": "pick", "object": "blue cube"},
            {"op": "place", "where": "bin"},
        ],
    ),
    # red put down, about to fetch blue
    (
        "やっぱり赤いキューブを青いキューブの上に置いて",
        RobotState(None, queue=[{"op": "pick", "object": "blue cube"}, {"op": "place", "where": "bin"}]),
        [{"op": "pick", "object": "red cube"}, {"op": "place", "where": "on:blue cube"}],
    ),
    # the same, said a moment later with blue already in the gripper
    (
        "やっぱり赤いキューブを青いキューブの上に置いて",
        RobotState("blue cube", queue=[{"op": "place", "where": "bin"}]),
        [
            {"op": "place", "where": "here"},
            {"op": "pick", "object": "red cube"},
            {"op": "place", "where": "on:blue cube"},
        ],
    ),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-5.5")
    ap.add_argument("--effort", default="low")
    ap.add_argument("--env-file", default=".env")
    args = ap.parse_args()
    parser = IntentParser(args.model, args.effort or None, env_file=args.env_file)
    ok = 0
    for text, state, expected in CASES:
        intent = parser.parse(text, state)
        match = intent.tasks == expected and intent.replace_queue
        ok += match
        print(f"「{text}」 held={state.held} queue={json.dumps(state.queue)}")
        print(
            f"  -> {json.dumps(intent.tasks, ensure_ascii=False)} replace_queue={intent.replace_queue} "
            f"clarify={intent.clarify!r}  [{intent.latency_s:.2f} s] {'OK' if match else 'UNEXPECTED'}"
        )
        for p in check_sequence(intent, state):
            print(f"  warning: {p}")
    print(f"{ok}/{len(CASES)} as expected ({args.model}, effort {args.effort or 'default'})")
    return 0 if ok == len(CASES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
