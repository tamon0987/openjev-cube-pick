"""Utterance + robot state -> task list, by gpt-5.5 with a strict JSON schema (OpenAI Responses API).

A task is ``{"op": "pick", "object": <name>}`` or ``{"op": "place", "where": "bin" | "here" | "on:<name>"}``.
The schema is built per call so that object names and places can only be ones the robot knows about.
``replace_queue`` says whether the tasks replace the current queue (an interruption: "やっぱり…", a new
command) or go after it ("それが終わったら…"). An empty task list with ``replace_queue`` is "stop".
``clarify`` is a question to ask back when the instruction is ambiguous; then no tasks are given.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from dlb.harness.marking import load_env_file

PLACES = ("bin", "here")
PROMPT = """You turn a spoken instruction (usually Japanese) for a small robot arm on a table into a task list.

Operations:
- {{"op": "pick", "object": <name>, "where": null}}: move to that object and grasp it. Only possible when the gripper is empty.
- {{"op": "place", "object": null, "where": <place>}}: put down the object in the gripper. <place> is "bin" (into the bin),
  "here" (put it down right where the gripper is now, e.g. その場に / ここに / 下ろして), or "on:<name>" (stack it on top of that object).

Rules:
- Use only the object names listed below ("bin" is the container things are put into). Map the user's words to them
  by meaning, kind, colour or sound: the words are usually Japanese and name things loosely (ビン/箱/ゴミ箱 -> the bin).
- それ / これ / 持っているもの refers to the object in the gripper.
- The gripper holds at most one object. If a task needs the gripper but it is (or will be) holding something, first place
  that object as the user said; if the user did not say where, place it "here".
- The robot is executing the current queue. replace_queue = true when the instruction changes or replaces what the robot is
  doing (a new command, やっぱり, 代わりに, 止まって/やめて); the tasks then start from the robot's current state.
  replace_queue = false only when the user adds work for afterwards (それが終わったら, 次に, あとで); the tasks then start
  from the state after the current queue.
- Stop / cancel (止まって, やめて, ストップ): tasks = [], replace_queue = true.
- The instruction comes from speech recognition and may be misheard (瓶 / びん / 水 for ビン, a similar-sounding word for
  an object's name): pick the listed object that the user most likely meant.
- If the instruction is ambiguous (e.g. an object that is not listed, or two objects match), give tasks = [] and a short
  Japanese question in "clarify". Otherwise clarify = null.

Current state:
- gripper holds: {held}
- known objects: {objects}
- current task queue: {queue}

Instruction: 「{utterance}」"""


@dataclass
class RobotState:
    held: str | None = None
    objects: list[str] = field(default_factory=lambda: ["bin"])  # what is on the table, plus the bin
    queue: list[dict[str, Any]] = field(default_factory=list)

    def places(self) -> list[str]:
        return [*PLACES, *(f"on:{o}" for o in self.objects if o != "bin")]

    def pickable(self) -> list[str]:
        return [o for o in self.objects if o != "bin"]


@dataclass
class Intent:
    tasks: list[dict[str, str]]
    replace_queue: bool
    clarify: str | None = None
    latency_s: float = 0.0
    utterance: str = ""

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def schema(state: RobotState) -> dict[str, Any]:
    task = {
        "type": "object",
        "properties": {
            "op": {"type": "string", "enum": ["pick", "place"]},
            "object": {"type": ["string", "null"], "enum": [*state.pickable(), None]},
            "where": {"type": ["string", "null"], "enum": [*state.places(), None]},
        },
        "required": ["op", "object", "where"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "tasks": {"type": "array", "items": task},
            "replace_queue": {"type": "boolean"},
            "clarify": {"type": ["string", "null"]},
        },
        "required": ["tasks", "replace_queue", "clarify"],
        "additionalProperties": False,
    }


def normalize(out: dict[str, Any], state: RobotState) -> Intent:
    """Checks the answer against the state (the schema already limits the names) and drops the unused field."""
    tasks = []
    for t in out.get("tasks", []):
        if t.get("op") == "pick" and t.get("object") in state.pickable():
            tasks.append({"op": "pick", "object": t["object"]})
        elif t.get("op") == "place" and t.get("where") in state.places():
            tasks.append({"op": "place", "where": t["where"]})
        else:
            raise ValueError(f"invalid task {t}")
    return Intent(tasks, bool(out.get("replace_queue", True)), out.get("clarify") or None)


def check_sequence(intent: Intent, state: RobotState) -> list[str]:
    """Problems with executing the tasks from ``state`` (pick with a full gripper, place with an empty one)."""
    held = state.held
    if not intent.replace_queue:  # appended: starts where the queue ends
        for t in state.queue:
            held = t.get("object") if t.get("op") == "pick" else None if t.get("op") == "place" else held
    problems = []
    for i, t in enumerate(intent.tasks):
        if t["op"] == "pick":
            if held is not None:
                problems.append(f"task {i}: pick {t['object']} while holding {held}")
            held = t["object"]
        else:
            if held is None:
                problems.append(f"task {i}: place with an empty gripper")
            elif t["where"] == f"on:{held}":
                problems.append(f"task {i}: place {held} on itself")
            held = None
    return problems


class IntentParser:
    def __init__(
        self,
        model: str = "gpt-5.5",
        effort: str | None = "low",
        timeout_s: float = 60.0,
        env_file: str = ".env",
        client=None,
    ):
        self.model, self.effort = model, effort
        if client is None:
            import httpx

            load_env_file(env_file)
            client = httpx.Client(
                base_url="https://api.openai.com/v1",
                timeout=timeout_s,
                headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
            )
        self.client = client

    def prompt(self, utterance: str, state: RobotState) -> str:
        return PROMPT.format(
            held=json.dumps(state.held),
            objects=json.dumps(state.objects),
            queue=json.dumps(state.queue, ensure_ascii=False),
            utterance=utterance,
        )

    def parse(self, utterance: str, state: RobotState) -> Intent:
        body: dict[str, Any] = {
            "model": self.model,
            "input": [
                {"role": "user", "content": [{"type": "input_text", "text": self.prompt(utterance, state)}]}
            ],
            "text": {
                "format": {"type": "json_schema", "name": "tasks", "schema": schema(state), "strict": True}
            },
        }
        if self.effort:
            body["reasoning"] = {"effort": self.effort}
        t0 = time.perf_counter()
        r = self.client.post("/responses", json=body)
        if r.status_code != 200:
            raise RuntimeError(f"{self.model}: {r.status_code} {r.text[:300]}")
        text = "".join(
            c.get("text", "")
            for o in r.json().get("output", [])
            if o.get("type") == "message"
            for c in o.get("content", [])
            if c.get("type") == "output_text"
        )
        intent = normalize(json.loads(text), state)
        intent.latency_s, intent.utterance = time.perf_counter() - t0, utterance
        return intent


def apply(intent: Intent, state: RobotState) -> RobotState:
    """The state with the queue rebuilt (``replace_queue``) or extended; ``held`` is left to the runner.
    A clarifying question changes nothing."""
    if intent.clarify:
        return RobotState(state.held, list(state.objects), [dict(t) for t in state.queue])
    queue = intent.tasks if intent.replace_queue else [*state.queue, *intent.tasks]
    return RobotState(state.held, list(state.objects), [dict(t) for t in queue])
