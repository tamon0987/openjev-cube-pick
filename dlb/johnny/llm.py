"""The model behind Johnny: one Responses API call with the system prompt, the chat history and the skills as
function tools. Returns the text and the tool calls of that one call; the loop (run tools, feed the results
back) is in ``agent.py``.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

from dlb.harness.marking import load_env_file

SYSTEM_PROMPT = """\
You are Johnny (ジョニー), the voice assistant of a small table-top robot rig: a ROBOTIS OMX-F arm, an overhead
camera and a wrist camera, a grasp VLM and the Jev decision layer, driven from one laptop. The user talks to you
by voice; each message begins with your name ("Hi, Johnny") and the transcript may contain mishearings.

Rules:
- Answer in Japanese, briefly (one or two short sentences; this is read aloud). Match the user's language if
  they clearly speak English.
- Anything that touches the rig (starting drivers or servers, checking cameras, moving the arm, tele-operation,
  picking and placing, stopping) is done ONLY through the tools. Never claim something happened or describe the
  rig's state unless a tool result says so. Never invent object names, positions or results.
- When a request is ambiguous or risky, ask one short question instead of guessing.
- Some tools run interactive, multi-step checks: a result may ask the user to do a physical step (for example
  to unplug a USB cable). Relay that instruction to the user concisely and faithfully (do not add steps or
  change its meaning), then wait. When the user reports back, call the same tool again with the next step as
  its parameters describe (the tool result tells you which step comes next). Do not skip steps or call
  another tool in between unless the user asks for it.
- If a tool failed, say what failed in plain words and what the user can do; do not retry blindly more than
  once.
- "止まって" / "stop" is handled before you see it; if it reaches you, call the stop tool if there is one.
- Keep the conversation going: earlier turns and tool results in the history are real and current unless a
  later one says otherwise.
"""


@dataclass
class ToolCall:
    call_id: str
    name: str
    arguments: dict[str, Any]
    raw_arguments: str = ""


@dataclass
class Reply:
    text: str
    calls: list[ToolCall] = field(default_factory=list)
    latency_s: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


class LLM:
    def __init__(
        self,
        model: str = "gpt-5.5",
        effort: str | None = "low",
        timeout_s: float = 120.0,
        env_file: str = ".env",
        instructions: str = SYSTEM_PROMPT,
        client=None,
    ):
        self.model, self.effort, self.instructions = model, effort, instructions
        if client is None:
            import httpx

            load_env_file(env_file)
            client = httpx.Client(
                base_url="https://api.openai.com/v1",
                timeout=timeout_s,
                headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
            )
        self.client = client

    def respond(self, items: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Reply:
        body: dict[str, Any] = {
            "model": self.model,
            "instructions": self.instructions,
            "input": items,
            "tools": tools,
            "store": False,
        }
        if self.effort:
            body["reasoning"] = {"effort": self.effort}
        t0 = time.perf_counter()
        r = self.client.post("/responses", json=body)
        if r.status_code != 200:
            raise RuntimeError(f"{self.model}: {r.status_code} {r.text[:300]}")
        out = r.json()
        reply = Reply("", latency_s=time.perf_counter() - t0, raw=out)
        texts = []
        for o in out.get("output", []):
            if o.get("type") == "message":
                texts += [c.get("text", "") for c in o.get("content", []) if c.get("type") == "output_text"]
            elif o.get("type") == "function_call":
                raw = o.get("arguments") or "{}"
                try:
                    args = json.loads(raw)
                except json.JSONDecodeError:
                    args = {}
                reply.calls.append(ToolCall(o.get("call_id", ""), o.get("name", ""), args, raw))
        reply.text = "".join(texts).strip()
        return reply
