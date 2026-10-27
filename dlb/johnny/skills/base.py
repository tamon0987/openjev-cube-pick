"""The skill interface: what the LLM sees (name, description, JSON-schema parameters) and what runs.

A skill is a small, self-contained action on the rig. It gets a ``Context`` (shared state: launched processes,
the robot when it has been started, how to speak to the user) and keyword arguments matching its parameters.
It returns a ``SkillResult`` whose ``text`` goes back to the LLM (and, when ``say`` is set, to the user).
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SkillResult:
    text: str  # what the LLM is told happened (plain language, short)
    ok: bool = True
    data: dict[str, Any] | None = None  # structured details for logs / tests


@dataclass
class Context:
    """Shared state for the skills of one Johnny session."""

    robot_config: str = "configs/robot/omx_f.yaml"
    env_file: str = ".env"
    say: Callable[[str], None] = print  # tell the user something right away (progress), not via the LLM
    log: Callable[[str], None] = print
    processes: dict[str, subprocess.Popen] = field(default_factory=dict)  # launched things by name
    robot: Any = None  # dlb.voice.agent.Agent once pick_place started it (owns RealOMX / runner / guide)
    state: dict[str, Any] = field(default_factory=dict)  # free-form, e.g. {"teleop": True}

    def running(self, name: str) -> bool:
        p = self.processes.get(name)
        return p is not None and p.poll() is None


class Skill:
    """Subclass and set ``name``, ``description``, ``parameters`` (JSON schema object); implement ``run``."""

    name: str = ""
    description: str = ""
    parameters: dict[str, Any] = {"type": "object", "properties": {}, "additionalProperties": False}

    def run(self, ctx: Context, **kwargs: Any) -> SkillResult:
        raise NotImplementedError

    def cancel(self, ctx: Context) -> None:
        """Called when the user says stop while this skill runs (optional)."""

    def tool_spec(self) -> dict[str, Any]:
        """OpenAI Responses API function tool definition."""
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "strict": False,
        }
