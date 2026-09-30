"""Decision contract shared by every backend.

The contract mirrors the TypeSafe Jev "System One" wire format (which djev and
openjev also speak) so that the *same* request can be sent to:

* TypeSafe Jev (hosted, text/JSON state only)
* openjev      (self-hosted, Jev-compatible, images optional)
* djev         (self-hosted, images native, endpoint differs)
* Oracle / Random (local, for ground truth and floor baselines)

Three question types exist:

* ``noul``   – yes/no, answered with a probability of "yes"
* ``choice`` – pick one option, answered with a distribution over options
* ``score``  – position on an ordered rubric, answered with a distribution over levels

Nothing here depends on the simulator or on HTTP.
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

QuestionType = Literal["noul", "choice", "score"]


# --------------------------------------------------------------------------- #
# Questions
# --------------------------------------------------------------------------- #
@dataclass
class Noul:
    instructions: str
    criteria: dict[str, str] | None = None  # {"true": "...", "false": "..."}

    type: QuestionType = field(default="noul", init=False)

    def to_wire(self) -> dict[str, Any]:
        d: dict[str, Any] = {"type": "noul", "instructions": self.instructions}
        if self.criteria:
            d["criteria"] = self.criteria
        return d


@dataclass
class Choice:
    instructions: str
    criteria: dict[str, str | dict[str, str]]  # option -> description (or {"image":..,"text":..})

    type: QuestionType = field(default="choice", init=False)

    def to_wire(self) -> dict[str, Any]:
        return {"type": "choice", "instructions": self.instructions, "criteria": self.criteria}

    @property
    def options(self) -> list[str]:
        return list(self.criteria.keys())


@dataclass
class Score:
    instructions: str
    criteria: list[str]  # level 0..N-1 descriptions (2..10 levels)

    type: QuestionType = field(default="score", init=False)

    def __post_init__(self) -> None:
        if not 2 <= len(self.criteria) <= 10:
            raise ValueError("score questions need 2..10 levels")

    def to_wire(self) -> dict[str, Any]:
        return {"type": "score", "instructions": self.instructions, "criteria": self.criteria}


Question = Noul | Choice | Score


# --------------------------------------------------------------------------- #
# Request
# --------------------------------------------------------------------------- #
def image_to_data_url(img: np.ndarray | bytes, fmt: str = "PNG") -> str:
    """Encode an HxWx3 uint8 array (or raw PNG/JPEG bytes) as a data URL."""
    from PIL import Image

    if isinstance(img, np.ndarray):
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, format=fmt)
        raw = buf.getvalue()
        mime = "image/png" if fmt.upper() == "PNG" else "image/jpeg"
    else:
        raw = img
        mime = "image/png" if raw[:4] == b"\x89PNG" else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(raw).decode("ascii")


@dataclass
class DecisionRequest:
    """One round-trip to a decision layer.

    ``state`` is what text-only backends see. ``images`` is what image-capable
    backends see in addition (data URLs). A backend that cannot take images
    must either drop them (``ImagePolicy = "drop"``) or refuse (``"error"``).
    """

    state: str | dict[str, Any] | list[Any]
    questions: dict[str, Question]
    images: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)  # not sent; for logging

    def to_wire(self, model: str | None = None, include_images: bool = True) -> dict[str, Any]:
        body: dict[str, Any] = {
            "state": self.state,
            "questions": {k: q.to_wire() for k, q in self.questions.items()},
        }
        if model:
            body["model"] = model
        if include_images and self.images:
            body["images"] = list(self.images)
        return body


# --------------------------------------------------------------------------- #
# Answers
# --------------------------------------------------------------------------- #
@dataclass
class Answer:
    type: QuestionType
    # noul
    noul: float | None = None
    # choice
    choice: str | None = None
    # score
    score: float | None = None
    legend: dict[str, str] | None = None
    # shared
    probabilities: dict[str, float] | None = None
    confidence: float | None = None

    # ---- convenience --------------------------------------------------- #
    @property
    def label(self) -> str:
        """Canonical hard label for accuracy computations."""
        if self.type == "noul":
            return "yes" if (self.noul or 0.0) >= 0.5 else "no"
        if self.type == "choice":
            return self.choice or ""
        # score: most likely level (not the expectation)
        if self.probabilities:
            return max(self.probabilities.items(), key=lambda kv: kv[1])[0]
        return str(int(round(self.score or 0)))

    @property
    def conf(self) -> float:
        """A confidence in [0,1] regardless of type (used by gating)."""
        if self.confidence is not None:
            return float(self.confidence)
        if self.type == "noul" and self.noul is not None:
            return abs(self.noul - 0.5) * 2.0
        if self.probabilities:
            return float(max(self.probabilities.values()))
        return 0.0

    @staticmethod
    def from_wire(d: dict[str, Any]) -> Answer:
        t = d.get("type")
        if t not in ("noul", "choice", "score"):
            raise ValueError(f"unknown answer type: {d}")
        return Answer(
            type=t,
            noul=d.get("noul"),
            choice=d.get("choice"),
            score=d.get("score"),
            legend=d.get("legend"),
            probabilities=d.get("probabilities"),
            confidence=d.get("confidence"),
        )

    def to_wire(self) -> dict[str, Any]:
        out: dict[str, Any] = {"type": self.type}
        for k in ("noul", "choice", "score", "legend", "probabilities", "confidence"):
            v = getattr(self, k)
            if v is not None:
                out[k] = v
        return out


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class DecisionResponse:
    answers: dict[str, Answer]
    model: str = ""
    usage: Usage = field(default_factory=Usage)
    latency_s: float = 0.0
    raw: dict[str, Any] | None = None

    @staticmethod
    def from_wire(d: dict[str, Any], latency_s: float = 0.0) -> DecisionResponse:
        usage = d.get("usage") or {}
        return DecisionResponse(
            answers={k: Answer.from_wire(v) for k, v in d.get("answers", {}).items()},
            model=str(d.get("model", "")),
            usage=Usage(int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0))),
            latency_s=latency_s,
            raw=d,
        )

    def to_wire(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "answers": {k: a.to_wire() for k, a in self.answers.items()},
            "usage": {"input_tokens": self.usage.input_tokens, "output_tokens": self.usage.output_tokens},
            "latency_s": self.latency_s,
        }


# --------------------------------------------------------------------------- #
# Helpers to build answers locally (oracle / random backends)
# --------------------------------------------------------------------------- #
def make_choice_answer(option: str, options: list[str], p: float = 1.0) -> Answer:
    rest = (1.0 - p) / max(1, len(options) - 1)
    probs = {o: (p if o == option else rest) for o in options}
    top2 = sorted(probs.values(), reverse=True)[:2]
    conf = top2[0] - (top2[1] if len(top2) > 1 else 0.0)
    return Answer(type="choice", choice=option, probabilities=probs, confidence=conf)


def make_noul_answer(p_yes: float) -> Answer:
    return Answer(type="noul", noul=float(p_yes))


def make_score_answer(level: int, criteria: list[str], p: float = 1.0) -> Answer:
    n = len(criteria)
    rest = (1.0 - p) / max(1, n - 1)
    probs = {str(i): (p if i == level else rest) for i in range(n)}
    expected = sum(i * probs[str(i)] for i in range(n))
    top2 = sorted(probs.values(), reverse=True)[:2]
    return Answer(
        type="score",
        score=expected,
        legend={str(i): c for i, c in enumerate(criteria)},
        probabilities=probs,
        confidence=top2[0] - (top2[1] if len(top2) > 1 else 0.0),
    )
