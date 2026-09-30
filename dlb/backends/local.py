"""Local (non-model) backends: oracle and random.

The harness attaches ground-truth labels computed from privileged simulator
state to ``req.meta["oracle"]`` for every request (this is how accuracy is
scored for real backends too). The oracle backend simply returns them, with an
optional label-noise rate so gating logic can be exercised offline.
"""

from __future__ import annotations

import random
import time

from dlb.backends.base import BackendInfo, DecisionBackend
from dlb.contract import (
    Choice,
    DecisionRequest,
    DecisionResponse,
    Noul,
    Score,
    make_choice_answer,
    make_noul_answer,
    make_score_answer,
)


class OracleBackend(DecisionBackend):
    image_policy = "drop"

    def __init__(self, name: str = "oracle", noise: float = 0.0, confidence: float = 0.95, seed: int = 0):
        self.noise = noise
        self.confidence = confidence
        self.rng = random.Random(seed)
        self.info = BackendInfo(name=name, modality="local", model="oracle")

    def decide(self, req: DecisionRequest) -> DecisionResponse:
        t0 = time.perf_counter()
        labels = req.meta.get("oracle") or {}
        answers = {}
        for key, q in req.questions.items():
            if key not in labels:
                raise KeyError(f"oracle label missing for question '{key}'")
            lab = labels[key]
            flip = self.rng.random() < self.noise
            if isinstance(q, Choice):
                opts = q.options
                if flip:
                    lab = self.rng.choice([o for o in opts if o != lab] or opts)
                answers[key] = make_choice_answer(lab, opts, p=self.confidence)
            elif isinstance(q, Noul):
                yes = lab in ("yes", True, 1)
                if flip:
                    yes = not yes
                answers[key] = make_noul_answer(self.confidence if yes else 1.0 - self.confidence)
            elif isinstance(q, Score):
                lvl = int(lab)
                if flip:
                    lvl = self.rng.randrange(len(q.criteria))
                answers[key] = make_score_answer(lvl, q.criteria, p=self.confidence)
        return DecisionResponse(answers=answers, model="oracle", latency_s=time.perf_counter() - t0)


class RandomBackend(DecisionBackend):
    image_policy = "drop"

    def __init__(self, name: str = "random", seed: int = 0):
        self.rng = random.Random(seed)
        self.info = BackendInfo(name=name, modality="local", model="random")

    def decide(self, req: DecisionRequest) -> DecisionResponse:
        t0 = time.perf_counter()
        answers = {}
        for key, q in req.questions.items():
            if isinstance(q, Choice):
                opts = q.options
                answers[key] = make_choice_answer(self.rng.choice(opts), opts, p=1.0 / len(opts) + 1e-6)
            elif isinstance(q, Noul):
                answers[key] = make_noul_answer(self.rng.random())
            elif isinstance(q, Score):
                n = len(q.criteria)
                answers[key] = make_score_answer(self.rng.randrange(n), q.criteria, p=1.0 / n + 1e-6)
        return DecisionResponse(answers=answers, model="random", latency_s=time.perf_counter() - t0)
