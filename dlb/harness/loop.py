"""Online control loop: observe -> ask decision layer -> gate -> execute -> log."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from dlb.backends.base import DecisionBackend
from dlb.contract import DecisionResponse
from dlb.harness.prompts import build_request
from dlb.sim.env import PRIMITIVES, PickPlaceEnv

Escalation = Literal["act", "oracle", "stop", "home"]


@dataclass
class GatingConfig:
    threshold: float = 0.0  # 0 disables gating
    on_low_confidence: Escalation = "act"


@dataclass
class EpisodeResult:
    episode: int
    seed: int
    backend: str
    modality: str
    success: bool
    steps: int
    wall_time_s: float
    decision_latency_s: list[float] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    escalations: int = 0
    action_accuracy: float = float("nan")  # vs oracle at the same state
    actions: list[str] = field(default_factory=list)
    stop_reason: str = ""


class EpisodeRunner:
    def __init__(
        self,
        env: PickPlaceEnv,
        backend: DecisionBackend,
        modality: str = "text",
        question_set: str = "action_only",
        gating: GatingConfig | None = None,
        log_dir: str | Path | None = None,
        save_images: bool = False,
        image_cameras: tuple[str, ...] | None = None,
    ):
        self.env = env
        self.backend = backend
        self.modality = modality
        self.question_set = question_set
        self.gating = gating or GatingConfig()
        self.log_dir = Path(log_dir) if log_dir else None
        self.save_images = save_images
        self.image_cameras = image_cameras
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)

    def run(self, episode: int, seed: int) -> EpisodeResult:
        env, be = self.env, self.backend
        need_images = self.modality != "text"
        obs = env.reset(seed=seed)
        if need_images and not obs.images:
            obs = env.observe(render=True)
        t0 = time.perf_counter()
        res = EpisodeResult(
            episode=episode,
            seed=seed,
            backend=be.info.name,
            modality=self.modality,
            success=False,
            steps=0,
            wall_time_s=0,
        )
        log_f = (
            open(self.log_dir / f"ep{episode:04d}_{be.info.name}.jsonl", "w", encoding="utf-8")
            if self.log_dir
            else None
        )
        correct = 0
        try:
            while not env.episode_over():
                oracle = env.oracle_labels()
                req = build_request(
                    obs,
                    question_set=self.question_set,
                    modality=self.modality,
                    image_cameras=self.image_cameras,
                    oracle=oracle,
                )
                resp: DecisionResponse = be.decide(req)
                ans = resp.answers["next_action"]
                proposed = ans.choice or ""
                if proposed not in PRIMITIVES:
                    proposed = "home"
                escalated = False
                executed = proposed
                if self.gating.threshold > 0 and ans.conf < self.gating.threshold:
                    escalated = True
                    res.escalations += 1
                    mode = self.gating.on_low_confidence
                    if mode == "oracle":
                        executed = oracle["next_action"]
                    elif mode == "home":
                        executed = "home"
                    elif mode == "stop":
                        res.stop_reason = "low_confidence_stop"
                        break
                correct += int(proposed == oracle["next_action"])
                info = env.execute(executed)
                res.actions.append(executed)
                res.decision_latency_s.append(resp.latency_s)
                res.input_tokens += resp.usage.input_tokens
                res.output_tokens += resp.usage.output_tokens
                if log_f:
                    rec = {
                        "episode": episode,
                        "step": obs.step,
                        "backend": be.info.name,
                        "modality": self.modality,
                        "state": obs.state_json,
                        "oracle": oracle,
                        "answers": {k: a.to_wire() for k, a in resp.answers.items()},
                        "proposed": proposed,
                        "confidence": ans.conf,
                        "escalated": escalated,
                        "executed": executed,
                        "exec_info": _jsonable(info),
                        "latency_s": resp.latency_s,
                        "usage": asdict(resp.usage),
                        "privileged_after": _jsonable(env.observe(render=False).privileged),
                    }
                    if self.save_images and obs.images:
                        from PIL import Image

                        for cam, img in obs.images.items():
                            p = self.log_dir / "images" / f"ep{episode:04d}_s{obs.step:02d}_{cam}.png"
                            p.parent.mkdir(exist_ok=True)
                            Image.fromarray(img).save(p)
                        rec["image_files"] = [
                            str(self.log_dir / "images" / f"ep{episode:04d}_s{obs.step:02d}_{cam}.png")
                            for cam in obs.images
                        ]
                    log_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                obs = env.observe(render=need_images)
        finally:
            if log_f:
                log_f.close()
        res.steps = env.step_count
        res.success = env.is_success()
        res.wall_time_s = time.perf_counter() - t0
        res.cost_usd = res.input_tokens / 1e6 * be.info.price_per_m_input_tokens_usd
        res.action_accuracy = correct / max(1, env.step_count)
        if not res.stop_reason:
            res.stop_reason = "done" if env.done_declared else "max_steps"
        return res


def _jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def summarize(results: list[EpisodeResult]) -> dict[str, Any]:
    if not results:
        return {}
    lat = [x for r in results for x in r.decision_latency_s]
    return {
        "backend": results[0].backend,
        "modality": results[0].modality,
        "episodes": len(results),
        "success_rate": float(np.mean([r.success for r in results])),
        "mean_steps": float(np.mean([r.steps for r in results])),
        "action_accuracy": float(np.nanmean([r.action_accuracy for r in results])),
        "latency_p50_s": float(np.percentile(lat, 50)) if lat else float("nan"),
        "latency_p95_s": float(np.percentile(lat, 95)) if lat else float("nan"),
        "decisions": len(lat),
        "input_tokens": int(sum(r.input_tokens for r in results)),
        "cost_usd": float(sum(r.cost_usd for r in results)),
        "escalations": int(sum(r.escalations for r in results)),
        "wall_time_s": float(sum(r.wall_time_s for r in results)),
    }
