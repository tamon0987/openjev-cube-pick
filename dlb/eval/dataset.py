"""Offline dataset: identical samples for every backend/modality.

Each sample = {images (PNG per camera), state_json, oracle labels, provenance}.
States are visited by running the oracle policy with an epsilon of random
primitives so off-nominal situations (jaw closed on nothing, hovering over the
wrong spot, cube dropped next to the bin) are represented.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from dlb.sim.env import PRIMITIVES, Observation, PickPlaceEnv


@dataclass
class Sample:
    id: str
    state_json: dict[str, Any]
    oracle: dict[str, Any]
    image_files: dict[str, str]
    meta: dict[str, Any]

    def to_observation(self, load_images: bool = True) -> Observation:
        images = {}
        if load_images:
            for cam, f in self.image_files.items():
                images[cam] = np.asarray(Image.open(f).convert("RGB"))
        return Observation(
            state_json=self.state_json, privileged={}, images=images, step=self.state_json["history"]["step"]
        )


def generate(
    out_dir: str | Path,
    n_episodes: int = 40,
    epsilon: float = 0.25,
    seed: int = 0,
    image_size: int = 320,
    cameras: tuple[str, ...] = ("front", "wrist"),
    max_steps: int = 14,
) -> Path:
    out = Path(out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    env = PickPlaceEnv(render=True, image_size=image_size, cameras=cameras, max_steps=max_steps)
    rng = np.random.default_rng(seed)
    prims = list(PRIMITIVES)
    manifest = out / "manifest.jsonl"
    n = 0
    label_counts: dict[str, int] = {}
    with open(manifest, "w", encoding="utf-8") as f:
        for ep in range(n_episodes):
            obs = env.reset(seed=seed * 100_000 + ep)
            while not env.episode_over():
                oracle = env.oracle_labels()
                sid = f"e{ep:04d}_s{obs.step:02d}"
                files = {}
                for cam, img in obs.images.items():
                    p = out / "images" / f"{sid}_{cam}.png"
                    Image.fromarray(img).save(p)
                    files[cam] = str(p.relative_to(out))
                explore = bool(rng.random() < epsilon)
                action = str(rng.choice(prims)) if explore else oracle["next_action"]
                if action == "done" and explore:  # would end the episode early without information
                    action = oracle["next_action"]
                rec = {
                    "id": sid,
                    "state_json": obs.state_json,
                    "oracle": oracle,
                    "image_files": files,
                    "meta": {"episode": ep, "step": obs.step, "action_taken": action, "explore": explore},
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
                label_counts[oracle["next_action"]] = label_counts.get(oracle["next_action"], 0) + 1
                env.execute(action)
                obs = env.observe(render=True)
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "samples": n,
                "episodes": n_episodes,
                "epsilon": epsilon,
                "seed": seed,
                "image_size": image_size,
                "cameras": list(cameras),
                "next_action_counts": label_counts,
            },
            f,
            indent=2,
        )
    return out


def load(dataset_dir: str | Path, limit: int | None = None) -> list[Sample]:
    d = Path(dataset_dir)
    samples: list[Sample] = []
    with open(d / "manifest.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            samples.append(
                Sample(
                    id=r["id"],
                    state_json=r["state_json"],
                    oracle=r["oracle"],
                    image_files={k: str(d / v) for k, v in r["image_files"].items()},
                    meta=r["meta"],
                )
            )
            if limit and len(samples) >= limit:
                break
    return samples
