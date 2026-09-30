"""Offline evaluation: run one backend over a fixed dataset and score every question."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from dlb.backends.base import DecisionBackend
from dlb.contract import Choice, Noul
from dlb.eval.dataset import Sample
from dlb.eval.metrics import latency_stats, summarize_question
from dlb.harness.prompts import build_request, questions


def run_offline(
    backend: DecisionBackend,
    samples: list[Sample],
    modality: str,
    question_set: str = "full",
    out_dir: str | Path | None = None,
    image_cameras: tuple[str, ...] | None = None,
    progress: bool = True,
    run_name: str | None = None,
) -> dict[str, Any]:
    qs = questions(question_set)
    kinds = {
        k: ("choice" if isinstance(q, Choice) else "noul" if isinstance(q, Noul) else "score")
        for k, q in qs.items()
    }
    rows: dict[str, list[dict[str, Any]]] = {k: [] for k in qs}
    latencies: list[float] = []
    in_tokens = 0
    errors: list[dict[str, Any]] = []
    name = run_name or f"{backend.info.name}__{modality}"
    out = Path(out_dir) if out_dir else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
    raw_f = open(out / f"{name}.jsonl", "w", encoding="utf-8") if out else None
    t_start = time.perf_counter()
    try:
        for i, s in enumerate(samples):
            obs = s.to_observation(load_images=(modality != "text"))
            req = build_request(
                obs,
                question_set=question_set,
                modality=modality,
                image_cameras=image_cameras,
                oracle=s.oracle,
            )
            try:
                resp = backend.decide(req)
            except Exception as e:  # keep going; record the failure
                errors.append({"id": s.id, "error": f"{e.__class__.__name__}: {e}"[:500]})
                if progress:
                    print(f"[{name}] {i + 1}/{len(samples)} ERROR {e}")
                continue
            latencies.append(resp.latency_s)
            in_tokens += resp.usage.input_tokens
            rec: dict[str, Any] = {
                "id": s.id,
                "latency_s": resp.latency_s,
                "usage": resp.usage.__dict__,
                "answers": {},
            }
            for k, kind in kinds.items():
                a = resp.answers[k]
                label = s.oracle[k]
                label_s = str(label) if kind != "noul" else ("yes" if label in ("yes", True, 1) else "no")
                row = {"id": s.id, "pred": a.label, "label": label_s, "conf": a.conf}
                if kind == "noul":
                    row["p_yes"] = float(a.noul if a.noul is not None else 0.5)
                rows[k].append(row)
                rec["answers"][k] = {**a.to_wire(), "label": label_s, "correct": a.label == label_s}
            if raw_f:
                raw_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if progress and (i + 1) % 10 == 0:
                print(f"[{name}] {i + 1}/{len(samples)}  p50={latency_stats(latencies)['p50'] * 1000:.0f} ms")
    finally:
        if raw_f:
            raw_f.close()
    summary: dict[str, Any] = {
        "run": name,
        "backend": backend.info.name,
        "model": backend.info.model,
        "modality": modality,
        "question_set": question_set,
        "samples": len(samples),
        "answered": len(latencies),
        "errors": len(errors),
        "latency_s": latency_stats(latencies),
        "input_tokens": in_tokens,
        "input_tokens_per_sample": in_tokens / max(1, len(latencies)),
        "cost_usd": in_tokens / 1e6 * backend.info.price_per_m_input_tokens_usd,
        "wall_time_s": time.perf_counter() - t_start,
        "questions": {k: summarize_question(kinds[k], rows[k]) for k in qs},
        "error_samples": errors[:20],
    }
    if out:
        with open(out / f"{name}.summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary
