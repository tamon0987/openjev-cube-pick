"""Metrics over (prediction, confidence, label) triples."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


def accuracy(preds: list[str], labels: list[str]) -> float:
    return float(np.mean([p == y for p, y in zip(preds, labels, strict=True)])) if preds else float("nan")


def macro_f1(preds: list[str], labels: list[str]) -> float:
    classes = sorted(set(labels) | set(preds))
    f1s = []
    for c in classes:
        tp = sum(1 for p, y in zip(preds, labels, strict=True) if p == c and y == c)
        fp = sum(1 for p, y in zip(preds, labels, strict=True) if p == c and y != c)
        fn = sum(1 for p, y in zip(preds, labels, strict=True) if p != c and y == c)
        if tp + fp + fn == 0:
            continue
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))
    return float(np.mean(f1s)) if f1s else float("nan")


def expected_calibration_error(confs: list[float], correct: list[bool], n_bins: int = 10) -> float:
    """Standard ECE: |acc - conf| weighted by bin mass."""
    if not confs:
        return float("nan")
    confs_a = np.clip(np.asarray(confs, float), 0, 1)
    corr_a = np.asarray(correct, float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:], strict=True):
        m = (confs_a > lo) & (confs_a <= hi) if lo > 0 else (confs_a >= lo) & (confs_a <= hi)
        if m.any():
            ece += m.mean() * abs(corr_a[m].mean() - confs_a[m].mean())
    return float(ece)


def brier(p_yes: list[float], labels_yes: list[bool]) -> float:
    if not p_yes:
        return float("nan")
    p = np.asarray(p_yes, float)
    y = np.asarray(labels_yes, float)
    return float(np.mean((p - y) ** 2))


def confusion(preds: list[str], labels: list[str]) -> dict[str, dict[str, int]]:
    m: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for p, y in zip(preds, labels, strict=True):
        m[y][p] += 1
    return {y: dict(row) for y, row in m.items()}


def coverage_accuracy_curve(
    confs: list[float], correct: list[bool], thresholds: list[float] | None = None
) -> list[dict[str, float]]:
    """For gating design: at each confidence threshold, what fraction of decisions
    would be taken automatically and how accurate are they."""
    thresholds = thresholds or [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    c = np.asarray(confs, float)
    ok = np.asarray(correct, float)
    out = []
    for t in thresholds:
        m = c >= t
        out.append(
            {
                "threshold": t,
                "coverage": float(m.mean()) if len(m) else float("nan"),
                "accuracy_covered": float(ok[m].mean()) if m.any() else float("nan"),
            }
        )
    return out


def latency_stats(lat: list[float]) -> dict[str, float]:
    if not lat:
        return {"p50": float("nan"), "p95": float("nan"), "mean": float("nan")}
    a = np.asarray(lat, float)
    return {"p50": float(np.percentile(a, 50)), "p95": float(np.percentile(a, 95)), "mean": float(a.mean())}


def summarize_question(kind: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """rows: [{pred, label, conf, p_yes?}]"""
    preds = [r["pred"] for r in rows]
    labels = [r["label"] for r in rows]
    confs = [r["conf"] for r in rows]
    correct = [p == y for p, y in zip(preds, labels, strict=True)]
    out: dict[str, Any] = {
        "n": len(rows),
        "accuracy": accuracy(preds, labels),
        "ece": expected_calibration_error(confs, correct),
        "mean_confidence": float(np.mean(confs)) if confs else float("nan"),
        "coverage_curve": coverage_accuracy_curve(confs, correct),
    }
    if kind == "choice":
        out["macro_f1"] = macro_f1(preds, labels)
        out["confusion"] = confusion(preds, labels)
    if kind == "noul":
        out["brier"] = brier([r["p_yes"] for r in rows], [y == "yes" for y in labels])
    if kind == "score":
        diffs = [abs(int(p) - int(y)) for p, y in zip(preds, labels, strict=True)]
        out["mean_abs_level_error"] = float(np.mean(diffs)) if diffs else float("nan")
    return out
