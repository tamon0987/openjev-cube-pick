"""Render Markdown comparison tables from offline summaries and online results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _f(x: Any, nd: int = 3) -> str:
    try:
        if x is None or x != x:  # NaN
            return "–"
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def offline_table(summaries: list[dict[str, Any]]) -> str:
    qkeys: list[str] = []
    for s in summaries:
        for k in s["questions"]:
            if k not in qkeys:
                qkeys.append(k)
    head = (
        ["run", "modality", "n", "err", "p50 ms", "p95 ms", "tok/sample", "cost $"]
        + [f"{k} acc" for k in qkeys]
        + [f"{k} ECE" for k in qkeys]
    )
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for s in summaries:
        row = [
            s["run"],
            s["modality"],
            str(s["answered"]),
            str(s["errors"]),
            _f(s["latency_s"]["p50"] * 1000, 0),
            _f(s["latency_s"]["p95"] * 1000, 0),
            _f(s["input_tokens_per_sample"], 0),
            _f(s["cost_usd"], 4),
        ]
        row += [_f(s["questions"].get(k, {}).get("accuracy")) for k in qkeys]
        row += [_f(s["questions"].get(k, {}).get("ece")) for k in qkeys]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def coverage_table(summary: dict[str, Any], question: str = "next_action") -> str:
    curve = summary["questions"][question]["coverage_curve"]
    lines = ["| threshold | coverage | accuracy (covered) |", "|---|---|---|"]
    for c in curve:
        lines.append(f"| {c['threshold']:.1f} | {_f(c['coverage'])} | {_f(c['accuracy_covered'])} |")
    return "\n".join(lines)


def confusion_table(summary: dict[str, Any], question: str = "next_action") -> str:
    conf = summary["questions"][question].get("confusion", {})
    labels = sorted(set(conf) | {p for row in conf.values() for p in row})
    lines = ["| label \\ pred | " + " | ".join(labels) + " |", "|" + "---|" * (len(labels) + 1)]
    for y in labels:
        lines.append(f"| {y} | " + " | ".join(str(conf.get(y, {}).get(p, 0)) for p in labels) + " |")
    return "\n".join(lines)


def online_table(summaries: list[dict[str, Any]]) -> str:
    head = [
        "backend",
        "modality",
        "episodes",
        "success",
        "mean steps",
        "action acc",
        "p50 ms",
        "p95 ms",
        "escalations",
        "cost $",
    ]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for s in summaries:
        lines.append(
            "| "
            + " | ".join(
                [
                    s["backend"],
                    s["modality"],
                    str(s["episodes"]),
                    _f(s["success_rate"]),
                    _f(s["mean_steps"], 1),
                    _f(s["action_accuracy"]),
                    _f(s["latency_p50_s"] * 1000, 0),
                    _f(s["latency_p95_s"] * 1000, 0),
                    str(s["escalations"]),
                    _f(s["cost_usd"], 4),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def build_report(results_dir: str | Path, out_file: str | Path | None = None) -> str:
    d = Path(results_dir)
    offline = [json.load(open(p, encoding="utf-8")) for p in sorted(d.glob("offline/*.summary.json"))]
    online = [json.load(open(p, encoding="utf-8")) for p in sorted(d.glob("online/*.summary.json"))]
    parts = [f"# decision-layer-bench report\n\nresults: `{d}`\n"]
    if offline:
        parts.append("## Offline (same samples for every backend)\n\n" + offline_table(offline))
        for s in offline:
            parts.append(f"\n### {s['run']} – next_action confusion\n\n" + confusion_table(s))
            parts.append(f"\n### {s['run']} – gating curve (next_action)\n\n" + coverage_table(s))
    if online:
        parts.append("\n## Online episodes\n\n" + online_table(online))
    md = "\n".join(parts) + "\n"
    if out_file:
        Path(out_file).write_text(md, encoding="utf-8")
    return md
