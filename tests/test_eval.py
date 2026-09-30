import json

import pytest

from dlb.backends import build_backend
from dlb.eval.dataset import generate, load
from dlb.eval.metrics import expected_calibration_error, macro_f1
from dlb.eval.offline import run_offline
from dlb.eval.report import build_report
from tests.conftest import render_available


def test_metrics_basic():
    assert macro_f1(["a", "b"], ["a", "b"]) == 1.0
    assert expected_calibration_error([1.0, 1.0], [True, True]) == 0.0
    assert expected_calibration_error([1.0, 1.0], [False, False]) == 1.0


@pytest.mark.skipif(not render_available(), reason="no offscreen GL")
def test_dataset_offline_report(tmp_path, mock_server):
    ds = generate(tmp_path / "data", n_episodes=2, epsilon=0.3, seed=1, image_size=64)
    samples = load(ds)
    assert len(samples) >= 10
    assert all((tmp_path / "data" / v).exists() for s in samples[:3] for v in [f"images/{s.id}_front.png"])

    out = tmp_path / "results" / "offline"
    s1 = run_offline(build_backend("oracle"), samples, modality="text", out_dir=out, progress=False)
    assert s1["questions"]["next_action"]["accuracy"] == 1.0
    s2 = run_offline(
        build_backend("djev", base_url=mock_server), samples, modality="image", out_dir=out, progress=False
    )
    assert s2["answered"] == len(samples) and s2["errors"] == 0
    assert s2["input_tokens_per_sample"] > 500  # mock charges 280 tokens per image
    s3 = run_offline(
        build_backend("typesafe", base_url=mock_server), samples, modality="text", out_dir=out, progress=False
    )
    assert s3["cost_usd"] > 0

    md = build_report(tmp_path / "results", out_file=tmp_path / "report.md")
    assert "oracle__text" in md and "djev__image" in md
    raw = [json.loads(line) for line in open(out / "oracle__text.jsonl")]
    assert raw[0]["answers"]["next_action"]["correct"] is True
