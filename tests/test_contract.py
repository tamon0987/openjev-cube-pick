import numpy as np

from dlb.contract import (
    Answer,
    Choice,
    DecisionRequest,
    DecisionResponse,
    Noul,
    Score,
    image_to_data_url,
    make_choice_answer,
    make_noul_answer,
    make_score_answer,
)


def test_wire_roundtrip():
    req = DecisionRequest(
        state={"a": 1},
        questions={
            "c": Choice(instructions="pick", criteria={"x": "X", "y": "Y"}),
            "n": Noul(instructions="yes?"),
            "s": Score(instructions="how much", criteria=["low", "mid", "high"]),
        },
        images=[image_to_data_url(np.zeros((8, 8, 3), dtype=np.uint8))],
    )
    body = req.to_wire(model="jev-latest")
    assert body["model"] == "jev-latest"
    assert body["questions"]["c"]["type"] == "choice"
    assert body["questions"]["s"]["criteria"] == ["low", "mid", "high"]
    assert body["images"][0].startswith("data:image/png;base64,")
    assert "images" not in req.to_wire(include_images=False)


def test_answer_parsing_and_labels():
    resp = DecisionResponse.from_wire(
        {
            "model": "jev-1.13.0",
            "answers": {
                "c": {
                    "type": "choice",
                    "choice": "y",
                    "probabilities": {"x": 0.2, "y": 0.8},
                    "confidence": 0.6,
                },
                "n": {"type": "noul", "noul": 0.93},
                "s": {
                    "type": "score",
                    "score": 1.3,
                    "confidence": 0.5,
                    "legend": {"0": "a", "1": "b", "2": "c"},
                    "probabilities": {"0": 0.0, "1": 0.7, "2": 0.3},
                },
            },
            "usage": {"input_tokens": 210, "output_tokens": 31},
        },
        latency_s=0.1,
    )
    assert resp.answers["c"].label == "y" and abs(resp.answers["c"].conf - 0.6) < 1e-9
    assert resp.answers["n"].label == "yes" and abs(resp.answers["n"].conf - 0.86) < 1e-9
    assert resp.answers["s"].label == "1"
    assert resp.usage.input_tokens == 210


def test_local_answer_builders():
    a = make_choice_answer("b", ["a", "b", "c"], p=0.9)
    assert a.choice == "b" and abs(sum(a.probabilities.values()) - 1) < 1e-9
    assert make_noul_answer(0.2).label == "no"
    s = make_score_answer(2, ["l0", "l1", "l2"], p=1.0)
    assert s.label == "2" and abs(s.score - 2.0) < 1e-9


def test_score_level_bounds():
    import pytest

    with pytest.raises(ValueError):
        Score(instructions="x", criteria=["only one"])


def test_answer_conf_fallbacks():
    assert Answer(type="choice", choice="a", probabilities={"a": 0.7, "b": 0.3}).conf == 0.7
    assert Answer(type="noul", noul=0.5).conf == 0.0
