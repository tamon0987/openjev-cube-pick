import pytest

from dlb.backends import build_backend, list_profiles
from dlb.contract import Choice, DecisionRequest, Noul, Score
from dlb.harness import EpisodeRunner, GatingConfig, summarize
from dlb.sim.env import PickPlaceEnv


def _req(images=None):
    return DecisionRequest(
        state={"ticket": "checkout is down"},
        questions={
            "route": Choice(instructions="team?", criteria={"billing": "b", "support": "s"}),
            "urgent": Noul(instructions="urgent?"),
            "sev": Score(instructions="sev", criteria=["low", "high"]),
        },
        images=images or [],
    )


def test_profiles_load():
    names = list_profiles()
    for n in ("typesafe", "openjev", "djev", "oracle", "random"):
        assert n in names
    be = build_backend("typesafe")
    assert be.info.modality == "text" and be.endpoint == "/v1/systemone"
    be = build_backend("djev")
    assert be.info.modality == "image" and be.endpoint == "/v1/request"


def test_http_backend_against_mock(mock_server):
    be = build_backend("typesafe", base_url=mock_server, api_key="x")
    resp = be.decide(_req())
    assert set(resp.answers) == {"route", "urgent", "sev"}
    assert resp.usage.input_tokens > 0
    assert resp.answers["route"].choice in ("billing", "support")


def test_http_backend_image_policies(mock_server):
    img = ["data:image/png;base64,iVBORw0KGgo="]
    text_be = build_backend("typesafe", base_url=mock_server)
    body = text_be._body(_req(images=img))
    assert "images" not in body  # dropped
    img_be = build_backend("djev", base_url=mock_server)
    body = img_be._body(_req(images=img))
    assert body["images"] == img and "model" not in body
    err_be = build_backend("typesafe", base_url=mock_server, image_policy="error")
    with pytest.raises(ValueError):
        err_be._body(_req(images=img))


def test_oracle_backend_and_gating():
    env = PickPlaceEnv(render=False)
    oracle = build_backend("oracle")
    r = EpisodeRunner(env, oracle, modality="text").run(0, seed=0)
    assert r.success and r.action_accuracy == 1.0

    noisy = build_backend("oracle_noisy", noise=0.5, confidence=0.6)
    runner = EpisodeRunner(
        env, noisy, modality="text", gating=GatingConfig(threshold=0.9, on_low_confidence="oracle")
    )
    results = [runner.run(i, seed=i) for i in range(3)]
    s = summarize(results)
    assert s["escalations"] == s["decisions"]  # every decision was below the gate -> oracle took over
    assert s["success_rate"] == 1.0


def test_random_backend_runs():
    env = PickPlaceEnv(render=False, max_steps=5)
    r = EpisodeRunner(env, build_backend("random"), modality="text").run(0, seed=3)
    assert r.steps <= 5


def test_nested_image_field(mock_server):
    img = ["data:image/png;base64,iVBORw0KGgo="]
    be = build_backend("djev", base_url=mock_server, image_field="state.images")
    body = be._body(_req(images=img))
    assert body["state"]["images"] == img and "images" not in body
