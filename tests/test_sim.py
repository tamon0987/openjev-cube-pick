import numpy as np
import pytest

from dlb.sim.env import PRIMITIVES, PickPlaceEnv
from tests.conftest import render_available


def test_oracle_solves_task_reliably():
    env = PickPlaceEnv(render=False)
    successes = 0
    for seed in range(12):
        env.reset(seed=seed)
        while not env.episode_over():
            env.execute(env.oracle_action())
        successes += env.is_success()
        assert env.step_count <= 9, f"oracle took too long on seed {seed}: {env.history}"
    assert successes == 12


def test_state_json_has_no_privileged_flags():
    env = PickPlaceEnv(render=False)
    obs = env.reset(seed=1)
    flat = str(obs.state_json).lower()
    for forbidden in ("held", "holding", "in_bin", "success", "aligned"):
        assert forbidden not in flat, forbidden
    assert obs.privileged["held"] is False
    assert all(k in PRIMITIVES for k in ["hover_object", "grasp", "done"])


def test_labels_track_phases():
    env = PickPlaceEnv(render=False)
    env.reset(seed=2)
    seen = []
    while not env.episode_over():
        lab = env.oracle_labels()
        seen.append(lab["progress"])
        env.execute(lab["next_action"])
    assert seen[0] == 0 and 2 in seen and 4 in seen
    assert env.oracle_labels()["task_complete"] == "yes"


def test_off_nominal_recovery():
    """Closing on nothing, then wandering: oracle must still finish."""
    env = PickPlaceEnv(render=False, max_steps=20)
    env.reset(seed=5)
    env.execute("grasp")
    assert env.oracle_action() == "release"
    env.execute("hover_bin")
    while not env.episode_over():
        env.execute(env.oracle_action())
    assert env.is_success()


@pytest.mark.skipif(not render_available(), reason="no offscreen GL")
def test_render_shapes():
    env = PickPlaceEnv(render=True, image_size=96)
    obs = env.reset(seed=0)
    assert set(obs.images) == {"front", "wrist"}
    assert obs.images["front"].shape == (96, 96, 3)
    assert obs.images["front"].dtype == np.uint8
    assert obs.images["wrist"].std() > 1.0  # not a blank frame
