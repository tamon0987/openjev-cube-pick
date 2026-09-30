from types import SimpleNamespace

import numpy as np

from dlb.harness.twotier import (
    GRASP_Z_TOL,
    SUBTASKS,
    JevBisectServoPolicy,
    OraclePolicy,
    ScriptedPlanner,
    Subtask,
    TwoTierRunner,
    summarize_twotier,
)
from dlb.sim.env import OPEN_MIN, Z_GRASP, PickPlaceEnv


def test_oracle_commands_complete_the_task_through_the_two_tier_harness():
    env = PickPlaceEnv(render=False)
    runner = TwoTierRunner(env, ScriptedPlanner(step_cm=2), OraclePolicy(), log_dir=None)
    results = [runner.run(i, seed=2000 + i) for i in range(6)]
    s = summarize_twotier(results)
    assert s["success_rate"] == 1.0, [(r.seed, r.subtasks, r.stop_reason) for r in results]
    assert s["command_accuracy"] == 1.0
    assert all(r.decisions <= 60 for r in results)


def test_relative_moves_follow_the_wrist_image_axes():
    env = PickPlaceEnv(render=False)
    env.reset(seed=0)
    fwd, right = env.wrist_axes()
    assert abs(float(fwd @ right)) < 1e-3
    start = env.tcp_pos.copy()
    env.move_relative([*(right * 0.02), 0.0])
    moved = env.tcp_pos[:2] - start[:2]
    assert float(moved @ right) > 0.015 and abs(float(moved @ fwd)) < 0.005


def _descend() -> Subtask:
    return Subtask("descend_grasp", "", "", SUBTASKS["descend_grasp"], step_cm=1.0)


def test_bisect_servo_descends_to_grasp_height_in_one_move_and_grasps_within_the_band():
    policy = JevBisectServoPolicy(backend=None)
    sub = _descend()

    def decide(z: float, aperture: float = OPEN_MIN + 0.01, held: bool = False):
        env = SimpleNamespace(tcp_pos=np.array([0.17, -0.09, z]), aperture=aperture, held=held)
        return policy.decide(env, sub, {})

    d = decide(0.105)  # from the observe height: one move of the whole way down
    assert d.command == "MV_DOWN"
    assert abs(d.step_scale * sub.step_m - (0.105 - Z_GRASP)) < 1e-9
    # the precise move rests a few mm high (residual sag): still a grasp, no second move
    assert decide(0.031).command == "GRASP"
    assert decide(Z_GRASP + GRASP_Z_TOL).command == "GRASP"
    # stopped well short: move again, still aimed at grasp height (never below it)
    d = decide(0.040)
    assert d.command == "MV_DOWN" and abs(d.step_scale * sub.step_m - (0.040 - Z_GRASP)) < 1e-9
    assert decide(0.03, aperture=0.02, held=True).command == "SUBTASK_DONE"
    assert decide(0.03, aperture=0.0).command == "ESCALATE"


def test_descent_moves_from_the_measured_tcp_not_a_stale_command():
    env = PickPlaceEnv(render=False)
    env.reset(seed=0)
    runner = TwoTierRunner(env, ScriptedPlanner(), JevBisectServoPolicy(backend=None), log_dir=None)
    tcp = env.tcp_pos.copy()
    env._tcp_cmd = tcp + np.array(
        [0.004, -0.003, 0.012]
    )  # an uncorrected fast move rested low of its command
    sub = _descend()
    runner._execute("MV_DOWN", sub, (tcp[2] - Z_GRASP) / sub.step_m)
    assert abs(env.tcp_pos[2] - Z_GRASP) < 0.004, env.tcp_pos
    assert np.linalg.norm(env.tcp_pos[:2] - tcp[:2]) < 0.004  # keeps the xy the servo aligned
