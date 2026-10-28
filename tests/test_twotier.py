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


# --------------------------------------------------------------------------- #
# the grasp and carry sequence, with fakes
# --------------------------------------------------------------------------- #
class _CarryEnv:
    """Just enough robot for the task-list loop: a gripper that closes on a cube when it is low enough."""

    has_truth = False

    def __init__(self):
        from dlb.harness.twotier import TRANSPORT_Z

        self.tcp_pos = np.array([0.15, 0.05, 0.12])
        self.held, self.precise, self.step_count = False, True, 0
        self.aperture = OPEN_MIN + 0.01
        self.moves: list[tuple[str, float]] = []
        self.transport_z = TRANSPORT_Z

    def wrist_axes(self):
        return np.array([1.0, 0.0]), np.array([0.0, -1.0])

    def move_relative(self, delta, from_measured=False):
        self.tcp_pos = self.tcp_pos + np.asarray(delta, float)
        self.step_count += 1
        self.moves.append(("rel", round(float(self.tcp_pos[2]), 3)))
        return {}

    def close_gripper(self):
        self.held = self.tcp_pos[2] < Z_GRASP + 0.01
        self.aperture = 0.03 if self.held else 0.0
        self.step_count += 1
        return self.held

    def open_gripper(self):
        self.held, self.aperture = False, OPEN_MIN + 0.01
        self.step_count += 1

    def render_all(self):
        return {"wrist": np.zeros((32, 32, 3), np.uint8)}

    def overhead_frame(self):
        return np.zeros((32, 32, 3), np.uint8)

    def is_success(self):
        return False


class _CarryGuide:
    container, bin_name, pending = True, "bin", ""

    def __init__(self, env):
        self.env, self.calls = env, []

    def goto(self, env, target, z, travel_z=None):
        from dlb.harness.marking import GotoResult

        self.calls.append((target, round(z, 3), travel_z))
        env.tcp_pos = np.array([0.25 if target == "bin" else 0.15, 0.0, z])
        return GotoResult(True, None, 1, 0.0, "moved", [])

    def set_targets(self, pick=None, place=None):
        pass

    def on_grasp(self, env):
        self.calls.append(("on_grasp", None, None))

    def reset(self):
        self.calls.append(("reset", None, None))

    def start_background(self, env, kind="refine"):
        self.calls.append(("mark", kind, None))

    def join(self):
        return None


class _CarryPolicy:
    name, needs_images = "carry", False

    def decide(self, env, sub, images):
        from dlb.harness.twotier import Decision

        if sub.name == "align_cube":
            return Decision("SUBTASK_DONE", 1.0)
        if env.tcp_pos[2] > Z_GRASP + 0.005:
            return Decision("MV_DOWN", 1.0, step_scale=(env.tcp_pos[2] - Z_GRASP) / sub.step_m)
        return Decision("GRASP", 1.0)


def test_grasp_carries_until_release_without_visual_hold_queries():
    from dlb.harness.twotier import TRANSPORT_Z, TaskPlanner, TwoTierRunner

    env = _CarryEnv()
    released = []
    original_open = env.open_gripper

    def record_open():
        released.append((env.held, env.tcp_pos.copy()))
        original_open()

    env.open_gripper = record_open
    guide = _CarryGuide(env)
    runner = TwoTierRunner(
        env,
        TaskPlanner([{"op": "pick", "object": "cube"}, {"op": "place", "where": "bin"}]),
        _CarryPolicy(),
        log_dir=None,
        guide=guide,
    )
    result = runner.run(0, seed=0, reset=False, keep_marks=True)
    assert result.stop_reason == "planner_complete" and result.missed_grasps == 0
    assert [call for call in guide.calls if call[0] == "bin"] == [("bin", TRANSPORT_Z, None)]
    assert len(released) == 1 and released[0][0]
    assert released[0][1][0] == 0.25  # only opens once it reaches the bin


def test_cross_point_shifts_along_the_wrist_up_axis():
    from dlb.harness.twotier import cross_point

    class Env:
        tcp_pos = np.array([0.2, 0.0, 0.1])
        cross_offset_fwd = 0.01

        def wrist_axes(self):
            return np.array([0.0, 1.0]), np.array([1.0, 0.0])

    p = cross_point(Env(), 0.015)
    assert np.allclose(p, [0.2, 0.01, 0.015])  # 1 cm along "up" (here +y), at the asked height
    Env.cross_offset_fwd = 0.0
    assert np.allclose(cross_point(Env(), 0.015), [0.2, 0.0, 0.015])


def test_bisect_servo_trusts_the_overhead_guide_over_a_not_visible_wrist_answer():
    """Up close the wrist model answered "red cube not visible" (p 0.02) with the cube under the cross; after
    a guided move the servo aligns on the side answers instead of escalating, once per guided move."""
    from dlb.contract import Answer

    def ask(env, sub, images, axes):
        side = {"left": 0.62, "right": 0.38}
        ans = {ax: {"side": Answer(type="choice", choice="left", probabilities=side)} for ax in axes}
        if "h" in ans:
            ans["h"]["visible"] = Answer(type="noul", noul=0.02)
        return ans, 0.0, 0

    policy = JevBisectServoPolicy(backend=None)
    policy._ask = ask
    env = SimpleNamespace(tcp_pos=np.array([0.16, -0.07, 0.114]), aperture=OPEN_MIN + 0.01, held=False)

    def align(uid: int) -> Subtask:
        return Subtask("align_cube", "", "", SUBTASKS["align_cube"], step_cm=1.0, uid=uid)

    assert policy.decide(env, align(1), {}).command == "ESCALATE"
    policy.overhead_on_target = True
    assert policy.decide(env, align(2), {}).command == "SUBTASK_DONE"  # unsure sides: close enough
    assert policy.decide(env, align(3), {}).command == "ESCALATE"  # the trust is used up
