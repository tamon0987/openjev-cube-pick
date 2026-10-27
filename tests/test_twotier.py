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
from dlb.sim.env import OPEN_MIN, Z_GRASP, Z_HOVER, PickPlaceEnv


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
# the visual held check along the carry (TwoTierRunner.held_check), with fakes
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


def test_runner_held_check_aborts_the_carry_and_picks_again(tmp_path):
    import json

    from dlb.harness.twotier import RELEASE_Z, TRANSPORT_Z, TaskPlanner, TwoTierRunner

    env = _CarryEnv()
    guide = _CarryGuide(env)
    tasks = [{"op": "pick", "object": "green cube"}, {"op": "place", "where": "bin"}]
    runner = TwoTierRunner(
        env, TaskPlanner(tasks), _CarryPolicy(), log_dir=tmp_path, save_images=False, guide=guide
    )
    seen = []
    answers = iter([True, False])  # grasp ok, then the lift check says the cube is gone

    def held_check(env_, images, where):
        seen.append((where, round(float(env_.tcp_pos[2]), 3)))
        ok = next(answers, True)
        return ok, 0.9 if ok else 0.1

    runner.held_check = held_check
    res = runner.run(0, seed=0, reset=False, keep_marks=True)
    assert res.stop_reason == "planner_complete" and res.drops == 1 and res.missed_grasps == 0
    assert res.held_checks == 6 and [w for w, _ in seen] == [
        "grasp",
        "lift",
        "grasp",
        "lift",
        "travel",
        "release",
    ]
    assert seen[1][1] == TRANSPORT_Z and seen[4][1] == TRANSPORT_Z and seen[5][1] == RELEASE_Z
    assert not env.held and env.aperture > OPEN_MIN  # released in the bin at the end
    # after the drop: opened, raised, the scene marked again, the planner restarted from the pick
    assert ("reset", None, None) in guide.calls and ("mark", "begin", None) in guide.calls
    assert [c for c in guide.calls if c[0] in ("cube", "bin")] == [
        ("cube", Z_HOVER, None),
        ("cube", Z_HOVER, None),
        ("bin", TRANSPORT_Z, None),
    ]
    recs = [json.loads(ln) for ln in (tmp_path / "ep0000.jsonl").read_text().splitlines()]
    checks = [r for r in recs if r["type"] == "held_check"]
    assert [(c["where"], c["held"], c["p_yes"]) for c in checks][:2] == [
        ("grasp", True, 0.9),
        ("lift", False, 0.1),
    ]
    drop = [r for r in recs if r["type"] == "event" and r["outcome"] == "dropped"]
    assert len(drop) == 1 and drop[0]["task_index"] == 0 and "lift" in drop[0]["reason"]


def test_runner_without_a_held_check_trusts_the_gripper(tmp_path):
    from dlb.harness.twotier import TaskPlanner, TwoTierRunner

    env = _CarryEnv()
    runner = TwoTierRunner(
        env,
        TaskPlanner([{"op": "pick", "object": "x"}, {"op": "place", "where": "bin"}]),
        _CarryPolicy(),
        log_dir=None,
        guide=_CarryGuide(env),
    )
    res = runner.run(0, seed=0, reset=False, keep_marks=True)
    assert res.stop_reason == "planner_complete" and res.drops == 0 and res.held_checks == 0


def test_runner_held_check_after_the_grasp_counts_as_a_missed_grasp(tmp_path):
    from dlb.harness.twotier import TaskPlanner, TwoTierRunner

    env = _CarryEnv()
    runner = TwoTierRunner(
        env,
        TaskPlanner([{"op": "pick", "object": "x"}, {"op": "place", "where": "bin"}]),
        _CarryPolicy(),
        log_dir=None,
        guide=_CarryGuide(env),
    )
    answers = iter([False])
    runner.held_check = lambda e, i, w: (next(answers, True), 0.5)
    res = runner.run(0, seed=0, reset=False, keep_marks=True)
    assert res.stop_reason == "planner_complete" and res.missed_grasps == 1 and res.drops == 0


def test_jev_held_check_asks_one_noul_on_the_wrist_image():
    from dlb.contract import make_noul_answer
    from dlb.harness.twotier import JevHeldCheck, set_object_names

    class Backend:
        def __init__(self):
            self.reqs = []

        def decide(self, req):
            self.reqs.append(req)
            return SimpleNamespace(answers={"held": make_noul_answer(0.2)})

    set_object_names("green cube", "tray")
    be = Backend()
    check = JevHeldCheck(be)
    env = SimpleNamespace(render_all=lambda: {"wrist": np.zeros((8, 8, 3), np.uint8)})
    held, p = check(env, {"wrist": np.zeros((8, 8, 3), np.uint8)}, "lift")
    assert (held, p) == (False, 0.2) and check.records == [{"where": "lift", "p_yes": 0.2}]
    req = be.reqs[0]
    assert list(req.questions) == ["held"] and "green cube" in req.questions["held"].instructions
    assert (
        req.state["stage"] == "lift" and len(req.images) == 1 and req.images[0].startswith("data:image/jpeg")
    )
