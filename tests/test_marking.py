import json
import time

import numpy as np

from dlb.harness.marking import Marks, OverheadGuide, TableMap, box_centred, fit_scale, summarize
from dlb.harness.twotier import (
    OVERHEAD_ORDER,
    TRANSPORT_Z,
    OraclePolicy,
    SequencePlanner,
    TwoTierResult,
    TwoTierRunner,
)
from dlb.sim.env import PickPlaceEnv


def _camera(xy: np.ndarray, s: float = 660.0, rot_deg: float = 8.0, c=(350.0, 400.0)) -> np.ndarray:
    """A synthetic overhead camera: forward ~ image up, left ~ image left, slightly rotated."""
    a = s * np.exp(1j * np.radians(rot_deg))
    w = a * complex(-xy[1], -xy[0]) + complex(*c)
    return np.array([w.real, w.imag])


def test_prior_map_points_the_right_way():
    m = TableMap(px_per_m=900.0)  # scale overestimated, as from marks of the gripper above the table
    m.add(np.array([0.19, 0.0]), _camera(np.array([0.19, 0.0])))
    cube = np.array([0.20, -0.04])
    est = m.to_table(_camera(cube))
    step, true = est - [0.19, 0.0], cube - [0.19, 0.0]
    assert step @ true > 0 and np.linalg.norm(step) < np.linalg.norm(
        true
    )  # undershoots in the right direction


def test_spread_marks_recover_the_camera():
    m = TableMap(px_per_m=900.0)
    for xy in ([0.19, 0.0], [0.17, -0.05], [0.18, 0.12]):
        m.add(np.array(xy), _camera(np.array(xy)), sd_px=1.0)  # exact marks: the pairs outweigh the prior
    cube = np.array([0.24, 0.10])
    assert np.linalg.norm(m.to_table(_camera(cube)) - cube) < 0.002
    assert abs(m.describe()["rotation_deg"] - 8.0) < 0.2


def test_close_noisy_marks_keep_the_prior_rotation():
    # the first real run: pairs 3-4 cm apart plus ~5 px marking noise fitted a -46 degree rotation
    m = TableMap(px_per_m=750.0)
    rng = np.random.default_rng(0)
    for xy in ([0.187, -0.001], [0.179, -0.035], [0.174, -0.054]):
        m.add(np.array(xy), _camera(np.array(xy), rot_deg=0.0) + rng.normal(0, 5, 2))
    assert m.describe()["rotation_deg"] == 0.0
    assert abs(m.describe()["px_per_cm"] - 6.6) < 1.0
    bin_ = np.array([0.20, 0.12])
    assert np.linalg.norm(m.to_table(_camera(bin_, rot_deg=0.0)) - bin_) < 0.03


def test_close_pairs_lean_on_the_prior_scale():
    """Gripper marks a few cm apart (15 px noise) plus the exact grasp pair: the bin 20 cm away.

    real_overhead2: three pairs within 9 cm fitted 5.5 px/cm (true ~6.6) and put the bin 6 cm off. With the prior
    in the fit (10% off) the bin lands within 3 cm in ~85% of the draws, the fit alone in ~45%.
    """
    rng = np.random.default_rng(1)
    bin_ = np.array([0.15, 0.13])
    hits = {True: 0, False: 0}
    for _ in range(300):
        prior = 660.0 * rng.normal(1.0, 0.1)
        for use_prior in (True, False):
            m = TableMap(px_per_m=prior if use_prior else None)
            for xy in ([0.184, -0.001], [0.171, -0.060]):
                m.add(
                    np.array(xy),
                    _camera(np.array(xy), rot_deg=0.0) + rng.normal(0, 15, 2),
                    sd_px=OverheadGuide.GRIPPER_SD_PX,
                )
            grasp = np.array([0.169, -0.096])
            m.add(grasp, _camera(grasp, rot_deg=0.0) + rng.normal(0, 3, 2), sd_px=OverheadGuide.CUBE_SD_PX)
            est = m.to_table(_camera(bin_, rot_deg=0.0) + rng.normal(0, 3, 2))
            hits[use_prior] += int(np.linalg.norm(est - bin_) < 0.03)
    assert hits[True] >= 0.8 * 300, hits
    assert hits[True] > hits[False] + 60, hits


def test_bin_centre_from_point_and_box():
    raw = [
        {
            "gripper": None,
            "cube": [500, 500],
            "cube_box": [490, 490, 510, 510],
            "bin": [300, 400],
            "bin_box": [250, 350, 370, 470],
        }
        for _ in range(3)
    ]
    m = summarize(raw, 1000, 1000)
    assert np.allclose(m.points["bin"], [300, 400]) and m.points["gripper"] is None  # the marked centres
    box_centred(m, "bin")  # what the guide does for the configured bin, whatever the objects are called
    assert np.allclose(m.points["bin"], [305, 405])  # mean of the marked centre and the box centre
    assert np.allclose(m.points["cube"], [500, 500])
    raw[0]["bin"] = raw[1]["bin"] = [100, 100]  # a centre outside its box: the box's centre alone
    m = summarize(raw, 1000, 1000)
    box_centred(m, "bin")
    assert np.allclose(m.points["bin"], [310, 410])


# --------------------------------------------------------------------------- #
# replay of the real runs (marks from results/twotier/logs/real_overhead{1,2}; the background marks, which the
# logs did not record, read back from the saved mark images)
# --------------------------------------------------------------------------- #
REAL = {
    # begin tcp, first mark (gripper, spread, cube, bin, px per 3.6 cm), tcp after goto_cube, second mark,
    # grasp tcp, bin centre from the release image (the gripper's position over the bin; +-1.5 cm)
    "real_overhead1": dict(
        begin=(0.1859, -0.0002, 0.0885),
        m1=((352, 276), 1.3, (366.7, 275.5), (243.2, 279.4), 22.7),
        after=(0.186, -0.0208, 0.119),
        m2=((357.1, 268.8), 1.9, (364.8, 266.9), (243.2, 279.4), None),
        grasp=(0.1606, -0.0853, 0.048),
        bin=(0.152, 0.136),
    ),
    "real_overhead2": dict(
        begin=(0.1838, -0.0008, 0.0856),
        m1=((336.0, 260.2), 11.0, (386.6, 271.2), (243.2, 266.9), 25.6),
        after=(0.1708, -0.0601, 0.12),
        m2=((389, 263), 3.0, (391, 265), (241, 267), None),
        grasp=(0.1692, -0.0956, 0.0329),
        bin=(0.146, 0.122),
    ),
}


def _marks(g, gs, c, b, size) -> Marks:
    pts = {"gripper": np.array(g, float), "cube": np.array(c, float), "bin": np.array(b, float)}
    return Marks(pts, {"gripper": gs, "cube": 1.0, "bin": 1.0}, 0.0, [], {"cube": None, "bin": None})


def _guide(marker, r) -> OverheadGuide:
    # the scale those runs worked with (it now comes from the start-up calibration)
    return OverheadGuide(marker, TableMap(px_per_m=r["m1"][4] / 0.036))


class _ReplayEnv:
    def __init__(self, begin, after):
        self.tcp_pos, self.after, self.moves = np.array(begin, float), np.array(after, float), []

    def overhead_frame(self):
        return np.zeros((480, 640, 3), np.uint8)

    def _move_tcp(self, target):
        self.moves.append(np.array(target))
        self.tcp_pos = self.after.copy() if len(self.moves) == 1 else np.array(target, float)


class _ListMarker:
    def __init__(self, marks, fail_first=False, delay=0.0):
        self.marks, self.fail_first, self.delay, self.calls = list(marks), fail_first, delay, 0

    def mark(self, img):
        self.calls += 1
        time.sleep(self.delay)
        if self.fail_first and self.calls == 1:
            raise RuntimeError("no answer")
        return self.marks.pop(0)


def test_replay_real_runs_release_near_the_bin():
    for name, r in REAL.items():
        env = _ReplayEnv(r["begin"], r["after"])
        marker = _ListMarker([_marks(*r["m1"]), _marks(*r["m2"])])
        g = _guide(marker, r)
        g.start_background(env, kind="begin")
        assert g.goto(env, "cube", 0.12).log[0]["begin_mark"] == "used"
        env.tcp_pos = np.array(r["grasp"])
        g.on_grasp(env)
        res = g.goto(env, "bin", 0.10)
        assert marker.calls == 2 and res.log[0]["bin_marks"] == 2
        err = res.target_xy - r["bin"]
        # left-right (where real_overhead2 missed by 6 cm, at 5.5 px/cm): within 2 cm; overall within 3 cm
        assert abs(err[1]) < 0.02 and np.linalg.norm(err) < 0.03, (name, res.target_xy)
        assert 5.0 < res.log[0]["map"]["px_per_cm"] < 7.5, (name, res.log[0]["map"])


def test_failed_begin_mark_falls_back_to_a_foreground_mark():
    r = REAL["real_overhead2"]
    env = _ReplayEnv(r["begin"], r["after"])
    marker = _ListMarker([_marks(*r["m1"]), _marks(*r["m2"])], fail_first=True)
    g = _guide(marker, r)
    g.start_background(env, kind="begin")
    rec = g.goto(env, "cube", 0.12).log[0]
    assert rec["begin_mark"] == "failed" and rec["mark_kind"] == "foreground"
    g.join()
    assert marker.calls == 3  # the failed one, the foreground one, the refinement after the move


def test_begin_mark_overlaps_the_reset():
    r = REAL["real_overhead2"]
    env = _ReplayEnv(r["begin"], r["after"])
    g = _guide(_ListMarker([_marks(*r["m1"]), _marks(*r["m2"])], delay=0.3), r)
    t0 = time.perf_counter()
    g.start_background(env, kind="begin")
    time.sleep(0.25)  # the rest of the reset
    rec = g.goto(env, "cube", 0.12).log[0]
    assert rec["begin_mark"] == "used" and rec["wait_s"] < 0.2 and time.perf_counter() - t0 < 0.5
    g.join()


# --------------------------------------------------------------------------- #
# the runner: begin pose at the start and the end of an episode, first mark during the reset
# --------------------------------------------------------------------------- #
class _PosedEnv(PickPlaceEnv):
    """The simulator with a robot's extras: saved poses, an overhead image, ``reset(on_begin=...)``."""

    BEGIN = np.array([0.184, 0.0, 0.086])

    def __init__(self):
        super().__init__(render=False)
        self.poses: list[str] = []
        self.frames: list[dict[str, np.ndarray]] = []

    def go_pose(self, name):
        self.poses.append(name)
        self._move_tcp(self.BEGIN.copy())

    def reset(self, seed=None, on_begin=None):
        obs = super().reset(seed)
        self.go_pose("begin")
        if on_begin is not None:
            on_begin()
        self.open_gripper()
        return obs

    def render_all(self):
        return {}

    def overhead_frame(self):
        self.frames.append(
            {"gripper": self.tcp_pos.copy(), "cube": self.cube_pos.copy(), "bin": self.bin_pos.copy()}
        )
        img = np.zeros((480, 640, 3), np.uint8)
        img[0, 0, 0] = len(self.frames) - 1
        return img


class _TruthMarker:
    """Marks the truth captured with the image through the synthetic camera."""

    def __init__(self, env):
        self.env, self.calls = env, 0

    def mark(self, img):
        self.calls += 1
        f = self.env.frames[int(img[0, 0, 0])]
        pts = {k: _camera(v, rot_deg=0.0) for k, v in f.items()}
        return Marks(pts, {k: 1.0 for k in pts}, 0.0, [], {})


def test_runner_starts_marking_at_the_begin_pose_and_returns_there(tmp_path):
    env = _PosedEnv()
    marker = _TruthMarker(env)
    runner = TwoTierRunner(
        env,
        SequencePlanner(overhead=True),
        OraclePolicy(),
        log_dir=tmp_path,
        save_images=False,
        guide=OverheadGuide(marker),
    )
    r = runner.run(0, seed=3)
    assert r.success and r.stop_reason == "planner_complete", (r.subtasks, r.stop_reason)
    # the first reset measures the scale: the calibration poses, back to the begin pose, then the two marks
    n = len(OverheadGuide.CALIB_OFFSETS)
    assert env.poses == ["begin", "begin", "begin"] and marker.calls == n + 2
    assert (
        abs(runner.guide.calibration["px_per_cm"] - 6.6) < 0.05 and runner.guide.calibration["inliers"] == n
    )
    r = runner.run(1, seed=4)  # the scale is measured once
    assert r.success and env.poses.count("begin") == 5 and marker.calls == n + 4
    recs = [json.loads(line) for line in open(tmp_path / "ep0000.jsonl")]
    guide = [x for x in recs if x["type"] == "guide"]
    assert guide[0]["log"][0]["begin_mark"] == "used"
    assert recs[-1]["type"] == "finish" and recs[-1]["action"] == "begin"
    assert np.allclose(env.tcp_pos, _PosedEnv.BEGIN, atol=0.01)


class _StopEnv:
    """Only what ``_finish`` touches."""

    def __init__(self, held, z=0.03):
        self.held, self.tcp_pos, self.calls = held, np.array([0.17, -0.09, z]), []

    def _move_tcp(self, target):
        self.calls.append(("move", *np.round(target, 3).tolist()))
        self.tcp_pos = np.array(target, float)

    def go_pose(self, name):
        self.calls.append(("pose", name))


def test_finish_returns_to_begin_unless_holding_after_a_failure():
    def finish(env, reason):
        runner = TwoTierRunner(env, SequencePlanner(overhead=True), OraclePolicy())
        res = TwoTierResult(0, 0, "p", "s", success=False, stop_reason=reason)
        out = []
        runner._finish(res, out.append)
        return out[-1]["action"]

    env = _StopEnv(held=False)
    assert finish(env, "planner_complete") == "begin"
    assert env.calls == [("move", 0.17, -0.09, TRANSPORT_Z), ("pose", "begin")]  # straight up first
    # with the begin pose's tcp known: straight up, level to above it, then the joint move
    env = _StopEnv(held=False)
    env.pose_tcp = {"begin": np.array([0.19, 0.0, 0.09])}
    assert finish(env, "planner_complete") == "begin"
    assert env.calls == [
        ("move", 0.17, -0.09, TRANSPORT_Z),
        ("move", 0.19, 0.0, TRANSPORT_Z),
        ("pose", "begin"),
    ]
    env = _StopEnv(held=False)
    assert finish(env, "max_decisions") == "begin" and env.calls[-1] == ("pose", "begin")
    env = _StopEnv(held=True)
    assert finish(env, "max_decisions") == "stay" and env.calls == []  # holding: stop where it is
    # the simulator (no saved poses) stays where it is
    runner = TwoTierRunner(PickPlaceEnv(render=False), SequencePlanner(), OraclePolicy())
    out = []
    runner._finish(TwoTierResult(0, 0, "p", "s", success=True, stop_reason="planner_complete"), out.append)
    assert out == []


def test_overhead_sequence_restarts_from_the_marks():
    p = SequencePlanner(overhead=True)

    class Env:
        held = False

    assert [s.name for s in p.plan(Env(), {}, []).subtasks] == OVERHEAD_ORDER
    ev = [{"outcome": "escalated", "subtask": "align_cube"}]
    assert p.plan(Env(), {}, ev).subtasks[0].name == "goto_cube"
    Env.held = True
    ev = [{"outcome": "escalated", "subtask": "goto_bin"}]
    assert [s.name for s in p.plan(Env(), {}, ev).subtasks] == ["goto_bin", "release"]


def test_task_planner_builds_and_restarts():
    from dlb.harness.twotier import TaskPlanner

    tasks = [
        {"op": "place", "where": "here"},
        {"op": "pick", "object": "blue cube"},
        {"op": "place", "where": "bin"},
    ]
    p = TaskPlanner(tasks, bin_name="black bin")

    class Env:
        held = True

    subs = p.plan(Env(), {}, []).subtasks
    assert [(s.name, s.target, s.place, s.task_index) for s in subs] == [
        ("place_at", "", "here", 0),
        ("goto_cube", "blue cube", "", 1),
        ("align_cube", "blue cube", "", 1),
        ("descend_grasp", "blue cube", "", 1),
        ("place_at", "", "bin", 2),
    ]
    Env.held = False
    ev = [{"outcome": "missed_grasp", "subtask": "descend_grasp", "task_index": 1}]
    assert p.plan(Env(), {}, ev).subtasks[0].name == "goto_cube"  # retry the pick
    p.plan(Env(), {}, ev)
    assert p.plan(Env(), {}, ev).task_complete  # third failure of the same task: give up
    assert p.plan(Env(), {}, [{"outcome": "queue_finished"}]).task_complete


def test_marks_keyed_by_object_name():
    from dlb.harness.marking import schema_for

    objs = ("red cube", "blue cube", "black bin")
    assert set(schema_for(objs)["required"]) == {
        "gripper",
        "red_cube",
        "blue_cube",
        "black_bin",
        "red_cube_box",
        "blue_cube_box",
        "black_bin_box",
    }
    raw = [
        {
            "gripper": [500, 800],
            "red_cube": [600, 500],
            "red_cube_box": [590, 490, 610, 510],
            "blue_cube": None,
            "blue_cube_box": None,
            "black_bin": [300, 400],
            "black_bin_box": [280, 380, 340, 440],
        }
    ] * 3
    m = summarize(raw, 1000, 1000, objects=objs)
    assert m.points["blue cube"] is None and np.allclose(m.points["red cube"], [600, 500])
    assert np.allclose(m.points["black bin"], [300, 400])


def test_goto_refuses_an_object_inside_the_bin():
    class M:
        names = ("orange cube", "black bin")

    g = OverheadGuide(M())
    pts = {
        "gripper": np.array([350.0, 280.0]),
        "orange cube": np.array([245.0, 280.0]),
        "black bin": np.array([243.0, 283.0]),
    }
    boxes = {"orange cube": np.array([235.0, 270, 255, 290]), "black bin": np.array([204.0, 245, 283, 320])}
    m = Marks(pts, {k: 1.0 for k in pts}, 0.0, [], boxes)
    g.marker.mark = lambda img: m

    class E:
        tcp_pos = np.array([0.187, 0.0, 0.093])

        def overhead_frame(self):
            return np.zeros((480, 640, 3), np.uint8)

        def _move_tcp(self, target):
            self.tcp_pos = np.array(target, float)

    r = g.goto(E(), "cube", 0.12)
    assert not r.ok and "inside the black bin" in r.reason
    # a flat mark as the place (not a container): an object lying on it can be picked again
    g = OverheadGuide(M(), TableMap(px_per_m=1150.0), container=False)
    g.marker.mark = lambda img: m
    r = g.goto(E(), "cube", 0.12)
    g.join()
    assert r.ok and r.reason == "moved"


def test_flat_place_sets_the_object_down():
    from dlb.harness.twotier import PLACE_Z_MARGIN, RELEASE_Z, Z_GRASP, Subtask

    class G:  # a guide that is already above the place
        bin_name, container = "red cross mark", False

        def set_targets(self, place=None):
            pass

        def goto(self, env, target, z, travel_z=None):
            env.tcp_pos = np.array([0.18, 0.11, z])
            return type("R", (), dict(ok=True, reason="moved", error_cm=0.0, log=[]))()

    class E:
        held, tcp_pos, calls = True, np.array([0.2, 0.0, 0.03]), []

        def move_relative(self, d, from_measured=False):
            self.tcp_pos = self.tcp_pos + d
            self.calls.append(("z", round(float(self.tcp_pos[2]), 4)))

        def open_gripper(self):
            self.calls.append(("open", round(float(self.tcp_pos[2]), 4)))

    def place(container):
        env, guide = E(), G()
        env.calls, guide.container = [], container
        runner = TwoTierRunner(env, SequencePlanner(overhead=True), OraclePolicy(), guide=guide)
        sub = Subtask("place_at", "", "", [], place="bin")
        res = TwoTierResult(0, 0, "p", "s", success=False)
        assert runner._place(sub, res, lambda *a: None, lambda rec: None) is None
        return env.calls

    low = round(Z_GRASP + PLACE_Z_MARGIN, 4)
    assert place(False) == [
        ("z", low),
        ("open", low),
        ("z", round(low + 0.03, 4)),
    ]  # set down, then clear of it
    assert place(True) == [("open", RELEASE_Z)]  # a bin: dropped in from the release height


def test_pick_target_keeps_its_marked_centre_whatever_it_is_called():
    class M:
        names = ("carrot plush toy", "white tape roll", "black bin")

    g = OverheadGuide(M(), roles={"cube": "carrot plush toy", "bin": "black bin"})
    pts = {
        "gripper": np.array([266.0, 250.0]),
        "carrot plush toy": np.array([240.0, 135.0]),
        "white tape roll": np.array([262.0, 215.0]),
        "black bin": np.array([160.0, 262.0]),
    }
    boxes = {
        "carrot plush toy": np.array([175.0, 105, 310, 167]),  # the leaves pull the box's centre off the body
        "white tape roll": np.array([207.0, 157, 322, 275]),
        "black bin": np.array([113.0, 207, 225, 315]),
    }
    g._use(None, np.array([0.187, 0.0, 0.093]), Marks(pts, {k: 1.0 for k in pts}, 0.0, [], boxes))
    assert np.allclose(g.last["cube"], [240, 135])  # not mixed with its box, although not a "cube"
    assert np.allclose(g.last["bin"], [164.5, 261.5])  # the bin: mean of the mark and the box's centre
    assert list(g._static) == ["black bin"]  # only the bin counts as static
    g.set_targets(place="white tape roll")  # stacking on another object: its latest mark, by role not by name
    assert np.allclose(g.last["bin"], [262, 215])


# --------------------------------------------------------------------------- #
# start-up calibration: the scale from the gripper's marks at known poses
# --------------------------------------------------------------------------- #
BEGIN = np.array([0.187, 0.0, 0.093])


def _ring(noise=0.0, seed=0, **cam):
    rng = np.random.default_rng(seed)
    xy = np.array([BEGIN[:2] + d for d in OverheadGuide.CALIB_OFFSETS])
    uv = np.array([_camera(p, **cam) for p in xy]) + rng.normal(0, noise, (len(xy), 2))
    return xy, uv


def test_fit_scale_recovers_the_camera():
    xy, uv = _ring(s=1150.0, rot_deg=4.0, c=(266.0, 465.0))
    f = fit_scale(xy, uv)
    assert f.ok and f.inlier.all() and abs(f.px_per_m - 1150) < 1e-6 and abs(f.rotation_deg - 4.0) < 1e-6
    for seed in range(20):  # gripper marks scatter by several pixels
        xy, uv = _ring(noise=6.0, seed=seed, s=1150.0, rot_deg=0.0, c=(266.0, 465.0))
        f = fit_scale(xy, uv)
        assert f.ok and abs(f.px_per_m / 1150 - 1) < 0.08, (seed, f.describe())


def test_fit_scale_rejects_a_wrong_mark():
    # the real failure: after a 6 cm sideways step the gripper was marked 70 px straight down the image; with the
    # begin pose alone that reads 4.4 px/cm instead of ~11.5
    xy, uv = _ring(noise=4.0, s=1150.0, rot_deg=0.0, c=(266.0, 465.0))
    xy = np.vstack([xy, BEGIN[:2] + [0.0, -0.061]])
    uv = np.vstack([uv, uv[0] + [0.0, 70.0]])
    f = fit_scale(xy, uv)
    assert f.ok and not f.inlier[-1] and f.inlier[:-1].all() and f.residual_px[-1] > 60
    assert abs(f.px_per_m / 1150 - 1) < 0.05
    # those two poses alone: the displacement points the wrong way, nothing is fitted
    f = fit_scale(xy[[0, -1]], uv[[0, -1]])
    assert not f.ok and "agree" in f.reason
    # two wrong marks among nine poses are rejected as well; four leave too few
    xy, uv = _ring(noise=4.0, s=1150.0, rot_deg=0.0, c=(266.0, 465.0))
    uv[3] += [60.0, 20.0]
    uv[6] += [-35.0, 50.0]
    f = fit_scale(xy, uv)
    assert f.ok and not f.inlier[3] and not f.inlier[6] and abs(f.px_per_m / 1150 - 1) < 0.06
    uv[1] += [0.0, 80.0]
    uv[8] += [90.0, 0.0]
    uv[4] += [-50.0, -50.0]
    assert not fit_scale(xy, uv).ok


def test_fit_scale_refuses_implausible_fits():
    xy, uv = _ring(s=1150.0, rot_deg=0.0, c=(266.0, 465.0))
    assert not fit_scale(xy, np.array([640.0, 480.0]) - uv).ok  # the image upside down (cameras.front.rotate)
    assert not fit_scale(*_ring(s=300.0, rot_deg=0.0)).ok  # 3 px/cm: not this table
    assert not fit_scale(xy[:, :1] * [1, 0] + [0, 0.0], uv).ok  # poses on a line: no spread in y
    assert fit_scale(xy, 2 * uv, width=1280).ok  # the limits scale with the image


class _CalEnv:
    """An arm that goes where it is told, seen by the synthetic camera."""

    def __init__(self, unreachable=()):
        self.tcp_pos, self.unreachable, self.poses, self.moves = BEGIN.copy(), unreachable, [], 0

    def overhead_frame(self):
        img = np.zeros((480, 640, 3), np.uint8)
        img[0, 0, :2] = np.round((self.tcp_pos[:2] - BEGIN[:2]) * 1000 + 128)  # the pose, for the fake marker
        return img

    def _move_tcp(self, target):
        if any(np.allclose(target[:2] - BEGIN[:2], u) for u in self.unreachable):
            raise RuntimeError("no IK solution")
        self.tcp_pos, self.moves = np.array(target, float), self.moves + 1

    def go_pose(self, name):
        self.poses.append(name)
        self.tcp_pos = BEGIN.copy()


class _GripperMarker:
    """Marks the gripper through the synthetic camera; ``wrong`` maps a pose index to the answers it gives there
    (pixel offsets, one per time it is asked; the last one repeats)."""

    names = ("carrot", "bin")

    def __init__(self, wrong=None):
        self.wrong, self.asked = wrong or {}, {}

    def mark_gripper(self, img):
        d = (img[0, 0, :2].astype(float) - 128) / 1000
        k = next((i for i, o in enumerate(OverheadGuide.CALIB_OFFSETS) if np.allclose(o, d)), -1)
        n = self.asked[k] = self.asked.get(k, 0) + 1
        off = self.wrong.get(k, [(0, 0)])
        uv = _camera(BEGIN[:2] + d, s=1150.0, rot_deg=0.0, c=(266.0, 465.0)) + off[min(n, len(off)) - 1]
        return Marks({"gripper": uv}, {"gripper": 2.0}, 0.0, [{"gripper": uv.tolist()}], {})


def test_calibration_sets_the_scale_and_logs(tmp_path):
    env, marker = _CalEnv(), _GripperMarker(wrong={7: [(0, 70)]})  # one pose is always marked wrongly
    g = OverheadGuide(marker, log_dir=tmp_path)
    assert g.needs_calibration and not g.map.ready
    rec = g.calibrate(env)
    n = len(OverheadGuide.CALIB_OFFSETS)
    assert rec["ok"] and rec["inliers"] == n - 1 and abs(rec["px_per_cm"] - 11.5) < 0.01
    assert [p["inlier"] for p in rec["poses"]] == [i != 7 for i in range(n)]
    assert marker.asked[7] == 1 and env.moves == n - 1  # accepted at once: no second mark, no extra poses
    assert env.poses == ["begin"] and np.allclose(env.tcp_pos, BEGIN)
    assert not g.needs_calibration and g.map.ready and len(g.map.xy) == n - 1
    assert abs(g.map.prior_scale - 1150) < 1 and abs(g.map.describe()["px_per_cm"] - 11.5) < 0.01
    target = np.array([0.24, -0.11])
    assert (
        np.linalg.norm(g.map.to_table(_camera(target, s=1150.0, rot_deg=0.0, c=(266.0, 465.0))) - target)
        < 1e-3
    )
    assert len(list(tmp_path.glob("calib_*.jpg"))) == n
    assert json.loads((tmp_path / "calib.jsonl").read_text())["poses"][7]["inlier"] is False


def test_calibration_marks_again_then_adds_poses():
    # five poses marked wrongly at first: too few agree, the same images are marked again and then agree
    off = [(0, 70), (60, -30), (-50, 40), (30, 90), (-70, -60)]
    wrong = {i: [o, (0, 0)] for i, o in zip((1, 2, 4, 6, 8), off, strict=True)}
    env, marker = _CalEnv(), _GripperMarker(wrong=wrong)
    g = OverheadGuide(marker)
    rec = g.calibrate(env)
    assert rec["ok"] and rec["inliers"] == 9 and marker.asked[2] == 2 and marker.asked[0] == 1
    assert env.poses == ["begin"]  # no extra poses needed
    # always wrong there, and one pose out of reach: the extra poses bring enough agreeing marks
    wrong = {i: [o] for i, o in zip((1, 2, 4, 6), off, strict=False)}
    env, marker = _CalEnv(unreachable=[OverheadGuide.CALIB_OFFSETS[8]]), _GripperMarker(wrong=wrong)
    g = OverheadGuide(marker)
    rec = g.calibrate(env)
    assert (
        rec["ok"]
        and rec["pairs"] == 8 + len(OverheadGuide.CALIB_EXTRA)
        and rec["inliers"] == rec["pairs"] - 4
    )
    assert env.poses == ["begin", "begin"] and abs(rec["px_per_cm"] - 11.5) < 0.01


def test_calibration_fails_loudly():
    import pytest

    wrong = {i: [(0, 70 + 15 * i)] for i in range(1, 9)}  # nothing agrees
    env = _CalEnv()
    g = OverheadGuide(_GripperMarker(wrong=wrong))
    with pytest.raises(RuntimeError, match="overhead calibration failed"):
        g.calibrate(env)
    assert g.needs_calibration and not g.map.xy and env.poses[-1] == "begin"  # nothing half-set, arm at begin
