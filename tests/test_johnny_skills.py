"""Johnny's skills against fakes: no ROS, no cameras, no network, no Docker, nothing moves."""

from __future__ import annotations

import importlib
import json
import os
import pkgutil
import signal
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import dlb.johnny.skills as skills_pkg
from dlb.johnny.skills import calibrate, control, identify, pick_place, rig, setup, teleop
from dlb.johnny.skills.base import Context, Skill, SkillResult
from dlb.johnny.teleop_relay import Relay, Settings
from dlb.real.omx_kinematics import OmxKinematics

REPO = Path(__file__).resolve().parents[1]
WRIST = "usb-Innomaker_Innomaker-U20CAM-720P_SN0001-video-index0"
FRONT = "usb-SunplusIT_Inc_j5_WebCam_JVU430_01.00.00-video-index0"
INTERNAL = "usb-C7FOF11V0724200CFAB0_Integrated_Webcam_FHD_200901010001-video-index0"
ARM_A = "usb-ROBOTIS_OpenRB-150_4B3F-if00"
ARM_B = "usb-ROBOTIS_OpenRB-150_F8A6-if00"


class FakePopen:
    def __init__(self, pid: int = 4242):
        self.pid, self.returncode, self.signals = pid, None, []

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        if self.returncode is None:
            raise rig.subprocess.TimeoutExpired("x", timeout)
        return self.returncode


class Rig:
    """What the fakes answer; tests change the attributes between calls."""

    def __init__(self):
        self.cameras: list[str] = [WRIST, FRONT]
        self.frames: dict[str, bool] = {WRIST: True, FRONT: True}
        self.ports: dict[str, str] = {ARM_A: "/dev/ttyACM0", ARM_B: "/dev/ttyACM1"}
        self.http: dict[str, bool] = {}
        self.docker: dict[str, str | None] = {"openjev": None, "stt": None}
        self.joint_states = False
        self.spawned: list[tuple[str, str, str]] = []
        self.stopped: list[str] = []
        self.commands: list[list[str]] = []


@pytest.fixture
def fake(monkeypatch, tmp_path):
    f = Rig()
    monkeypatch.setattr(rig, "V4L_BY_ID", tmp_path / "v4l")
    monkeypatch.setattr(rig, "OUT_DIR", tmp_path / "out")
    monkeypatch.setattr(rig, "RIG_JSON", tmp_path / "out" / "rig.json")
    monkeypatch.setattr(rig, "list_cameras", lambda: [tmp_path / "v4l" / c for c in f.cameras])
    monkeypatch.setattr(rig, "list_arm_ports", lambda: [tmp_path / "serial" / p for p in f.ports])
    monkeypatch.setattr(rig, "resolve", lambda d: f.ports.get(Path(d).name, str(d)))
    monkeypatch.setattr(
        rig,
        "grab_frame",
        lambda dev, settle_s=1.0: np.zeros((48, 64, 3), np.uint8) if f.frames.get(Path(dev).name) else None,
    )
    monkeypatch.setattr(rig, "http_ok", lambda url, timeout=2.0: f.http.get(url, False))
    monkeypatch.setattr(rig, "docker_state", lambda name: f.docker.get(name))

    def docker_start(name):
        f.docker[name] = "running"
        return True

    monkeypatch.setattr(rig, "docker_start", docker_start)
    monkeypatch.setattr(rig, "joint_states_up", lambda cfg, timeout=10.0: f.joint_states)

    def run(cmd, timeout=10.0, env=None):
        f.commands.append(cmd)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(rig, "run", run)

    def spawn(ctx, name, cmd, env):
        f.spawned.append((name, cmd, env.get("ROS_DOMAIN_ID", "")))
        p = FakePopen()
        ctx.processes[name] = p
        if name == "follower":
            f.joint_states = True
        return p

    monkeypatch.setattr(rig, "spawn", spawn)

    def stop(ctx, name, timeout=10.0):
        p = ctx.processes.pop(name, None)
        if p is None or p.poll() is not None:
            return "not running"
        p.returncode = 0
        f.stopped.append(name)
        return "stopped"

    monkeypatch.setattr(rig, "stop", stop)
    monkeypatch.setattr(pick_place.time, "sleep", lambda s: None)
    monkeypatch.setattr(teleop.time, "sleep", lambda s: None)
    monkeypatch.setattr(identify, "save_frame", lambda frame, path: None)
    monkeypatch.setattr(identify, "save_front_sheet", lambda frame, path: None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    return f


@pytest.fixture
def ctx(tmp_path):
    cfg = tmp_path / "omx_f.yaml"
    # the repo's config with its camera paths under the fake by-id directory (same tmp_path as ``fake``)
    cfg.write_text(
        (REPO / "configs/robot/omx_f.yaml").read_text().replace("/dev/v4l/by-id", str(tmp_path / "v4l"))
    )
    env = tmp_path / ".env"
    env.write_text("OPENAI_API_KEY=sk-test\n")
    said: list[str] = []
    c = Context(robot_config=str(cfg), env_file=str(env), say=said.append, log=lambda s: None)
    c.said = said
    return c


# ------------------------------------------------------------------ #
# registry shape
# ------------------------------------------------------------------ #
def all_skills() -> list[Skill]:
    out = []
    for m in pkgutil.iter_modules(skills_pkg.__path__):
        mod = importlib.import_module(f"dlb.johnny.skills.{m.name}")
        for v in vars(mod).values():
            if (
                isinstance(v, type)
                and issubclass(v, Skill)
                and v is not Skill
                and v.__module__ == mod.__name__
            ):
                out.append(v())
    return out


def test_every_skill_instantiates_with_a_valid_tool_spec():
    skills = all_skills()
    names = [s.name for s in skills]
    assert len(names) == len(set(names)) and all(names)
    assert {
        "rig_status",
        "rig_setup",
        "rig_shutdown",
        "identify_cameras",
        "identify_arms",
        "teleop_start",
        "teleop_stop",
        "pick_place",
        "robot_home",
        "robot_release",
        "calibrate_overhead",
        "stop_all",
        "status",
    } <= set(names)
    for s in skills:
        spec = s.tool_spec()
        assert spec["type"] == "function" and spec["name"] == s.name and len(s.description) > 40
        assert spec["parameters"]["type"] == "object" and spec["parameters"]["additionalProperties"] is False
        json.dumps(spec)  # serializable for the API


# ------------------------------------------------------------------ #
# rig helpers
# ------------------------------------------------------------------ #
def test_arm_ports_default_then_identified(fake, ctx):
    ports = rig.arm_ports(ctx)
    assert ports == {
        "follower": "/dev/ttyACM0",
        "leader": "/dev/ttyACM1",
        "identified": False,
        "found": ["/dev/ttyACM0", "/dev/ttyACM1"],
    }
    # identified the other way round, and the numbers swapped after a replug
    ctx.state["arms"] = {"follower": {"by_id": ARM_B}, "leader": {"by_id": ARM_A}}
    fake.ports = {ARM_A: "/dev/ttyACM3", ARM_B: "/dev/ttyACM2"}
    ports = rig.arm_ports(ctx)
    assert (ports["follower"], ports["leader"], ports["identified"]) == ("/dev/ttyACM2", "/dev/ttyACM3", True)
    fake.ports = {}
    assert rig.arm_ports(ctx)["follower"] is None


def test_stop_signals_the_process_group(monkeypatch):
    sent = []
    monkeypatch.setattr(os, "killpg", lambda pid, sig: sent.append((pid, sig)))
    c = Context(log=lambda s: None)
    p = FakePopen(pid=77)
    c.processes["leader"] = p
    orig_wait = p.wait

    def wait(timeout=None):
        if sent:  # exits on the first signal
            p.returncode = 0
        return orig_wait(timeout)

    p.wait = wait
    assert rig.stop(c, "leader") == "stopped"
    assert sent == [(77, signal.SIGINT)] and "leader" not in c.processes
    assert rig.stop(c, "leader") == "not running"


def test_ros_shell_sources_ros_and_execs():
    cmd = rig.ros_shell(rig.FOLLOWER_LAUNCH.format(port="/dev/ttyACM0"))
    assert cmd[:2] == ["bash", "-lc"]
    assert (
        "source /opt/ros/jazzy/setup.bash" in cmd[2]
        and "exec ros2 launch open_manipulator_bringup omx_f_follower_ai.launch.py port_name:=/dev/ttyACM0"
        in cmd[2]
    )


# ------------------------------------------------------------------ #
# setup
# ------------------------------------------------------------------ #
def test_rig_status_reports_missing_and_asks_for_hands(fake, ctx):
    fake.frames[WRIST] = False
    r = setup.RigStatus().run(ctx)
    assert not r.ok and r.data["ask_user"]
    assert "手首カメラから画像が来ません" in r.text and "follower driver not running" in r.text
    assert r.data["checks"] == {
        "camera wrist": False,
        "camera front": True,
        "arms": True,
        "follower": False,
        "openjev": False,
        "stt": False,
        "openai": True,
    }
    assert "identify_arms" in r.data["suggest"]  # the ports are assumed from the numbers


def test_rig_status_all_ok(fake, ctx):
    fake.joint_states = True
    fake.http = {f"{rig.OPENJEV_URL}/health": True, f"{rig.STT_URL}/health": True}
    ctx.state["arms"] = {"follower": {"by_id": ARM_A}, "leader": {"by_id": ARM_B}}
    r = setup.RigStatus().run(ctx)
    assert r.ok and "ask_user" not in r.data and r.data["suggest"] == []


def test_rig_status_unknown_camera_suggests_identify(fake, ctx):
    fake.cameras = [FRONT]
    r = setup.RigStatus().run(ctx)
    assert r.data["checks"]["camera wrist"] is False and "identify_cameras" in r.data["suggest"]


def test_rig_setup_starts_what_is_missing(fake, ctx):
    fake.docker["openjev"] = "exited"  # container exists -> docker start
    r = setup.RigSetup().run(ctx)
    assert [s[0] for s in fake.spawned] == ["follower"]
    assert (
        fake.spawned[0][1]
        == "ros2 launch open_manipulator_bringup omx_f_follower_ai.launch.py port_name:=/dev/ttyACM0"
    )
    assert fake.spawned[0][2] == "42"
    assert ctx.running("follower") and ctx.state["follower_port"] == "/dev/ttyACM0"
    assert fake.docker["openjev"] == "running"
    assert ["bash", "scripts/stt_server.sh"] in fake.commands  # no stt container yet -> the script
    assert (
        "started the follower driver" in r.text and "docker start openjev" in r.text and "1-3 min" in r.text
    )
    assert r.data["checks"]["follower"] is True and r.data["checks"]["openjev"] is False  # not healthy yet


def test_rig_setup_only_what_was_asked(fake, ctx):
    r = setup.RigSetup().run(ctx, start=["stt"])
    assert fake.spawned == [] and fake.docker["openjev"] is None
    assert ["bash", "scripts/stt_server.sh"] in fake.commands and r.data["checks"]["follower"] is False


def test_rig_setup_follower_launch_dies(fake, ctx, monkeypatch):
    def spawn(ctx_, name, cmd, env):
        p = FakePopen()
        p.returncode = 1
        ctx_.processes[name] = p
        return p

    monkeypatch.setattr(rig, "spawn", spawn)
    r = setup.RigSetup().run(ctx, start=["follower"])
    assert not r.ok and "exited right away" in r.text


def test_rig_setup_no_follower_port(fake, ctx):
    fake.ports = {}
    r = setup.RigSetup().run(ctx, start=["follower"])
    assert not r.ok and "no follower arm port" in r.text and r.data["ask_user"]


def test_rig_shutdown_stops_launches_and_releases_the_robot(fake, ctx):
    ctx.processes["leader"], ctx.processes["follower"] = FakePopen(1), FakePopen(2)
    ctx.state["teleop"] = True
    closed = []
    ctx.robot = SimpleNamespace(env=SimpleNamespace(close=lambda: closed.append(1)))
    fake.docker = {"openjev": "running", "stt": "running"}
    r = setup.RigShutdown().run(ctx, docker=True)
    assert fake.stopped == ["leader", "follower"] and closed == [1] and ctx.robot is None
    assert ctx.state["teleop"] is False and r.data["docker"] == ["openjev", "stt"]
    assert ["docker", "stop", "openjev", "stt"] in fake.commands


# ------------------------------------------------------------------ #
# interactive identification
# ------------------------------------------------------------------ #
def test_identify_cameras_full_flow_writes_config(fake, ctx):
    fake.cameras = [WRIST, FRONT, INTERNAL]  # the internal webcam is already filtered by list_cameras in rig;
    fake.cameras = [WRIST, FRONT]  # the fake listing stands for the filtered one
    s = identify.IdentifyCameras()
    r = s.run(ctx, step="start")
    assert r.data["ask_user"] and r.data["next"] == "wrist_unplugged" and "手首カメラのUSBを抜いて" in r.text
    # nothing unplugged yet: same instruction again
    r = s.run(ctx, step="wrist_unplugged")
    assert r.data["next"] == "wrist_unplugged" and "まだ" in r.text
    fake.cameras = [FRONT]
    r = s.run(ctx, step="wrist_unplugged")
    assert r.data["next"] == "plugged" and r.data["wrist"] == WRIST and r.data["front"] == FRONT
    # still unplugged
    r = s.run(ctx, step="plugged")
    assert r.data["next"] == "plugged" and "手首カメラ" in r.text and "まだ見つかりません" in r.text
    fake.cameras = [WRIST, FRONT]
    fake.frames[FRONT] = False
    r = s.run(ctx, step="plugged")
    assert r.data["next"] == "plugged" and "俯瞰カメラから画像が来ません" in r.text
    fake.frames[FRONT] = True
    r = s.run(ctx, step="plugged")
    assert r.data["next"] == "rotation" and r.data["current_rotate"] == 180 and ctx.state["cameras_changed"]
    cfg = rig.load_config(ctx.robot_config)["cameras"]
    assert cfg["wrist"]["device"] == str(rig.V4L_BY_ID / WRIST) and cfg["front"]["device"] == str(
        rig.V4L_BY_ID / FRONT
    )
    r = s.run(ctx, step="rotation")  # no angle given
    assert r.data["next"] == "rotation"
    r = s.run(ctx, step="rotation", rotate=90)
    assert r.ok and "ask_user" not in r.data and "calibrate_overhead" in r.data["suggest"]
    assert rig.load_config(ctx.robot_config)["cameras"]["front"]["rotate"] == 90
    assert json.loads(rig.RIG_JSON.read_text())["cameras"]["rotate"] == 90
    assert ctx.state["identify_cameras"] == {}
    # the comments of the config survive the in-place write
    assert "# USB cameras" in Path(ctx.robot_config).read_text()


def test_identify_cameras_three_cameras_and_cancel(fake, ctx):
    third = "usb-Other_Cam-video-index0"
    fake.cameras = [WRIST, FRONT, third]
    s = identify.IdentifyCameras()
    s.run(ctx, step="start")
    fake.cameras = [FRONT, third]
    r = s.run(ctx, step="wrist_unplugged")
    assert r.data["next"] == "front_unplugged" and "俯瞰" in r.text
    fake.cameras = [WRIST, third]
    r = s.run(ctx, step="front_unplugged")
    assert r.data["next"] == "plugged" and r.data["front"] == FRONT
    r = s.run(ctx, step="cancel")
    assert "cancelled" in r.text and ctx.state["identify_cameras"] == {}
    # a step without a session starts over
    fake.cameras = [WRIST]
    r = s.run(ctx, step="plugged")
    assert r.data["next"] == "start" and "1 台" in r.text


def test_identify_arms_flow(fake, ctx):
    s = identify.IdentifyArms()
    r = s.run(ctx, step="start")
    assert r.data["next"] == "leader_unplugged" and "リーダーアーム" in r.text
    fake.ports = {ARM_A: "/dev/ttyACM0"}
    r = s.run(ctx, step="leader_unplugged")
    assert r.data["next"] == "plugged"
    assert r.data["arms"]["leader"]["by_id"] == ARM_B and r.data["arms"]["follower"]["by_id"] == ARM_A
    fake.ports = {ARM_A: "/dev/ttyACM0", ARM_B: "/dev/ttyACM2"}  # came back with another number
    r = s.run(ctx, step="plugged")
    assert r.ok and "ask_user" not in r.data and "leader on /dev/ttyACM2" in r.text
    assert json.loads(rig.RIG_JSON.read_text())["arms"]["leader"]["dev"] == "/dev/ttyACM2"
    assert rig.arm_ports(ctx) == {
        "follower": "/dev/ttyACM0",
        "leader": "/dev/ttyACM2",
        "identified": True,
        "found": ["/dev/ttyACM0", "/dev/ttyACM2"],
    }


def test_identify_arms_one_arm_only(fake, ctx):
    fake.ports = {ARM_B: "/dev/ttyACM0"}
    r = identify.IdentifyArms().run(ctx, step="start")
    assert r.ok and "ask_user" not in r.data and ctx.state["arms"]["follower"]["by_id"] == ARM_B


# ------------------------------------------------------------------ #
# teleop (relay + leader launch)
# ------------------------------------------------------------------ #
@pytest.fixture
def relay_phase(fake, monkeypatch):
    """The relay's status file as the teleop skill sees it; the test sets ``fake.relay``."""
    fake.relay = {
        "phase": "following",
        "reason": "moving 17 deg at most in 2.0 s",
        "messages_out": 98,
        "clamped": 3,
    }
    monkeypatch.setattr(teleop, "relay_status", lambda: fake.relay)
    return fake


def test_teleop_start_launches_follower_relay_and_leader(relay_phase, ctx):
    fake = relay_phase
    r = teleop.TeleopStart().run(ctx)
    assert r.ok, r.text
    assert ctx.state["teleop"] and ctx.state["arm_moved_externally"]
    assert [s[0] for s in fake.spawned] == ["follower", "relay", "leader"]
    relay_cmd, leader_cmd = fake.spawned[1][1], fake.spawned[2][1]
    assert relay_cmd.startswith(f"{sys.executable} -m dlb.johnny.teleop_relay --config {ctx.robot_config}")
    assert "--raw /leader/joint_trajectory_raw" in relay_cmd and "--status" in relay_cmd
    assert leader_cmd == (
        f"ros2 launch {teleop.LAUNCH_FILE} port_name:=/dev/ttyACM1 raw_topic:=/leader/joint_trajectory_raw"
    )
    assert teleop.LAUNCH_FILE.exists() and fake.spawned[2][2] == "42"
    assert "reached the leader's pose" in r.text and "clear" in r.text and "1.0 rad/s" in r.text
    assert any("ゆっくり" in s for s in ctx.said)
    r = teleop.TeleopStart().run(ctx)
    assert "following the leader" in r.text and "98 commands" in r.text
    assert "following the leader" in teleop.TeleopStatus().run(ctx).text
    r = teleop.TeleopStop().run(ctx)
    assert fake.stopped == ["relay", "leader"] and ctx.state["teleop"] is False and "holds its pose" in r.text
    assert teleop.TeleopStop().run(ctx).text == "tele-operation was not running"
    assert teleop.TeleopStatus().run(ctx).text == "tele-operation off"


def test_teleop_start_refused_approach_stops_everything(relay_phase, ctx):
    fake = relay_phase
    fake.joint_states = True
    fake.relay = {
        "phase": "refused",
        "reason": "the leader's pose [0.0, -1.86, ...] is outside the follower's joint limits",
    }
    r = teleop.TeleopStart().run(ctx)
    assert not r.ok and "refused the approach" in r.text and "joint limits" in r.text
    assert fake.stopped == ["relay", "leader"] and ctx.state["teleop"] is False
    assert not ctx.running("relay") and not ctx.running("leader")


def test_teleop_start_times_out_without_a_leader_stream(relay_phase, ctx, monkeypatch):
    fake = relay_phase
    fake.joint_states = True
    fake.relay = {"phase": "waiting"}
    clock = iter(range(0, 100000, 10))
    monkeypatch.setattr(teleop.time, "monotonic", lambda: float(next(clock)))
    r = teleop.TeleopStart().run(ctx)
    assert (
        not r.ok
        and "did not start" in r.text
        and "relay waiting" in r.text
        and fake.stopped == ["relay", "leader"]
    )


def test_teleop_start_says_approach_once_and_waits_for_following(relay_phase, ctx, monkeypatch):
    fake = relay_phase
    fake.joint_states = True
    phases = iter(
        [
            {"phase": "waiting"},
            {"phase": "approach", "reason": "moving 40 deg at most in 2.0 s"},
            {"phase": "approach"},
            {"phase": "following", "reason": "moving 40 deg at most in 2.0 s"},
        ]
    )
    monkeypatch.setattr(teleop, "relay_status", lambda: next(phases))
    r = teleop.TeleopStart().run(ctx)
    assert r.ok and "40 deg" in r.text
    assert sum(s.endswith("ゆっくり動きます。") for s in ctx.said) == 1


def test_teleop_start_refused_during_a_task_or_without_leader(fake, ctx):
    ctx.state["task_running"] = True
    r = teleop.TeleopStart().run(ctx)
    assert not r.ok and "stop first" in r.text
    ctx.state["task_running"] = False
    fake.ports = {ARM_A: "/dev/ttyACM0"}
    r = teleop.TeleopStart().run(ctx)
    assert not r.ok and "leader arm" in r.text and fake.spawned == []


def test_teleop_start_dead_launch(relay_phase, ctx, monkeypatch):
    fake = relay_phase
    fake.joint_states = True
    fake.relay = {"phase": "waiting"}

    def spawn(ctx_, name, cmd, env):
        p = FakePopen()
        if name == "leader":
            p.returncode = 2
        ctx_.processes[name] = p
        return p

    monkeypatch.setattr(rig, "spawn", spawn)
    r = teleop.TeleopStart().run(ctx)
    assert not r.ok and "leader launch exited" in r.text and not ctx.state["teleop"]


def test_teleop_line_reports_a_broken_pair(relay_phase, ctx):
    ctx.processes["leader"] = FakePopen()
    assert "BROKEN: the relay is not running" in teleop.teleop_line(ctx)
    ctx.processes["relay"] = FakePopen()
    assert teleop.teleop_line(ctx).startswith("tele-operation on: following")
    relay_phase.relay = {"phase": "paused"}
    assert "paused" in control.Status().run(ctx).text


# ------------------------------------------------------------------ #
# pick and place with a fake Agent
# ------------------------------------------------------------------ #
class FakeEnv:
    def __init__(self):
        self.held, self.calls, self._stamp = False, [], 0.0
        self.tcp_pos = np.array([0.2, 0.0, 0.1])

    def reset(self, seed=None):
        self.calls.append("reset")

    def go_pose(self, name):
        self.calls.append(f"go_pose {name}")

    def at_pose(self, name):
        return False

    def open_gripper(self):
        self.calls.append("open")

    def close(self):
        self.calls.append("close")


class FakeRunner:
    max_decisions, max_planner_calls = 40, 12

    def run(self, episode, seed, reset=True, keep_marks=False):
        return SimpleNamespace(stop_reason="planner_complete", decisions=7, missed_grasps=0)


class FakeAgent:
    def __init__(self, args):
        from dlb.voice.intent import Intent, RobotState

        self.args, self.Intent = args, Intent
        self.objects = ["carrot", "red cube"]
        self.state = RobotState(None, [*self.objects, "bin"], [])
        self.env, self.runner, self.guide = None, None, None
        self.answer = Intent([{"op": "pick", "object": "carrot"}, {"op": "place", "where": "bin"}], True)
        self.executed, self.marks = [], 0
        self.fail_start = False

    def start_robot(self):
        if self.fail_start:
            raise SystemExit("no objects or no container found on the table")
        self.env, self.runner = FakeEnv(), FakeRunner()
        self.guide = SimpleNamespace(
            calibration={
                "px_per_cm": 11.9,
                "rotation_deg": 1.5,
                "inliers": 8,
                "pairs": 9,
                "rms_px": 4.2,
                "seconds": 22.0,
                "poses": [{"tcp": [0.21, 0.03], "mark": None, "inlier": False}],
            },
            reset=lambda: None,
        )
        self.env.calls.append("start")

    def parse(self, text):
        return self.answer

    def execute(self, tasks):
        self.executed.append((tasks, self.runner.max_decisions))
        self.runner.run(0, seed=0, reset=False, keep_marks=True)  # as Agent.execute does
        self.state = self.state.__class__("carrot" if self.env.held else None, self.state.objects, [])

    def idle_mark(self):
        self.marks += 1


@pytest.fixture
def agent(fake, monkeypatch):
    made = []

    def build(args):
        a = FakeAgent(args)
        made.append(a)
        return a

    monkeypatch.setattr(pick_place, "build_agent", build)
    fake.http[f"{rig.OPENJEV_URL}/health"] = True
    fake.joint_states = True
    return made


def test_pick_place_starts_the_robot_once_and_runs_the_tasks(fake, ctx, agent):
    s = pick_place.PickPlace()
    r = s.run(ctx, instruction="人参をトレーに入れて", objects=["carrot", "red cube"], place="white tray")
    assert r.ok, r.text
    a = agent[0]
    assert ctx.robot is a and a.args.objects == "carrot,red cube" and a.args.bin_name == "white tray"
    assert (
        a.args.robot_config == ctx.robot_config
        and a.args.out == str(rig.OUT_DIR)
        and a.args.intent_model == "gpt-5.5"
    )
    assert a.args.text and not a.args.dry_run and not a.args.flat_place
    assert any("自己校正" in m for m in ctx.said)
    assert a.executed == [([{"op": "pick", "object": "carrot"}, {"op": "place", "where": "bin"}], 40)]
    assert (
        r.data["tasks"] == a.answer.tasks
        and not r.data["cancelled"]
        and "done: pick carrot, place bin" in r.text
    )
    assert r.data["stop_reason"] == "planner_complete" and "7 decisions" in r.text  # the wrapped runner.run
    assert ctx.state["task_running"] is False and ctx.state["running_skill"] is None
    # second call: no new agent, same robot
    s.run(ctx, instruction="赤いキューブもお願い")
    assert len(agent) == 1 and len(a.executed) == 2
    assert os.environ["OPENAI_API_KEY"] == "sk-test"  # the env file was loaded for the parser


def test_pick_place_clarify_and_stop_words(fake, ctx, agent):
    s = pick_place.PickPlace()
    s.run(ctx, instruction="x")
    a = agent[0]
    a.answer = a.Intent([], True, clarify="緑のキューブは見当たりません。赤いキューブですか？")
    r = s.run(ctx, instruction="緑のキューブをビンに")
    assert (
        not r.ok
        and r.data["clarify"] == a.answer.clarify
        and r.data["ask_user"]
        and r.text == a.answer.clarify
    )
    a.answer = a.Intent([], True)
    r = s.run(ctx, instruction="止まって")
    assert r.ok and r.data["tasks"] == [] and len(a.executed) == 1


def test_pick_place_preconditions(fake, ctx, agent):
    s = pick_place.PickPlace()
    fake.http[f"{rig.OPENJEV_URL}/health"] = False
    fake.docker["openjev"] = "running"
    r = s.run(ctx, instruction="人参をトレーに入れて")
    assert not r.ok and r.text == pick_place.NOT_HEALTHY and agent == []
    fake.docker["openjev"] = None
    r = s.run(ctx, instruction="人参をトレーに入れて")
    assert not r.ok and "rig_setup" in r.text
    fake.http[f"{rig.OPENJEV_URL}/health"] = True
    Path(ctx.env_file).write_text("")
    os.environ.pop("OPENAI_API_KEY", None)  # load_env_file only sets what is unset
    r = s.run(ctx, instruction="x")
    assert not r.ok and "OPENAI_API_KEY" in r.text
    Path(ctx.env_file).write_text("OPENAI_API_KEY=sk-test\n")
    ctx.state["task_running"] = True
    r = s.run(ctx, instruction="x")
    assert not r.ok and "already running" in r.text
    ctx.state["task_running"] = False
    r = s.run(ctx, instruction="   ")
    assert not r.ok


def test_pick_place_starts_the_follower_and_stops_teleop_first(fake, ctx, agent):
    fake.joint_states = False
    ctx.processes["leader"] = FakePopen()
    ctx.state["teleop"] = True
    ctx.state["arm_moved_externally"] = True
    r = pick_place.PickPlace().run(ctx, instruction="人参をトレーに入れて")
    assert r.ok and r.text.startswith("tele-operation stopped first.")
    assert fake.stopped == ["leader"] and [s[0] for s in fake.spawned] == ["follower"]
    assert ctx.state["teleop"] is False and "arm_moved_externally" not in ctx.state
    # the robot had just started (begin pose already): no extra reset before the task
    assert agent[0].env.calls == ["start"]


def test_pick_place_returns_to_begin_after_teleop(fake, ctx, agent):
    s = pick_place.PickPlace()
    s.run(ctx, instruction="x")
    a = agent[0]
    ctx.state["arm_moved_externally"] = True  # teleop happened in between
    s.run(ctx, instruction="y")
    assert a.env.calls == ["start", "reset"] and a.marks == 1 and "arm_moved_externally" not in ctx.state


def test_pick_place_start_failure_closes_the_env(fake, ctx, monkeypatch):
    def build(args):
        a = FakeAgent(args)
        a.fail_start = True
        a.env = FakeEnv()
        return a

    monkeypatch.setattr(pick_place, "build_agent", build)
    fake.http[f"{rig.OPENJEV_URL}/health"] = True
    fake.joint_states = True
    r = pick_place.PickPlace().run(ctx, instruction="x")
    assert not r.ok and "could not start the robot" in r.text and "no objects" in r.text and ctx.robot is None


def test_pick_place_cancel_zeroes_the_budgets_and_restores_them(fake, ctx, agent):
    s = pick_place.PickPlace()
    s.run(ctx, instruction="x")
    a = agent[0]
    seen = {}

    def execute(tasks):
        assert ctx.state["task_running"] and ctx.state["running_skill"] is s
        s.cancel(ctx)  # "stop" arrives while the task runs
        seen["budgets"] = (a.runner.max_decisions, a.runner.max_planner_calls)

    a.execute = execute
    r = s.run(ctx, instruction="y")
    assert seen["budgets"] == (0, 0)
    assert (a.runner.max_decisions, a.runner.max_planner_calls) == (40, 12)
    assert r.ok and r.data["cancelled"] and "stopped on request" in r.text
    assert ctx.state["task_running"] is False
    # a stale cancel does not affect the next task
    a.execute = lambda tasks: seen.setdefault("next", (a.runner.max_decisions, s._cancel.is_set()))
    s.run(ctx, instruction="z")
    assert seen["next"] == (40, False)


def test_pick_place_task_failure_is_reported(fake, ctx, agent):
    s = pick_place.PickPlace()
    s.run(ctx, instruction="x")

    def execute(tasks):
        raise RuntimeError("move rested 0.1 rad from its goal")

    agent[0].execute = execute
    r = s.run(ctx, instruction="y")
    assert not r.ok and "the task failed" in r.text and ctx.state["task_running"] is False


def test_robot_home_and_release(fake, ctx, agent):
    r = pick_place.RobotHome().run(ctx)
    assert r.ok and r.data["started"] and len(agent) == 1
    a = agent[0]
    ctx.state["arm_moved_externally"] = True
    r = pick_place.RobotHome().run(ctx)
    assert (
        r.ok
        and a.env.calls == ["start", "go_pose begin"]
        and a.marks == 1
        and "arm_moved_externally" not in ctx.state
    )
    a.state = a.state.__class__("carrot", a.state.objects, [])
    r = pick_place.RobotRelease().run(ctx)
    assert r.ok and a.env.calls[-1] == "open" and a.state.held is None
    ctx.state["task_running"] = True
    assert not pick_place.RobotHome().run(ctx).ok and not pick_place.RobotRelease().run(ctx).ok


# ------------------------------------------------------------------ #
# calibration
# ------------------------------------------------------------------ #
def test_calibrate_overhead_reports_the_startup_calibration(fake, ctx, agent):
    r = calibrate.CalibrateOverhead().run(ctx)
    assert r.ok and "robot started" in r.text and "11.9 px/cm" in r.text and "8 of 9 poses agree" in r.text
    assert (
        "pose 1 at (21.0, 3.0) cm: gripper not found" in r.text and r.data["calibration"]["seconds"] == 22.0
    )


def test_calibrate_overhead_reruns_from_the_begin_pose(fake, ctx, agent):
    pick_place.RobotHome().run(ctx)
    a = agent[0]
    rec = {
        "px_per_cm": 12.3,
        "rotation_deg": -0.4,
        "inliers": 9,
        "pairs": 9,
        "rms_px": 3.0,
        "seconds": 21.5,
        "poses": [],
    }
    calls = []
    a.guide = SimpleNamespace(
        calibration=None,
        reset=lambda: calls.append("reset"),
        calibrate=lambda env: (calls.append("calibrate"), rec)[1],
        map=None,
    )
    ctx.state["cameras_changed"] = True
    ctx.said.clear()
    r = calibrate.CalibrateOverhead().run(ctx)
    assert r.ok and "12.3 px/cm" in r.text and "9 of 9 poses agree" in r.text and "rejected" not in r.text
    assert a.env.calls[-1] == "go_pose begin" and calls == ["reset", "calibrate"] and a.marks == 1
    assert a.guide.map is not None and "cameras_changed" not in ctx.state
    assert any("手を離して" in s for s in ctx.said)


def test_calibrate_overhead_failure_and_preconditions(fake, ctx, agent):
    pick_place.RobotHome().run(ctx)
    a = agent[0]
    bad = {
        "px_per_cm": 3.0,
        "rotation_deg": 90.0,
        "inliers": 2,
        "pairs": 9,
        "rms_px": 40.0,
        "seconds": 30.0,
        "poses": [{"tcp": [0.2, 0.0], "mark": [10, 10], "inlier": False, "residual_px": 55.0}],
    }

    def fail(env):
        a.guide.calibration = bad
        raise RuntimeError("overhead calibration failed (too few inliers)")

    a.guide = SimpleNamespace(calibration=None, reset=lambda: None, calibrate=fail, map=None)
    r = calibrate.CalibrateOverhead().run(ctx)
    assert (
        not r.ok
        and "too few inliers" in r.text
        and "55 px off the fit" in r.text
        and "cameras_front.jpg" in r.text
    )
    a.env.held = True
    r = calibrate.CalibrateOverhead().run(ctx)
    assert not r.ok and "holds something" in r.text
    ctx.state["task_running"] = True
    assert not calibrate.CalibrateOverhead().run(ctx).ok


# ------------------------------------------------------------------ #
# control
# ------------------------------------------------------------------ #
def test_stop_all_cancels_the_task_and_ends_teleop(fake, ctx):
    cancelled = []

    class Running(Skill):
        name = "pick_place"

        def cancel(self, ctx_):
            cancelled.append(ctx_)

    ctx.state.update(task_running=True, running_skill=Running())
    ctx.processes["leader"] = FakePopen()
    r = control.StopAll().run(ctx)
    assert cancelled == [ctx] and fake.stopped == ["leader"] and ctx.state["teleop"] is False
    assert "pick_place cancelled" in r.text and "tele-operation stopped" in r.text and "holds" in r.text
    ctx.state["task_running"] = False
    assert control.StopAll().run(ctx).text.startswith("nothing was running")


def test_status_line(fake, ctx):
    fake.http = {f"{rig.OPENJEV_URL}/health": True}
    ctx.processes["leader"], ctx.processes["relay"] = FakePopen(), FakePopen()
    ctx.state["identify_arms"] = {"seen": {}}
    r = control.Status().run(ctx)
    assert isinstance(r, SkillResult) and r.ok
    assert (
        "tele-operation on" in r.text
        and "robot not started" in r.text
        and "openjev ok" in r.text
        and "stt down" in r.text
    )
    assert (
        "cameras wrist, front" in r.text
        and "launched: leader, relay" in r.text
        and "in progress: identify_arms" in r.text
    )
    assert r.data["ports"]["follower"] == "/dev/ttyACM0" and r.data["teleop"] is True
    ctx.robot = SimpleNamespace(
        state=SimpleNamespace(held="carrot"), env=SimpleNamespace(held=True, _stamp=0.0)
    )
    assert "holding carrot" in control.Status().run(ctx).text


# ------------------------------------------------------------------ #
# the relay's logic (no ROS)
# ------------------------------------------------------------------ #
JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5"]
LEADER_ORDER = [
    "joint2",
    "joint3",
    "joint1",
    "joint4",
    "joint5",
    "gripper_joint_1",
]  # as the broadcaster sends


def leader_msg(q, grip):
    d = dict(zip(JOINTS, q, strict=True))
    return LEADER_ORDER, [d.get(n, grip) for n in LEADER_ORDER]


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def make_relay(tmp_path, **over):
    cfg = {
        "arm_joints": JOINTS,
        "gripper_joint": "gripper_joint_1",
        "gripper_open": 0.45,
        "gripper_closed": 0.0,
    }
    sent = []
    clock = Clock()
    relay = Relay(
        Settings.from_config(cfg, **over),
        OmxKinematics(),
        lambda q, g, t: sent.append((list(q), g, t)),
        now=clock,
        status_path=tmp_path / "s.json",
    )
    return relay, sent, clock


def test_relay_approaches_slowly_then_follows_with_a_velocity_limit(tmp_path):
    relay, sent, clock = make_relay(tmp_path)
    follower = [0.0, -0.5, 0.7, 0.6, 0.0]
    relay.on_leader(*leader_msg([0.0, -0.9, 0.7, 0.6, 0.0], -0.29))  # before the follower is known: waits
    assert relay.phase == "waiting" and sent == []
    relay.on_follower([*JOINTS, "gripper_joint_1"], [*follower, 0.4])
    relay.on_leader(*leader_msg([0.0, -0.9, 0.7, 0.6, 0.0], -0.29))
    assert relay.phase == "approach" and len(sent) == 1
    q, g, t = sent[0]
    assert (
        q == [0.0, -0.9, 0.7, 0.6, 0.0] and g == 0.0 and t == pytest.approx(2.0)
    )  # 0.4 rad < 2 s x 0.4 rad/s
    assert "23 deg" in relay.reason
    # more leader messages during the approach: nothing is sent until it is over
    clock.t += 1.0
    relay.on_leader(*leader_msg([0.0, -0.9, 0.7, 0.6, 0.0], -0.29))
    assert len(sent) == 1
    clock.t += 1.6  # 2.0 s + 0.5 s margin passed
    relay.tick()
    assert relay.phase == "following"
    # the leader jumps 0.3 rad on joint1: forwarded 0.02 rad per message at most, 50 Hz
    for i in range(1, 6):
        clock.t += 0.02
        relay.on_leader(*leader_msg([0.3, -0.9, 0.7, 0.6, 0.0], 0.7))
        assert sent[-1][0][0] == pytest.approx(0.02 * i) and sent[-1][2] == pytest.approx(0.02)
        assert sent[-1][1] == 0.45  # gripper beyond open: clamped
    assert relay.n_clamped == 5 and len(sent) == 6
    # messages faster than the rate are dropped
    relay.on_leader(*leader_msg([0.3, -0.9, 0.7, 0.6, 0.0], 0.2))
    assert len(sent) == 6
    # the status file (written at most twice a second)
    clock.t += 0.5
    relay.on_follower(JOINTS, follower)  # the follower's joint states keep coming
    relay.tick()
    st = json.loads((tmp_path / "s.json").read_text())
    assert (
        st["phase"] == "following"
        and st["messages_out"] == 6
        and st["clamped"] == 5
        and st["gripper"] == 0.45
    )
    assert st["leader_q"][0] == 0.3 and st["cmd_q"][0] == pytest.approx(0.1)


def test_relay_refuses_poses_outside_the_limits_or_too_low(tmp_path):
    relay, sent, clock = make_relay(tmp_path)
    relay.on_follower(JOINTS, [0.0, -0.5, 0.7, 0.6, 0.0])
    relay.on_leader(*leader_msg([0.0, -1.86, 0.7, 0.6, 0.0], 0.0))  # joint2 past the firmware limit
    assert relay.phase == "refused" and "joint limits" in relay.reason and sent == []
    assert json.loads((tmp_path / "s.json").read_text())["phase"] == "refused"
    relay, sent, clock = make_relay(tmp_path)
    relay.on_follower(JOINTS, [0.0, -0.5, 0.7, 0.6, 0.0])
    low = [0.0, 1.2, 0.9, 0.5, 0.0]  # arm stretched out and down: fingertips below the base plane
    assert OmxKinematics().within_limits(np.array(low)) and OmxKinematics().fk(np.array(low))[0][2] < 0.02
    relay.on_leader(*leader_msg(low, 0.0))
    assert relay.phase == "refused" and "floor" in relay.reason and sent == []


def test_relay_pauses_without_leader_data_and_resumes(tmp_path):
    relay, sent, clock = make_relay(tmp_path)
    relay.on_follower(JOINTS, [0.0, -0.5, 0.7, 0.6, 0.0])
    relay.on_leader(*leader_msg([0.0, -0.5, 0.7, 0.6, 0.0], 0.3))
    clock.t += 3.0
    relay.tick()
    assert relay.phase == "following"
    clock.t += 0.6  # no leader message for 0.6 s
    relay.tick()
    assert relay.phase == "paused"
    relay.on_follower(JOINTS, [0.0, -0.5, 0.7, 0.6, 0.0])
    relay.on_leader(*leader_msg([0.0, -0.5, 0.7, 0.6, 0.0], 0.3))
    assert relay.phase == "following" and sent[-1][1] == 0.3
    # follower joint states gone (driver died): paused, and the leader alone cannot resume it
    clock.t += 1.5
    relay.tick()
    assert relay.phase == "paused"
    relay.on_leader(*leader_msg([0.0, -0.5, 0.7, 0.6, 0.0], 0.3))
    assert relay.phase == "paused"
    relay.stop()
    assert json.loads((tmp_path / "s.json").read_text())["phase"] == "stopped"


def test_relay_approach_time_follows_the_distance(tmp_path):
    relay, sent, clock = make_relay(tmp_path, approach_speed=0.4)
    relay.on_follower(JOINTS, [0.0, -0.5, 0.7, 0.6, 0.0])
    relay.on_leader(*leader_msg([1.6, -0.5, 0.7, 0.6, 0.0], 0.3))  # 1.6 rad on joint1
    assert sent[0][2] == pytest.approx(4.0) and relay.t_arrive == pytest.approx(clock.t + 4.5)
