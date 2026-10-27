"""Tele-operation: the leader arm (OMX-L) drives the follower (OMX-F) -- through a relay, never directly.

ROBOTIS's leader launch publishes the leader's joints as ``/leader/joint_trajectory`` with ``time_from_start = 0``
at 100 Hz, and the follower launch remaps its ``arm_controller`` to that topic: the follower jumps to the leader's
pose at full speed the moment the leader comes up (on this rig that overloaded two motors). So the leader is
launched through ``dlb/johnny/launch/omx_l_leader_relay.launch.py``, which remaps the stream to
``/leader/joint_trajectory_raw``, and ``python -m dlb.johnny.teleop_relay`` (a child process) first moves the
follower to the leader's pose slowly, then forwards the stream with a velocity limit and the gripper clamped to
the config's range. Two child processes: ``ctx.processes["relay"]`` and ``ctx.processes["leader"]``.

Our own bridge (``RealOMX``) publishes to the same ``/leader/joint_trajectory``, hence the mutual exclusion with
``pick_place``: ``ctx.state["teleop"]`` and ``ctx.state["task_running"]`` never hold at the same time.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

from dlb.johnny.skills import rig
from dlb.johnny.skills.base import Context, Skill, SkillResult
from dlb.johnny.teleop_relay import Settings, read_status

LAUNCH_FILE = Path(__file__).resolve().parents[1] / "launch" / "omx_l_leader_relay.launch.py"
RAW_TOPIC = "/leader/joint_trajectory_raw"
STATUS_FILE = "teleop_status.json"  # under rig.OUT_DIR
START_TIMEOUT_S = 40.0  # the approach alone can take a few seconds (0.4 rad/s)
SAFETY = "Keep hands and objects clear of the follower arm: it moves whenever the leader moves."
PHASES = {
    "waiting": "waiting for the leader's pose and the follower's joints",
    "approach": "approaching the leader's pose slowly",
    "following": "following the leader",
    "paused": "paused (no leader data or no follower joint states; the follower holds)",
    "refused": "refused",
    "stopped": "stopped",
}


def status_path() -> Path:
    return rig.OUT_DIR / STATUS_FILE


def relay_status() -> dict[str, Any] | None:
    return read_status(status_path())


def teleop_running(ctx: Context) -> bool:
    alive = ctx.running("leader") or ctx.running("relay")
    ctx.state["teleop"] = alive  # a launch that died on its own counts as stopped
    return alive


def teleop_line(ctx: Context) -> str:
    """One phrase for status lines: what the follower does right now."""
    if not teleop_running(ctx):
        return "tele-operation off"
    st = relay_status() or {}
    phase = st.get("phase", "unknown")
    if not ctx.running("relay"):
        return "tele-operation BROKEN: the relay is not running (the follower holds); teleop_stop"
    if not ctx.running("leader"):
        return "tele-operation BROKEN: the leader driver is not running; teleop_stop"
    line = f"tele-operation on: {PHASES.get(phase, phase)}"
    if phase == "following":
        line += f" ({st.get('messages_out', 0)} commands, {st.get('clamped', 0)} velocity-clamped)"
    return line


def stop_teleop(ctx: Context) -> dict[str, str]:
    """Stop the forwarding first (the relay), then SIGINT the leader launch; the follower holds its last pose."""
    r = {"relay": rig.stop(ctx, "relay"), "leader": rig.stop(ctx, "leader")}
    ctx.state["teleop"] = False
    return r


def wait_for_following(ctx: Context, timeout_s: float = START_TIMEOUT_S) -> tuple[bool, str]:
    """Poll the relay's status until it follows, refuses, or something dies."""
    t0 = time.monotonic()
    last = ""
    while time.monotonic() - t0 < timeout_s:
        relay, leader = ctx.processes.get("relay"), ctx.processes.get("leader")
        st = relay_status() or {}
        phase = st.get("phase", "")
        if phase == "refused":
            return False, "the relay refused the approach: " + str(st.get("reason", ""))
        if phase == "following":
            return True, str(st.get("reason", ""))
        if relay is not None and relay.poll() is not None:
            return False, f"the relay exited (code {relay.returncode}); see {rig.OUT_DIR}/relay.log"
        if leader is not None and leader.poll() is not None:
            return False, f"the leader launch exited (code {leader.returncode}); see {rig.OUT_DIR}/leader.log"
        if phase == "approach" and phase != last:
            ctx.say("フォロワーがリーダーの姿勢までゆっくり動きます。")
        last = phase
        time.sleep(0.5)
    st = relay_status() or {}
    return False, (
        f"tele-operation did not start within {timeout_s:.0f} s (relay {st.get('phase') or 'silent'}): "
        "no leader joint stream? check the leader's port and power"
    )


class TeleopStart(Skill):
    name = "teleop_start"
    description = (
        "Start tele-operation: the follower robot arm mirrors the leader arm that a person moves by hand. Starts "
        "the follower driver first when needed. The follower first moves SLOWLY to the leader's current pose "
        "(refused when that pose is outside the joint limits or too low), then follows with a speed limit. Use "
        "for 'teleop', 'テレオペしたい', 'リーダーアームで動かしたい', '手で操作する'. Not possible while a "
        "pick-and-place task runs."
    )
    parameters = {"type": "object", "properties": {}, "additionalProperties": False}

    def run(self, ctx: Context, **_: Any) -> SkillResult:
        if ctx.state.get("task_running"):
            return SkillResult("a robot task is running; say stop first, then start tele-operation", ok=False)
        if teleop_running(ctx):
            return SkillResult(f"{teleop_line(ctx)}. {SAFETY}", data={"teleop": True})
        cfg = rig.load_config(ctx.robot_config)
        ports = rig.arm_ports(ctx, cfg)
        if not ports["leader"]:
            return SkillResult(
                f"the leader arm's USB port was not found (arm ports: {ports['found'] or 'none'}); plug the "
                "leader in, or run identify_arms when which port is which is unknown",
                ok=False,
                data=ports,
            )
        notes = []
        if not rig.follower_running(ctx, cfg):
            ctx.say("フォロワーのドライバを起動します。")
            ok, msg = rig.start_follower(ctx, cfg)
            if not ok:
                return SkillResult(f"cannot start tele-operation: {msg}", ok=False)
            notes.append(msg)
        ctx.say(
            "テレオペを始めます。フォロワーアームがリーダーの姿勢までゆっくり動いてから追従します。手を離してください。"
        )
        try:
            status_path().unlink()
        except OSError:
            pass
        env = rig.ros_env(cfg)
        # the relay first (it waits for the stream), then the leader with its stream parked on the raw topic
        rig.spawn(
            ctx,
            "relay",
            f"{sys.executable} -m dlb.johnny.teleop_relay --config {ctx.robot_config} --raw {RAW_TOPIC} "
            f"--status {status_path()}",
            env,
        )
        rig.spawn(
            ctx,
            "leader",
            f"ros2 launch {LAUNCH_FILE} port_name:={ports['leader']} raw_topic:={RAW_TOPIC}",
            env,
        )
        ok, msg = wait_for_following(ctx)
        if not ok:
            stop_teleop(ctx)
            return SkillResult(f"tele-operation not started: {msg}", ok=False, data={"teleop": False})
        ctx.state["teleop"] = True
        ctx.state["arm_moved_externally"] = True  # the next robot task returns to the begin pose first
        s = Settings
        return SkillResult(
            f"tele-operation is on: the follower reached the leader's pose ({msg}) and now mirrors the leader "
            f"(leader on {ports['leader']}, stream {RAW_TOPIC} -> relay -> {cfg.get('command_topic')}; at most "
            f"{s.max_step * s.rate_hz:.1f} rad/s per joint, gripper kept within the config's open/closed range). "
            f"{SAFETY} Say stop to end it." + (" " + "; ".join(notes) if notes else ""),
            data={"teleop": True, "leader_port": ports["leader"], "follower_port": ports["follower"]},
        )


class TeleopStop(Skill):
    name = "teleop_stop"
    description = (
        "End tele-operation: stop forwarding the leader's motion and the leader arm driver; the follower holds "
        "its current pose (torque stays on). Use for 'stop teleop', 'テレオペ終わり', 'もう手で動かさない'."
    )
    parameters = {"type": "object", "properties": {}, "additionalProperties": False}

    def run(self, ctx: Context, **_: Any) -> SkillResult:
        r = stop_teleop(ctx)
        if all(v == "not running" for v in r.values()):
            return SkillResult("tele-operation was not running", data={"teleop": False, **r})
        return SkillResult(
            f"tele-operation stopped (relay {r['relay']}, leader {r['leader']}); the follower holds its pose "
            "where the leader left it",
            data={"teleop": False, **r},
        )


class TeleopStatus(Skill):
    name = "teleop_status"
    description = (
        "Whether tele-operation (leader arm mirroring) is on and what the follower does: approaching, "
        "following, paused. Cheap. 'テレオペ中?', 'ついてきてる?'"
    )
    parameters = {"type": "object", "properties": {}, "additionalProperties": False}

    def run(self, ctx: Context, **_: Any) -> SkillResult:
        line = teleop_line(ctx)
        return SkillResult(line, data={"teleop": ctx.state.get("teleop", False), "relay": relay_status()})
