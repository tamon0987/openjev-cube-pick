"""Tele-operation: the leader arm (OMX-L) drives the follower (OMX-F).

The leader launch (``omx_l_leader_ai.launch.py``) runs a ``joint_trajectory_command_broadcaster`` in the
``/leader`` namespace that publishes the leader's joints as ``/leader/joint_trajectory``; the follower launch
remaps its ``arm_controller``'s ``/arm_controller/joint_trajectory`` to that topic, so the follower follows as
soon as both run. Our own bridge (``RealOMX``) publishes to the same topic, hence the mutual exclusion with
``pick_place``: ``ctx.state["teleop"]`` and ``ctx.state["task_running"]`` never hold at the same time.
"""

from __future__ import annotations

import time
from typing import Any

from dlb.johnny.skills import rig
from dlb.johnny.skills.base import Context, Skill, SkillResult

SAFETY = "Keep hands and objects clear of the follower arm: it moves whenever the leader moves."


def teleop_running(ctx: Context) -> bool:
    alive = ctx.running("leader")
    ctx.state["teleop"] = alive  # a launch that died on its own counts as stopped
    return alive


def stop_teleop(ctx: Context) -> str:
    """SIGINT to the leader launch and wait; the follower holds its last pose."""
    r = rig.stop(ctx, "leader")
    ctx.state["teleop"] = False
    return r


class TeleopStart(Skill):
    name = "teleop_start"
    description = (
        "Start tele-operation: the follower robot arm mirrors the leader arm that a person moves by hand. Starts "
        "the follower driver first when needed. Use for 'teleop', 'テレオペしたい', 'リーダーアームで動かしたい', "
        "'手で操作する'. Not possible while a pick-and-place task runs. The follower jumps to the leader's pose "
        "the moment it starts."
    )
    parameters = {"type": "object", "properties": {}, "additionalProperties": False}

    def run(self, ctx: Context, **_: Any) -> SkillResult:
        if ctx.state.get("task_running"):
            return SkillResult("a robot task is running; say stop first, then start tele-operation", ok=False)
        if teleop_running(ctx):
            return SkillResult(f"tele-operation is already on. {SAFETY}", data={"teleop": True})
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
        ctx.say("テレオペを始めます。フォロワーアームがリーダーの姿勢に動きます。手を離してください。")
        p = rig.spawn(ctx, "leader", rig.LEADER_LAUNCH.format(port=ports["leader"]), rig.ros_env(cfg))
        time.sleep(3.0)  # a wrong port or a busy one makes the launch die at once
        if p.poll() is not None:
            ctx.state["teleop"] = False
            return SkillResult(
                f"the leader launch exited right away (code {p.returncode}); see {rig.OUT_DIR}/leader.log "
                f"(port {ports['leader']}: is the leader powered and on that port?)",
                ok=False,
            )
        ctx.state["teleop"] = True
        ctx.state["arm_moved_externally"] = True  # the next robot task returns to the begin pose first
        return SkillResult(
            f"tele-operation is on: the follower now mirrors the leader arm (leader on {ports['leader']}, topic "
            f"/leader/joint_trajectory). {SAFETY} Say stop to end it."
            + (" " + "; ".join(notes) if notes else ""),
            data={"teleop": True, "leader_port": ports["leader"], "follower_port": ports["follower"]},
        )


class TeleopStop(Skill):
    name = "teleop_stop"
    description = (
        "End tele-operation: stop the leader arm driver; the follower holds its current pose (torque stays on). "
        "Use for 'stop teleop', 'テレオペ終わり', 'もう手で動かさない'."
    )
    parameters = {"type": "object", "properties": {}, "additionalProperties": False}

    def run(self, ctx: Context, **_: Any) -> SkillResult:
        r = stop_teleop(ctx)
        if r == "not running":
            return SkillResult("tele-operation was not running", data={"teleop": False})
        return SkillResult(
            f"tele-operation {r}; the follower holds its pose where the leader left it",
            data={"teleop": False},
        )


class TeleopStatus(Skill):
    name = "teleop_status"
    description = "Whether tele-operation (leader arm mirroring) is on. Cheap. 'テレオペ中?'"
    parameters = {"type": "object", "properties": {}, "additionalProperties": False}

    def run(self, ctx: Context, **_: Any) -> SkillResult:
        on = teleop_running(ctx)
        return SkillResult("tele-operation is on" if on else "tele-operation is off", data={"teleop": on})
