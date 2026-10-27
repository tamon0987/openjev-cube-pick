"""Stop: at once (``stop_all``: hold where it is) or safely (``safe_stop``: open, rest pose, torque off); status.

The core runs ``stop_all`` on bare stop words without the LLM. 「止まって」/「ストップ」 mean "freeze now";
「停止して」/「終了して」/「今日は終わり」 mean "finish up": the LLM maps those to ``safe_stop``, and ``stop_all``
does the same when the caller passes such a word while nothing runs (``word``).
"""

from __future__ import annotations

import time
from typing import Any

from dlb.johnny.skills import rig
from dlb.johnny.skills.base import Context, Skill, SkillResult
from dlb.johnny.skills.teleop import stop_teleop, teleop_line, teleop_running

SAFE_WORDS = ("停止", "終了", "finish", "shut down", "shutdown", "今日は終わり", "おしまい")
TASK_END_WAIT_S = 90.0  # a cancelled task ends after its move in flight (a guided approach can take a while)


def is_safe_word(word: str) -> bool:
    w = (word or "").strip().lower()
    return bool(w) and any(k in w for k in SAFE_WORDS)


def safe_stop_sequence(ctx: Context, docker: bool = False) -> tuple[list[str], bool]:
    """Cancel, end teleop, open the gripper, rest pose, release the robot, torque off; (steps, all ok)."""
    steps: list[str] = []
    # 1. a running task: cancel, and wait for its move in flight to end before touching the arm
    skill = ctx.state.get("running_skill")
    if ctx.state.get("task_running") and skill is not None:
        ctx.say("作業を止めます。")
        skill.cancel(ctx)
        t0 = time.monotonic()
        while ctx.state.get("task_running") and time.monotonic() - t0 < TASK_END_WAIT_S:
            time.sleep(0.2)
        if ctx.state.get("task_running"):
            steps.append(
                f"{skill.name} cancelled but still running after {TASK_END_WAIT_S:.0f} s: the arm is left alone"
            )
            return steps, False
        steps.append(f"{skill.name} cancelled (ended after its move in flight)")
    # 2. tele-operation: the relay first, then the leader driver
    if teleop_running(ctx):
        ctx.say("テレオペを止めます。")
        stop_teleop(ctx)
        steps.append("tele-operation stopped")
    # 3./4. the robot: open the gripper right away (the arm is never high), then slowly to the rest pose
    agent = ctx.robot
    if agent is not None and getattr(agent, "env", None) is not None:
        env = agent.env
        try:
            was_held = bool(env.held)
            ctx.say("グリッパーを開きます。")
            env.open_gripper()
            steps.append("gripper opened" + (" (the held object was let go)" if was_held else ""))
            if hasattr(agent, "state"):
                agent.state = type(agent.state)(None, agent.state.objects, [], agent.state.place_name)
            pose = "rest" if (rig.load_config(ctx.robot_config).get("poses") or {}).get("rest") else "begin"
            ctx.say("アームを休止姿勢にゆっくり動かします。")
            env.go_pose(pose)
            steps.append(f"arm at the {pose} pose")
        except Exception as e:  # noqa: BLE001 - a failed move: leave the torque on, the arm where it is
            steps.append(f"moving the arm failed ({e}): the arm holds where it is, torque left ON")
            return steps, False
        # 5. release the robot (cameras, ROS node) and cut the torque with the arm resting
        try:
            env.close()
        except Exception as e:  # noqa: BLE001 - shutting down anyway
            ctx.log(f"robot close: {e}")
        ctx.robot = None
        steps.append("robot released (cameras closed)")
    follower = rig.stop(ctx, "follower")
    if follower != "not running":
        ctx.say("フォロワーのドライバを止めます。トルクが切れます。")
        steps.append(f"follower driver {follower} (torque off)")
    ctx.state["teleop"] = False
    if docker:
        stopped = rig.docker_stop(["openjev", "stt"])
        steps.append("docker stopped: " + (", ".join(stopped) or "nothing was running"))
    return steps, True


class StopAll(Skill):
    name = "stop_all"
    description = (
        "IMMEDIATE STOP: cancel the running robot task (it halts after the move in flight) and end "
        "tele-operation; the arm holds its position, nothing else moves. Use for '止まって', 'ストップ', 'やめて', "
        "'stop', '待って' whenever something is running or the user sounds alarmed. Safe to call when nothing "
        "runs. For a planned shutdown ('停止して', '終了して', '今日は終わり') use safe_stop instead."
    )
    parameters = {
        "type": "object",
        "properties": {
            "word": {
                "type": "string",
                "description": "Optional: the stop word heard. 停止/終了 while nothing runs means a planned stop: "
                "safe_stop runs instead (gripper open, rest pose, torque off).",
            }
        },
        "additionalProperties": False,
    }

    def run(self, ctx: Context, word: str = "", **_: Any) -> SkillResult:
        done = []
        idle = not ctx.state.get("task_running") and not teleop_running(ctx)
        if idle and is_safe_word(word):
            return SafeStop().run(ctx)
        skill = ctx.state.get("running_skill")
        if ctx.state.get("task_running") and skill is not None:
            skill.cancel(ctx)
            done.append(f"{skill.name} cancelled (it stops after the current move)")
        if teleop_running(ctx):
            stop_teleop(ctx)
            done.append("tele-operation stopped (relay and leader driver)")
        text = "; ".join(done) if done else "nothing was running"
        return SkillResult(f"{text}; the arm holds where it is", data={"stopped": done})


class SafeStop(Skill):
    name = "safe_stop"
    description = (
        "PLANNED, SAFE STOP of the robot, in order: cancel a running task, end tele-operation, open the gripper, "
        "move the arm slowly to its rest pose, release the robot and cut the follower driver's torque with the "
        "arm resting. Use for '停止して', '終了して', '今日は終わり', 'ロボットを止めて', 'finish', 'shut the robot "
        "down', 'おしまい'. openjev and STT keep running unless docker=true. For an IMMEDIATE halt ('止まって', "
        "'ストップ') use stop_all."
    )
    parameters = {
        "type": "object",
        "properties": {
            "docker": {
                "type": "boolean",
                "description": "Also stop the openjev and stt containers (minutes to come back). Default false.",
            }
        },
        "additionalProperties": False,
    }

    def run(self, ctx: Context, docker: bool = False, **_: Any) -> SkillResult:
        steps, ok = safe_stop_sequence(ctx, docker)
        text = "; ".join(f"{i + 1}. {s}" for i, s in enumerate(steps)) or (
            "nothing was running and no robot was started"
        )
        return SkillResult(
            ("safe stop done: " if ok else "safe stop INCOMPLETE: ") + text, ok=ok, data={"steps": steps}
        )


class Status(Skill):
    name = "status"
    description = (
        "One-line status of the rig and the robot: what runs (task, tele-operation, follower driver), whether the "
        "robot is started and what it holds, openjev / STT health, cameras and arm ports present. Cheap. Use for "
        "'今どうなってる?', 'status', '何してる?', 'ロボットは動いてる?'."
    )
    parameters = {"type": "object", "properties": {}, "additionalProperties": False}

    def run(self, ctx: Context, **_: Any) -> SkillResult:
        cfg = rig.load_config(ctx.robot_config)
        agent = ctx.robot
        task = bool(ctx.state.get("task_running"))
        teleop = teleop_running(ctx)
        follower = rig.follower_running(ctx, cfg, probe=not task)  # no ROS CLI while the robot works
        ports = rig.arm_ports(ctx, cfg)
        cams = {str(p) for p in rig.list_cameras()}
        cameras = {
            cam: str(cfg.get("cameras", {}).get(cam, {}).get("device")) in cams for cam in ("wrist", "front")
        }
        held = None
        if agent is not None:
            try:
                held = agent.state.held or ("something" if agent.env.held else None)
            except Exception:  # noqa: BLE001 - status must not fail
                held = None
        openjev, stt = rig.openjev_ok(), rig.stt_ok()
        parts = [
            "task running" if task else (teleop_line(ctx) if teleop else "idle"),
            ("robot started, holding " + (held or "nothing")) if agent is not None else "robot not started",
            "follower driver " + ("up" if follower else "down"),
            "openjev " + ("ok" if openjev else "down"),
            "stt " + ("ok" if stt else "down"),
            "cameras " + ", ".join(c for c, ok in cameras.items() if ok)
            if any(cameras.values())
            else "no rig camera",
            f"arm ports follower {ports['follower'] or '?'} leader {ports['leader'] or '?'}",
        ]
        procs = [n for n in ctx.processes if ctx.running(n)]
        if procs:
            parts.append("launched: " + ", ".join(procs))
        pending = [k for k in ("identify_cameras", "identify_arms") if ctx.state.get(k)]
        if pending:
            parts.append("in progress: " + ", ".join(pending))
        data = {
            "task_running": task,
            "teleop": teleop,
            "robot_started": agent is not None,
            "held": held,
            "follower": follower,
            "openjev": openjev,
            "stt": stt,
            "cameras": cameras,
            "ports": ports,
            "processes": procs,
        }
        return SkillResult("; ".join(parts), data=data)
