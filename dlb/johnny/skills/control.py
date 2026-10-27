"""Stop everything, and a one-line status."""

from __future__ import annotations

from typing import Any

from dlb.johnny.skills import rig
from dlb.johnny.skills.base import Context, Skill, SkillResult
from dlb.johnny.skills.teleop import stop_teleop, teleop_running


class StopAll(Skill):
    name = "stop_all"
    description = (
        "STOP: cancel the running robot task (it halts after the move in flight) and end tele-operation; the arm "
        "holds its position. Use for '止まって', 'ストップ', 'やめて', 'stop', '待って' whenever something is "
        "running or the user sounds alarmed. Safe to call when nothing runs."
    )
    parameters = {"type": "object", "properties": {}, "additionalProperties": False}

    def run(self, ctx: Context, **_: Any) -> SkillResult:
        done = []
        skill = ctx.state.get("running_skill")
        if ctx.state.get("task_running") and skill is not None:
            skill.cancel(ctx)
            done.append(f"{skill.name} cancelled (it stops after the current move)")
        if teleop_running(ctx):
            stop_teleop(ctx)
            done.append("tele-operation stopped")
        text = "; ".join(done) if done else "nothing was running"
        return SkillResult(f"{text}; the arm holds where it is", data={"stopped": done})


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
            "task running" if task else ("tele-operation on" if teleop else "idle"),
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
