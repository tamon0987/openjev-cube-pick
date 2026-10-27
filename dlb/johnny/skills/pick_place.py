"""Pick and place by voice: the instruction goes through ``dlb.voice.agent.Agent`` (intent parser -> tasks ->
two-tier runner with the overhead guide and the Jev servo).

The Agent (and with it the RealOMX bridge, the cameras, the one openjev client) is built on the first robot
skill and kept in ``ctx.robot``; starting it moves the arm to the begin pose and runs the ~22 s overhead
self-calibration. ``robot_home`` and ``robot_release`` are the small moves one asks for by voice.

Cancel: the runner has no cancel hook, so ``cancel`` zeroes its decision and planner budgets -- it stops at the
next decision (an in-flight move, or a guided approach / place in progress, finishes first) and the arm holds
where it is.
"""

from __future__ import annotations

import argparse
import threading
import time
from dataclasses import replace
from typing import Any

from dlb.johnny.skills import rig
from dlb.johnny.skills.base import Context, Skill, SkillResult
from dlb.johnny.skills.teleop import stop_teleop, teleop_running

NOT_HEALTHY = "openjev がまだ起動中です（あと1〜3分）。少し待ってからもう一度言ってください。"


def agent_args(ctx: Context, objects: list[str] | None, place: str, flat_place: bool) -> argparse.Namespace:
    """The fields of ``python -m dlb.voice.agent``'s argparse (see its ``main``), as Johnny runs it."""
    return argparse.Namespace(
        text=True,
        dry_run=False,
        objects=",".join(o.strip() for o in (objects or []) if o.strip()),
        bin_name=place or "",
        flat_place=bool(flat_place),
        robot_config=ctx.robot_config,
        device="default",
        stt_url=f"{rig.STT_URL}/v1/",
        stt_model=None,
        intent_model="gpt-5.5",
        effort="low",
        mark_model="gpt-5.5",
        mark_n=5,
        remark_s=30.0,
        max_decisions=40,
        out=str(rig.OUT_DIR),
        run_name=time.strftime("session_%Y%m%d_%H%M%S"),
    )


def build_agent(args: argparse.Namespace) -> Any:
    from dlb.voice.agent import Agent

    return Agent(args)


def ensure_robot(
    ctx: Context, objects: list[str] | None = None, place: str = "", flat_place: bool = False
) -> tuple[Any, str]:
    """``ctx.robot`` (the Agent with the robot started), building and starting it when needed; (None, why)."""
    if ctx.robot is not None:
        return ctx.robot, ""
    cfg = rig.load_config(ctx.robot_config)
    if not rig.openai_key(ctx.env_file):
        return None, "OPENAI_API_KEY is not set (.env): the instruction parser and the overhead marks need it"
    if not rig.openjev_ok():
        return None, NOT_HEALTHY if rig.docker_state("openjev") == "running" else (
            "openjev is not running: rig_setup starts it (healthy after 1-3 min)"
        )
    if not rig.follower_running(ctx, cfg):
        ctx.say("フォロワーのドライバを起動します。")
        ok, msg = rig.start_follower(ctx, cfg)
        if not ok:
            return None, msg
    rig.load_env_file(ctx.env_file)
    ctx.say(
        "ロボットを起動します。アームがはじめの姿勢に動き、30秒ほどかけて俯瞰カメラを自己校正します。手を離してください。"
    )
    agent = build_agent(agent_args(ctx, objects, place, flat_place))
    try:
        agent.start_robot()
    except BaseException as e:  # SystemExit from discover(), RuntimeError from the calibration, Ctrl-C
        if agent.env is not None:
            try:
                agent.env.close()
            except Exception:  # noqa: BLE001 - already failing
                pass
        if isinstance(e, KeyboardInterrupt):
            raise
        return None, f"could not start the robot: {e}"
    _record_results(agent)
    ctx.robot = agent
    ctx.state.pop("arm_moved_externally", None)
    ctx.log(f"robot started: objects {agent.objects}, place {agent.args.bin_name!r}")
    return agent, ""


def _record_results(agent: Any) -> None:
    """Keep the runner's last result on the agent (``Agent.execute`` prints it but does not return it)."""
    runner = agent.runner
    if runner is None or getattr(runner, "_johnny_wrapped", False):
        return
    orig = runner.run

    def run(*a: Any, **k: Any) -> Any:
        r = orig(*a, **k)
        agent.last_result = r
        return r

    runner.run, runner._johnny_wrapped = run, True


def return_to_begin(ctx: Context, agent: Any) -> None:
    """After tele-operation the arm is anywhere: back to the begin pose, gripper open, marks afresh."""
    if ctx.state.pop("arm_moved_externally", False):
        ctx.say("はじめの姿勢に戻ります。")
        agent.env.reset(seed=0)
        agent.state = replace(agent.state, held=None, queue=[])
        agent.idle_mark()


def leave_teleop(ctx: Context) -> str:
    if teleop_running(ctx):
        ctx.say("テレオペを止めます。")
        stop_teleop(ctx)
        return "tele-operation stopped first. "
    return ""


class PickPlace(Skill):
    name = "pick_place"
    description = (
        "Have the robot arm pick objects on the table and place them (into the container, on another object, or "
        "put down where it is), from the user's own words: 'put the carrot in the tray', '人参をトレーに入れて', "
        "'赤いのをビンに', 'それを青いキューブの上に', 'その場に置いて'. Pass the instruction as said. The first "
        "call starts the robot (begin pose + ~30 s camera self-calibration); the objects and the container are "
        "found in the overhead image unless given. Blocks until the task ends; if the instruction is ambiguous "
        "the result is a question to ask the user (ok=false, data.clarify). Not while tele-operation runs "
        "(it is stopped first) or another task runs."
    )
    parameters = {
        "type": "object",
        "properties": {
            "instruction": {
                "type": "string",
                "description": "What the user asked for, verbatim (usually Japanese), e.g. 「人参をトレーに入れて」.",
            },
            "objects": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional: names of the objects on the table (English, e.g. 'carrot', 'red cube'); "
                "used only when the robot starts, else they are found in the overhead image.",
            },
            "place": {
                "type": "string",
                "description": "Optional: how the container / place looks (e.g. 'white tray'); used only when the "
                "robot starts, else found in the image.",
            },
            "flat_place": {
                "type": "boolean",
                "description": "The place is a flat mark on the table (objects are set down on it), not a container. "
                "Default false. Robot start only.",
            },
        },
        "required": ["instruction"],
        "additionalProperties": False,
    }

    def __init__(self) -> None:
        self._cancel = threading.Event()

    def run(
        self,
        ctx: Context,
        instruction: str = "",
        objects: list[str] | None = None,
        place: str = "",
        flat_place: bool = False,
        **_: Any,
    ) -> SkillResult:
        if not instruction.strip():
            return SkillResult("no instruction given", ok=False)
        if ctx.state.get("task_running"):
            return SkillResult("a robot task is already running; say stop first", ok=False)
        prefix = leave_teleop(ctx)
        agent, err = ensure_robot(ctx, objects, place, flat_place)
        if agent is None:
            return SkillResult(prefix + err, ok=False)
        if not rig.openjev_ok():
            return SkillResult(prefix + NOT_HEALTHY, ok=False)
        intent = agent.parse(instruction)
        if intent is None:
            return SkillResult(
                prefix + "could not parse the instruction (OpenAI request failed); say it again", ok=False
            )
        if intent.clarify:
            return SkillResult(
                prefix + intent.clarify, ok=False, data={"clarify": intent.clarify, "ask_user": True}
            )
        if not intent.tasks:
            return SkillResult(
                prefix + "nothing to do (the instruction was a stop or had no task)", data={"tasks": []}
            )
        self._cancel.clear()
        ctx.state["task_running"], ctx.state["running_skill"] = True, self
        runner = agent.runner
        budgets = (runner.max_decisions, runner.max_planner_calls)
        try:
            return_to_begin(ctx, agent)
            ctx.say("実行します。")
            agent.execute(intent.tasks)
        except Exception as e:  # noqa: BLE001 - a failed motion leaves the arm where it is
            return SkillResult(
                prefix + f"the task failed: {e}; the arm holds where it is",
                ok=False,
                data={"tasks": intent.tasks},
            )
        finally:
            runner.max_decisions, runner.max_planner_calls = budgets
            ctx.state["task_running"], ctx.state["running_skill"] = False, None
        r = getattr(agent, "last_result", None)
        held = agent.state.held
        data = {
            "tasks": intent.tasks,
            "held": held,
            "cancelled": self._cancel.is_set(),
            "stop_reason": getattr(r, "stop_reason", ""),
            "decisions": getattr(r, "decisions", None),
            "missed_grasps": getattr(r, "missed_grasps", None),
        }
        if self._cancel.is_set():
            return SkillResult(
                prefix + f"stopped on request; the arm holds where it is, holding {held or 'nothing'}",
                data=data,
            )
        done = ", ".join(f"{t['op']} {t.get('object') or t.get('where')}" for t in intent.tasks)
        return SkillResult(
            prefix + f"done: {done} ({data['stop_reason'] or 'finished'}, {data['decisions']} decisions, "
            f"{data['missed_grasps']} missed grasps); holding {held or 'nothing'}",
            data=data,
        )

    def cancel(self, ctx: Context) -> None:
        """Stop at the runner's next decision / planner call; the move in flight finishes and the arm holds."""
        self._cancel.set()
        agent = ctx.robot
        if agent is not None and agent.runner is not None:
            agent.runner.max_decisions = 0
            agent.runner.max_planner_calls = 0
        ctx.log("pick_place: cancel requested (stops after the current move)")


class RobotHome(Skill):
    name = "robot_home"
    description = (
        "Move the robot arm to its begin pose (over the table, camera looking down), gripper as it is. Starts the "
        "robot when it is not started yet. Use for 'go home', 'はじめの姿勢に', 'ホームに戻って', '元の位置へ'."
    )
    parameters = {"type": "object", "properties": {}, "additionalProperties": False}

    def run(self, ctx: Context, **_: Any) -> SkillResult:
        if ctx.state.get("task_running"):
            return SkillResult("a robot task is running; say stop first", ok=False)
        prefix = leave_teleop(ctx)
        started = ctx.robot is None
        agent, err = ensure_robot(ctx)
        if agent is None:
            return SkillResult(prefix + err, ok=False)
        if started:
            return SkillResult(prefix + "the robot started and is at the begin pose", data={"started": True})
        try:
            ctx.say("はじめの姿勢に動きます。")
            agent.env.go_pose("begin")
            ctx.state.pop("arm_moved_externally", None)
            agent.idle_mark()
        except Exception as e:  # noqa: BLE001
            return SkillResult(prefix + f"could not reach the begin pose: {e}", ok=False)
        return SkillResult(prefix + "the arm is at the begin pose", data={"started": False})


class RobotRelease(Skill):
    name = "robot_release"
    description = (
        "Open the gripper where the arm is (lets go of whatever it holds). Use for 'open the gripper', '放して', "
        "'離して', 'グリッパー開けて'. Starts the robot when needed."
    )
    parameters = {"type": "object", "properties": {}, "additionalProperties": False}

    def run(self, ctx: Context, **_: Any) -> SkillResult:
        if ctx.state.get("task_running"):
            return SkillResult("a robot task is running; say stop first", ok=False)
        prefix = leave_teleop(ctx)
        agent, err = ensure_robot(ctx)
        if agent is None:
            return SkillResult(prefix + err, ok=False)
        try:
            agent.env.open_gripper()
        except Exception as e:  # noqa: BLE001
            return SkillResult(prefix + f"could not open the gripper: {e}", ok=False)
        agent.state = replace(agent.state, held=None)
        return SkillResult(prefix + "the gripper is open", data={"held": None})
