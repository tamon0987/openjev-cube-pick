"""Pick and place by voice: the instruction goes through ``dlb.voice.agent.Agent`` (intent parser -> tasks ->
two-tier runner with the overhead guide and the Jev servo), and the outcome is verified before it is reported.

The Agent (and with it the RealOMX bridge, the cameras, the one openjev client) is built on the first robot
skill -- ``rig_setup`` by default -- and kept in ``ctx.robot``; starting it moves the arm to the begin pose and
runs the ~22 s overhead self-calibration. ``robot_home`` and ``robot_release`` are the small moves one asks for
by voice.

After a task, one fresh overhead mark must show the object inside the place's box (a flat place: within
4 cm of its mark); otherwise the result says so (ok=False) after one retry of the whole task.

Cancel: the runner has no cancel hook, so ``cancel`` zeroes its decision and planner budgets -- it stops at the
next decision (an in-flight move, or a guided approach / place in progress, finishes first) and the arm holds
where it is.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from dlb.johnny.skills import rig
from dlb.johnny.skills.base import Context, Skill, SkillResult
from dlb.johnny.skills.teleop import stop_teleop, teleop_running

NOT_HEALTHY = "openjev がまだ起動中です（あと1〜3分）。少し待ってからもう一度言ってください。"
VERIFY_RETRIES = 1  # the whole task once more when the overhead image does not show the object in its place
FLAT_TOL_M = 0.04  # a flat place: the object counts as placed within this distance of the place's mark


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
        return None, (
            NOT_HEALTHY
            if rig.docker_state("openjev") == "running"
            else "openjev is not running: rig_setup starts it (healthy after 1-3 min)"
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


# ------------------------------------------------------------------ #
# verification of the place by the overhead image
# ------------------------------------------------------------------ #
@dataclass
class Verification:
    ok: bool | None  # True: seen in its place; False: seen elsewhere; None: could not tell
    text: str
    record: dict[str, Any]


def _inside(pt: np.ndarray, box: np.ndarray | None) -> bool:
    return box is not None and bool(box[0] <= pt[0] <= box[2] and box[1] <= pt[1] <= box[3])


def verify_place(agent: Any, tasks: list[dict[str, str]]) -> Verification | None:
    """After the runner returned to the begin pose: does one fresh overhead mark show the last picked object in
    its place? Uses the idle mark the agent starts after every task (and starts the next one afterwards).
    None when the task list does not end with a place."""
    if not tasks or tasks[-1].get("op") != "place":
        return None
    picks = [t["object"] for t in tasks if t.get("op") == "pick" and t.get("object")]
    guide, env = agent.guide, agent.env
    if not picks or guide is None:
        return None
    obj, where = picks[-1], tasks[-1].get("where") or "bin"
    place = agent.args.bin_name if where == "bin" else where.removeprefix("on:")
    flat = where == "bin" and bool(getattr(agent.args, "flat_place", False))
    rec: dict[str, Any] = {"type": "verify", "object": obj, "where": where, "place": place}
    t0 = time.perf_counter()
    try:
        shot = guide.join() if guide.pending else None
        marks = shot.marks if shot is not None else guide.marker.mark(env.overhead_frame())
    except Exception as e:  # noqa: BLE001 - no mark, no verdict
        rec.update(ok=None, reason=f"marking failed: {e}")
        _log_verify(agent, rec)
        return Verification(None, f"could not verify the place (the overhead mark failed: {e})", rec)
    rec["seconds"] = round(time.perf_counter() - t0, 1)
    pt = marks.points.get(obj)
    rec["object_px"] = None if pt is None else np.round(pt, 1).tolist()
    if where == "here":
        ok = pt is not None
        text = (
            f"set down here: the {obj} is {'seen on the table' if ok else 'NOT seen in the overhead image'}"
        )
    elif pt is None:
        ok = None
        text = f"placed, but the {obj} is not seen in the overhead image: cannot confirm it is in the {place}"
    else:
        box, ppt = marks.boxes.get(place), marks.points.get(place)
        rec["place_box_px"] = None if box is None else np.round(box, 1).tolist()
        dist = None
        if ppt is not None and getattr(guide.map, "ready", False):
            dist = float(np.linalg.norm(guide.map.to_table(pt) - guide.map.to_table(ppt)))
            rec["distance_cm"] = round(dist * 100, 1)
        if flat:
            ok = dist is not None and dist <= FLAT_TOL_M
        else:
            ok = _inside(pt, box) if box is not None else (dist is not None and dist <= FLAT_TOL_M)
        if ok:
            text = (
                f"placed, confirmed by the overhead image: the {obj} is {'on' if flat else 'in'} the {place}"
            )
        else:
            lies = ""
            if getattr(guide.map, "ready", False):
                xy = guide.map.to_table(pt)
                lies = f"it lies at ({xy[0] * 100:.0f}, {xy[1] * 100:.0f}) cm on the table"
                if dist is not None:
                    lies += f", {dist * 100:.0f} cm from the {place}"
            text = f"placed, but the {obj} is NOT {'on' if flat else 'in'} the {place}" + (
                f": {lies}" if lies else ""
            )
    rec.update(ok=ok, text=text)
    _log_verify(agent, rec)
    agent.idle_mark()  # the mark was consumed: start the next one so the next instruction finds a fresh one
    return Verification(ok, text, rec)


def _log_verify(agent: Any, rec: dict[str, Any]) -> None:
    """Append the verdict to the episode log the runner wrote (``ep<n>.jsonl``)."""
    log_dir = getattr(agent.runner, "log_dir", None)
    if not log_dir:
        return
    try:
        with open(log_dir / f"ep{max(0, agent.episode - 1):04d}.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except OSError:
        pass


class PickPlace(Skill):
    name = "pick_place"
    description = (
        "Have the robot arm pick objects on the table and place them (into the container, on another object, or "
        "put down where it is), from the user's own words: 'put the carrot in the tray', '人参をトレーに入れて', "
        "'赤いのをビンに', 'それを青いキューブの上に', 'その場に置いて'. Pass the instruction as said. Starts the "
        "robot first when rig_setup did not (begin pose + ~30 s camera self-calibration); the objects and the "
        "container are found in the overhead image unless given. Blocks until the task ends. The place is checked "
        "with the overhead image afterwards: ok=false with "
        "data.verified=false means the object is NOT where it should be -- never report success then. If the "
        "instruction is ambiguous the result is a question to ask the user (ok=false, data.clarify). Not while "
        "tele-operation runs (it is stopped first) or another task runs."
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
            # an object the user names may have been put down after start-up: look again, then parse again
            try:
                added = agent.refresh_objects()
            except Exception as e:  # noqa: BLE001 - the clarify question stands
                added, _ = [], ctx.log(f"refresh_objects failed: {e}")
            if added:
                intent = agent.parse(instruction) or intent
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
        attempts, verdict, r = 0, None, None
        try:
            return_to_begin(ctx, agent)
            agent.ensure_fresh_marks()  # the overhead image is of the table as it is now
            ctx.say("実行します。")
            while True:
                attempts += 1
                agent.execute(intent.tasks)
                r = getattr(agent, "last_result", None)
                complete = getattr(r, "stop_reason", "") == "planner_complete"
                verdict = (
                    verify_place(agent, intent.tasks) if complete and not self._cancel.is_set() else None
                )
                if self._cancel.is_set() or not complete or verdict is None or verdict.ok is True:
                    break
                if attempts > VERIFY_RETRIES:
                    break
                ctx.say("置けていないので、もう一度やります。")
        except Exception as e:  # noqa: BLE001 - a failed motion leaves the arm where it is
            return SkillResult(
                prefix + f"the task failed: {e}; the arm holds where it is",
                ok=False,
                data={"tasks": intent.tasks},
            )
        finally:
            runner.max_decisions, runner.max_planner_calls = budgets
            ctx.state["task_running"], ctx.state["running_skill"] = False, None
        held = agent.state.held
        complete = getattr(r, "stop_reason", "") == "planner_complete"
        data = {
            "tasks": intent.tasks,
            "held": held,
            "cancelled": self._cancel.is_set(),
            "stop_reason": getattr(r, "stop_reason", ""),
            "decisions": getattr(r, "decisions", None),
            "missed_grasps": getattr(r, "missed_grasps", None),
            "attempts": attempts,
            "verified": None if verdict is None else verdict.ok,
        }
        if self._cancel.is_set():
            return SkillResult(
                prefix + f"stopped on request; the arm holds where it is, holding {held or 'nothing'}",
                data=data,
            )
        done = ", ".join(f"{t['op']} {t.get('object') or t.get('where')}" for t in intent.tasks)
        stats = f"{data['decisions']} decisions, {data['missed_grasps']} missed grasps" + (
            f", {attempts} attempts" if attempts > 1 else ""
        )
        if not complete:
            return SkillResult(
                prefix + f"NOT done: {done} stopped early ({data['stop_reason'] or 'unknown'}; {stats}); "
                f"holding {held or 'nothing'}",
                ok=False,
                data=data,
            )
        if verdict is None:
            return SkillResult(prefix + f"done: {done} ({stats}); holding {held or 'nothing'}", data=data)
        ok = verdict.ok is True
        return SkillResult(
            prefix + f"{'done' if ok else 'NOT confirmed'}: {done} ({stats}); {verdict.text}; "
            f"holding {held or 'nothing'}",
            ok=ok,
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
