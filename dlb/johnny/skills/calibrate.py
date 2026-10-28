"""Overhead-camera self-calibration on request (``OverheadGuide.calibrate``: 9 poses around the begin pose)."""

from __future__ import annotations

from typing import Any

from dlb.johnny.skills.base import Context, Skill, SkillResult
from dlb.johnny.skills.pick_place import ensure_robot, leave_teleop


def describe(rec: dict[str, Any] | None) -> str:
    if not rec:
        return "no calibration record"
    poses = rec.get("poses") or []
    bad = []
    for i, p in enumerate(poses):
        if p.get("inlier"):
            continue
        xy = p.get("tcp") or [0, 0]
        where = f"pose {i + 1} at ({xy[0] * 100:.1f}, {xy[1] * 100:.1f}) cm"
        if p.get("mark") is None:
            bad.append(f"{where}: gripper not found in the image")
        else:
            res = p.get("residual_px")
            bad.append(f"{where}: mark rejected" + ("" if res is None else f" ({res:.0f} px off the fit)"))
    text = (
        f"{rec.get('px_per_cm')} px/cm, rotation {rec.get('rotation_deg')} deg, {rec.get('inliers')} of "
        f"{rec.get('pairs')} poses agree"
        + (f" (residual {rec['rms_px']} px)" if rec.get("rms_px") is not None else "")
        + f", {rec.get('seconds')} s"
    )
    if bad:
        text += "; rejected: " + "; ".join(bad)
    return text


class CalibrateOverhead(Skill):
    name = "calibrate_overhead"
    description = (
        "Measure the overhead camera's scale again with the arm (it visits 9 poses around the begin pose, ~25 s; "
        "the gripper is marked in each image). Needed after the overhead camera was moved, replaced or turned "
        "(identify_cameras). Use for 'calibrate the camera', 'カメラのキャリブレーション', '俯瞰カメラを校正して'. "
        "Starts the robot when it is not started (that already calibrates). Needs an empty gripper, no "
        "tele-operation, no running task."
    )
    parameters = {"type": "object", "properties": {}, "additionalProperties": False}

    def run(self, ctx: Context, **_: Any) -> SkillResult:
        if ctx.state.get("task_running"):
            return SkillResult("a robot task is running; say stop first", ok=False)
        prefix = leave_teleop(ctx)
        if ctx.robot is None:
            agent, err = ensure_robot(ctx)
            if agent is None:
                return SkillResult(prefix + err, ok=False)
            ctx.state.pop("cameras_changed", None)
            rec = agent.guide.calibration
            return SkillResult(
                prefix + "the robot started; its start-up calibration: " + describe(rec),
                data={"calibration": rec},
            )
        agent = ctx.robot
        env, guide = agent.env, agent.guide
        if env.held:
            return SkillResult(
                prefix + "the gripper holds something: place it first (or robot_release), then calibrate",
                ok=False,
            )
        ctx.say("俯瞰カメラを校正します。アームがはじめの姿勢の周りを動き回ります。手を離してください。")
        try:
            if not env.at_pose("begin"):
                env.go_pose("begin")
            guide.reset()
            from dlb.harness.marking import TableMap

            guide.map = TableMap()  # the old scale and pairs belong to the old camera placement
            rec = guide.calibrate(env)
        except RuntimeError as e:
            rec = guide.calibration
            return SkillResult(
                prefix
                + f"calibration failed: {e}. Details: {describe(rec)}. Check results/johnny/cameras_front.jpg "
                "(rig_status / identify_cameras): the robot must be at the bottom of the overhead image and the "
                "gripper visible around the begin pose.",
                ok=False,
                data={"calibration": rec},
            )
        except Exception as e:  # noqa: BLE001 - a motion error leaves the arm where it is
            return SkillResult(prefix + f"calibration aborted: {e}; the arm holds where it is", ok=False)
        ctx.state.pop("cameras_changed", None)
        ctx.state.pop("arm_moved_externally", None)
        agent.park()
        agent.idle_mark()
        return SkillResult(prefix + "overhead calibration done: " + describe(rec), data={"calibration": rec})
