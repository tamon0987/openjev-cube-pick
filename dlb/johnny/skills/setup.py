"""Rig set-up: what is connected and running, start what is missing, shut it down.

``rig_status`` only looks; ``rig_setup`` also starts the follower driver, openjev and the STT server. Both are
idempotent: when something needs the user's hands (a camera that gives no frame) the result carries an
instruction with ``ask_user`` and the next call checks again.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any

from dlb.johnny.skills import rig
from dlb.johnny.skills.base import Context, Skill, SkillResult

SERVICES = ("follower", "openjev", "stt")


@dataclass
class Check:
    name: str
    ok: bool | None  # None: unknown / cannot tell
    text: str
    ask: str = ""  # an instruction for the user when their hands are needed (Japanese, relayed as is)
    suggest: str = ""  # a skill to run instead of guessing


def check_cameras(cfg: dict[str, Any], ctx: Context, frames: bool = True) -> list[Check]:
    """Each configured camera: known device, present, and (``frames``) gives a frame."""
    present = {str(p) for p in rig.list_cameras()}
    out = []
    jp = {"wrist": "手首カメラ", "front": "俯瞰カメラ"}
    for cam in ("wrist", "front"):
        dev = str(cfg.get("cameras", {}).get(cam, {}).get("device") or "")
        if not dev or not dev.startswith(str(rig.V4L_BY_ID)):
            out.append(
                Check(
                    f"camera {cam}", None, f"{cam} camera not set in the config", suggest="identify_cameras"
                )
            )
        elif dev not in present:
            out.append(
                Check(
                    f"camera {cam}",
                    False,
                    f"{cam} camera {rig.camera_label(dev)} is not connected",
                    ask=f"{jp[cam]}（{rig.camera_label(dev)}）が見つかりません。USBを挿し直してください。"
                    "別のカメラに替えたのなら identify_cameras で設定し直します。",
                    suggest="identify_cameras",
                )
            )
        elif frames and rig.grab_frame(dev) is None:
            out.append(
                Check(
                    f"camera {cam}",
                    False,
                    f"{cam} camera gives no frame",
                    ask=f"{jp[cam]}から画像が来ません。USBを挿し直してください。",
                )
            )
        else:
            out.append(Check(f"camera {cam}", True, f"{cam} camera ok"))
    return out


def check_arms(ctx: Context, cfg: dict[str, Any]) -> Check:
    ports = rig.arm_ports(ctx, cfg)
    if not ports["found"]:
        return Check(
            "arms",
            False,
            "no arm found on USB",
            ask="アームのUSB（OpenRB-150）が見つかりません。フォロワーアームのUSBをつないで電源を入れてください。",
        )
    text = f"follower {ports['follower'] or '?'}, leader {ports['leader'] or '?'}"
    if not ports["follower"]:
        return Check(
            "arms",
            False,
            f"arm ports {ports['found']}: the follower's port is unknown; {text}",
            suggest="identify_arms",
        )
    if not ports["identified"]:
        return Check(
            "arms",
            True,
            f"{text} (assumed from the port numbers; identify_arms makes sure which arm is which)",
            suggest="identify_arms",
        )
    return Check("arms", True, text)


def check_services(ctx: Context, cfg: dict[str, Any]) -> list[Check]:
    follower = rig.follower_running(ctx, cfg)
    openjev = rig.openjev_ok()
    stt = rig.stt_ok()
    return [
        Check("follower", follower, "follower driver running" if follower else "follower driver not running"),
        Check(
            "openjev",
            openjev,
            "openjev healthy"
            if openjev
            else (
                "openjev starting (1-3 min after start; not healthy yet)"
                if rig.docker_state("openjev") == "running"
                else "openjev not running"
            ),
        ),
        Check("stt", stt, "STT server healthy" if stt else "STT server not running"),
        Check(
            "openai",
            rig.openai_key(ctx.env_file),
            "OPENAI_API_KEY set" if rig.openai_key(ctx.env_file) else "OPENAI_API_KEY missing (.env)",
        ),
    ]


def summarize(checks: list[Check], started: list[str]) -> str:
    ok = [c.text for c in checks if c.ok]
    bad = [c.text for c in checks if c.ok is False]
    unknown = [c.text for c in checks if c.ok is None]
    parts = []
    if started:
        parts.append("started: " + "; ".join(started))
    if ok:
        parts.append("ok: " + "; ".join(ok))
    if bad:
        parts.append("missing: " + "; ".join(bad))
    if unknown:
        parts.append("unknown: " + "; ".join(unknown))
    suggest = sorted({c.suggest for c in checks if c.suggest})
    if suggest:
        parts.append("suggested: " + ", ".join(suggest))
    return ". ".join(parts) + "."


def result(checks: list[Check], started: list[str], extra: dict[str, Any] | None = None) -> SkillResult:
    asks = [c.ask for c in checks if c.ask and c.ok is False]
    data: dict[str, Any] = {
        "checks": {c.name: c.ok for c in checks},
        "started": started,
        "suggest": sorted({c.suggest for c in checks if c.suggest}),
        **(extra or {}),
    }
    text = summarize(checks, started)
    if asks:
        data["ask_user"] = True
        text += " " + " ".join(asks) + " (done? -> call again to check)"
    return SkillResult(text, ok=all(c.ok for c in checks), data=data)


class RigStatus(Skill):
    name = "rig_status"
    description = (
        "Check the rig without changing anything: the two USB cameras (wrist, overhead) give frames, the arms' "
        "USB ports (follower, leader), the ROS 2 follower driver, the openjev decision server, the "
        "speech-to-text server, the OpenAI key. Use for 'is everything ready?', 'check the robot', "
        "'セットアップの状態', 'カメラは大丈夫?'. deep=true also sends one test request to openjev (~1 s)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "deep": {
                "type": "boolean",
                "description": "Also run a one-shot test request against openjev (dlb smoke). Default false.",
            }
        },
        "additionalProperties": False,
    }

    def run(self, ctx: Context, deep: bool = False, **_: Any) -> SkillResult:
        cfg = rig.load_config(ctx.robot_config)
        checks = [*check_cameras(cfg, ctx), check_arms(ctx, cfg), *check_services(ctx, cfg)]
        extra: dict[str, Any] = {}
        if deep and not ctx.state.get("task_running"):
            r = rig.run([sys.executable, "-m", "dlb.cli", "smoke", "--backend", "openjev"], timeout=90)
            ok = r is not None and r.returncode == 0
            checks.append(
                Check(
                    "openjev smoke",
                    ok,
                    "openjev answers test questions" if ok else "openjev smoke test failed",
                )
            )
            extra["smoke"] = (r.stdout + r.stderr)[-800:] if r is not None else "timed out"
        extra["robot_started"] = ctx.robot is not None
        extra["teleop"] = bool(ctx.state.get("teleop"))
        if ctx.state.get("cameras_changed") and ctx.robot is not None:
            checks.append(
                Check(
                    "overhead map",
                    None,
                    "the overhead camera changed: run calibrate_overhead",
                    suggest="calibrate_overhead",
                )
            )
        return result(checks, [], extra)


class RigSetup(Skill):
    name = "rig_setup"
    description = (
        "Set the rig up: check everything (as rig_status) and start what is missing: the ROS 2 follower arm "
        "driver (torque on, the arm holds its pose), the openjev decision server (Docker; healthy after 1-3 min), "
        "the speech-to-text server. Use for 'set up the robot', 'ロボットのセットアップして', 'start everything', "
        "'ドライバを起動して'. Does not move the arm. Cameras that give no frame become an instruction for the user."
    )
    parameters = {
        "type": "object",
        "properties": {
            "start": {
                "type": "array",
                "items": {"type": "string", "enum": list(SERVICES)},
                "description": "What to start when missing: follower (arm driver), openjev, stt. Default: all.",
            }
        },
        "additionalProperties": False,
    }

    def run(self, ctx: Context, start: list[str] | None = None, **_: Any) -> SkillResult:
        cfg = rig.load_config(ctx.robot_config)
        want = set(start or SERVICES)
        started: list[str] = []
        checks = [*check_cameras(cfg, ctx), check_arms(ctx, cfg)]
        if "follower" in want and not rig.follower_running(ctx, cfg):
            ok, msg = rig.start_follower(ctx, cfg)
            started.append(msg) if ok else checks.append(Check("follower start", False, msg))
        if "openjev" in want and not rig.openjev_ok():
            started.append(self._start_openjev())
        if "stt" in want and not rig.stt_ok():
            started.append(self._start_stt())
        checks.extend(check_services(ctx, cfg))
        extra: dict[str, Any] = {"robot_started": ctx.robot is not None}
        if ctx.state.get("cameras_changed"):
            checks.append(
                Check(
                    "overhead map",
                    None,
                    "the overhead camera was changed: calibrate_overhead (done anyway when the robot starts)",
                    suggest="calibrate_overhead",
                )
            )
        res = result(checks, started, extra)
        if any("openjev" in s for s in started):
            res.text += " openjev takes 1-3 min to become healthy: check again with rig_status later."
        return res

    @staticmethod
    def _start_openjev() -> str:
        state = rig.docker_state("openjev")
        if state == "running":
            return "openjev container is running, waiting for it to become healthy"
        if state is not None:
            return "docker start openjev" if rig.docker_start("openjev") else "docker start openjev FAILED"
        r = rig.run(
            ["bash", "scripts/serve_openjev.sh"], timeout=120, env={**os.environ, "HF_HUB_OFFLINE": "1"}
        )
        return "scripts/serve_openjev.sh" + ("" if r is not None and r.returncode == 0 else " FAILED")

    @staticmethod
    def _start_stt() -> str:
        state = rig.docker_state("stt")
        if state == "running":
            return "stt container is running, waiting for it"
        if state is not None:
            return "docker start stt" if rig.docker_start("stt") else "docker start stt FAILED"
        r = rig.run(["bash", "scripts/stt_server.sh"], timeout=300)
        return "scripts/stt_server.sh" + ("" if r is not None and r.returncode == 0 else " FAILED")


class RigShutdown(Skill):
    name = "rig_shutdown"
    description = (
        "Shut the rig down: stop tele-operation and the follower arm driver (torque off: hold the arm or let it "
        "rest first), release the robot, and optionally stop the openjev and STT Docker containers. Use for "
        "'shut down', '終了して', '電源を落とす前に止めて'. Not for 'stop' during a task (use stop_all)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "docker": {
                "type": "boolean",
                "description": "Also stop the openjev and stt containers (they take minutes to come back). Default false.",
            }
        },
        "additionalProperties": False,
    }

    def run(self, ctx: Context, docker: bool = False, **_: Any) -> SkillResult:
        done: dict[str, Any] = {}
        running = ctx.state.get("running_skill")
        if ctx.state.get("task_running") and running is not None:
            running.cancel(ctx)
            done["task"] = "cancelled"
        done["leader"] = rig.stop(ctx, "leader")
        ctx.state["teleop"] = False
        if ctx.robot is not None:
            try:
                ctx.robot.env.close()
            except Exception as e:  # noqa: BLE001 - shutting down anyway
                ctx.log(f"robot close: {e}")
            ctx.robot = None
            done["robot"] = "released"
        if ctx.running("follower"):
            ctx.say("フォロワーのドライバを止めます。トルクが切れるのでアームを支えてください。")
        done["follower"] = rig.stop(ctx, "follower")
        if docker:
            done["docker"] = rig.docker_stop(["openjev", "stt"])
        text = "; ".join(f"{k}: {v}" for k, v in done.items() if v != "not running") or "nothing was running"
        return SkillResult(f"shut down: {text}.", data=done)
