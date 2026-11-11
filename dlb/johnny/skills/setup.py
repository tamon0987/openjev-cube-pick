"""Rig set-up: what is connected and running, start what is missing, shut it down.

``rig_status`` only looks; ``rig_setup`` also starts the follower driver, openjev and the STT server. Both are
idempotent: when something needs the user's hands (a camera that gives no frame) the result carries an
instruction with ``ask_user`` and the next call checks again.
"""

from __future__ import annotations

import os
import sys
import time
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
    # the started robot holds the cameras open (a second open gives nothing): judge by its latest frames
    env = getattr(ctx.robot, "env", None)
    held = getattr(env, "frame_t", None)
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
        elif frames and held is not None:
            if time.monotonic() - held.get(cam, float("-inf")) < 2.0:
                out.append(Check(f"camera {cam}", True, f"{cam} camera ok (streaming to the robot)"))
            else:
                out.append(
                    Check(
                        f"camera {cam}",
                        False,
                        f"{cam} camera stopped streaming to the robot (unplugged while in use?)",
                        ask=f"{jp[cam]}の映像が止まっています。「停止して」でロボットを止めてから、"
                        "もう一度セットアップしてください。",
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


def robot_check(ctx: Context) -> Check:
    """Started (begin pose, overhead calibrated) or not: pick_place can only start right away when it is."""
    agent = ctx.robot
    if agent is None:
        return Check(
            "robot", False, "robot not started (rig_setup starts it: begin pose + overhead calibration)"
        )
    cal = getattr(getattr(agent, "guide", None), "calibration", None) or {}
    if cal:
        return Check(
            "robot",
            True,
            f"robot started, overhead calibrated ({cal.get('px_per_cm')} px/cm, {cal.get('inliers')} of "
            f"{cal.get('pairs')} poses)",
        )
    return Check("robot", True, "robot started")


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
        checks = [*check_cameras(cfg, ctx), check_arms(ctx, cfg), *check_services(ctx, cfg), robot_check(ctx)]
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
        "'ドライバを起動して'. Then, by default, it also starts the robot: the arm moves to the begin pose and around it "
        "for ~25 s to calibrate the overhead camera (so pick_place can start right away); robot=false skips "
        "that. Before starting an arm, verify its saved USB identity. If the roles are unknown, guide the user "
        "through unplugging and reconnecting the FOLLOWER USB, then its WRIST CAMERA USB. Confirm the overhead camera rotation too. Relay the question and WAIT for the user's "
        "answer, then call rig_setup again with step=data.next. Do not switch to another skill to continue. "
        "start/robot options are remembered during this dialogue. step=cancel aborts. "
        "Cameras that give no frame become an instruction for the user."
    )
    parameters = {
        "type": "object",
        "properties": {
            "step": {
                "type": "string",
                "enum": [
                    "start",
                    "follower_unplugged",
                    "follower_plugged",
                    "camera_start",
                    "camera_wrist_unplugged",
                    "camera_front_unplugged",
                    "camera_plugged",
                    "camera_rotation",
                    "cancel",
                ],
                "description": "Default start. Continue with the previous result's next step only after the user answers.",
            },
            "rotate": {
                "type": "integer",
                "enum": [0, 90, 180, 270],
                "description": "camera_rotation only: the angle confirmed by the user.",
            },
            "start": {
                "type": "array",
                "items": {"type": "string", "enum": list(SERVICES)},
                "description": "What to start when missing: follower (arm driver), openjev, stt. Default: all.",
            },
            "robot": {
                "type": "boolean",
                "description": "Also start the robot (begin pose + overhead calibration, the arm moves) once the "
                "services are up. Default true.",
            },
        },
        "additionalProperties": False,
    }

    def cancel(self, ctx: Context) -> None:
        st = ctx.state.pop("rig_setup_dialogue", {})
        if st.get("camera_started"):
            from dlb.johnny.skills.identify import IdentifyCameras

            IdentifyCameras().run(ctx, step="cancel")

    def _confirm_arms(self, ctx: Context, st: dict[str, Any], step: str) -> SkillResult | None:
        from dlb.johnny.skills.identify import ask

        devs = {p.name: rig.resolve(p) for p in rig.list_arm_ports()}
        saved = ctx.state.get("arms") or rig.load_rig().get("arms") or {}
        ids = {role: (saved.get(role) or {}).get("by_id") for role in ("follower", "leader")}
        if "seen" not in st and len(set(ids.values())) == 2 and all(i in devs for i in ids.values()):
            arms = {role: {"by_id": i, "dev": devs[i]} for role, i in ids.items()}
        else:
            if ctx.running("leader") or ctx.running("relay") or ctx.state.get("teleop"):
                return SkillResult(
                    "テレオペを終了してから、もう一度セットアップしてください。アームの識別はまだ行っていません。",
                    ok=False,
                )
            if ctx.robot is not None or rig.follower_running(ctx, rig.load_config(ctx.robot_config)):
                return SkillResult(
                    "アームのドライバが起動中なので、USBの取り外し確認を始められません。"
                    "ドライバを停止してから、もう一度セットアップしてください。",
                    ok=False,
                )
            expected = st.get("next", "start")
            if step != expected:
                return ask("前の案内に従って接続を確認し、できたら教えてください。", expected)
            if expected == "start":
                if len(devs) != 2:
                    return ask(
                        f"アームを {len(devs)} 台検出しました。リーダーとフォロワーの2台をUSB接続して、できたら教えてください。"
                        "ポート番号だけでは役割を決めません。",
                        "start",
                    )
                st.update(seen=devs, next="follower_unplugged")
                return ask(
                    "ポートの役割を確認します。フォロワーアーム（実際に作業する方）のUSBだけを抜いてください。"
                    "リーダーは接続したままにして、抜けたら教えてください。",
                    "follower_unplugged",
                )
            if expected == "follower_unplugged":
                seen = st["seen"]
                gone = set(seen) - set(devs)
                if len(gone) != 1 or len(devs) != 1 or not set(devs) < set(seen):
                    return ask(
                        "フォロワー1台だけが外れた状態を確認できません。リーダーだけを接続した状態にして、教えてください。",
                        "follower_unplugged",
                    )
                st.update(
                    ids={"follower": next(iter(gone)), "leader": next(iter(devs))}, next="follower_plugged"
                )
                return ask(
                    "フォロワーのポートが確認できました。フォロワーのUSBを挿し直して、できたら教えてください。",
                    "follower_plugged",
                )
            ids = st["ids"]
            if not all(i in devs for i in ids.values()):
                return ask(
                    "確認した2台のUSB接続がまだ揃っていません。両方を接続してから教えてください。",
                    "follower_plugged",
                )
            arms = {role: {"by_id": i, "dev": devs[i]} for role, i in ids.items()}
            # Commit only after reconnection; cancelling a partial dialogue changes no mapping.
            rig.save_rig(arms=arms)
        ctx.state["arms"] = arms
        follower, leader = arms["follower"]["dev"], arms["leader"]["dev"]
        ctx.say(f"フォロワーは {follower}、リーダーは {leader} と確認できました。")
        if ctx.running("follower") and ctx.state.get("follower_port") != follower:
            return SkillResult(
                "起動中のフォロワードライバのポートが、確認したポートと一致しません。"
                "ドライバを停止してから、もう一度セットアップしてください。",
                ok=False,
            )
        st.pop("seen", None)
        st.pop("next", None)
        return None

    def _confirm_cameras(
        self, ctx: Context, st: dict[str, Any], step: str, rotate: int | None
    ) -> SkillResult | None:
        from dlb.johnny.skills.identify import IdentifyCameras, ask

        cfg = rig.load_config(ctx.robot_config)
        saved = rig.load_rig().get("cameras") or {}
        present = {str(p) for p in rig.list_cameras()}
        known = (
            saved.get("wrist") != saved.get("front")
            and all(
                saved.get(c) in present and saved.get(c) == cfg.get("cameras", {}).get(c, {}).get("device")
                for c in ("wrist", "front")
            )
            and saved.get("rotate") == cfg.get("cameras", {}).get("front", {}).get("rotate", 0)
        )
        if known and not st.get("camera_started"):
            return None
        if ctx.robot is not None:
            return SkillResult(
                "カメラを使用中です。ロボットを終了してから、もう一度セットアップしてください。", ok=False
            )
        expected = st.get("camera_next", "camera_start")
        if st.get("camera_started") and step != expected:
            return ask("前のカメラ確認の案内に従い、できたら教えてください。", expected)
        st["camera_started"] = True
        res = IdentifyCameras().run(ctx, step=expected.removeprefix("camera_"), rotate=rotate)
        if res.data and res.data.get("ask_user"):
            st["camera_next"] = "camera_" + res.data["next"]
            res.data["next"] = st["camera_next"]
            return res
        if not res.ok:
            return res
        st.pop("camera_started", None)
        st.pop("camera_next", None)
        ctx.say("アームとカメラの接続を確認できました。セットアップを続けます。")
        return None

    def run(
        self,
        ctx: Context,
        start: list[str] | None = None,
        robot: bool | None = None,
        step: str = "start",
        rotate: int | None = None,
        **_: Any,
    ) -> SkillResult:
        if step == "cancel":
            self.cancel(ctx)
            return SkillResult("セットアップの確認を中止しました。確認済みで保存した機器設定は保持します。")
        if step not in self.parameters["properties"]["step"]["enum"]:
            return SkillResult(f"unknown setup step: {step}", ok=False)
        cfg = rig.load_config(ctx.robot_config)
        st = ctx.state.setdefault("rig_setup_dialogue", {})
        st.setdefault("start", list(start or SERVICES))
        st.setdefault("robot", True if robot is None else robot)
        want, robot = set(st["start"]), st["robot"]
        if "follower" in want or robot:
            # Camera steps start only after the arms have been identified. Recheck the saved IDs on every turn.
            confirmation = self._confirm_arms(ctx, st, "start" if step.startswith("camera_") else step)
            if confirmation is not None:
                return confirmation
            if robot:
                confirmation = self._confirm_cameras(ctx, st, step, rotate)
                if confirmation is not None:
                    return confirmation
        self.cancel(ctx)
        cfg = rig.load_config(ctx.robot_config)
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
        if robot:
            checks.append(self._start_robot(ctx, checks, started))
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
    def _start_robot(ctx: Context, checks: list[Check], started: list[str]) -> Check:
        """Start the robot (begin pose + ~22 s overhead calibration) once the services it needs are up."""
        if ctx.robot is not None:
            return robot_check(ctx)
        ready = {c.name: c.ok for c in checks}
        missing = [n for n in ("follower", "openjev", "openai") if not ready.get(n)]
        if missing:
            return Check(
                "robot",
                False,
                f"robot not started yet ({', '.join(missing)} not ready); the next rig_setup, calibrate_overhead "
                "or pick_place starts it",
            )
        from dlb.johnny.skills.calibrate import describe
        from dlb.johnny.skills.pick_place import ensure_robot

        agent, err = ensure_robot(ctx)
        if agent is None:
            return Check("robot", False, f"robot start failed: {err}")
        # A freshly started robot has calibrated the camera selected by this dialogue.
        ctx.state.pop("cameras_changed", None)
        env = getattr(agent, "env", None)
        waiting = env is not None and hasattr(env, "at_pose") and env.at_pose("wait", tol=0.1)
        started.append(
            "the robot: "
            + ("waiting pose (pulled back, out of the overhead view)" if waiting else "begin pose")
            + ", overhead calibrated ("
            + describe(getattr(agent.guide, "calibration", None))
            + ")"
        )
        return robot_check(ctx)

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
        "Shut the rig down (same as safe_stop: gripper open, rest pose, release the robot, follower driver "
        "torque off with the arm resting), and optionally stop the openjev and STT Docker containers. "
        "Use for 'shut down the rig', '電源を落とす前に止めて', 'リグを落として'. Not for 'stop' during a task "
        "(use stop_all)."
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
        from dlb.johnny.skills.control import safe_stop_sequence

        steps, ok = safe_stop_sequence(ctx, docker)
        text = "; ".join(steps) or "nothing was running"
        return SkillResult(
            ("shut down: " if ok else "shutdown INCOMPLETE: ") + text + ".", ok=ok, data={"steps": steps}
        )
