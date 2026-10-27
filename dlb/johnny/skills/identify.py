"""Which camera and which arm is which, found by asking the user to unplug one at a time.

Both skills are step machines: each call does one step, keeps its state in ``ctx.state``, and returns the
instruction for the user with ``data={"ask_user": True, "next": <step>}``. The LLM relays the text; when the
user says they did it, it calls the skill again with ``step=<next>``. Every step can be repeated (nothing
changed yet -> the same instruction again) and ``cancel`` forgets everything.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from dlb.johnny.skills import rig
from dlb.johnny.skills.base import Context, Skill, SkillResult

CAMERA_JP = {"wrist": "手首カメラ", "front": "俯瞰カメラ"}
ROTATIONS = (0, 90, 180, 270)


def _setup_cameras() -> Any:
    """``scripts/setup_cameras.py`` as a module (``write_config``, ``rotations``); the repo root is the cwd."""
    path = Path(__file__).resolve().parents[3] / "scripts" / "setup_cameras.py"
    spec = importlib.util.spec_from_file_location("setup_cameras", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def write_camera_config(path: str | Path, values: dict[tuple[str, str], Any]) -> None:
    _setup_cameras().write_config(Path(path), values)


def save_front_sheet(frame: Any, path: Path) -> None:
    """The overhead frame turned 0/90/180/270 degrees side by side, for the user to pick the right one."""
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), _setup_cameras().rotations(frame))


def save_frame(frame: Any, path: Path) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), frame)


def ask(text: str, next_step: str, **data: Any) -> SkillResult:
    return SkillResult(text, ok=True, data={"ask_user": True, "next": next_step, **data})


def _names(paths: list[Path]) -> list[str]:
    return [p.name for p in paths]


class IdentifyCameras(Skill):
    name = "identify_cameras"
    description = (
        "Find out which USB camera is the wrist camera and which is the overhead camera, by asking the user to "
        "unplug one at a time, then write them into the robot config and check the overhead image's rotation. "
        "Use when cameras were replaced or reattached, when rig_status says a camera is unknown, or for 'set up "
        "the cameras', 'カメラの設定', 'どっちがどのカメラか分からない'. Step machine: call with step='start' "
        "first; the result's text is the instruction for the user and data.next the step to call once they say "
        "they did it. Takes about a minute. step='cancel' aborts."
    )
    parameters = {
        "type": "object",
        "properties": {
            "step": {
                "type": "string",
                "enum": ["start", "wrist_unplugged", "front_unplugged", "plugged", "rotation", "cancel"],
                "description": "start: begin. Afterwards the 'next' of the previous result, once the user did it.",
            },
            "rotate": {
                "type": "integer",
                "enum": list(ROTATIONS),
                "description": "step='rotation' only: the clockwise turn (0/90/180/270) that puts the robot at the "
                "bottom of the overhead image, as the user answered.",
            },
        },
        "required": ["step"],
        "additionalProperties": False,
    }

    def run(self, ctx: Context, step: str = "start", rotate: int | None = None, **_: Any) -> SkillResult:
        st = ctx.state.setdefault("identify_cameras", {})
        if step == "cancel":
            written = st.get("written", False)
            st.clear()
            return SkillResult(
                "camera identification cancelled"
                + (
                    "; the camera devices were already written to the config"
                    if written
                    else "; config unchanged"
                )
            )
        if step != "start" and "seen" not in st:
            step = "start"
        handler = getattr(self, f"_{step}", None)
        if handler is None:
            return SkillResult(f"unknown step {step!r}", ok=False)
        return handler(ctx, st, rotate)

    def _start(self, ctx: Context, st: dict[str, Any], _: Any) -> SkillResult:
        devs = _names(rig.list_cameras())
        if len(devs) < 2:
            return ask(
                f"リグのカメラが {len(devs)} 台しか見つかりません（{', '.join(map(rig.camera_label, devs)) or 'なし'}）。"
                "手首カメラと俯瞰カメラの両方のUSBをつないでから、もう一度言ってください。",
                "start",
                found=devs,
            )
        st.clear()
        st.update(seen=devs, step="wrist_unplugged")
        return ask(
            f"カメラを {len(devs)} 台検出しました（{', '.join(map(rig.camera_label, devs))}）。"
            "フォロワーアームの手首カメラのUSBを抜いてください。抜いたら教えてください。",
            "wrist_unplugged",
            found=devs,
        )

    def _wrist_unplugged(self, ctx: Context, st: dict[str, Any], _: Any) -> SkillResult:
        now = _names(rig.list_cameras())
        gone = [d for d in st["seen"] if d not in now]
        if not gone:
            return ask(
                "まだどのカメラも外れていません。手首カメラのUSBを抜いてから、もう一度教えてください。",
                "wrist_unplugged",
            )
        if len(gone) > 1:
            return ask(
                f"カメラが {len(gone)} 台外れました（{', '.join(map(rig.camera_label, gone))}）。全部挿し直してから、"
                "手首カメラだけを抜いて教えてください。",
                "wrist_unplugged",
            )
        st["wrist"] = gone[0]
        rest = [d for d in st["seen"] if d != gone[0]]
        if len(rest) == 1:
            st["front"], st["step"] = rest[0], "plugged"
            return ask(
                f"手首カメラは {rig.camera_label(gone[0])} です。俯瞰カメラは残りの {rig.camera_label(rest[0])} にします。"
                "手首カメラのUSBを挿し直してください。挿したら教えてください。",
                "plugged",
                wrist=gone[0],
                front=rest[0],
            )
        st["step"] = "front_unplugged"
        return ask(
            f"手首カメラは {rig.camera_label(gone[0])} です。手首カメラを挿し直して、次に俯瞰（上の）カメラのUSBを抜いてください。"
            "抜いたら教えてください。",
            "front_unplugged",
            wrist=gone[0],
        )

    def _front_unplugged(self, ctx: Context, st: dict[str, Any], _: Any) -> SkillResult:
        if "wrist" not in st:
            return self._start(ctx, st, None)
        now = _names(rig.list_cameras())
        cands = [d for d in st["seen"] if d != st["wrist"]]
        gone = [d for d in cands if d not in now]
        if not gone:
            return ask(
                "俯瞰カメラがまだ外れていません。俯瞰カメラのUSBを抜いてから、もう一度教えてください。",
                "front_unplugged",
            )
        if len(gone) > 1:
            return ask(
                f"カメラが {len(gone)} 台外れています（{', '.join(map(rig.camera_label, gone))}）。俯瞰カメラ以外を挿し直してから教えてください。",
                "front_unplugged",
            )
        st["front"], st["step"] = gone[0], "plugged"
        return ask(
            f"俯瞰カメラは {rig.camera_label(gone[0])} です。USBを挿し直してください。挿したら教えてください。",
            "plugged",
            wrist=st["wrist"],
            front=gone[0],
        )

    def _plugged(self, ctx: Context, st: dict[str, Any], _: Any) -> SkillResult:
        if "wrist" not in st or "front" not in st:
            return self._start(ctx, st, None)
        now = _names(rig.list_cameras())
        for cam in ("wrist", "front"):
            if st[cam] not in now:
                return ask(
                    f"{CAMERA_JP[cam]}（{rig.camera_label(st[cam])}）がまだ見つかりません。USBを挿し直してから教えてください。",
                    "plugged",
                )
        frames = {cam: rig.grab_frame(rig.V4L_BY_ID / st[cam]) for cam in ("wrist", "front")}
        for cam, f in frames.items():
            if f is None:
                return ask(
                    f"{CAMERA_JP[cam]}から画像が来ません。USBを挿し直してから教えてください。", "plugged"
                )
        save_frame(frames["wrist"], rig.OUT_DIR / "cameras_wrist.jpg")
        sheet = rig.OUT_DIR / "cameras_front.jpg"
        save_front_sheet(frames["front"], sheet)
        wrist, front = str(rig.V4L_BY_ID / st["wrist"]), str(rig.V4L_BY_ID / st["front"])
        write_camera_config(ctx.robot_config, {("wrist", "device"): wrist, ("front", "device"): front})
        st["written"], st["step"] = True, "rotation"
        ctx.state["cameras_changed"] = True
        current = rig.load_config(ctx.robot_config).get("cameras", {}).get("front", {}).get("rotate", 0)
        return ask(
            f"カメラを設定に書きました（手首: {rig.camera_label(wrist)}、俯瞰: {rig.camera_label(front)}）。"
            f"俯瞰画像を 0/90/180/270 度回したものを {sheet} に並べて保存しました。ロボットが画像の下側に来るのは"
            f"どの角度ですか？（今の設定は {current} 度。そのままでよければ {current} と答えてください）",
            "rotation",
            wrist=wrist,
            front=front,
            sheet=str(sheet),
            current_rotate=current,
        )

    def _rotation(self, ctx: Context, st: dict[str, Any], rotate: int | None) -> SkillResult:
        if rotate not in ROTATIONS:
            return ask(
                "角度を 0、90、180、270 のどれかで教えてください（ロボットが画像の下側に来る回転）。",
                "rotation",
            )
        write_camera_config(ctx.robot_config, {("front", "rotate"): int(rotate)})
        wrist, front = str(rig.V4L_BY_ID / st["wrist"]), str(rig.V4L_BY_ID / st["front"])
        rig.save_rig(cameras={"wrist": wrist, "front": front, "rotate": int(rotate)})
        st.clear()
        ctx.state["cameras_changed"] = True
        return SkillResult(
            f"cameras set: wrist = {rig.camera_label(wrist)}, overhead = {rig.camera_label(front)}, overhead "
            f"image turned {rotate} degrees. The overhead map must be measured again: calibrate_overhead (the "
            "robot does it at start-up anyway).",
            data={"wrist": wrist, "front": front, "rotate": int(rotate), "suggest": ["calibrate_overhead"]},
        )


class IdentifyArms(Skill):
    name = "identify_arms"
    description = (
        "Find out which USB serial port is the follower arm (the robot) and which is the leader arm (the one a "
        "person moves for tele-operation), by asking the user to unplug the leader's USB for a moment. The "
        "follower driver and tele-operation then use the right ports. Use when rig_status says the arms are "
        "assumed/unknown, after replugging, or for 'アームの設定', 'どっちがリーダー?'. Step machine like "
        "identify_cameras: step='start' first, then the 'next' of the previous result; 'cancel' aborts."
    )
    parameters = {
        "type": "object",
        "properties": {
            "step": {
                "type": "string",
                "enum": ["start", "leader_unplugged", "plugged", "cancel"],
                "description": "start: begin. Afterwards the 'next' of the previous result, once the user did it.",
            }
        },
        "required": ["step"],
        "additionalProperties": False,
    }

    def run(self, ctx: Context, step: str = "start", **_: Any) -> SkillResult:
        st = ctx.state.setdefault("identify_arms", {})
        if step == "cancel":
            st.clear()
            return SkillResult("arm identification cancelled; nothing changed")
        if step != "start" and "seen" not in st:
            step = "start"
        handler = getattr(self, f"_{step}", None)
        if handler is None:
            return SkillResult(f"unknown step {step!r}", ok=False)
        return handler(ctx, st)

    def _start(self, ctx: Context, st: dict[str, Any]) -> SkillResult:
        ports = rig.list_arm_ports()
        devs = {p.name: rig.resolve(p) for p in ports}
        if not devs:
            return ask(
                "アームのUSB（OpenRB-150）が1台も見つかりません。両方のアームのUSBをつないで電源を入れてから教えてください。",
                "start",
            )
        if len(devs) == 1:
            ((name, dev),) = devs.items()
            arms = {"follower": {"by_id": name, "dev": dev}}
            ctx.state["arms"] = arms
            rig.save_rig(arms=arms)
            st.clear()
            return SkillResult(
                f"only one arm is connected ({name} -> {dev}); taken as the follower. Plug the leader arm in too "
                "and run identify_arms again to tell them apart.",
                data={"arms": arms},
            )
        st.clear()
        st.update(seen=devs, step="leader_unplugged")
        return ask(
            f"アームを {len(devs)} 台検出しました（{', '.join(devs.values())}）。"
            "リーダーアーム（手で動かす方）のUSBを抜いてください。抜いたら教えてください。",
            "leader_unplugged",
            found=devs,
        )

    def _leader_unplugged(self, ctx: Context, st: dict[str, Any]) -> SkillResult:
        now = _names(rig.list_arm_ports())
        seen: dict[str, str] = st["seen"]
        gone = [n for n in seen if n not in now]
        if not gone:
            return ask(
                "まだどのアームも外れていません。リーダーアームのUSBを抜いてから、もう一度教えてください。",
                "leader_unplugged",
            )
        if len(gone) > 1:
            return ask(
                f"アームが {len(gone)} 台外れています。リーダーアーム以外を挿し直してから教えてください。",
                "leader_unplugged",
            )
        rest = [n for n in seen if n != gone[0]]
        if len(rest) != 1:
            return ask(
                f"リーダー以外に {len(rest)} 台残っています。フォロワーアームだけをつないだ状態にしてから教えてください。",
                "leader_unplugged",
            )
        arms = {
            "leader": {"by_id": gone[0], "dev": seen[gone[0]]},
            "follower": {"by_id": rest[0], "dev": seen[rest[0]]},
        }
        st.update(arms=arms, step="plugged")
        ctx.state["arms"] = arms
        rig.save_rig(arms=arms)
        return ask(
            f"リーダーアームは {seen[gone[0]]}、フォロワーアームは {seen[rest[0]]} です。"
            "リーダーアームのUSBを挿し直してください。挿したら教えてください。",
            "plugged",
            arms=arms,
        )

    def _plugged(self, ctx: Context, st: dict[str, Any]) -> SkillResult:
        arms = st.get("arms") or {}
        if not arms:
            return self._start(ctx, st)
        devs = {p.name: rig.resolve(p) for p in rig.list_arm_ports()}
        missing = [role for role in ("follower", "leader") if arms[role]["by_id"] not in devs]
        if missing:
            return ask(
                f"{'リーダー' if 'leader' in missing else 'フォロワー'}アームのUSBがまだ見つかりません。挿し直してから教えてください。",
                "plugged",
            )
        for role in ("follower", "leader"):
            arms[role]["dev"] = devs[arms[role]["by_id"]]  # the number can change after a replug
        ctx.state["arms"] = arms
        rig.save_rig(arms=arms)
        st.clear()
        note = ""
        if ctx.running("follower") and arms["follower"]["dev"] != ctx.state.get("follower_port"):
            note = " The follower driver that runs now was started on another port: rig_shutdown, then rig_setup."
        return SkillResult(
            f"arms identified: follower on {arms['follower']['dev']}, leader on {arms['leader']['dev']} "
            "(remembered by their USB ids in results/johnny/rig.json)." + note,
            data={"arms": arms},
        )
