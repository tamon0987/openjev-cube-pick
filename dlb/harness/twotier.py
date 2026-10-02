"""Two-tier control: a slow planner sets subtasks, a fast decision layer picks relative moves.

Inspired by Show-Harness (https://showlab.github.io/Show-Harness/): the fast layer chooses from a
parameter-free command vocabulary (``MV_FWD`` ... ``GRASP``) and the interpreter supplies metric
step sizes. Two things differ from the primitive harness in ``loop.py``:

* The fast layer never sees object coordinates. It gets camera images, the robot's own state
  (gripper height, jaw open/closed) and the planner's subtask text. The only geometric aid is a
  green cross drawn on the wrist image where the point directly below the gripper appears; it is
  computed from the robot's own kinematics, not from the objects.
* Long-horizon reasoning lives in a planner (scripted with simulator truth, or an OpenAI VLM) that
  is called only at subtask boundaries and when the fast layer escalates.

Directions follow the wrist image: ``MV_FWD`` moves toward the top of the wrist image,
``MV_RIGHT`` toward its right. ``MV_UP``/``MV_DOWN`` are world vertical.
"""

from __future__ import annotations

import inspect
import itertools
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from dlb.backends.base import DecisionBackend
from dlb.contract import Choice, DecisionRequest, Noul, image_to_data_url
from dlb.sim.env import OPEN_MIN, TASK_TEXT, Z_GRASP, Z_HOVER, Z_TRAVEL, PickPlaceEnv

# Object names used in every prompt. The texts below say "red cube" / "blue bin" (the simulator's objects);
# `_n` swaps in the names set with `set_object_names` (on the real table: the object named in the instruction or
# found in the overhead image, e.g. "carrot plush toy", "black bin"). Nothing depends on what the names say.
OBJ = {"cube": "red cube", "bin": "blue bin"}


def set_object_names(cube: str, bin_: str) -> None:
    OBJ.update(cube=cube, bin=bin_)


def _n(text: str) -> str:
    return text.replace("red cube", OBJ["cube"]).replace("blue bin", OBJ["bin"])


MOVES = ["MV_FWD", "MV_BACK", "MV_LEFT", "MV_RIGHT", "MV_UP", "MV_DOWN"]
COMMANDS = [*MOVES, "GRASP", "RELEASE", "SUBTASK_DONE", "ESCALATE"]
SUBTASKS = {
    "reach_cube": "Move horizontally until the gripper is roughly above the red cube (coarse, 2 cm steps).",
    "align_cube": "Fine-align at the current height until the gripper is directly above the centre of the red cube (1 cm steps).",
    "descend_grasp": "Lower the open gripper around the red cube and close it on the cube.",
    "transport_to_bin": "Carry the held cube up to travel height and horizontally until it is above the blue bin.",
    "release": "Open the gripper so the cube drops into the bin.",
    "recover": "Open the gripper and raise it to a safe height, e.g. after a missed grasp or a dropped cube.",
    # executed by the overhead guide (dlb/harness/marking.py), not by the fast layer
    "goto_cube": "Move above the cube marked in the overhead image (coarse, by marks).",
    "goto_bin": "Carry the held cube above the bin marked in the overhead image (coarse, by marks).",
    "place_at": "Put the held object down at a place: into the bin, here, or on top of another object.",
}
GUIDED = {"goto_cube": "cube", "goto_bin": "bin"}
# place_at (task lists): carry the held object to a place and let go; executed by the harness
PLACE_Z_MARGIN = 0.005  # above the surface the held object is set down on
# Stacking ("on:<object>") sets the held object down this far above the grasp height: the one remaining size
# assumption (objects ~3 cm tall, as the cubes). Without it the heights of both objects would be needed.
CUBE_EDGE = 0.03
DEFAULT_COMMANDS = {
    "reach_cube": ["MV_FWD", "MV_BACK", "MV_LEFT", "MV_RIGHT", "SUBTASK_DONE", "ESCALATE"],
    "align_cube": ["MV_FWD", "MV_BACK", "MV_LEFT", "MV_RIGHT", "SUBTASK_DONE", "ESCALATE"],
    "descend_grasp": [*MOVES, "GRASP", "SUBTASK_DONE", "ESCALATE"],
    "transport_to_bin": [*MOVES, "SUBTASK_DONE", "ESCALATE"],
    "release": ["MV_DOWN", "RELEASE", "SUBTASK_DONE", "ESCALATE"],
    "recover": ["MV_UP", "RELEASE", "SUBTASK_DONE", "ESCALATE"],
    "goto_cube": [],
    "goto_bin": [],
    "place_at": [],
}
CUBE_Z, BIN_RIM_Z = 0.015, 0.04  # heights used for the "below the gripper" cross
# Grasps succeed with the tcp up to ~1.0-1.4 cm off along the wrist image's vertical axis and
# ~1.5-2 cm along its horizontal axis (measured over seeds 2000-2002); keep a margin.
GRASP_TOL_FWD, GRASP_TOL_RIGHT = 0.009, 0.015
# reach only needs the cube roughly below the gripper; descend_grasp refines while lowering, when the
# cube looks larger in the wrist image and small offsets are easier to see
REACH_TOL = 0.02
# the cube lands in the bin when it is released within ~3 cm of the bin centre on each axis
BIN_TOL = 0.022
# Carry high: the held cube hangs at the bottom of the wrist image and hides the part of the bin behind the
# gripper; from 14 cm the bin sits higher in the image (top-bottom 0.51 -> 0.65 in the probe; 16 cm is out of
# reach for some bin positions).
TRANSPORT_Z = 0.14
# Below this height the finger bar hides the lower half of the cube, so top-bottom judgements are unreliable
# (probe: most top-bottom errors at 3.5-5 cm); forward/backward alignment is finished in align_cube.
NO_FWD_BELOW_Z = 0.06
# Close the gripper once the tcp is within this of grasp height (the move clamps at Z_GRASP, never lower). On the
# real arm the tcp sits 2 cm behind the fingertips on a tool pitched 52 deg down, so the fingertips are 1.6 cm below
# the tcp: 1.2 cm above the table at Z_GRASP, 2.0 cm at the edge of this band, i.e. at least the top centimetre of
# the 3 cm cube is between the pads. Real grasps held from 3.35 cm (real_grasp_top) and 3.6 cm (real_overhead1);
# the precise descent came to rest 1-6 mm above its target (residual gravity sag), so it normally grasps next.
GRASP_Z_TOL = 0.008
# drop the cube from lower than the carrying height: less drift while falling, and the arm is not at full reach
RELEASE_Z = 0.10


def target_is_bin(sub: Subtask) -> bool:
    return sub.name in ("transport_to_bin", "release")


_SUBTASK_IDS = itertools.count()


@dataclass
class Subtask:
    name: str
    goal: str
    done_when: str
    allowed: list[str]
    step_cm: float = 2.0
    max_steps: int = 20
    hint: str = ""
    search: str = ""  # move to make when the target is not visible in the wrist image ("" = escalate)
    target: str = ""  # object to pick (task lists); "" = the configured cube
    place: str = ""  # place_at: "bin" | "here" | "on:<object>"
    task_index: int = -1  # position of the task this subtask belongs to (TaskPlanner)
    uid: int = field(default_factory=lambda: next(_SUBTASK_IDS))  # stable identity (id() gets reused)

    @property
    def step_m(self) -> float:
        return float(np.clip(self.step_cm, 0.5, 4.0)) / 100


@dataclass
class Plan:
    task_complete: bool
    subtasks: list[Subtask]
    pre_moves: list[str] = field(default_factory=list)  # executed by the harness before the first subtask
    pre_move_step_cm: float = 1.0
    scene: str = ""
    diagnosis: str = ""
    latency_s: float = 0.0
    usage: dict[str, int] = field(default_factory=dict)

    @property
    def subtask(self) -> Subtask | None:
        return self.subtasks[0] if self.subtasks else None


# --------------------------------------------------------------------------- #
# observation helpers
# --------------------------------------------------------------------------- #
def cross_height(sub: Subtask | None) -> float:
    return BIN_RIM_Z if sub is not None and sub.name in ("transport_to_bin", "release") else CUBE_Z


def wrist_with_cross(env: PickPlaceEnv, img: np.ndarray, height: float) -> np.ndarray:
    """Draw a green cross where the point directly below the tcp at ``height`` appears."""
    px = env.wrist_pixel(np.array([*env.tcp_pos[:2], height]), size=img.shape[0])
    if px is None:
        return img
    out = img.copy()
    x, y = int(round(px[0])), int(round(px[1]))
    h, w = out.shape[:2]
    arm, th = max(6, w // 32), max(1, w // 160)
    col = np.array([30, 200, 60], np.uint8)
    out[max(0, y - th) : min(h, y + th + 1), max(0, x - arm) : min(w, x + arm + 1)] = col
    out[max(0, y - arm) : min(h, y + arm + 1), max(0, x - th) : min(w, x + th + 1)] = col
    return out


def zoom_on_cross(env: PickPlaceEnv, img: np.ndarray, height: float, half: int | None = None) -> np.ndarray:
    """Crop a square around the below-gripper point and upscale it to the input size.

    The crop covers about 10.5 cm of table at the reach height, like the 64 px crop in simulation; a real
    camera with a different field of view sets ``env.zoom_half`` from its calibration.
    """
    from PIL import Image

    half = int(half or getattr(env, "zoom_half", 64) * img.shape[0] / 320)

    px = env.wrist_pixel(np.array([*env.tcp_pos[:2], height]), size=img.shape[0])
    h, w = img.shape[:2]
    x, y = (w // 2, h // 2) if px is None else (int(px[0]), int(px[1]))
    x0, y0 = int(np.clip(x - half, 0, w - 2 * half)), int(np.clip(y - half, 0, h - 2 * half))
    tile = Image.fromarray(img[y0 : y0 + 2 * half, x0 : x0 + 2 * half]).resize((w, h), Image.LANCZOS)
    return np.asarray(tile)


def front_forward_view(env: PickPlaceEnv, front: np.ndarray, height: float, half: int = 48) -> np.ndarray:
    """Front-camera crop around the below-gripper point, rotated so the gripper's forward axis points right.

    Uses only the fixed front camera's calibration and the robot's own kinematics. Lets the top-bottom
    (forward/backward) question be asked as left-right from a viewpoint where the held cube does not hide
    the bin.
    """
    from PIL import Image, ImageDraw

    size = front.shape[0]
    fwd, _ = env.wrist_axes()
    p0 = np.array([*env.tcp_pos[:2], height])
    a = env.camera_pixel("front", p0, size)
    b = env.camera_pixel("front", p0 + np.array([*(fwd * 0.03), 0.0]), size)
    if a is None or b is None:
        return front
    im = Image.fromarray(front)
    d = ImageDraw.Draw(im)
    x, y = a
    r = max(4, size // 60)
    d.line([(x - r, y), (x + r, y)], fill=(30, 200, 60), width=2)
    d.line([(x, y - r), (x, y + r)], fill=(30, 200, 60), width=2)
    ang = float(np.degrees(np.arctan2(b[1] - a[1], b[0] - a[0])))
    im = im.rotate(ang, center=(x, y), resample=Image.BICUBIC, fillcolor=(255, 255, 255))
    x0, y0 = int(np.clip(x - half, 0, size - 2 * half)), int(np.clip(y - half, 0, size - 2 * half))
    return np.asarray(im.crop((x0, y0, x0 + 2 * half, y0 + 2 * half)).resize((size, size), Image.LANCZOS))


def top_view_crop(env: PickPlaceEnv, top: np.ndarray, world_dir: np.ndarray, half: int = 70) -> np.ndarray:
    """Overhead-camera crop centred on the point below the gripper, rotated so ``world_dir`` points RIGHT.

    With a calibrated overhead camera, "is the target left or right of the cross" in this crop answers where the
    target lies along ``world_dir`` (the MV_RIGHT or MV_FWD direction), whatever the arm's pose.
    """
    from PIL import Image, ImageDraw

    size = top.shape[0]
    p0 = np.array([*env.tcp_pos[:2], 0.0])
    a = env.camera_pixel("front", p0, size)
    b = env.camera_pixel("front", p0 + np.array([*(np.asarray(world_dir)[:2] * 0.05), 0.0]), size)
    im = Image.fromarray(top)
    x, y = a
    ang = float(np.degrees(np.arctan2(b[1] - a[1], b[0] - a[0])))
    im = im.rotate(ang, center=(x, y), resample=Image.BICUBIC, fillcolor=(255, 255, 255))
    x0, y0 = int(round(x - half)), int(round(y - half))
    crop = im.crop((x0, y0, x0 + 2 * half, y0 + 2 * half)).resize((size, size), Image.LANCZOS)
    d = ImageDraw.Draw(crop)
    c, r = size / 2, max(6, size // 32)
    d.line([(c - r, c), (c + r, c)], fill=(30, 200, 60), width=3)
    d.line([(c, c - r), (c, c + r)], fill=(30, 200, 60), width=3)
    return np.asarray(crop)


def robot_state(env: PickPlaceEnv) -> dict[str, Any]:
    """Proprioception only: what the robot knows about itself."""
    ap = env.aperture
    return {
        "gripper_height_cm": round(float(env.tcp_pos[2]) * 100, 1),
        "grasp_height_cm": Z_GRASP * 100,
        "travel_height_cm": TRANSPORT_Z * 100,
        # Gripper servo reading: a real jaw that stops early is closed on something. The simulated grasp
        # is a weld and the fingers close fully either way, so the weld stands in for that reading.
        "jaw": "open" if ap > OPEN_MIN else ("closed_on_object" if env.held else "closed_empty"),
    }


# --------------------------------------------------------------------------- #
# ground truth (for scoring and for the scripted planner / oracle policy)
# --------------------------------------------------------------------------- #
def oracle_command(env: PickPlaceEnv, sub: Subtask) -> str:
    tcp, cube, binp = env.tcp_pos, env.cube_pos, env.bin_pos
    # a step can only land within half a step of the target, so never ask for less than that
    tol = 0.55 * sub.step_m
    cube_tol = (max(tol, GRASP_TOL_FWD), max(tol, GRASP_TOL_RIGHT))

    def horiz(target: np.ndarray, tols: tuple[float, float]) -> str | None:
        ef, er = offset_fr(env, target)
        out_f, out_r = abs(ef) - tols[0], abs(er) - tols[1]
        if out_f <= 0 and out_r <= 0:
            return None
        if out_f >= out_r:
            return "MV_FWD" if ef > 0 else "MV_BACK"
        return "MV_RIGHT" if er > 0 else "MV_LEFT"

    open_ = env.aperture > OPEN_MIN
    if sub.name == "reach_cube":
        if env.held or env.cube_in_bin():
            return "ESCALATE"
        return horiz(cube, (max(tol, REACH_TOL), max(tol, REACH_TOL))) or "SUBTASK_DONE"
    if sub.name == "align_cube":
        if env.held or env.cube_in_bin():
            return "ESCALATE"
        return horiz(cube, cube_tol) or "SUBTASK_DONE"
    if sub.name == "descend_grasp":
        if env.held:
            return "SUBTASK_DONE"
        if not open_:
            return "ESCALATE"  # closed on nothing
        h = horiz(cube, cube_tol)
        if h:
            return h
        return "MV_DOWN" if tcp[2] > Z_GRASP + GRASP_Z_TOL else "GRASP"
    if sub.name == "transport_to_bin":
        if not env.held:
            return "ESCALATE"
        if tcp[2] < TRANSPORT_Z - 0.01:
            return "MV_UP"
        return horiz(binp, (max(tol, BIN_TOL), max(tol, BIN_TOL))) or "SUBTASK_DONE"
    if sub.name == "release":
        if not open_ and tcp[2] > RELEASE_Z + 0.005:
            return "MV_DOWN"
        return "RELEASE" if not open_ else "SUBTASK_DONE"
    if sub.name == "recover":
        if not open_:
            return "RELEASE"
        return "MV_UP" if tcp[2] < Z_HOVER - 0.01 else "SUBTASK_DONE"
    return "ESCALATE"


def offset_fr(env: PickPlaceEnv, target: np.ndarray) -> tuple[float, float]:
    """Target minus tcp, horizontally, along the wrist image's (up, right) axes."""
    fwd, right = env.wrist_axes()
    e = np.asarray(target)[:2] - env.tcp_pos[:2]
    return float(e @ fwd), float(e @ right)


def cube_below(env: PickPlaceEnv) -> bool:
    """Roughly below the gripper: close enough to start descend_grasp."""
    ef, er = offset_fr(env, env.cube_pos)
    return max(abs(ef), abs(er)) <= REACH_TOL + 0.003


def cube_aligned(env: PickPlaceEnv) -> bool:
    """Within the grasp window: ready to descend."""
    ef, er = offset_fr(env, env.cube_pos)
    return abs(ef) <= GRASP_TOL_FWD + 0.002 and abs(er) <= GRASP_TOL_RIGHT + 0.002


def oracle_stage(env: PickPlaceEnv) -> str | None:
    """Which subtask the task needs next, from simulator truth (None = complete)."""
    if env.is_success():
        return None
    open_ = env.aperture > OPEN_MIN
    if env.cube_in_bin():
        return "release"
    if env.held:
        binp, tcp = env.bin_pos, env.tcp_pos
        ef, er = offset_fr(env, binp)
        above = max(abs(ef), abs(er)) <= BIN_TOL + 0.002 and tcp[2] > Z_TRAVEL - 0.015
        return "release" if above else "transport_to_bin"
    if not open_:
        return "recover"
    if not cube_below(env):
        return "reach_cube"
    return "descend_grasp" if cube_aligned(env) else "align_cube"


# --------------------------------------------------------------------------- #
# planners
# --------------------------------------------------------------------------- #
class Planner(Protocol):
    name: str

    def plan(
        self, env: PickPlaceEnv, images: dict[str, np.ndarray], events: list[dict[str, Any]]
    ) -> Plan: ...


SCRIPTED_TEXT = {
    "reach_cube": (
        "Move horizontally until the green cross in the wrist image is on or right next to the red cube.",
        "The green cross is on or touching the red cube.",
    ),
    "align_cube": (
        "Make the green cross sit on the centre of the red cube.",
        "The green cross is on the centre of the red cube.",
    ),
    "descend_grasp": (
        "Lower the gripper onto the red cube, keeping the green cross on the cube left to right, then close the gripper at grasp height.",
        "The gripper has been closed (GRASP ends this subtask).",
    ),
    "transport_to_bin": (
        "Raise the held cube to about 14 cm, then move horizontally until the green cross is inside the blue bin.",
        "The gripper is at travel height and the green cross is inside the blue bin.",
    ),
    "release": ("Open the gripper over the bin.", "The jaw is open."),
    "goto_cube": (
        "Move above the cube marked in the overhead image.",
        "The gripper is marked over the cube.",
    ),
    "goto_bin": (
        "Carry the cube above the bin marked in the overhead image.",
        "The gripper is marked over the bin.",
    ),
    "recover": (
        "Open the gripper and raise it to about 9 cm.",
        "The jaw is open and the gripper is at about 9 cm.",
    ),
}


class ScriptedPlanner:
    """Stage-1 planner: picks the correct stage from simulator truth and queues the rest of the task.

    With ``corrections`` it also emulates a planner that reads the remaining offset off the images after a
    failure and sends explicit corrective moves (an upper bound for what a VLM planner could add).
    """

    name = "scripted"

    # Corrective text hints ("the cube is still left of the cross") are off by default: with them, command-mode
    # reach accuracy fell from 42% to 5% because the controller moved the opposite way.
    def __init__(
        self, step_cm: float = 2.0, fine_cm: float = 1.0, hints: bool = False, corrections: bool = False
    ):
        self.step_cm, self.fine_cm, self.hints, self.corrections = step_cm, fine_cm, hints, corrections

    def _subtask(self, env: PickPlaceEnv, stage: str) -> Subtask:
        goal, done = (_n(t) for t in SCRIPTED_TEXT[stage])
        # coarse moves to get above the cube and the bin, fine steps while descending onto the cube
        step = self.fine_cm if stage in ("align_cube", "descend_grasp") else self.step_cm
        search = ""
        if stage == "reach_cube":
            search = _direction(*offset_fr(env, env.cube_pos))
        elif stage == "transport_to_bin":
            # where the bin lies as seen from the cube, as a planner would read it off the front camera
            fwd, right = env.wrist_axes()
            e = env.bin_pos[:2] - env.cube_pos[:2]
            search = _direction(float(e @ fwd), float(e @ right))
        return Subtask(
            stage,
            goal,
            done,
            list(DEFAULT_COMMANDS[stage]),
            step_cm=step,
            search=search,
            max_steps=30 if stage == "transport_to_bin" else 20,
        )

    def plan(self, env: PickPlaceEnv, images: dict[str, np.ndarray], events: list[dict[str, Any]]) -> Plan:
        stage = oracle_stage(env)
        if stage is None:
            return Plan(task_complete=True, subtasks=[], scene="task complete (simulator truth)")
        order = TASK_ORDER[TASK_ORDER.index(stage) :] if stage in TASK_ORDER else [stage, *TASK_ORDER]
        pre: list[str] = []
        last = events[-1] if events else None
        if (
            self.corrections
            and last
            and last["outcome"] in ("missed_grasp", "escalated", "stuck", "oscillating")
        ):
            pre = _truth_corrections(env)
        return Plan(
            task_complete=False,
            subtasks=[self._subtask(env, s) for s in order],
            pre_moves=pre,
            scene=f"stage from simulator truth: {stage}",
        )


class SequencePlanner:
    """Planner without simulator truth (real robot): the fixed task order, restarted where a failure occurred.

    ``search_cube`` / ``search_bin`` are the wrist-image directions to move while the target is not in view,
    given by whoever looks at the front camera before the run (a person, or Claude acting as the planner).
    """

    name = "sequence"

    def __init__(
        self,
        step_cm: float = 2.0,
        fine_cm: float = 1.0,
        search_cube: str = "",
        search_bin: str = "",
        overhead: bool = False,
    ):
        self.step_cm, self.fine_cm, self.search = step_cm, fine_cm, {"cube": search_cube, "bin": search_bin}
        # with the overhead guide: coarse moves by marks replace the searches and the transport servo
        self.order = OVERHEAD_ORDER if overhead else TASK_ORDER

    def _sub(self, stage: str) -> Subtask:
        goal, done = (_n(t) for t in SCRIPTED_TEXT[stage])
        step = self.fine_cm if stage in ("align_cube", "descend_grasp") else self.step_cm
        search = (
            self.search["bin"]
            if stage == "transport_to_bin"
            else self.search["cube"]
            if stage == "reach_cube"
            else ""
        )
        return Subtask(
            stage,
            goal,
            done,
            list(DEFAULT_COMMANDS[stage]),
            step_cm=step,
            search=search,
            max_steps=30 if stage == "transport_to_bin" else 20,
        )

    def plan(self, env: PickPlaceEnv, images: dict[str, np.ndarray], events: list[dict[str, Any]]) -> Plan:
        last = events[-1] if events else None
        full = self.order
        if last is None:
            order = full
        elif last["outcome"] == "queue_finished":
            return Plan(
                task_complete=True, subtasks=[], scene="queue finished; success is judged by a person"
            )
        elif last["outcome"] == "missed_grasp":
            order = full  # the cube may have moved (or been pushed): start again from finding it
        else:
            failed = last.get("subtask", full[0])
            order = full[full.index(failed) :] if failed in full else full
            if failed in ("reach_cube", "align_cube", "descend_grasp") and full is OVERHEAD_ORDER:
                order = full  # lost the cube in the wrist view: mark it again from above
            elif failed in ("align_cube", "descend_grasp"):
                order = full
            if failed in ("goto_bin", "transport_to_bin", "release") and not env.held:
                order = full  # dropped on the way
        return Plan(task_complete=False, subtasks=[self._sub(s) for s in order], scene="fixed sequence")


class TaskPlanner:
    """Planner for a task list from an instruction, e.g. [{"op": "pick", "object": "red cube"},
    {"op": "place", "where": "on:blue cube"}] (dlb/voice/intent.py).

    pick X -> goto_cube, align_cube, descend_grasp with target X; place W -> place_at W. After a failure it
    restarts from the task that failed (a pick that already holds its object counts as done), at most
    ``retries`` times per task.
    """

    name = "tasks"

    def __init__(
        self, tasks: list[dict[str, str]], bin_name: str = "bin", fine_cm: float = 1.0, retries: int = 2
    ):
        self.tasks, self.bin_name, self.fine_cm, self.retries = list(tasks), bin_name, fine_cm, retries
        self.tries: dict[int, int] = {}
        self.held: str | None = None  # object the last pick grasped (for the caller's state)

    def _subs(self, i: int, t: dict[str, str]) -> list[Subtask]:
        if t["op"] == "pick":
            out = []
            for stage in ("goto_cube", "align_cube", "descend_grasp"):
                goal, done = (x.replace("red cube", t["object"]) for x in SCRIPTED_TEXT[stage])
                out.append(
                    Subtask(
                        stage,
                        goal,
                        done,
                        list(DEFAULT_COMMANDS[stage]),
                        step_cm=self.fine_cm,
                        target=t["object"],
                        task_index=i,
                    )
                )
            return out
        where = t.get("where", "bin")
        return [
            Subtask(
                "place_at",
                f"Put the held object down: {where}.",
                "The jaw is open.",
                [],
                place=where,
                task_index=i,
            )
        ]

    def plan(self, env: PickPlaceEnv, images: dict[str, np.ndarray], events: list[dict[str, Any]]) -> Plan:
        last = events[-1] if events else None
        start = 0
        if last is not None:
            if last["outcome"] == "queue_finished":
                return Plan(task_complete=True, subtasks=[], scene="task list finished")
            start = max(0, int(last.get("task_index", 0)))
            if " is inside the " in str(last.get("reason", "")):
                return Plan(task_complete=True, subtasks=[], scene=last["reason"], diagnosis="nothing to do")
            t = self.tasks[start] if start < len(self.tasks) else None
            if (
                t is not None
                and t["op"] == "pick"
                and env.held
                and last.get("subtask") not in ("descend_grasp",)
            ):
                start += 1  # already holding it
            self.tries[start] = self.tries.get(start, 0) + 1
            if self.tries[start] > self.retries:
                return Plan(
                    task_complete=True,
                    subtasks=[],
                    scene=f"gave up on task {start}: {self.tasks[start]}",
                    diagnosis="retries exhausted",
                )
        subs = [s for i, t in enumerate(self.tasks) if i >= start for s in self._subs(i, t)]
        return Plan(task_complete=False, subtasks=subs, scene=f"tasks from {start}: {self.tasks[start:]}")


TASK_ORDER = ["reach_cube", "align_cube", "descend_grasp", "transport_to_bin", "release"]
# the marks bring the gripper within ~1-2 cm of the cube: the coarse wrist search (reach_cube) is not needed
OVERHEAD_ORDER = ["goto_cube", "align_cube", "descend_grasp", "goto_bin", "release"]


def _direction(ef: float, er: float) -> str:
    if abs(er) >= abs(ef):
        return "MV_RIGHT" if er > 0 else "MV_LEFT"
    return "MV_FWD" if ef > 0 else "MV_BACK"


def _truth_corrections(env: PickPlaceEnv, step: float = 0.01, limit: int = 4) -> list[str]:
    """1 cm moves that bring the gripper over the cube (used only by the scripted upper-bound planner)."""
    ef, er = offset_fr(env, env.cube_pos)
    moves = ["MV_FWD" if ef > 0 else "MV_BACK"] * int(round(abs(ef) / step))
    moves += ["MV_RIGHT" if er > 0 else "MV_LEFT"] * int(round(abs(er) / step))
    return moves[:limit]


PLANNER_SYSTEM = f"""You are the slow, deliberate planner for a small 5-DoF robot arm (ROBOTIS OMX) on a table.
Task: {TASK_TEXT}

A fast low-level controller executes your subtasks one step at a time. It can only choose among these commands:
- MV_FWD / MV_BACK: move the gripper a fixed step toward the top / bottom of the WRIST image
- MV_LEFT / MV_RIGHT: move toward the left / right of the WRIST image
- MV_UP / MV_DOWN: raise / lower the gripper (it cannot go below grasp height)
- GRASP (close the fingers), RELEASE (open the fingers)
- SUBTASK_DONE (it believes the completion condition holds), ESCALATE (it hands control back to you)
The controller sees the wrist image and the robot's own state, never object coordinates. On the wrist image a
green cross marks the point directly below the gripper, so "the green cross is on the red cube" means the
gripper is above the cube.

You give a QUEUE of subtasks. The harness moves to the next subtask by itself when the controller reports
SUBTASK_DONE, when GRASP closes on the cube, or after RELEASE. You are called again only when the queue is
empty or something goes wrong. The harness detects these failures and reports them in the event log:
- missed_grasp: GRASP closed on nothing. The harness has already opened the gripper and raised it to ~9 cm.
- stuck: a subtask ran past max_steps. oscillating: the controller alternated opposite moves.
- escalated: the controller chose ESCALATE.
Each event records where the gripper was (gripper_xy_cm, from the robot's own joint angles), so you can tell
whether an attempt repeated the same spot.

Subtask types:
{json.dumps(SUBTASKS, indent=1)}

Write goals and completion conditions in visual terms the controller can check in the images. Do NOT steer
the controller with directions in text (e.g. "the cube is left of the cross"): it tends to move the wrong
way. To correct a position, use pre_moves: up to 4 commands the harness executes directly, with
pre_move_step_cm, before the first subtask starts. Judge the needed correction yourself from the images,
especially after a missed grasp (the controller usually misjudges forward/backward offsets, i.e. the
top-bottom direction of the wrist image). Do not repeat an approach that already failed at the same spot.
`search` is the wrist-image direction to move while the target is not visible (e.g. the bin while carrying
the cube), judged from the front camera. Use step_cm 2 for reach_cube / transport_to_bin and 1 for
align_cube and descend_grasp. Forward/backward (top-bottom) alignment must be finished in align_cube at the
current height: while descending, the finger bar hides the cube and only left-right is corrected. The
normal queue is reach_cube -> align_cube -> descend_grasp -> transport_to_bin -> release. Set task_complete only when the cube is inside
the bin and the gripper is open."""

SUBTASK_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["name", "goal", "done_when", "hint", "search", "step_cm", "max_steps"],
    "properties": {
        "name": {"type": "string", "enum": list(SUBTASKS)},
        "goal": {"type": "string"},
        "done_when": {"type": "string"},
        "hint": {"type": "string", "description": "Optional short note for the controller; may be empty."},
        "search": {"type": "string", "enum": ["", "MV_FWD", "MV_BACK", "MV_LEFT", "MV_RIGHT"]},
        "step_cm": {"type": "number", "enum": [1, 2, 3]},
        "max_steps": {"type": "integer", "minimum": 2, "maximum": 30},
    },
}
PLAN_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["scene", "diagnosis", "task_complete", "pre_moves", "pre_move_step_cm", "subtasks"],
    "properties": {
        "scene": {"type": "string", "description": "What you see: gripper, cube and bin, relative layout."},
        "diagnosis": {
            "type": "string",
            "description": "What went wrong so far and what you change; empty at the start.",
        },
        "task_complete": {"type": "boolean"},
        "pre_moves": {"type": "array", "maxItems": 4, "items": {"type": "string", "enum": MOVES}},
        "pre_move_step_cm": {"type": "number", "enum": [1, 2]},
        "subtasks": {"type": "array", "minItems": 0, "maxItems": 4, "items": SUBTASK_SCHEMA},
    },
}


class OpenAIPlanner:
    """Stage-2 planner: an OpenAI vision model called through the Responses API."""

    name = "openai"

    def __init__(
        self, model: str = "gpt-5.5", reasoning_effort: str | None = "low", timeout_s: float = 120.0
    ):
        import httpx

        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.client = httpx.Client(
            base_url="https://api.openai.com/v1",
            headers={"Authorization": f"Bearer {key}"},
            timeout=timeout_s,
        )

    def plan(self, env: PickPlaceEnv, images: dict[str, np.ndarray], events: list[dict[str, Any]]) -> Plan:
        content: list[dict[str, Any]] = [
            {
                "type": "input_text",
                "text": json.dumps(
                    {"robot_state": robot_state(env), "event_log": events[-12:]}, ensure_ascii=False
                ),
            }
        ]
        for cam, img in images.items():
            content.append({"type": "input_text", "text": f"{cam} camera:"})
            content.append({"type": "input_image", "image_url": image_to_data_url(img, fmt="JPEG")})
        body: dict[str, Any] = {
            "model": self.model,
            "instructions": _n(PLANNER_SYSTEM),
            "input": [{"role": "user", "content": content}],
            "text": {
                "format": {"type": "json_schema", "name": "plan", "schema": PLAN_SCHEMA, "strict": True}
            },
        }
        if self.reasoning_effort:
            body["reasoning"] = {"effort": self.reasoning_effort}
        t0 = time.perf_counter()
        r = self.client.post("/responses", json=body)
        latency = time.perf_counter() - t0
        if r.status_code != 200:
            raise RuntimeError(f"OpenAI {r.status_code}: {r.text[:500]}")
        d = r.json()
        text = "".join(
            c.get("text", "")
            for o in d.get("output", [])
            if o.get("type") == "message"
            for c in o.get("content", [])
            if c.get("type") == "output_text"
        )
        p = json.loads(text)
        u = d.get("usage") or {}
        subs = [
            Subtask(
                name=s["name"],
                goal=s["goal"],
                done_when=s["done_when"],
                allowed=list(DEFAULT_COMMANDS[s["name"]]),
                step_cm=float(s["step_cm"]),
                max_steps=int(s["max_steps"]),
                hint=s.get("hint", ""),
                search=s.get("search", ""),
            )
            for s in p["subtasks"]
        ]
        return Plan(
            task_complete=bool(p["task_complete"]),
            subtasks=subs,
            pre_moves=list(p["pre_moves"]),
            pre_move_step_cm=float(p["pre_move_step_cm"]),
            scene=p["scene"],
            diagnosis=p["diagnosis"],
            latency_s=latency,
            usage={
                "input_tokens": int(u.get("input_tokens", 0)),
                "output_tokens": int(u.get("output_tokens", 0)),
            },
        )


# --------------------------------------------------------------------------- #
# fast policies
# --------------------------------------------------------------------------- #
@dataclass
class Decision:
    command: str
    confidence: float
    answers: dict[str, Any] = field(default_factory=dict)
    latency_s: float = 0.0
    input_tokens: int = 0
    step_scale: float = 1.0  # fraction of the subtask step for this move (the servo's half-step back)
    xy: tuple[float, float] | None = None  # MV_XY: (forward, right) in metres along the wrist image axes


class OraclePolicy:
    """Harness sanity check: executes the ground-truth command."""

    name = "oracle"
    needs_images = False

    def decide(self, env: PickPlaceEnv, sub: Subtask, images: dict[str, np.ndarray]) -> Decision:
        return Decision(oracle_command(env, sub), 1.0)


CAMERA_NOTE = (
    "The wrist camera looks down past the two dark finger tips (at the bottom of the image). "
    "The green cross marks the point directly below the gripper. "
    "'Forward' means toward the top of the wrist image, 'right' toward its right."
)


def command_descriptions(sub: Subtask) -> dict[str, str]:
    s = f"{sub.step_cm:g} cm"
    return {
        "MV_FWD": f"Move the gripper {s} toward the top of the wrist image.",
        "MV_BACK": f"Move the gripper {s} toward the bottom of the wrist image.",
        "MV_LEFT": f"Move the gripper {s} toward the left of the wrist image.",
        "MV_RIGHT": f"Move the gripper {s} toward the right of the wrist image.",
        "MV_UP": f"Raise the gripper {s}.",
        "MV_DOWN": f"Lower the gripper {s} (it stops at grasp height).",
        "GRASP": "Close the fingers on whatever is between them.",
        "RELEASE": "Open the fingers.",
        "SUBTASK_DONE": f"The subtask is finished: {sub.done_when}",
        "ESCALATE": "The scene does not fit the subtask (target not visible, cube dropped, stuck). Ask the planner.",
    }


class JevCommandPolicy:
    """Mode A: the decision layer picks one command directly."""

    needs_images = True

    def __init__(
        self,
        backend: DecisionBackend,
        cameras: tuple[str, ...] = ("wrist",),
        cross: bool = True,
        black: bool = False,
        show_hint: bool = True,
        zoom: bool = False,
    ):
        self.be, self.cameras, self.cross, self.black, self.show_hint = (
            backend,
            cameras,
            cross,
            black,
            show_hint,
        )
        # "zoom" in cameras means the zoomed crop around the cross stands in for a camera image
        self.zoom = zoom and "wrist" in cameras
        tags = [*cameras] + (["zoom"] if self.zoom else [])
        flags = ("" if cross else ",nocross") + (",black" if black else "")
        self.name = f"jev_command[{'+'.join(tags)}{flags}]"

    def _image_names(self) -> list[str]:
        names = [
            "zoomed wrist view around the green cross" if c == "zoom" else f"{c} camera" for c in self.cameras
        ]
        if self.zoom:
            names.append("zoomed view of the wrist image around the green cross (same cross, 2.5x)")
        return names

    def _images(
        self, env: PickPlaceEnv, sub: Subtask, images: dict[str, np.ndarray], rotate: bool = False
    ) -> list[str]:
        out = []
        for cam in self.cameras:
            img = images["wrist" if cam == "zoom" else cam]
            if cam in ("wrist", "zoom") and self.cross:
                img = wrist_with_cross(env, img, cross_height(sub))
            if cam == "zoom":
                img = zoom_on_cross(env, img, cross_height(sub))
            out.append(np.zeros_like(img) if self.black else img)
        if self.zoom:
            z = zoom_on_cross(
                env,
                wrist_with_cross(env, images["wrist"], cross_height(sub)) if self.cross else images["wrist"],
                cross_height(sub),
            )
            out.append(np.zeros_like(z) if self.black else z)
        if rotate:  # 90 degrees clockwise: the image top becomes the right side
            out = [np.ascontiguousarray(np.rot90(i, k=-1)) for i in out]
        return [image_to_data_url(i, fmt="JPEG") for i in out]

    def state(self, env: PickPlaceEnv, sub: Subtask) -> dict[str, Any]:
        st: dict[str, Any] = {
            "role": "You are the fast controller of a robot arm. Each step you choose one command that makes "
            "progress on the current subtask, judging only from the images and the robot state.",
            "task": _n(TASK_TEXT),
            "subtask": sub.goal,
            "done_when": sub.done_when,
            "robot_state": robot_state(env),
            "images": [f"image {i + 1}: {n}" for i, n in enumerate(self._image_names())],
            "camera_note": CAMERA_NOTE
            if self.cross
            else CAMERA_NOTE.replace("The green cross marks the point directly below the gripper. ", ""),
        }
        if self.show_hint and sub.hint:
            st["planner_hint"] = sub.hint
        if sub.search:
            st["if_target_not_visible"] = f"choose {sub.search}"
        return st

    def decide(self, env: PickPlaceEnv, sub: Subtask, images: dict[str, np.ndarray]) -> Decision:
        desc = command_descriptions(sub)
        q = Choice(
            instructions="Which command should the robot execute now?",
            criteria={c: desc[c] for c in sub.allowed},
        )
        req = DecisionRequest(
            state=self.state(env, sub), questions={"command": q}, images=self._images(env, sub, images)
        )
        resp = self.be.decide(req)
        a = resp.answers["command"]
        cmd = a.choice if a.choice in sub.allowed else "ESCALATE"
        return Decision(cmd, a.conf, {"command": a.to_wire()}, resp.latency_s, resp.usage.input_tokens)


class JevPerceptionPolicy(JevCommandPolicy):
    """Mode B: the decision layer answers what it sees; a small rule maps answers to a command."""

    def __init__(self, *a: Any, rotate_v: bool = True, **kw: Any):
        super().__init__(*a, **kw)
        # Top-bottom is asked again on the image rotated 90 degrees clockwise, as a left-right question: the
        # model is much better at left-right (probe at reach height: top-bottom 0.65 -> 0.87). "on" needs both
        # views to agree, which cuts premature "done" answers.
        self.rotate_v = rotate_v
        self.name = self.name.replace("jev_command", "jev_perception") + ("" if rotate_v else "[norot]")
        self._pool = None

    def decide(self, env: PickPlaceEnv, sub: Subtask, images: dict[str, np.ndarray]) -> Decision:
        target = OBJ["bin"] if target_is_bin(sub) else OBJ["cube"]
        ref = "the green cross" if self.cross else "the point directly below the gripper"
        qs: dict[str, Any] = {
            "visible": Noul(instructions=f"Is the {target} visible in the wrist image?"),
            "horizontal": Choice(
                instructions=f"In the wrist image, where is the {target} relative to {ref}, left to right?",
                criteria={
                    "left": f"The {target} is clearly left of {ref}.",
                    "on": f"{ref[0].upper() + ref[1:]} is on the {target} (left-right).",
                    "right": f"The {target} is clearly right of {ref}.",
                },
            ),
            "vertical": Choice(
                instructions=f"In the wrist image, where is the {target} relative to {ref}, top to bottom?",
                criteria={
                    "above": f"The {target} is clearly above {ref} (toward the top of the image).",
                    "on": f"{ref[0].upper() + ref[1:]} is on the {target} (top-bottom).",
                    "below": f"The {target} is clearly below {ref} (toward the bottom of the image).",
                },
            ),
        }
        st = self.state(env, sub)
        st["role"] = "You are the eyes of a robot arm. Answer what you see in the images."
        imgs = self._images(env, sub, images)
        req = DecisionRequest(state=st, questions=qs, images=imgs)
        if not (self.rotate_v and len(imgs) == 1):
            resp = self.be.decide(req)
            ans = dict(resp.answers)
            latency, tokens = resp.latency_s, resp.usage.input_tokens
        else:
            from concurrent.futures import ThreadPoolExecutor

            rot = self._images(env, sub, images, rotate=True)
            st_r = {**st, "images": ["the same view rotated 90 degrees clockwise"]}
            req_r = DecisionRequest(state=st_r, questions={"horizontal": qs["horizontal"]}, images=rot)
            if self._pool is None:
                self._pool = ThreadPoolExecutor(2)
            f_main, f_rot = self._pool.submit(self.be.decide, req), self._pool.submit(self.be.decide, req_r)
            resp, resp_r = f_main.result(), f_rot.result()
            ans = dict(resp.answers)
            ans["vertical_upright"] = ans["vertical"]
            ans["vertical_rotated"] = resp_r.answers["horizontal"]
            ans["vertical"] = combine_vertical(ans["vertical_upright"], ans["vertical_rotated"])
            latency, tokens = (
                max(resp.latency_s, resp_r.latency_s),
                resp.usage.input_tokens + resp_r.usage.input_tokens,
            )
        cmd, conf = self._rule(env, sub, ans)
        return Decision(cmd, conf, {k: v.to_wire() for k, v in ans.items()}, latency, tokens)

    @staticmethod
    def _rule(env: PickPlaceEnv, sub: Subtask, ans: dict[str, Any]) -> tuple[str, float]:
        rs = robot_state(env)
        z, jaw_open = rs["gripper_height_cm"] / 100, rs["jaw"] == "open"
        if sub.name == "release":
            return ("RELEASE", 1.0) if not jaw_open else ("SUBTASK_DONE", 1.0)
        if sub.name == "recover":
            if not jaw_open:
                return "RELEASE", 1.0
            return ("MV_UP", 1.0) if z < Z_HOVER - 0.01 else ("SUBTASK_DONE", 1.0)
        held = rs["jaw"] == "closed_on_object"
        if sub.name == "descend_grasp" and not jaw_open:
            return ("SUBTASK_DONE", 1.0) if held else ("ESCALATE", 1.0)
        if sub.name == "transport_to_bin":
            if not held:
                return "ESCALATE", 1.0
            if z < TRANSPORT_Z - 0.01:
                return "MV_UP", 1.0
        if ans["visible"].noul < 0.5:
            return (sub.search, ans["visible"].conf) if sub.search else ("ESCALATE", ans["visible"].conf)
        h, v = ans["horizontal"], ans["vertical"]
        off_h = 1 - (h.probabilities or {}).get("on", 0.0)
        off_v = 1 - (v.probabilities or {}).get("on", 0.0)
        if sub.name == "descend_grasp" and z < NO_FWD_BELOW_Z:
            off_v = (
                0.0  # the finger bar hides the cube's lower half; forward/backward was settled in align_cube
            )
        if max(off_h, off_v) > 0.5:
            if off_h >= off_v:
                return ("MV_LEFT" if h.choice == "left" else "MV_RIGHT"), off_h
            return ("MV_FWD" if v.choice == "above" else "MV_BACK"), off_v
        conf = 1 - max(off_h, off_v)
        if sub.name == "descend_grasp":
            return ("MV_DOWN", conf) if z > Z_GRASP + GRASP_Z_TOL else ("GRASP", conf)
        return "SUBTASK_DONE", conf


class JevServoPolicy(JevCommandPolicy):
    """Binary servo: ask only which side of the cross the target is on, per axis, and stop at the flip.

    Left-right is asked on the upright crop, top-bottom as left-right on the crop rotated 90 degrees (the
    probe: sign accuracy >= 0.92 on both axes for offsets over 1 cm, near chance under 1 cm). An axis is done
    when its answer flips between consecutive moves along it; the servo then steps back half a step, which
    leaves at most half a step of error. The subtask is done when both axes are. Descending goes straight
    down (alignment is finished before) and grasps at grasp height.
    """

    SURE = 0.8  # side answers below this probability count as "close to the cross"

    def __init__(self, *a: Any, **kw: Any):
        super().__init__(*a, **kw)
        self.name = self.name.replace("jev_command", "jev_servo")
        self._pool = None
        self._key: Any = None
        self._state: dict[str, Any] = {}

    def _reset(self, sub: Subtask) -> None:
        self._key = sub.uid
        # per axis: last side answered, whether converged, whether a half step back is pending
        self._state = {ax: {"last": None, "done": False} for ax in ("h", "v")}
        self._turn = "h"

    def _ask(
        self, env: PickPlaceEnv, sub: Subtask, images: dict[str, np.ndarray], axes: list[str]
    ) -> tuple[dict, float, int]:
        from concurrent.futures import ThreadPoolExecutor

        target = OBJ["bin"] if target_is_bin(sub) else OBJ["cube"]
        ref = "the green cross"
        side = Choice(
            instructions=f"Is the centre of the {target} left or right of {ref}?",
            criteria={
                "left": f"The centre of the {target} is left of {ref}.",
                "right": f"The centre of the {target} is right of {ref}.",
            },
        )
        note = {
            "note": "Zoomed wrist camera view of a robot gripper looking down. The green cross marks the point "
            "directly below the gripper." + (f" The gripper holds a {OBJ['cube']}." if env.held else "")
        }
        reqs = {}
        if target_is_bin(sub) and getattr(env, "top_view", False) and "front" in images:
            # The held cube hides the wrist view: judge the bin from the overhead camera, one rotated crop per axis.
            fwd, right = env.wrist_axes()
            note_top = {
                "note": "Overhead camera view of a robot arm, rotated. The green cross marks the point directly "
                "below the gripper."
            }
            for ax in axes:
                crop = top_view_crop(env, images["front"], right if ax == "h" else fwd)
                qs = {"side": side}
                if ax == "h":
                    qs["visible"] = Noul(instructions=f"Is the {target} visible?")
                reqs[ax] = DecisionRequest(
                    state=note_top, questions=qs, images=[image_to_data_url(crop, fmt="JPEG")]
                )
            if self._pool is None:
                self._pool = ThreadPoolExecutor(3)
            futs = {ax: self._pool.submit(self.be.decide, r) for ax, r in reqs.items()}
            out, lat, tok = {}, 0.0, 0
            for ax, f in futs.items():
                r = f.result()
                out[ax] = r.answers
                lat, tok = max(lat, r.latency_s), tok + r.usage.input_tokens
            return out, lat, tok
        if "h" in axes:
            reqs["h"] = DecisionRequest(
                state=note,
                questions={"side": side, "visible": Noul(instructions=f"Is the {target} visible?")},
                images=self._images(env, sub, images),
            )
        if "v" in axes:
            reqs["v"] = DecisionRequest(
                state={**note, "rotated": "the view is rotated"},
                questions={"side": side},
                images=self._images(env, sub, images, rotate=True),
            )
            if target_is_bin(sub) and "front" in images and getattr(env, "front_calibrated", True):
                # second opinion from the front camera (the held cube can hide the bin in the wrist view)
                fv = front_forward_view(env, images["front"], cross_height(sub))
                reqs["v_front"] = DecisionRequest(
                    state={
                        "note": "Close-up of a front camera view of a robot arm, rotated. The green cross marks the "
                        "point directly below the gripper."
                    },
                    questions={"side": side},
                    images=[image_to_data_url(fv, fmt="JPEG")],
                )
        if self._pool is None:
            self._pool = ThreadPoolExecutor(3)
        futs = {ax: self._pool.submit(self.be.decide, r) for ax, r in reqs.items()}
        out, lat, tok = {}, 0.0, 0
        for ax, f in futs.items():
            r = f.result()
            out[ax] = r.answers
            lat, tok = max(lat, r.latency_s), tok + r.usage.input_tokens
        return out, lat, tok

    def decide(self, env: PickPlaceEnv, sub: Subtask, images: dict[str, np.ndarray]) -> Decision:
        if self._key != sub.uid:
            self._reset(sub)
        rs = robot_state(env)
        z, jaw = rs["gripper_height_cm"] / 100, rs["jaw"]
        if sub.name == "release":
            if jaw != "open" and z > RELEASE_Z + 0.005:
                return Decision("MV_DOWN", 1.0)
            return Decision("RELEASE" if jaw != "open" else "SUBTASK_DONE", 1.0)
        if sub.name == "recover":
            if jaw != "open":
                return Decision("RELEASE", 1.0)
            return Decision("MV_UP" if z < Z_HOVER - 0.01 else "SUBTASK_DONE", 1.0)
        if sub.name == "descend_grasp":
            if jaw != "open":
                return Decision("SUBTASK_DONE" if jaw == "closed_on_object" else "ESCALATE", 1.0)
            return Decision("MV_DOWN" if z > Z_GRASP + GRASP_Z_TOL else "GRASP", 1.0)
        if sub.name == "transport_to_bin":
            if jaw != "closed_on_object":
                return Decision("ESCALATE", 1.0)
            if z < TRANSPORT_Z - 0.01:
                return Decision("MV_UP", 1.0)
        obs_z = getattr(env, "observe_z", None)
        if obs_z and sub.name in ("reach_cube", "align_cube") and z < obs_z - 0.01:
            # align from higher up, where the cube looks small enough to judge its centre (real wrist camera)
            return Decision("MV_UP", 1.0, step_scale=min(1.0, (obs_z - z) / sub.step_m))
        st = self._state
        todo = [ax for ax in ("h", "v") if not st[ax]["done"]]
        if not todo:
            return Decision("SUBTASK_DONE", 1.0)
        # ask about the axis whose turn it is (alternate between unfinished axes)
        ax = self._turn if self._turn in todo else todo[0]
        ans, lat, tok = self._ask(env, sub, images, [ax])
        a = ans[ax]
        wire = {f"{ax}_{k}": v.to_wire() for k, v in a.items()}
        if "visible" in a and a["visible"].noul < 0.5:
            cmd = sub.search or "ESCALATE"
            return Decision(cmd, a["visible"].conf, wire, lat, tok)
        probs = dict(a["side"].probabilities or {})
        if ax == "v" and "v_front" in ans:
            # average with the front camera's opinion (the held cube can hide the bin in the wrist view)
            wire.update({f"v_front_{k}": v.to_wire() for k, v in ans["v_front"].items()})
            pf = ans["v_front"]["side"].probabilities or {}
            probs = {k: 0.5 * (probs.get(k, 0.0) + pf.get(k, 0.0)) for k in ("left", "right")}
        side = max(("left", "right"), key=lambda k: probs.get(k, 0.0))  # rotated view: right = above
        conf = probs.get(side, 0.0)
        if len(todo) == 2:
            self._turn = "v" if ax == "h" else "h"
        bin_fwd = ax == "v" and target_is_bin(sub) and not getattr(env, "top_view", False)
        if bin_fwd:
            # A bin behind the gripper is partly hidden (held cube in the wrist view, arm in the front view):
            # there the answers are rarely a confident "ahead" (7%) but often unsure (30%). So only a confident
            # "ahead" moves forward; anything else moves back, and only a confident flip ends the axis.
            side = "right" if probs.get("right", 0.0) >= self.SURE else "left"
            conf = max(probs.get("right", 0.0), 1.0 - probs.get("right", 0.0))
        big_cube = getattr(env, "observe_z", None) is not None and not target_is_bin(sub)
        if conf < self.SURE and big_cube:
            # A large cube in view makes "left or right of the centre" hard once the cross is on it, so an unsure
            # answer does not mean centred (the real grasp landed on the cube's edge): creep half a step and ask again.
            cmd = (
                {"right": "MV_RIGHT", "left": "MV_LEFT"}
                if ax == "h"
                else {"right": "MV_FWD", "left": "MV_BACK"}
            )[side]
            st[ax]["unsure"] = st[ax].get("unsure", 0) + 1
            if st[ax]["unsure"] >= 4:
                st[ax]["done"] = True
            return Decision(cmd, conf, wire, lat, tok, step_scale=0.5)
        if conf < self.SURE and not bin_fwd:
            # Wrong side answers come with low probability (probe of episode logs: mean 0.67 vs 0.9 for right
            # ones). An unsure answer means the target is close to the cross: this axis is done, no move.
            st[ax]["done"] = True
            return (
                self.decide(env, sub, images)
                if [x for x in ("h", "v") if not st[x]["done"]]
                else Decision("SUBTASK_DONE", conf, wire, lat, tok)
            )
        if ax == "h":
            cmd = "MV_RIGHT" if side == "right" else "MV_LEFT"
        else:
            cmd = "MV_FWD" if side == "right" else "MV_BACK"
        last = st[ax]["last"]
        st[ax]["last"] = side
        if last is not None and side != last and (conf >= self.SURE or bin_fwd):
            # passed the centre since the last move on this axis: step back half a step, axis done
            st[ax]["done"] = True
            return Decision(cmd, conf, wire, lat, tok, step_scale=0.5)
        return Decision(cmd, conf, wire, lat, tok)


def combine_vertical(upright: Any, rotated: Any) -> Any:
    """Merge the upright top-bottom answer with the rotated view's left-right answer (right = above).

    "on" only when both views say so; otherwise trust the rotated view's off-side answer, falling back to the
    upright one when only it sees an offset.
    """
    from dlb.contract import make_choice_answer

    rp = rotated.probabilities or {}
    r = {"above": rp.get("right", 0.0), "on": rp.get("on", 0.0), "below": rp.get("left", 0.0)}
    u = upright.probabilities or {}
    opts = ["above", "on", "below"]
    if upright.choice == "on" and rotated.choice == "on":
        return make_choice_answer("on", opts, min(u.get("on", 0.0), r["on"]))
    if rotated.choice != "on":
        side = max(("above", "below"), key=lambda k: r[k])
        return make_choice_answer(side, opts, r[side] + 0.5 * r["on"])
    return make_choice_answer(upright.choice, opts, u.get(upright.choice, 0.0))


# --------------------------------------------------------------------------- #
# runner
# --------------------------------------------------------------------------- #


class JevBisectServoPolicy(JevServoPolicy):
    """Fast servo for the real robot: both axes per step, the step halves at every flip, straight descent.

    The binary servo asked one axis per step and moved a fixed step, so it wandered (15 moves to align on the
    first real run, with the target already within 1 cm). Here each step asks left/right and (rotated)
    top/bottom in parallel and makes one diagonal move. Each axis starts at ``step0`` and halves its step when
    its answer flips; it is done once the step is below ``min_step`` or an answer is unsure (the target is
    near the cross). After ``max_moves`` moves the servo descends anyway (a miss is detected and recovered).
    Descent and release are one move each.
    """

    def __init__(self, *a: Any, step0: float = 0.015, min_step: float = 0.004, max_moves: int = 6, **kw: Any):
        super().__init__(*a, **kw)
        self.name = self.name.replace("jev_servo", "jev_bisect")
        self.step0, self.min_step, self.max_moves = step0, min_step, max_moves

    def _reset(self, sub: Subtask) -> None:
        super()._reset(sub)
        for ax in ("h", "v"):
            self._state[ax].update(step=self.step0, flipped=False)
        self._moves = 0

    def decide(self, env: PickPlaceEnv, sub: Subtask, images: dict[str, np.ndarray]) -> Decision:
        if self._key != sub.uid:
            self._reset(sub)
        rs = robot_state(env)
        z, jaw = rs["gripper_height_cm"] / 100, rs["jaw"]
        if sub.name == "descend_grasp":
            if jaw != "open":
                return Decision("SUBTASK_DONE" if jaw == "closed_on_object" else "ESCALATE", 1.0)
            if z > Z_GRASP + GRASP_Z_TOL:
                # one precise move straight down to grasp height, measured from where the arm rests (see _execute)
                return Decision("MV_DOWN", 1.0, step_scale=(z - Z_GRASP) / sub.step_m)
            return Decision("GRASP", 1.0)
        if sub.name == "release":
            if jaw != "open" and z > RELEASE_Z + 0.01:
                return Decision("MV_DOWN", 1.0, step_scale=(z - RELEASE_Z) / sub.step_m)
            return Decision("RELEASE" if jaw != "open" else "SUBTASK_DONE", 1.0)
        if sub.name not in ("reach_cube", "align_cube"):
            return super().decide(env, sub, images)
        st = self._state
        todo = [ax for ax in ("h", "v") if not st[ax]["done"]]
        if not todo or self._moves >= self.max_moves:
            return Decision("SUBTASK_DONE", 1.0)
        ans, lat, tok = self._ask(env, sub, images, todo)
        wire = {f"{ax}_{k}": v.to_wire() for ax in todo for k, v in ans[ax].items()}
        if "h" in ans and "visible" in ans["h"] and ans["h"]["visible"].noul < 0.5:
            return Decision(sub.search or "ESCALATE", ans["h"]["visible"].conf, wire, lat, tok)
        move = {"h": 0.0, "v": 0.0}
        confs = []
        for ax in todo:
            probs = ans[ax]["side"].probabilities or {}
            side = max(("left", "right"), key=lambda k: probs.get(k, 0.0))  # rotated view: right = ahead
            conf = probs.get(side, 0.0)
            confs.append(conf)
            a = st[ax]
            sign = 1.0 if side == "right" else -1.0
            if conf < self.SURE:
                # Within ~1 cm the side answers are near chance (sim: 0.5-0.7 at 0.2-1 cm); creeping on them walked
                # the gripper away from the cube. An unsure answer means close enough.
                a["done"] = True
                continue
            if a["last"] is not None and side != a["last"]:
                a["flipped"] = True
                a["step"] /= 2
            a["last"] = side
            move[ax] = sign * a["step"]
            if a["flipped"] and a["step"] < self.min_step:
                a["done"] = True  # this last half step leaves under min_step of error
        if move["h"] == 0.0 and move["v"] == 0.0:
            return Decision("SUBTASK_DONE", min(confs) if confs else 1.0, wire, lat, tok)
        self._moves += 1
        return Decision("MV_XY", min(confs) if confs else 1.0, wire, lat, tok, xy=(move["v"], move["h"]))


@dataclass
class TwoTierResult:
    episode: int
    seed: int
    policy: str
    planner: str
    success: bool
    decisions: int = 0
    motions: int = 0
    planner_calls: int = 0
    escalations: int = 0
    command_accuracy: float = float("nan")
    wall_time_s: float = 0.0
    fast_latency_s: list[float] = field(default_factory=list)
    planner_latency_s: list[float] = field(default_factory=list)
    planner_tokens: dict[str, int] = field(default_factory=lambda: {"input_tokens": 0, "output_tokens": 0})
    subtasks: list[str] = field(default_factory=list)
    missed_grasps: int = 0
    pre_moves: int = 0
    stop_reason: str = ""
    reset_s: float = 0.0  # the reset (to the begin pose), before wall_time_s starts
    return_s: float = 0.0  # the return to the begin pose after the episode, not in wall_time_s


class TwoTierRunner:
    """Runs one episode: planner queue -> fast policy steps, with auto-advance and failure detection.

    The harness returns to the planner only when the queue is empty or a failure is detected:
    missed_grasp (then it opens the gripper and rises to hover height first), stuck (max_steps),
    oscillating (alternating opposite moves) or escalated (the controller chose ESCALATE).

    A robot with saved poses (``go_pose``) starts every episode at the begin pose (the reset) and returns
    there at the end: after the task is complete it rises to travel height first; after a failure it returns
    only when it holds nothing (a held cube stays where it is). With an overhead guide the first mark starts
    in the background as soon as the arm rests at the begin pose, overlapping the rest of the reset.
    """

    OPPOSITE = {
        "MV_FWD": "MV_BACK",
        "MV_BACK": "MV_FWD",
        "MV_LEFT": "MV_RIGHT",
        "MV_RIGHT": "MV_LEFT",
        "MV_UP": "MV_DOWN",
        "MV_DOWN": "MV_UP",
    }

    def __init__(
        self,
        env: PickPlaceEnv,
        planner: Planner,
        policy: Any,
        max_decisions: int = 80,
        max_planner_calls: int = 12,
        conf_threshold: float = 0.0,
        log_dir: str | Path | None = None,
        save_images: bool = True,
        guide: Any = None,
    ):
        self.env, self.planner, self.policy = env, planner, policy
        self.guide = guide  # OverheadGuide for the goto_* subtasks
        self.max_decisions, self.max_planner_calls, self.conf_threshold = (
            max_decisions,
            max_planner_calls,
            conf_threshold,
        )
        self.log_dir = Path(log_dir) if log_dir else None
        self.save_images = save_images
        if self.log_dir:
            (self.log_dir / "images").mkdir(parents=True, exist_ok=True)

    def _observe(self) -> dict[str, np.ndarray]:
        need = (
            self.policy.needs_images or self.planner.name != "scripted" or (self.log_dir and self.save_images)
        )
        return self.env.render_all() if need else {}

    def _save(self, tag: str, images: dict[str, np.ndarray], sub: Subtask | None) -> dict[str, str]:
        if not (self.log_dir and self.save_images):
            return {}
        from PIL import Image

        files = {}
        for cam, img in images.items():
            if cam == "wrist":
                img = wrist_with_cross(self.env, img, cross_height(sub))
            p = self.log_dir / "images" / f"{tag}_{cam}.jpg"
            Image.fromarray(img).save(p, quality=85)
            files[cam] = str(p.relative_to(self.log_dir))
        return files

    def _oscillating(self, moves: list[str]) -> bool:
        m = moves[-4:]
        return len(m) == 4 and all(self.OPPOSITE.get(a) == b for a, b in zip(m, m[1:], strict=False))

    def run(self, episode: int, seed: int, reset: bool = True, keep_marks: bool = False) -> TwoTierResult:
        """One episode. ``keep_marks``: keep the guide's marks (and a pending mark started while idle)."""
        env = self.env
        if self.guide is not None and not keep_marks:
            self.guide.reset()
        t_reset = time.perf_counter()
        if reset:
            self._reset_env(seed)
        res = TwoTierResult(episode, seed, self.policy.name, self.planner.name, success=False)
        res.reset_s = time.perf_counter() - t_reset
        events: list[dict[str, Any]] = []
        misses: list[np.ndarray] = []
        log = open(self.log_dir / f"ep{episode:04d}.jsonl", "w", encoding="utf-8") if self.log_dir else None
        correct = 0
        t0 = time.perf_counter()

        def write(rec: dict[str, Any]) -> None:
            if log:
                log.write(json.dumps(_jsonable(rec), ensure_ascii=False) + "\n")

        def where() -> dict[str, Any]:
            rs = robot_state(env)
            return {
                "gripper_xy_cm": [round(float(v) * 100, 1) for v in env.tcp_pos[:2]],
                "gripper_height_cm": rs["gripper_height_cm"],
                "jaw": rs["jaw"],
            }

        def event(sub: Subtask, outcome: str, reason: str, steps: int, answer: Any = None) -> None:
            ev = {
                "after_decision": res.decisions,
                "subtask": sub.name,
                "goal": sub.goal,
                "outcome": outcome,
                "reason": reason,
                "steps": steps,
                "task_index": sub.task_index,
                **where(),
            }
            if answer is not None:
                ev["controller_last_answer"] = answer
            events.append(ev)
            write({"type": "event", **ev})

        def call_planner() -> tuple[Plan, list[Subtask]] | None:
            if res.planner_calls >= self.max_planner_calls:
                res.stop_reason = "max_planner_calls"
                return None
            images = self._observe()
            plan = self.planner.plan(env, images, events)
            res.planner_calls += 1
            res.planner_latency_s.append(plan.latency_s)
            for k, v in plan.usage.items():
                res.planner_tokens[k] = res.planner_tokens.get(k, 0) + v
            write(
                {
                    "type": "plan",
                    "decision": res.decisions,
                    "task_complete": plan.task_complete,
                    "scene": plan.scene,
                    "diagnosis": plan.diagnosis,
                    "pre_moves": plan.pre_moves,
                    "pre_move_step_cm": plan.pre_move_step_cm,
                    "subtasks": [asdict(s) for s in plan.subtasks],
                    "subtask": asdict(plan.subtask) if plan.subtask else None,
                    "latency_s": plan.latency_s,
                    "oracle_stage": oracle_stage(env) if getattr(env, "has_truth", True) else None,
                    "images": self._save(f"ep{episode:04d}_p{res.planner_calls:02d}", images, plan.subtask),
                }
            )
            # corrective moves chosen by the planner, executed without the fast layer
            step = Subtask("pre", "", "", [], step_cm=plan.pre_move_step_cm)
            for m in plan.pre_moves:
                if m in MOVES:
                    self._execute(m, step)
                    res.pre_moves += 1
            return plan, list(plan.subtasks)

        try:
            got = call_planner()
            while got is not None:
                plan, queue = got
                if plan.task_complete or not queue:
                    res.stop_reason = "planner_complete" if plan.task_complete else "empty_plan"
                    break
                failure = None
                while queue and failure is None:
                    sub = queue.pop(0)
                    res.subtasks.append(sub.name)
                    sub_steps, moves = 0, []
                    if sub.target:
                        set_object_names(sub.target, OBJ["bin"])  # the fast layer's prompts name this object
                        if self.guide is not None:
                            self.guide.set_targets(pick=sub.target)
                    if sub.name in GUIDED:
                        failure = self._guided(sub, res, event, write)
                        continue
                    if sub.name == "place_at":
                        failure = self._place(sub, res, event, write)
                        continue
                    images = self._observe()
                    while True:
                        if res.decisions >= self.max_decisions:
                            failure = "max_decisions"
                            break
                        truth = oracle_command(env, sub) if getattr(env, "has_truth", True) else ""
                        dec = self.policy.decide(env, sub, images)
                        res.decisions += 1
                        sub_steps += 1
                        res.fast_latency_s.append(dec.latency_s)
                        correct += int(dec.command == truth)
                        cmd = dec.command
                        detected = ""
                        if (
                            self.conf_threshold > 0
                            and dec.confidence < self.conf_threshold
                            and cmd != "ESCALATE"
                        ):
                            detected, cmd = f"low confidence {dec.confidence:.2f} for {cmd}", "ESCALATE"
                        elif (cmd in MOVES or cmd == "MV_XY") and sub_steps > sub.max_steps:
                            detected, cmd = f"no completion after {sub.max_steps} steps", "ESCALATE"
                        elif cmd in MOVES and self._oscillating([*moves, cmd]):
                            detected, cmd = f"alternating {cmd} / {self.OPPOSITE[cmd]}", "ESCALATE"
                        info = self._execute(cmd, sub, dec.step_scale, dec.xy)
                        if cmd in MOVES:
                            moves.append(cmd)
                        write(
                            {
                                "type": "step",
                                "decision": res.decisions,
                                "subtask": sub.name,
                                "sub_step": sub_steps,
                                "command": cmd,
                                "proposed": dec.command,
                                "truth": truth,
                                "confidence": dec.confidence,
                                "detected": detected,
                                "answers": dec.answers,
                                "latency_s": dec.latency_s,
                                "exec": info,
                                "robot_state": robot_state(env),
                                "truth_state": (
                                    {
                                        "tcp": env.tcp_pos,
                                        "cube": env.cube_pos,
                                        "bin": env.bin_pos,
                                        "held": env.held,
                                    }
                                    if getattr(env, "has_truth", True)
                                    else {"tcp": env.tcp_pos, "held": env.held}
                                ),
                                "images": self._save(f"ep{episode:04d}_d{res.decisions:03d}", images, sub),
                            }
                        )
                        images = self._observe()
                        if cmd == "SUBTASK_DONE" or cmd == "RELEASE":
                            break  # auto-advance to the next queued subtask
                        if cmd == "GRASP":
                            if env.held:
                                if self.guide is not None:
                                    self.guide.on_grasp(env)
                                break
                            # missed grasp: recover automatically, then let the planner rethink
                            xy = env.tcp_pos[:2].copy()
                            same = [i + 1 for i, p in enumerate(misses) if np.linalg.norm(p - xy) < 0.015]
                            misses.append(xy)
                            res.missed_grasps += 1
                            self._recover()
                            reason = (
                                "GRASP closed on nothing; harness opened the gripper and raised it to ~9 cm"
                            )
                            if same:
                                reason += f"; same spot as missed attempt(s) {same} — change the approach"
                            event(sub, "missed_grasp", reason, sub_steps, dec.answers)
                            failure = "missed_grasp"
                            break
                        if cmd == "ESCALATE":
                            res.escalations += 1
                            outcome = (
                                "escalated"
                                if not detected
                                else ("oscillating" if "alternating" in detected else "stuck")
                            )
                            event(
                                sub,
                                outcome,
                                detected or "the controller chose ESCALATE",
                                sub_steps,
                                dec.answers,
                            )
                            failure = outcome
                            break
                if failure == "max_decisions":
                    res.stop_reason = "max_decisions"
                    break
                if failure is None:
                    events.append({"after_decision": res.decisions, "outcome": "queue_finished", **where()})
                got = call_planner()
            res.wall_time_s = time.perf_counter() - t0
            self._finish(res, write)  # not after an exception: a failed motion leaves the robot where it is
        finally:
            if log:
                log.close()
            if self.guide is not None:
                self.guide.join()
        res.motions = env.step_count
        res.success = (
            env.is_success() if getattr(env, "has_truth", True) else False
        )  # real: judged by a person
        res.command_accuracy = correct / max(1, res.decisions)
        res.wall_time_s = res.wall_time_s or time.perf_counter() - t0
        return res

    def _reset_env(self, seed: int) -> None:
        """Reset the env; with an overhead guide, start the first mark once the arm rests at the begin pose.

        The first reset also measures the overhead map's scale (``OverheadGuide.calibrate``: the arm visits a few
        poses around the begin pose and comes back), unless the guide's map was given a scale."""
        env, guide = self.env, self.guide
        if guide is None or not hasattr(env, "overhead_frame"):
            env.reset(seed=seed)
            return

        def begin_mark() -> None:
            try:
                guide.start_background(env, kind="begin")
            except Exception as e:  # noqa: BLE001 - goto_cube then marks in the foreground
                print("  could not start the first overhead mark:", e, flush=True)

        if getattr(guide, "needs_calibration", False):
            env.reset(seed=seed)
            guide.calibrate(env)  # ends at the begin pose
            begin_mark()
        elif "on_begin" in inspect.signature(env.reset).parameters:
            env.reset(seed=seed, on_begin=begin_mark)
        else:
            env.reset(seed=seed)
            begin_mark()

    def _finish(self, res: TwoTierResult, write: Any) -> None:
        """Back to the begin pose after the episode (robots with saved poses only; the simulator stays)."""
        env = self.env
        if not hasattr(env, "go_pose"):
            return
        complete = res.stop_reason == "planner_complete"
        if not complete and env.held:
            write({"type": "finish", "action": "stay", "reason": f"{res.stop_reason} while holding the cube"})
            return
        t0 = time.perf_counter()
        env.precise = False  # travel moves: no sag correction needed
        try:
            # Straight up, level to above the begin pose, then the joint move. A joint move from a release over the
            # bin swept low and caught the bin's rim and the object just dropped in it.
            here = env.tcp_pos.copy()
            if here[2] < TRANSPORT_Z - 0.005:
                env._move_tcp(np.array([*here[:2], TRANSPORT_Z]))
            above = getattr(env, "pose_tcp", {}).get("begin")
            if above is not None and np.linalg.norm(above[:2] - here[:2]) > 0.03:
                env._move_tcp(np.array([*above[:2], max(TRANSPORT_Z, float(above[2]))]))
            env.go_pose("begin")
        except Exception as e:  # noqa: BLE001 - the episode's result stands; the robot stops where it is
            print("  return to the begin pose failed:", e, flush=True)
            write({"type": "finish", "action": "failed", "reason": str(e)})
            return
        finally:
            env.precise = True
        res.return_s = time.perf_counter() - t0
        write(
            {
                "type": "finish",
                "action": "begin",
                "reason": res.stop_reason,
                "seconds": round(res.return_s, 2),
            }
        )

    def _guided(self, sub: Subtask, res: TwoTierResult, event: Any, write: Any) -> str | None:
        """A coarse move by overhead marks; returns the failure outcome, or None."""
        if self.guide is None:
            event(sub, "escalated", "no overhead guide configured", 0)
            return "escalated"
        target = GUIDED[sub.name]
        z = RELEASE_Z if target == "bin" else (getattr(self.env, "observe_z", None) or Z_HOVER)
        t0 = time.perf_counter()
        self.env.precise = False  # travel: the wrist servo or the release corrects afterwards
        try:
            r = self.guide.goto(self.env, target, z, travel_z=TRANSPORT_Z if target == "bin" else None)
        finally:
            self.env.precise = True
        res.planner_latency_s.append(time.perf_counter() - t0)
        write(
            {
                "type": "guide",
                "decision": res.decisions,
                "subtask": sub.name,
                "ok": r.ok,
                "reason": r.reason,
                "iterations": r.iterations,
                "error_cm": round(r.error_cm * 100, 1),
                "log": r.log,
            }
        )
        if r.ok:
            return None
        event(sub, "escalated", f"overhead guide: {r.reason}", r.iterations)
        res.escalations += 1
        return "escalated"

    def _place(self, sub: Subtask, res: TwoTierResult, event: Any, write: Any) -> str | None:
        """place_at: into the bin (released from RELEASE_Z), here (set down on the table), or on another object
        (approach above it by marks, then set the held cube down on its top face)."""
        env, where = self.env, sub.place or "bin"
        if not env.held:
            event(sub, "escalated", "place_at: the gripper holds nothing", 0)
            return "escalated"
        t0 = time.perf_counter()
        try:
            if where == "here":
                z = Z_GRASP + PLACE_Z_MARGIN
            else:
                name = self.guide_bin_name if where == "bin" else where.removeprefix("on:")
                if self.guide is None:
                    event(sub, "escalated", "no overhead guide configured", 0)
                    return "escalated"
                self.guide.set_targets(place=name)
                env.precise = False
                r = self.guide.goto(env, "bin", RELEASE_Z, travel_z=TRANSPORT_Z)
                env.precise = True
                write(
                    {
                        "type": "guide",
                        "decision": res.decisions,
                        "subtask": sub.name,
                        "place": where,
                        "ok": r.ok,
                        "reason": r.reason,
                        "error_cm": round(r.error_cm * 100, 1),
                        "log": r.log,
                    }
                )
                if not r.ok:
                    event(sub, "escalated", f"overhead guide: {r.reason}", 1)
                    res.escalations += 1
                    return "escalated"
                z = None if where == "bin" else Z_GRASP + CUBE_EDGE + PLACE_Z_MARGIN
            if z is not None:
                env.precise = True  # set it down gently at the right height
                env.move_relative(np.array([0.0, 0.0, z - float(env.tcp_pos[2])]), from_measured=True)
            env.open_gripper()
            if z is not None:
                env.precise = False
                env.move_relative(np.array([0.0, 0.0, 0.03]))  # clear of the object before any travel
        finally:
            env.precise = True
        res.planner_latency_s.append(time.perf_counter() - t0)
        write({"type": "place", "where": where, "tcp": np.round(env.tcp_pos, 4).tolist(), "held": env.held})
        return None

    @property
    def guide_bin_name(self) -> str:
        # the bin the guide was set up with (not guessed from the names: a carrot listed before the bin was once
        # taken for the bin)
        return getattr(self.guide, "bin_name", "bin")

    def _recover(self) -> None:
        self.env.open_gripper()
        up = max(0.0, Z_HOVER - float(self.env.tcp_pos[2]))
        if up > 0.005:
            self.env.move_relative(np.array([0.0, 0.0, up]))

    def _execute(
        self, cmd: str, sub: Subtask, scale: float = 1.0, xy: tuple[float, float] | None = None
    ) -> dict[str, Any]:
        # only the descent to the grasp needs the gravity-sag correction; the servo re-checks after other moves
        self.env.precise = cmd == "MV_DOWN" and sub.name == "descend_grasp"
        try:
            return self._execute_move(cmd, sub, scale, xy)
        finally:
            self.env.precise = True

    def _execute_move(
        self, cmd: str, sub: Subtask, scale: float, xy: tuple[float, float] | None
    ) -> dict[str, Any]:
        env, s = self.env, sub.step_m * scale
        fwd, right = env.wrist_axes()
        if cmd == "MV_XY" and xy is not None:
            return env.move_relative(np.array([*(fwd * xy[0] + right * xy[1]), 0.0]))
        vec = {
            "MV_FWD": np.array([*(fwd * s), 0.0]),
            "MV_BACK": np.array([*(-fwd * s), 0.0]),
            "MV_RIGHT": np.array([*(right * s), 0.0]),
            "MV_LEFT": np.array([*(-right * s), 0.0]),
            "MV_UP": np.array([0.0, 0.0, s]),
            "MV_DOWN": np.array([0.0, 0.0, -s]),
        }
        if cmd == "MV_DOWN" and sub.name in ("descend_grasp", "release"):
            # From the measured tcp, not the last commanded target: after the fast (uncorrected) servo moves the arm
            # rests up to ~1.5 cm off its command, mostly low from sag. Adding the policy's step (measured z - grasp
            # height) to that higher command stopped the descent short (real_overhead2: 10.5 -> 3.7 cm, then a
            # second move to 3.1). This keeps the xy the wrist servo aligned and aims exactly at Z_GRASP (or, for the
            # release, at RELEASE_Z).
            return env.move_relative(vec[cmd], from_measured=True)
        if cmd in vec:
            return env.move_relative(vec[cmd])
        if cmd == "GRASP":
            return {"grasped": env.close_gripper()}
        if cmd == "RELEASE":
            env.open_gripper()
            return {}
        return {}


def summarize_twotier(results: list[TwoTierResult]) -> dict[str, Any]:
    if not results:
        return {}
    fast = [x for r in results for x in r.fast_latency_s]
    plan = [x for r in results for x in r.planner_latency_s]
    return {
        "policy": results[0].policy,
        "planner": results[0].planner,
        "episodes": len(results),
        "success_rate": float(np.mean([r.success for r in results])),
        "mean_decisions": float(np.mean([r.decisions for r in results])),
        "mean_planner_calls": float(np.mean([r.planner_calls for r in results])),
        "mean_escalations": float(np.mean([r.escalations for r in results])),
        "mean_missed_grasps": float(np.mean([r.missed_grasps for r in results])),
        "mean_pre_moves": float(np.mean([r.pre_moves for r in results])),
        "command_accuracy": float(np.mean([r.command_accuracy for r in results])),
        "fast_p50_ms": float(np.percentile(fast, 50) * 1000) if fast else 0.0,
        "planner_p50_s": float(np.percentile(plan, 50)) if plan else 0.0,
        "planner_tokens": {
            k: int(sum(r.planner_tokens.get(k, 0) for r in results))
            for k in ("input_tokens", "output_tokens")
        },
        "mean_wall_time_s": float(np.mean([r.wall_time_s for r in results])),
        "stop_reasons": {
            s: sum(r.stop_reason == s for r in results) for s in {r.stop_reason for r in results}
        },
    }


def _jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return [round(float(v), 4) for v in x.ravel()]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.bool_):
        return bool(x)
    return x


__all__ = [
    "COMMANDS",
    "SUBTASKS",
    "JevCommandPolicy",
    "JevPerceptionPolicy",
    "OpenAIPlanner",
    "OraclePolicy",
    "ScriptedPlanner",
    "Subtask",
    "TwoTierRunner",
    "oracle_command",
    "summarize_twotier",
]
