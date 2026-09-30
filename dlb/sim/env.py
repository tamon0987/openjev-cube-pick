"""Pick-and-place environment driven by discrete primitives.

The decision layer never outputs joint targets. It picks one *primitive* per
cycle (a ``choice`` question) and optionally answers side questions (``noul`` /
``score``). Execution is deterministic: IK + position control in MuJoCo.

Robot: the official ROBOTIS OMX (OMX-F follower) MJCF from
``ROBOTIS-GIT/robotis_mujoco_menagerie`` (Apache-2.0), vendored under
``assets/omx`` with a tcp site, fingertip sites and a wrist camera added.

Observation modalities:

* ``state_json``  – what text-only backends (TypeSafe Jev) see. Geometric,
  rounded, **no privileged flags** such as "holding" (that must be inferred).
* ``images``      – what image backends (djev / openjev) see: ``front`` and ``wrist``.
* ``privileged``  – simulator truth used only by the oracle policy and for labels.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from importlib import resources
from typing import Any

import mujoco
import numpy as np

from dlb.sim.ik import SiteIK

# OMX joint / actuator names (official MJCF). Joint5 is the wrist roll about the tool axis.
ARM_JOINTS = ["Joint1", "Joint2", "Joint3", "Joint4", "Joint5"]
ARM_ACTUATORS = ["Joint1", "Joint2", "Joint3", "Joint4", "Joint5"]
JAW_JOINT = "Gripper"  # hinge, 0 = closed, 1.745 = fully open; Gripper_mimic follows via equality
JAW_ACTUATOR = "Gripper"
GRIPPER_BODY = "link5"  # wrist link carrying the tcp site and the finger pivots

JAW_OPEN = 0.8  # rad -> ~7.5 cm between finger pads (plenty for the 3 cm cube)
JAW_CLOSED = 0.0
OPEN_MIN = 0.065  # aperture (m) above this counts as "open"
CLOSED_MAX = 0.006  # aperture (m) below this counts as "closed on nothing"

# tcp sits ~2.2 cm behind the fingertips along the tool axis
Z_GRASP = 0.028
Z_HOVER = 0.090
Z_TRAVEL = 0.120
# workspace clamp for relative moves (tcp): height, radius from the base axis, azimuth
Z_MAX = 0.160
R_MIN, R_MAX = 0.10, 0.26
TH_MAX = np.deg2rad(70)

PRIMITIVES: dict[str, str] = {
    "hover_object": "Move the gripper to 6 cm directly above the red cube (gripper stays as it is).",
    "descend": "Lower the gripper straight down to grasp height at its current xy position.",
    "grasp": "Close the gripper on whatever is between the fingers.",
    "lift": "Raise the gripper straight up to the safe travel height.",
    "hover_bin": "Move the gripper at travel height to directly above the centre of the blue bin.",
    "release": "Open the gripper (drops the cube if it is held).",
    "home": "Return the arm to its home pose without changing the gripper.",
    "done": "Declare the task complete: the cube rests inside the bin and the gripper is open. No motion.",
}

TASK_TEXT = "Pick up the red cube and place it inside the blue bin, then declare done."


@dataclass
class Observation:
    state_json: dict[str, Any]
    privileged: dict[str, Any]
    images: dict[str, np.ndarray] = field(default_factory=dict)
    step: int = 0


class PickPlaceEnv:
    def __init__(
        self,
        xml_path: str | None = None,
        image_size: int = 320,
        cameras: tuple[str, ...] = ("front", "wrist"),
        max_steps: int = 15,
        render: bool = True,
        seed: int = 0,
    ):
        if xml_path is None:
            xml_path = str(resources.files("dlb.sim").joinpath("assets/omx/omx_scene.xml"))
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.image_size = image_size
        self.cameras = cameras
        self.max_steps = max_steps
        self.render_enabled = render
        self.rng = np.random.default_rng(seed)
        self._renderer: mujoco.Renderer | None = None

        m = self.model
        self.tcp_site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "tcp")
        self.bin_site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "bin_center")
        self.cube_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "cube")
        self.bin_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "bin")
        self.gripper_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, GRIPPER_BODY)
        self.finger_a = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "finger_a")
        self.finger_b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "finger_b")
        self.cube_qadr = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")]
        self.cube_dadr = m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")]
        self.jaw_qadr = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, JAW_JOINT)]
        self.weld_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, "grasp_weld")
        self.arm_act = np.array(
            [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a) for a in ARM_ACTUATORS]
        )
        self.jaw_act = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, JAW_ACTUATOR)
        self.arm_qadr = np.array(
            [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in ARM_JOINTS]
        )
        self.home_key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
        self.home_ctrl = m.key_ctrl[self.home_key].copy()
        self.bin_inner_half = 0.045 - 0.003  # floor half-size minus wall thickness
        self.bin_wall_top = 0.040

        self.ik = SiteIK(m, "tcp", ARM_JOINTS)
        self.step_count = 0
        self.history: list[str] = []
        self.done_declared = False
        self.last_exec: dict[str, Any] = {}

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #
    def reset(self, seed: int | None = None) -> Observation:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        m, d = self.model, self.data
        mujoco.mj_resetDataKeyframe(m, d, self.home_key)
        d.eq_active[self.weld_id] = 0
        d.ctrl[:] = self.home_ctrl

        # sample cube and bin poses in a reachable arc in front of the base
        cube_xy = self._sample_xy()
        for _ in range(100):
            bin_xy = self._sample_xy()
            if np.linalg.norm(bin_xy - cube_xy) > 0.13:
                break
        yaw = self.rng.uniform(-np.pi, np.pi)
        d.qpos[self.cube_qadr : self.cube_qadr + 3] = [cube_xy[0], cube_xy[1], 0.0151]
        d.qpos[self.cube_qadr + 3 : self.cube_qadr + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        m.body_pos[self.bin_body][:2] = bin_xy
        mujoco.mj_forward(m, d)
        self._settle(0.5)

        self.step_count = 0
        self.history = []
        self.done_declared = False
        self.last_exec = {}
        self._tcp_cmd = None  # last commanded tcp target (see move_relative)
        return self.observe()

    def _sample_xy(self) -> np.ndarray:
        r = self.rng.uniform(0.14, 0.225)
        th = self.rng.uniform(-np.deg2rad(50), np.deg2rad(50))
        return np.array([r * np.cos(th), r * np.sin(th)])

    # ------------------------------------------------------------------ #
    # physics helpers
    # ------------------------------------------------------------------ #
    def _settle(self, seconds: float) -> None:
        n = int(seconds / self.model.opt.timestep)
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)

    def _move_tcp(self, target: np.ndarray, seg: float = 0.03, timeout: float = 1.2) -> float:
        """Straight-line Cartesian move via IK waypoints. Returns final position error."""
        self._tcp_cmd = np.asarray(target, float).copy()
        start = self.tcp_pos.copy()
        dist = float(np.linalg.norm(target - start))
        n = max(1, int(np.ceil(dist / seg)))
        err = 0.0
        for i in range(1, n + 1):
            wp = start + (target - start) * (i / n)
            q, err = self.ik.solve(self.data, wp)
            self.data.ctrl[self.arm_act] = q
            self._step_until(q, timeout=timeout)
        return float(np.linalg.norm(self.tcp_pos - target))

    def _step_until(self, q_target: np.ndarray, timeout: float, tol: float = 0.01) -> None:
        n = int(timeout / self.model.opt.timestep)
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)
            if np.max(np.abs(self.data.qpos[self.arm_qadr] - q_target)) < tol:
                break
        self._settle(0.05)

    def _gripper_yaw(self, data: mujoco.MjData | None = None) -> float:
        """World yaw of the finger opening axis (tcp site y-axis)."""
        d = data or self.data
        R = d.site_xmat[self.tcp_site].reshape(3, 3)
        return float(np.arctan2(R[1, 1], R[0, 1]))

    def _align_roll_to_cube(self) -> None:
        """Rotate the wrist roll so the finger axis is parallel to a cube face normal."""
        q = self.data.xquat[self.cube_body]
        cube_yaw = float(np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2)))
        cur = self._gripper_yaw()
        delta = (cube_yaw - cur + np.pi / 4) % (np.pi / 2) - np.pi / 4  # wrap to (-45deg, 45deg]
        roll_idx = self.arm_qadr[4]
        scratch = self.ik._scratch
        best_q, best_err = None, np.inf
        for sign in (1.0, -1.0):
            scratch.qpos[:] = self.data.qpos
            cand = float(np.clip(self.data.qpos[roll_idx] + sign * delta, self.ik.lo[4], self.ik.hi[4]))
            scratch.qpos[roll_idx] = cand
            mujoco.mj_kinematics(self.model, scratch)
            err = abs((cube_yaw - self._gripper_yaw(scratch) + np.pi / 4) % (np.pi / 2) - np.pi / 4)
            if err < best_err:
                best_q, best_err = cand, err
        q_target = self.data.qpos[self.arm_qadr].copy()
        q_target[4] = best_q
        self.data.ctrl[self.arm_act] = q_target
        self._step_until(q_target, timeout=0.8)

    def _set_jaw(self, ctrl: float, seconds: float = 0.4) -> None:
        self.data.ctrl[self.jaw_act] = ctrl
        self._settle(seconds)

    def _activate_weld(self) -> None:
        m, d = self.model, self.data
        p1, R1 = d.xpos[self.gripper_body], d.xmat[self.gripper_body].reshape(3, 3)
        p2, q2 = d.xpos[self.cube_body], d.xquat[self.cube_body]
        rel_pos = R1.T @ (p2 - p1)
        q1 = d.xquat[self.gripper_body]
        q1_inv = np.array([q1[0], -q1[1], -q1[2], -q1[3]])
        rel_quat = np.zeros(4)
        mujoco.mju_mulQuat(rel_quat, q1_inv, q2)
        m.eq_data[self.weld_id][0:3] = 0.0  # anchor
        m.eq_data[self.weld_id][3:6] = rel_pos
        m.eq_data[self.weld_id][6:10] = rel_quat
        m.eq_data[self.weld_id][10] = 1.0
        d.eq_active[self.weld_id] = 1

    # ------------------------------------------------------------------ #
    # state accessors
    # ------------------------------------------------------------------ #
    @property
    def tcp_pos(self) -> np.ndarray:
        return self.data.site_xpos[self.tcp_site]

    @property
    def cube_pos(self) -> np.ndarray:
        return self.data.xpos[self.cube_body]

    @property
    def bin_pos(self) -> np.ndarray:
        return self.data.site_xpos[self.bin_site]

    @property
    def aperture(self) -> float:
        """Distance between the two finger-pad sites (m)."""
        return float(np.linalg.norm(self.data.site_xpos[self.finger_a] - self.data.site_xpos[self.finger_b]))

    @property
    def held(self) -> bool:
        return bool(self.data.eq_active[self.weld_id])

    def cube_in_bin(self) -> bool:
        c, b = self.cube_pos, self.bin_pos
        inside_xy = np.all(np.abs(c[:2] - b[:2]) < self.bin_inner_half - 0.012)
        return bool(inside_xy and c[2] < self.bin_wall_top and not self.held)

    def is_success(self) -> bool:
        return self.cube_in_bin() and self.aperture > OPEN_MIN

    def cube_between_fingers(self) -> bool:
        """Cube centre inside the pad region, expressed in the tcp site frame
        (x: across the finger width, y: finger opening axis, z: tool axis)."""
        R = self.data.site_xmat[self.tcp_site].reshape(3, 3)
        rel = R.T @ (self.cube_pos - self.tcp_pos)
        return bool(abs(rel[0]) < 0.016 and abs(rel[1]) < 0.025 and abs(rel[2]) < 0.028)

    # ------------------------------------------------------------------ #
    # primitives
    # ------------------------------------------------------------------ #
    def execute(self, primitive: str) -> dict[str, Any]:
        if primitive not in PRIMITIVES:
            raise ValueError(f"unknown primitive {primitive!r}")
        info: dict[str, Any] = {"primitive": primitive}
        tcp = self.tcp_pos.copy()
        if primitive == "hover_object":
            tgt = np.array([*self.cube_pos[:2], Z_HOVER])
            if tcp[2] < Z_HOVER - 0.01 and np.linalg.norm(tcp[:2] - tgt[:2]) > 0.01:
                self._move_tcp(np.array([*tcp[:2], Z_HOVER]))
            info["ik_err"] = self._move_tcp(tgt)
            self._align_roll_to_cube()
        elif primitive == "descend":
            info["ik_err"] = self._move_tcp(np.array([*tcp[:2], Z_GRASP]))
        elif primitive == "grasp":
            self._set_jaw(JAW_CLOSED, 0.5)
            if self.cube_between_fingers() and not self.held:
                self._activate_weld()
                self._settle(0.1)
            info["grasped"] = self.held
        elif primitive == "lift":
            info["ik_err"] = self._move_tcp(np.array([*tcp[:2], Z_TRAVEL]))
        elif primitive == "hover_bin":
            tgt = np.array([*self.bin_pos[:2], Z_TRAVEL])
            if tcp[2] < Z_TRAVEL - 0.01:
                self._move_tcp(np.array([*tcp[:2], Z_TRAVEL]))
            info["ik_err"] = self._move_tcp(tgt)
        elif primitive == "release":
            self.data.eq_active[self.weld_id] = 0
            self._set_jaw(JAW_OPEN, 0.3)
            self._settle(0.6)
        elif primitive == "home":
            self.data.ctrl[self.arm_act] = self.home_ctrl[self.arm_act]
            self._step_until(self.home_ctrl[self.arm_act], timeout=1.5)
        elif primitive == "done":
            self.done_declared = True
        self.step_count += 1
        self.history.append(primitive)
        info["success"] = self.is_success()
        self.last_exec = info
        return info

    def episode_over(self) -> bool:
        return self.done_declared or self.step_count >= self.max_steps

    # ------------------------------------------------------------------ #
    # low-level motion (two-tier harness: relative moves instead of primitives)
    # ------------------------------------------------------------------ #
    def wrist_axes(self) -> tuple[np.ndarray, np.ndarray]:
        """Horizontal unit vectors for "toward the top" and "toward the right" of the wrist image."""
        R = self.data.cam_xmat[self._cam_id("wrist")].reshape(3, 3)
        axes = []
        for v in (R[:, 1], R[:, 0]):
            h = np.array([v[0], v[1]])
            axes.append(h / max(np.linalg.norm(h), 1e-9))
        return axes[0], axes[1]

    def move_relative(self, delta: np.ndarray, from_measured: bool = False) -> dict[str, Any]:
        """Straight-line tcp move by ``delta`` (m, world frame), clamped to a safe workspace.

        The move is relative to the last commanded target, not the measured tcp: the arm settles a few mm
        short of each target (IK tolerance, sag), and adding deltas to the measured position let that error
        accumulate (about 2 mm of height lost per horizontal step). ``from_measured`` moves from the measured
        tcp instead, for a single move to an absolute height computed from the measured one (the descent).
        """
        tcp = self.tcp_pos.copy()
        cmd = None if from_measured else getattr(self, "_tcp_cmd", None)
        base = cmd if cmd is not None and np.linalg.norm(cmd - tcp) < 0.02 else tcp
        tgt = base + np.asarray(delta, float)
        tgt[2] = float(np.clip(tgt[2], Z_GRASP, Z_MAX))
        r, th = float(np.hypot(tgt[0], tgt[1])), float(np.arctan2(tgt[1], tgt[0]))
        r, th = float(np.clip(r, R_MIN, R_MAX)), float(np.clip(th, -TH_MAX, TH_MAX))
        tgt[:2] = [r * np.cos(th), r * np.sin(th)]
        clamped = bool(np.linalg.norm(tgt - (base + np.asarray(delta, float))) > 1e-4)
        err = self._move_tcp(tgt)
        self.step_count += 1
        return {"ik_err": err, "clamped": clamped}

    def close_gripper(self) -> bool:
        self._set_jaw(JAW_CLOSED, 0.5)
        if self.cube_between_fingers() and not self.held:
            self._activate_weld()
            self._settle(0.1)
        self.step_count += 1
        return self.held

    def open_gripper(self) -> None:
        # Open the fingers before dropping the weld. Without the primitive's roll alignment the fingers
        # can close into a rotated cube; releasing the weld first lets that penetration fling the cube.
        self._set_jaw(JAW_OPEN, 0.3)
        self.data.eq_active[self.weld_id] = 0
        self.data.qvel[self.cube_dadr : self.cube_dadr + 6] = 0
        self._settle(0.6)
        self.step_count += 1

    def wrist_pixel(self, point: np.ndarray, size: int | None = None) -> tuple[float, float] | None:
        """Project a world point into the wrist image (pixels), or None if behind the camera."""
        return self.camera_pixel("wrist", point, size)

    def camera_pixel(
        self, camera: str, point: np.ndarray, size: int | None = None
    ) -> tuple[float, float] | None:
        """Project a world point into a camera image (pixels), or None if behind the camera."""
        cid = self._cam_id(camera)
        size = size or self.image_size
        R = self.data.cam_xmat[cid].reshape(3, 3)
        v = R.T @ (np.asarray(point, float) - self.data.cam_xpos[cid])
        if v[2] >= -1e-6:
            return None
        f = 0.5 * size / np.tan(np.deg2rad(self.model.cam_fovy[cid]) / 2)
        return size / 2 + f * v[0] / -v[2], size / 2 - f * v[1] / -v[2]

    def _cam_id(self, name: str) -> int:
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, name)

    # ------------------------------------------------------------------ #
    # observation
    # ------------------------------------------------------------------ #
    def observe(self, render: bool | None = None) -> Observation:
        cm = lambda v: [round(float(x) * 100, 1) for x in np.atleast_1d(v)]  # noqa: E731
        tcp, cube, binp = self.tcp_pos, self.cube_pos, self.bin_pos
        ap = self.aperture
        gstate = "open" if ap > OPEN_MIN else ("closed" if ap < CLOSED_MAX else "partially_closed")
        state_json = {
            "task": TASK_TEXT,
            "units": "centimetres in the world frame; z is up; the table surface is z=0",
            "gripper": {
                "robot": "ROBOTIS OMX",
                "tcp_xyz": cm(tcp),
                "aperture_cm": round(ap * 100, 1),
                "jaw_state": gstate,
            },
            "objects": {
                "red_cube": {"xyz": cm(cube), "edge_cm": 3.0},
                "blue_bin": {
                    "center_xyz": cm(binp),
                    "inner_half_width_cm": round(self.bin_inner_half * 100, 1),
                    "wall_height_cm": round(self.bin_wall_top * 100, 1),
                },
            },
            "relative": {
                "cube_minus_tcp_xyz": cm(cube - tcp),
                "bin_minus_tcp_xyz": cm(binp - tcp),
                "cube_tcp_xy_distance": round(float(np.linalg.norm(cube[:2] - tcp[:2])) * 100, 1),
                "cube_bin_xy_distance": round(float(np.linalg.norm(cube[:2] - binp[:2])) * 100, 1),
            },
            "reference_heights_cm": {
                "grasp": Z_GRASP * 100,
                "hover": Z_HOVER * 100,
                "travel": Z_TRAVEL * 100,
            },
            "history": {"step": self.step_count, "last_actions": self.history[-4:]},
        }
        privileged = {
            "held": self.held,
            "cube_in_bin": self.cube_in_bin(),
            "aligned": bool(np.linalg.norm(cube[:2] - tcp[:2]) < 0.012),
            "tcp": tcp.tolist(),
            "cube": cube.tolist(),
            "bin": binp.tolist(),
            "aperture": ap,
            "success": self.is_success(),
        }
        images: dict[str, np.ndarray] = {}
        if render if render is not None else self.render_enabled:
            images = self.render_all()
        return Observation(state_json=state_json, privileged=privileged, images=images, step=self.step_count)

    def render_all(self) -> dict[str, np.ndarray]:
        if self._renderer is None:
            if "MUJOCO_GL" not in os.environ:
                os.environ.setdefault("MUJOCO_GL", "egl")
            self._renderer = mujoco.Renderer(self.model, self.image_size, self.image_size)
        out = {}
        for cam in self.cameras:
            self._renderer.update_scene(self.data, camera=cam)
            out[cam] = self._renderer.render().copy()
        return out

    # ------------------------------------------------------------------ #
    # oracle (ground truth) policy and labels
    # ------------------------------------------------------------------ #
    def oracle_action(self) -> str:
        tcp, cube, binp = self.tcp_pos, self.cube_pos, self.bin_pos
        ap = self.aperture
        if self.cube_in_bin():
            return "done" if ap > OPEN_MIN else "release"
        if self.held:
            if tcp[2] < Z_TRAVEL - 0.015:
                return "lift"
            if np.linalg.norm(tcp[:2] - binp[:2]) > 0.015:
                return "hover_bin"
            return "release"
        if ap < OPEN_MIN:  # not fully open (e.g. closed on nothing) -> open first
            return "release"
        dxy = np.linalg.norm(tcp[:2] - cube[:2])
        if dxy > 0.012:
            return "hover_object"
        if tcp[2] > Z_GRASP + 0.015:
            return "descend"
        return "grasp"

    def oracle_labels(self) -> dict[str, Any]:
        """Ground truth for every question the harness may ask (see harness.prompts)."""
        tcp, cube, binp = self.tcp_pos, self.cube_pos, self.bin_pos
        held, in_bin = self.held, self.cube_in_bin()
        aligned = bool(np.linalg.norm(cube[:2] - tcp[:2]) < 0.012)
        above_bin = held and np.linalg.norm(tcp[:2] - binp[:2]) < 0.015
        if in_bin:
            progress = 4
        elif above_bin:
            progress = 3
        elif held:
            progress = 2
        elif aligned:
            progress = 1
        else:
            progress = 0
        return {
            "next_action": self.oracle_action(),
            "holding": "yes" if held else "no",
            "aligned": "yes" if aligned else "no",
            "task_complete": "yes" if in_bin and self.aperture > OPEN_MIN else "no",
            "progress": progress,
        }
