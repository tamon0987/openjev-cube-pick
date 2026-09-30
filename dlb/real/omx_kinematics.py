# From the author's port of Show-Harness to the OMX-F (core/omx/kinematics.py), Apache-2.0.
# Upstream: https://github.com/showlab/Show-Harness ; see NOTICE.
"""ROBOTIS OMX-F (5-DoF + gripper) kinematics: FK, analytic IK, joint limits.

Pure math (numpy/scipy only) so the interpreter, the mock robot and the tests share one
model. The chain is taken verbatim from the official URDF
(``open_manipulator_description/urdf/omx_f/omx_f.urdf``, ROBOTIS ``open_manipulator``
repo, ``jazzy`` branch); joint values are the URDF joint angles, which are what the
DYNAMIXEL ticks map to 1:1 through ``rad = (tick - 2048) * 2*pi / 4096`` (verified
against ROBOTIS's own ``dynamixel_hardware_interface`` model files).

Chain (each row: parent-frame offset to the joint origin, then the joint axis)::

    joint1  (-0.01125, 0, 0.034)   Z   base yaw           (DXL 11, extended-position mode)
    joint2  (0, 0, 0.0635)         Y   shoulder pitch     (DXL 12)
    joint3  (0.0415, 0, 0.11315)   Y   elbow pitch        (DXL 13)  <- link2 has a 20 deg bend
    joint4  (0.162, 0, 0)          Y   wrist pitch        (DXL 14)
    joint5  (0.0287, 0, 0)         X   wrist roll         (DXL 15)
    ee      (0.09193, -0.0016, 0)      end_effector_link  (between the fingertips)

At q = 0 the upper arm points up (tilted ~20 deg forward), the forearm and the tool point
horizontally forward (+X). Positive q2/q3/q4 pitch the following link DOWN (rotation about
+Y takes +X toward -Z). The TOOL AXIS -- the gripper's approach direction -- is the
end-effector frame's +X, not +Z as on the Franka/Piper flanges; the OMX interpreter sets
``TOOL_AXIS = (1, 0, 0)`` accordingly.

Why analytic: joints 2-4 are parallel (a planar 3R chain in the vertical "arm plane"
that joint 1 swings around), joint 5 rolls about the tool axis. So a Cartesian target has
exactly one free orientation family: the tool's heading is pinned to the arm plane
(= joint 1), while its pitch and its roll about itself are free. :meth:`ik_bounded`
solves position EXACTLY and realizes the requested orientation as closely as the
mechanism allows, reporting the residual as ``ori_deviation_rad`` -- a 5-DoF arm cannot
do better, so the interpreter treats that residual as information, not failure.

The two tiny non-planar offsets (joint 1 axis 11.25 mm behind the base origin, the
1.6 mm lateral EE offset) are handled exactly: the first by solving in cylindrical
coordinates about the joint-1 axis, the second by a 3-iteration fixed-point correction
against the full FK.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R

OMX_DOF = 5

# --- URDF geometry (meters) --------------------------------------------------------
J1_XYZ = np.array([-0.01125, 0.0, 0.034])
J2_XYZ = np.array([0.0, 0.0, 0.0635])
J3_XYZ = np.array([0.0415, 0.0, 0.11315])
J4_XYZ = np.array([0.162, 0.0, 0.0])
J5_XYZ = np.array([0.0287, 0.0, 0.0])
EE_XYZ = np.array([0.09193, -0.0016, 0.0])

# Planar constants derived from the chain (see the module docstring).
L2 = float(np.hypot(J3_XYZ[0], J3_XYZ[2]))          # 0.12052 m, joint2 -> joint3
BETA2 = float(math.atan2(J3_XYZ[2], J3_XYZ[0]))      # 69.9 deg: link2 elevation at q2 = 0
L3 = float(J4_XYZ[0])                                # 0.162 m,   joint3 -> joint4
L4 = float(J5_XYZ[0] + EE_XYZ[0])                    # 0.12063 m, joint4 -> fingertips (along tool)
J2_HEIGHT = float(J1_XYZ[2] + J2_XYZ[2])             # 0.0975 m, joint2 axis height above base

# Joint limits (rad). joint2/joint3 are the firmware position limits ROBOTIS ships in the
# OMX-F ros2_control description (DXL Min/Max Position Limit 830..3129 and 1024..3140
# ticks); joint1 is an extended-position motor (no firmware limit) bounded here to one
# turn for cable sanity; joint4/joint5 are full-turn motors bounded just short of a
# half-turn (the wrist collides with the forearm well before that -- keep an eye on it).
JOINT_LIMITS_RAD = np.array(
    [
        [-math.pi, math.pi],
        [(830 - 2048) * 2 * math.pi / 4096, (3129 - 2048) * 2 * math.pi / 4096],
        [(1024 - 2048) * 2 * math.pi / 4096, (3140 - 2048) * 2 * math.pi / 4096],
        [-3.0, 3.0],
        [-math.pi, math.pi],
    ]
)

TICKS_PER_TURN = 4096
TICK_ZERO = 2048


def tick_to_rad(tick: float) -> float:
    return (float(tick) - TICK_ZERO) * (2.0 * math.pi / TICKS_PER_TURN)


def rad_to_tick(rad: float) -> int:
    return int(round(float(rad) * (TICKS_PER_TURN / (2.0 * math.pi)) + TICK_ZERO))


def _rz(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _ry(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rx(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def wrap_pi(a: float) -> float:
    return (float(a) + math.pi) % (2.0 * math.pi) - math.pi


def unwrap_near(a: float, ref: float) -> float:
    """Shift ``a`` by multiples of 2*pi so it is nearest ``ref`` (continuity for q1/q5)."""
    return float(ref) + wrap_pi(float(a) - float(ref))


class OmxKinematics:
    """FK / analytic IK for the OMX-F chain described in the module docstring."""

    def __init__(self, joint_limits_rad: Optional[np.ndarray] = None) -> None:
        self.joint_limits = (
            JOINT_LIMITS_RAD.copy()
            if joint_limits_rad is None
            else np.asarray(joint_limits_rad, dtype=float).reshape(OMX_DOF, 2)
        )

    # -- forward ------------------------------------------------------------
    def fk(self, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """End-effector (fingertip) position [m] and rotation matrix in the base frame."""
        q = np.asarray(q, dtype=float).reshape(-1)[:OMX_DOF]
        r1 = _rz(q[0])
        r2 = r1 @ _ry(q[1])
        r3 = r2 @ _ry(q[2])
        r4 = r3 @ _ry(q[3])
        r5 = r4 @ _rx(q[4])
        p = J1_XYZ + r1 @ J2_XYZ + r2 @ J3_XYZ + r3 @ J4_XYZ + r4 @ J5_XYZ + r5 @ EE_XYZ
        return p, r5

    def fk_pose7(self, q: np.ndarray) -> np.ndarray:
        """FK as the harness pose7 ``[x, y, z, qx, qy, qz, qw]``."""
        p, rot = self.fk(q)
        return np.concatenate([p, R.from_matrix(rot).as_quat()])

    def tool_axis(self, q: np.ndarray) -> np.ndarray:
        """Unit approach direction of the gripper (base frame)."""
        _, rot = self.fk(q)
        return rot[:, 0]

    def tool_pitch_down_rad(self, q: np.ndarray) -> float:
        """Tool pitch measured DOWN from horizontal (pi/2 = pointing straight down)."""
        q = np.asarray(q, dtype=float).reshape(-1)
        return float(q[1] + q[2] + q[3])

    # -- limits ----------------------------------------------------------------
    def within_limits(self, q: np.ndarray, margin_rad: float = 0.0) -> bool:
        q = np.asarray(q, dtype=float).reshape(-1)[:OMX_DOF]
        lo = self.joint_limits[:, 0] + margin_rad
        hi = self.joint_limits[:, 1] - margin_rad
        return bool(np.all(q >= lo) and np.all(q <= hi))

    def clamp(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=float).reshape(-1)[:OMX_DOF]
        return np.clip(q, self.joint_limits[:, 0], self.joint_limits[:, 1])

    def limit_margins_deg(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=float).reshape(-1)[:OMX_DOF]
        return np.degrees(np.minimum(q - self.joint_limits[:, 0], self.joint_limits[:, 1] - q))

    # -- inverse ---------------------------------------------------------------
    @staticmethod
    def _planar_2r(rw: float, zw: float, elbow_sign: float) -> Optional[Tuple[float, float]]:
        """Solve joint2/joint3 so the wrist center (joint 4) sits at planar ``(rw, zw)``
        relative to joint 2. ``elbow_sign`` picks the branch. Returns ``(q2, q3)``."""
        d2 = rw * rw + zw * zw
        c = (d2 - L2 * L2 - L3 * L3) / (2.0 * L2 * L3)
        # A few mm beyond full extension is the straight-arm singularity (or the planar
        # approximation ignoring the 1.6 mm fingertip offset), not "unreachable": clamp
        # it -- the fixed-point correction and the caller's 1 mm position check decide.
        if c < -1.0 - 1e-9 or c > 1.05:
            return None
        c = max(-1.0, min(1.0, c))
        delta = elbow_sign * math.acos(c)  # relative turn from link2 to link3 (elevation sense)
        # Elevation of link2 = direction to the wrist minus the interior offset.
        e2 = math.atan2(zw, rw) - math.atan2(L3 * math.sin(delta), L2 + L3 * math.cos(delta))
        q2 = BETA2 - e2
        q3 = -delta - BETA2
        return q2, q3

    def _solve_planar(
        self,
        target_pos: np.ndarray,
        tool_axis: Optional[np.ndarray],
        pitch_seed: float,
        roll_seed: float,
        target_rot: Optional[np.ndarray],
        q1_ref: float,
        back: bool,
        elbow_sign: float,
    ) -> Optional[np.ndarray]:
        """One analytic solution for a (front/back, elbow) branch, corrected for the small
        non-planar offsets by a fixed-point iteration against the full FK.

        ``back`` selects the arm plane pointing AWAY from the target (the arm reaching
        over its own base, joint 1 turned half a turn); ``elbow_sign`` the elbow branch.
        The tool pitch is taken from ``tool_axis`` (its elevation is preserved exactly,
        only the forward/backward sense follows the branch); ``None`` keeps ``pitch_seed``.
        """
        target = np.asarray(target_pos, dtype=float).reshape(3)
        p_eff = target.copy()
        q = None
        for _ in range(6):
            dx = p_eff[0] - J1_XYZ[0]
            dy = p_eff[1] - J1_XYZ[1]
            heading = math.atan2(dy, dx) + (math.pi if back else 0.0)
            q1 = unwrap_near(heading, q1_ref)
            r = math.hypot(dx, dy) * (-1.0 if back else 1.0)
            z = p_eff[2] - J2_HEIGHT
            if tool_axis is not None:
                t = np.asarray(tool_axis, dtype=float)
                h_xy = math.hypot(float(t[0]), float(t[1]))
                # Sense of the tool along the arm plane's forward direction.
                fwd = float(t[0]) * math.cos(heading) + float(t[1]) * math.sin(heading)
                sense = -1.0 if fwd < 0.0 else 1.0
                pitch_down = math.atan2(-float(t[2]), sense * h_xy)
            else:
                pitch_down = pitch_seed
            rw = r - L4 * math.cos(pitch_down)
            zw = z + L4 * math.sin(pitch_down)
            sol = self._planar_2r(rw, zw, elbow_sign)
            if sol is None:
                return None
            q2, q3 = sol
            q4 = pitch_down - q2 - q3
            # Roll: best fit of the requested orientation within the reachable family
            # Rz(q1) Ry(pitch) Rx(roll)  (see the module docstring).
            if target_rot is not None:
                m = (_rz(q1) @ _ry(pitch_down)).T @ np.asarray(target_rot, dtype=float)
                q5 = math.atan2(m[2, 1] - m[1, 2], m[1, 1] + m[2, 2])
            else:
                q5 = roll_seed
            q5 = unwrap_near(q5, roll_seed)
            q = np.array([q1, q2, q3, wrap_pi(q4), q5])
            p_fk, _ = self.fk(q)
            err = target - p_fk
            if float(np.linalg.norm(err)) < 2e-5:
                break
            p_eff = p_eff + err  # the offsets are tiny; the correction converges in 2-3 steps
        return q

    def _candidates(
        self,
        target_pos: np.ndarray,
        tool_axis: Optional[np.ndarray],
        target_rot: Optional[np.ndarray],
        seed: np.ndarray,
        pos_tol_m: float,
    ) -> list:
        pitch_seed = self.tool_pitch_down_rad(seed)
        out = []
        for back in (False, True):
            for elbow_sign in (1.0, -1.0):
                q = self._solve_planar(
                    target_pos, tool_axis, pitch_seed, float(seed[4]), target_rot,
                    float(seed[0]), back, elbow_sign,
                )
                if q is None or not np.all(np.isfinite(q)):
                    continue
                if not self.within_limits(q):
                    continue
                p_fk, _ = self.fk(q)
                if float(np.linalg.norm(p_fk - np.asarray(target_pos, dtype=float))) > pos_tol_m:
                    continue
                out.append(q)
        return out

    def ik_bounded(
        self,
        target_pos: np.ndarray,
        target_rot: np.ndarray,
        q_seed: np.ndarray,
        max_ori_dev_rad: float = math.radians(90.0),
        pos_tol_m: float = 1e-3,
    ) -> Tuple[Optional[np.ndarray], float]:
        """Position-exact IK with the orientation realized as closely as the 5-DoF chain
        allows. Returns ``(joints, ori_deviation_rad)`` or ``(None, 0.0)`` when the
        position is unreachable / outside the joint limits (the true workspace boundary).

        The requested orientation fixes the tool PITCH (from its approach axis) and the
        ROLL about the tool (best-fit); the tool HEADING is whatever the arm plane at that
        position dictates. ``ori_deviation_rad`` is the angle between the requested and the
        achieved orientation; ``max_ori_dev_rad`` above it makes the solve fail (the
        interpreter passes a large budget and merely reports the deviation, because on this
        arm the residual is structural, not a sign of a bad target).

        All four branches (arm plane facing the target or reaching back over the base x
        elbow up/down) are solved; the one nearest ``q_seed`` in joint space wins, so a
        stream of nearby targets stays on one branch.
        """
        seed = np.asarray(q_seed, dtype=float).reshape(-1)[:OMX_DOF]
        rot = np.asarray(target_rot, dtype=float).reshape(3, 3)
        cands = self._candidates(target_pos, rot[:, 0], rot, seed, pos_tol_m)
        if not cands:
            return None, 0.0
        best = min(cands, key=lambda q: float(np.linalg.norm(q - seed)))
        _, rot_fk = self.fk(best)
        dev = float((R.from_matrix(rot_fk) * R.from_matrix(rot).inv()).magnitude())
        if dev > float(max_ori_dev_rad):
            return None, 0.0
        return best, dev

    def ik_position(
        self, target_pos: np.ndarray, q_seed: np.ndarray, pitch_down: Optional[float] = None
    ) -> Optional[np.ndarray]:
        """Position-only IK keeping the seed's tool pitch (or ``pitch_down``) and roll."""
        seed = np.asarray(q_seed, dtype=float).reshape(-1)[:OMX_DOF]
        if pitch_down is not None:
            seed = seed.copy()
            seed[3] = float(pitch_down) - seed[1] - seed[2]
        cands = self._candidates(target_pos, None, None, seed, 1e-3)
        if not cands:
            return None
        return min(cands, key=lambda q: float(np.linalg.norm(q - seed)))

    def jacobian(self, q: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """6x5 numeric geometric Jacobian (position rows 0-2, rotation-vector rows 3-5)."""
        q = np.asarray(q, dtype=float).reshape(-1)[:OMX_DOF]
        p0, r0 = self.fk(q)
        jac = np.zeros((6, OMX_DOF))
        for j in range(OMX_DOF):
            dq = q.copy()
            dq[j] += eps
            p1, r1 = self.fk(dq)
            jac[:3, j] = (p1 - p0) / eps
            jac[3:, j] = R.from_matrix(r1 @ r0.T).as_rotvec() / eps
        return jac

    def reach_ok(self, target_pos: np.ndarray, q_seed: np.ndarray) -> bool:
        """Quick reachability probe at the seed's tool pitch (no orientation fit)."""
        return self.ik_position(target_pos, q_seed) is not None
