"""Damped least-squares IK for a site: position + "tool points down" orientation.

Works on any MJCF with a named site and a list of arm joints (hinge/slide,
1 DoF each). The orientation task only constrains the site's local z-axis to
align with ``down`` (world -z), which is 2 DoF, leaving roll free.
"""

from __future__ import annotations

import mujoco
import numpy as np


class SiteIK:
    def __init__(
        self,
        model: mujoco.MjModel,
        site: str,
        joints: list[str],
        damping: float = 0.05,
        pos_weight: float = 1.0,
        ori_weight: float = 0.3,
        max_iters: int = 100,
        tol: float = 1e-3,
        step: float = 0.7,
    ):
        self.model = model
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site)
        self.joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in joints]
        self.qpos_idx = np.array([model.jnt_qposadr[j] for j in self.joint_ids])
        self.dof_idx = np.array([model.jnt_dofadr[j] for j in self.joint_ids])
        self.lo = model.jnt_range[self.joint_ids, 0].copy()
        self.hi = model.jnt_range[self.joint_ids, 1].copy()
        self.damping = damping
        self.pos_weight = pos_weight
        self.ori_weight = ori_weight
        self.max_iters = max_iters
        self.tol = tol
        self.step = step
        self._scratch = mujoco.MjData(model)
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))

    def solve(
        self,
        data: mujoco.MjData,
        target_pos: np.ndarray,
        q_init: np.ndarray | None = None,
        down: np.ndarray | None = None,
    ) -> tuple[np.ndarray, float]:
        """Return (q for the arm joints, final position error in metres)."""
        d = self._scratch
        d.qpos[:] = data.qpos
        d.qvel[:] = 0
        q = (q_init if q_init is not None else data.qpos[self.qpos_idx]).astype(float).copy()
        down = np.array([0.0, 0.0, -1.0]) if down is None else np.asarray(down, float)
        target_pos = np.asarray(target_pos, float)
        err_norm = np.inf
        for _ in range(self.max_iters):
            d.qpos[self.qpos_idx] = q
            mujoco.mj_kinematics(self.model, d)
            mujoco.mj_comPos(self.model, d)
            p = d.site_xpos[self.site_id]
            R = d.site_xmat[self.site_id].reshape(3, 3)
            z_axis = R[:, 2]
            e_pos = target_pos - p
            # rotation that brings z_axis onto `down`: axis = z x down, |axis| = sin(theta)
            e_rot = np.cross(z_axis, down)
            err_norm = float(np.linalg.norm(e_pos))
            if err_norm < self.tol and np.linalg.norm(e_rot) < 0.02:
                break
            mujoco.mj_jacSite(self.model, d, self._jacp, self._jacr, self.site_id)
            Jp = self._jacp[:, self.dof_idx]
            Jr = self._jacr[:, self.dof_idx]
            J = np.vstack([self.pos_weight * Jp, self.ori_weight * Jr])
            e = np.concatenate([self.pos_weight * e_pos, self.ori_weight * e_rot])
            JJt = J @ J.T + (self.damping**2) * np.eye(J.shape[0])
            dq = J.T @ np.linalg.solve(JJt, e)
            q = np.clip(q + self.step * dq, self.lo, self.hi)
        return q, err_norm
