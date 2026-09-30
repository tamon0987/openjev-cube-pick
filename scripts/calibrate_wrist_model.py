"""Wrist-camera model from the robot's own motion: intrinsics + hand-eye, no hand placement needed.

The cube sits anywhere on the table in the wrist view. The arm visits a grid of poses around the begin pose
(same tool pitch); in each the cube's colour blob gives one pixel. A pinhole camera rigidly attached to link5
(focal length, principal point, rotation and translation in the link5 frame) and the cube's position are fitted
together by least squares. The harness then projects any point exactly: the cross ("directly below the gripper")
and the image's "up"/"right" directions follow from the model at every pose and pitch.
The colour is used only for this measurement; nothing here reaches the decision layer.

    uv run python scripts/calibrate_wrist_model.py
Writes the camera model into configs/robot/calib_wrist.yaml (the old cross table is kept for reference).
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import mujoco
import numpy as np
import yaml
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot

from scripts.calibrate_wrist import blob


def link5_pose(env) -> tuple[np.ndarray, np.ndarray]:
    bid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "link5")
    return env.data.xpos[bid].copy(), env.data.xmat[bid].reshape(3, 3).copy()


def project(
    params: np.ndarray, p5: np.ndarray, R5: np.ndarray, X: np.ndarray, c: float = 160.0
) -> np.ndarray:
    f, cx, cy = params[0], c, c  # principal point at the image centre: pure translations cannot pin it down
    Rc = Rot.from_rotvec(
        params[1:4]
    ).as_matrix()  # camera axes in the link5 frame (columns: x right, y down, z fwd)
    tc = params[4:7]
    Xl = R5.T @ (X - p5)  # point in link5
    Xc = Rc.T @ (Xl - tc)  # point in the camera frame
    return np.array([cx + f * Xc[0] / Xc[2], cy + f * Xc[1] / Xc[2]])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/robot/omx_f.yaml")
    ap.add_argument(
        "--hsv-lo", type=int, nargs=3, default=[3, 60, 80], help="cube colour, OpenCV HSV (pale orange ok)"
    )
    ap.add_argument("--hsv-hi", type=int, nargs=3, default=[25, 255, 255])
    ap.add_argument("--d", type=float, default=0.02, help="grid step (m)")
    ap.add_argument(
        "--center",
        type=float,
        nargs=2,
        default=[0.0, 0.0],
        help="grid centre offset (m, x y) from the begin tcp",
    )
    ap.add_argument(
        "--tag",
        default="",
        help="suffix for the saved observations; all calib_wrist_obs*.npy are fitted together",
    )
    a = ap.parse_args()

    from dlb.real.omx import RealOMX

    env = RealOMX(a.config, camera_source="usb", cameras=("wrist",), image_size=320)
    env.reset(seed=0)  # begin pose, gripper open
    base = env.tcp_pos.copy() + np.array([*a.center, 0.0])
    obs = []
    out = Path(env.cfg["cameras"]["wrist"]["calibration"])
    for dz, dx, dy in itertools.product((0.0, 0.03), (-a.d, 0.0, a.d), (-a.d, 0.0, a.d)):
        env._move_tcp(base + np.array([dx, dy, dz]))
        img = env.render_all()["wrist"]
        u, v, area = blob(img, a.hsv_lo, a.hsv_hi)
        p5, R5 = link5_pose(env)
        print(
            f"  offset ({dx * 100:+.0f},{dy * 100:+.0f},{dz * 100:+.0f}) cm -> cube ({u:.0f},{v:.0f}) px, {area} px"
        )
        if area > 400 and 5 < u < 315 and 5 < v < 315:
            obs.append((p5, R5, np.array([u, v])))
    env._move_tcp(base)
    np.save(
        out.with_name(f"calib_wrist_obs{a.tag}.npy"),
        np.array([np.r_[p5, R5.ravel(), uv] for p5, R5, uv in obs]),
    )
    # fit every saved set together: views at different tool pitches break the depth / focal-length ambiguity
    env.go_pose("begin")
    table_z = env.table_z
    env.close()
    fit(load_sets(out.parent), table_z, out)


def load_sets(folder: Path) -> list:
    """Every saved observation set; each set had the cube somewhere else, so each gets its own cube position."""
    sets = []
    for f in sorted(folder.glob("calib_wrist_obs*.npy")):
        # blobs cut by the image border have a shifted centroid: keep views with the cube well inside the frame
        sets.append(
            [
                (r[:3], r[3:12].reshape(3, 3), r[12:14])
                for r in np.load(f)
                if 50 < r[12] < 270 and 50 < r[13] < 270
            ]
        )
    print(f"fitting {sum(map(len, sets))} views from {len(sets)} sets")
    return sets


# end_effector_link (between the closed fingertips) in the link5 frame (URDF) and where it appears in the wrist
# image with the gripper closed (read off an image, 2026-09-28). It pins the depth/focal-length ambiguity that
# pure translations leave.
EE_LINK5 = np.array([0.09193, -0.0016, 0.0])
EE_PIXEL = np.array([189.0, 268.0])


def fit(sets: list, table_z: float, out: Path, ee_weight: float = 4.0) -> None:
    if sets and not isinstance(sets[0], list):
        sets = [sets]
    obs = [o for s in sets for o in s]
    if len(obs) < 8:
        raise SystemExit(f"only {len(obs)} usable views: is the cube in the wrist view at the begin pose?")

    # initial guess: camera looking along the tool axis (link5 +x), image right = link5 -y, image down = link5 -z
    Rc0 = np.column_stack([[0, -1, 0], [0, 0, -1], [1, 0, 0]]).astype(float)
    cube_z = 0.015 + table_z  # the cube's centre: it rests on the table
    g = np.concatenate([(s[len(s) // 2][0] + s[len(s) // 2][1][:, 0] * 0.12)[:2] for s in sets])
    best = None
    # the fit has local minima: start from a few focal lengths and camera offsets
    for f0, tz in itertools.product((300.0, 500.0, 800.0), (0.03, 0.06)):
        x0 = np.r_[f0, Rot.from_matrix(Rc0).as_rotvec(), [0.03, 0.0, tz], g]
        lo = np.r_[
            250.0, -np.pi * np.ones(3), [-0.08, -0.08, -0.08], g - 0.2
        ]  # the camera sits on the gripper
        hi = np.r_[900.0, np.pi * np.ones(3), [0.08, 0.08, 0.08], g + 0.2]

        def resid(x: np.ndarray) -> np.ndarray:
            r = []
            for k, s in enumerate(sets):
                X = np.r_[x[7 + 2 * k : 9 + 2 * k], cube_z]
                r += [project(x[:7], p5, R5, X) - uv for p5, R5, uv in s]
            r.append(ee_weight * (project(x[:7], np.zeros(3), np.eye(3), EE_LINK5) - EE_PIXEL))
            return np.concatenate(r)

        sol = least_squares(resid, x0, bounds=(lo, hi), loss="soft_l1", f_scale=3.0)
        if best is None or sol.cost < best[0].cost:
            best = (sol, resid)
    sol, resid = best
    r = resid(sol.x).reshape(-1, 2)
    print(f"  fingertip constraint residual {np.linalg.norm(r[-1]) / ee_weight:.1f} px")
    r = r[:-1]
    err = np.linalg.norm(r, axis=1)
    x = sol.x
    print(f"fit: {len(obs)} views, reprojection mean {err.mean():.1f} px, max {err.max():.1f} px")
    print(
        f"  f={x[0]:.0f}px cam in link5 t={np.round(x[4:7] * 100, 1).tolist()} cm, cubes at "
        f"{[np.round(x[7 + 2 * k : 9 + 2 * k] * 100, 1).tolist() for k in range(len(sets))]} cm (base frame)"
    )
    calib = yaml.safe_load(out.read_text()) if out.exists() else {}
    calib.update(
        {
            "square_size": 320,
            "model": {
                "f": round(float(x[0]), 2),
                "cx": 160.0,
                "cy": 160.0,
                "R_link5_cam": np.round(Rot.from_rotvec(x[1:4]).as_matrix(), 6).tolist(),
                "t_link5_cam": np.round(x[4:7], 5).tolist(),
                "reprojection_px": {"mean": round(float(err.mean()), 2), "max": round(float(err.max()), 2)},
                "views": len(obs),
            },
        }
    )
    out.write_text(yaml.safe_dump(calib, sort_keys=False))
    print("wrote the camera model to", out)


if __name__ == "__main__":
    main()
