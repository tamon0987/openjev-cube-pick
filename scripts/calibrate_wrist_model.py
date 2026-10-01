"""Wrist-camera model from the robot's own motion: intrinsics + hand-eye, no hand placement needed.

The cube sits on the table, near the middle of the wrist view at the begin pose. The arm visits a grid of poses
around the begin pose (same tool pitch); in each the cube's colour blob gives one pixel. A pinhole camera rigidly
attached to link5 (focal length, rotation and translation in the link5 frame) and the cube's position are fitted
together by least squares. The harness then projects any point exactly: the cross ("directly below the gripper")
and the image's "up"/"right" directions follow from the model at every pose and pitch.
The colour is used only for this measurement; nothing here reaches the decision layer.

Pure translations leave depth and focal length ambiguous, so the fit also needs one known point: the closed
fingertips. First the gripper closes and the wrist image is saved with a pixel grid; type the pixel midway between
the two fingertip ends (or pass --ee-pixel U V). The value is kept in calib_wrist.yaml and offered as the default next
time; measure it again whenever the camera is re-mounted.

    python scripts/calibrate_wrist_model.py           # calibrate (writes configs/robot/calib_wrist.yaml)
    python scripts/calibrate_wrist_model.py --check   # does the current calib_wrist.yaml fit this rig?
Images and observations go to results/calib_wrist/.

--check needs no cube: it closes the gripper at the begin pose and marks where the current model puts the
closed fingertips. If the mark sits midway between the fingertip ends in the image, the file fits this camera mount.
"""

from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path

import mujoco
import numpy as np
import yaml
from calibrate_wrist import blob  # sibling module (run as python scripts/calibrate_wrist_model.py)
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot


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
        "--ee-pixel",
        type=float,
        nargs=2,
        help="pixel (u v, 320 px square) of the closed fingertips in the wrist image; asked for when omitted",
    )
    ap.add_argument("--out-dir", default="results/calib_wrist", help="images and observations of this run")
    ap.add_argument(
        "--check", action="store_true", help="mark the model's fingertips on the wrist image only"
    )
    a = ap.parse_args()

    from dlb.real.omx import RealOMX

    env = RealOMX(a.config, camera_source="usb", cameras=("wrist",), image_size=320)
    out = Path(env.cfg["cameras"]["wrist"]["calibration"])
    run_dir = Path(a.out_dir) / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    env.reset(seed=0)  # begin pose, gripper open
    if a.check:
        check(env, out, run_dir)
        return

    img = env.render_all()["wrist"]
    save_rgb(run_dir / "begin.jpg", img)
    u, v, area = blob(img, a.hsv_lo, a.hsv_hi)
    if not (area > 400 and 60 < u < 260 and 60 < v < 260):
        env.close()
        raise SystemExit(
            f"the cube is not near the middle of the wrist view at the begin pose (blob at ({u:.0f},{v:.0f}) px,"
            f" {area} px). Move it there and run again; the wrist image is in {run_dir / 'begin.jpg'}"
        )

    ee_pixel = measure_ee_pixel(env, out, run_dir, a.ee_pixel)
    env.open_gripper()

    base = env.tcp_pos.copy() + np.array([*a.center, 0.0])
    obs = []
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
    np.save(run_dir / "obs.npy", np.array([np.r_[p5, R5.ravel(), uv] for p5, R5, uv in obs]))
    env.go_pose("begin")
    table_z = env.table_z
    env.close()
    # blobs cut by the image border have a shifted centroid: keep views with the cube well inside the frame
    obs = [o for o in obs if 50 < o[2][0] < 270 and 50 < o[2][1] < 270]
    fit([obs], table_z, out, ee_pixel)


def check(env, out: Path, run_dir: Path) -> None:
    """Mark where the current camera model puts the closed fingertips; the user compares it with the image."""
    import cv2

    model = (yaml.safe_load(out.read_text()) or {}).get("model") if out.exists() else None
    if not model:
        env.close()
        raise SystemExit(f"no camera model in {out}: run scripts/calibrate_wrist_model.py without --check")
    params = np.r_[
        model["f"], Rot.from_matrix(np.asarray(model["R_link5_cam"])).as_rotvec(), model["t_link5_cam"]
    ]
    u, v = project(params, np.zeros(3), np.eye(3), EE_LINK5)
    env.close_gripper()
    img = cv2.cvtColor(env.render_all()["wrist"], cv2.COLOR_RGB2BGR)
    env.open_gripper()
    env.close()
    img = cv2.resize(img, (640, 640))
    cv2.circle(img, (round(2 * u), round(2 * v)), 10, (0, 255, 0), 2)
    cv2.drawMarker(img, (round(2 * u), round(2 * v)), (0, 255, 0), cv2.MARKER_CROSS, 30, 1)
    path = run_dir / "check.jpg"
    cv2.imwrite(str(path), img)
    print(f"model fingertips at ({u:.0f}, {v:.0f}) px (320 px square): {path}")
    print(
        "If the green mark sits midway between the closed fingertip ends (within ~10 px), keep calib_wrist.yaml."
        " Otherwise run scripts/calibrate_wrist_model.py."
    )


def save_rgb(path: Path, img_rgb: np.ndarray) -> None:
    import cv2

    cv2.imwrite(str(path), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))


def measure_ee_pixel(env, out: Path, run_dir: Path, given: list[float] | None) -> np.ndarray:
    """Pixel of the closed fingertips: from --ee-pixel, or read off a gridded wrist image by the user."""
    import cv2

    if given:
        return np.array(given, float)
    env.close_gripper()
    img = cv2.cvtColor(env.render_all()["wrist"], cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (640, 640), interpolation=cv2.INTER_NEAREST)  # 2x, so the grid labels stay readable
    for p in range(0, 321, 20):
        c = (0, 255, 255) if p % 100 == 0 else (90, 90, 90)
        cv2.line(img, (2 * p, 0), (2 * p, 639), c, 1)
        cv2.line(img, (0, 2 * p), (639, 2 * p), c, 1)
        if p % 40 == 0:
            cv2.putText(img, str(p), (2 * p + 2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            cv2.putText(img, str(p), (2, 2 * p + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    path = run_dir / "fingertips.jpg"
    cv2.imwrite(str(path), img)
    old = (yaml.safe_load(out.read_text()) or {}).get("ee_pixel") if out.exists() else None
    print(f"gripper closed; wrist image with a pixel grid (labels in 320 px units): {path}")
    while True:
        prompt = "pixel midway between the closed fingertip ends, 'u v'" + (
            f" [Enter = {old[0]:g} {old[1]:g}]" if old else ""
        )
        s = input(prompt + ": ").strip()
        if not s and old:
            return np.array(old, float)
        try:
            u, v = (float(t) for t in s.replace(",", " ").split())
        except ValueError:
            continue
        if 0 <= u <= 320 and 0 <= v <= 320:
            return np.array([u, v])


# end_effector_link (between the closed fingertips) in the link5 frame (URDF). Where it appears in the wrist image
# (measured per rig) pins the depth/focal-length ambiguity that pure translations leave.
EE_LINK5 = np.array([0.09193, -0.0016, 0.0])


def fit(sets: list, table_z: float, out: Path, ee_pixel: np.ndarray, ee_weight: float = 4.0) -> None:
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
            r.append(ee_weight * (project(x[:7], np.zeros(3), np.eye(3), EE_LINK5) - ee_pixel))
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
            "ee_pixel": [round(float(c), 1) for c in ee_pixel],
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
