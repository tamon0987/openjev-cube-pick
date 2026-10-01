"""Wrist-camera calibration for the real OMX-F (step R2 of docs/real_robot.md), a few minutes with the cube.

Measures two things the harness needs, from the robot's own motion (nothing here is given to the decision
layer; the cube's colour is used only to measure):

* where a point directly below the tcp appears in the wrist image, for several heights (the green cross),
* which horizontal world directions correspond to "up" and "right" in the wrist image (MV_FWD / MV_RIGHT),
  stored in the link5 frame so they follow the wrist as the base turns.

Procedure (the robot's follower launch must be running):
    python scripts/calibrate_wrist.py
1. the arm moves to the begin pose (the pitch the IK keeps afterwards); place the cube directly below the gripper, centred between the
   fingertips, then press Enter,
2. the arm rises and lowers a little (cube stays), then moves 2 cm along world x and y and back.
Writes configs/robot/calib_wrist.yaml and debug images next to it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw


def blob(img_rgb: np.ndarray, hsv_lo: list[int], hsv_hi: list[int]) -> tuple[float, float, int]:
    import cv2

    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    m = cv2.inRange(hsv, np.array(hsv_lo), np.array(hsv_hi))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, cent = cv2.connectedComponentsWithStats(m)
    if n <= 1:
        return float("nan"), float("nan"), 0
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return float(cent[k][0]), float(cent[k][1]), int(stats[k, cv2.CC_STAT_AREA])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/robot/omx_f.yaml")
    ap.add_argument("--cube-size", type=float, default=0.03, help="cube edge (m)")
    ap.add_argument(
        "--hsv-lo", type=int, nargs=3, default=[3, 120, 90], help="cube colour, OpenCV HSV (orange)"
    )
    ap.add_argument("--hsv-hi", type=int, nargs=3, default=[25, 255, 255])
    ap.add_argument("--jog", type=float, default=0.02)
    ap.add_argument(
        "--dz",
        type=float,
        nargs="+",
        default=[0.0, 0.02, 0.04, -0.02, -0.04],
        help="tcp height changes (m) from the begin pose; keep the fingertips above the cube",
    )
    ap.add_argument("--yes", action="store_true")
    a = ap.parse_args()

    from dlb.real.omx import RealOMX

    env = RealOMX(a.config, camera_source="usb", cameras=("wrist",), image_size=320)
    out = Path(env.cfg["cameras"]["wrist"]["calibration"])
    if not a.yes:
        input("The arm will move slowly to the begin pose. Clear the area, then press Enter ")
    env.reset(seed=0)
    if not a.yes:
        input("Place the cube directly below the gripper, centred between the fingertips. Press Enter ")
    cube_z = a.cube_size / 2

    areas = {}

    def shoot(tag: str) -> tuple[float, float]:
        img = env.render_all()["wrist"]
        u, v, area = blob(img, a.hsv_lo, a.hsv_hi)
        im = Image.fromarray(img)
        if area:
            ImageDraw.Draw(im).ellipse([u - 4, v - 4, u + 4, v + 4], outline=(0, 255, 0), width=2)
        im.save(out.with_name(f"calib_wrist_{tag}.jpg"))
        print(f"  {tag}: tcp z {env.tcp_pos[2] * 100:.1f} cm, cube at ({u:.0f}, {v:.0f}) px, {area} px area")
        if not area:
            raise SystemExit(f"cube not found in {tag}; check --hsv-lo/--hsv-hi and the debug image")
        areas[tag] = area
        return u, v

    # 1. the point below the tcp at several heights (the cube stays put, the tcp moves straight up/down)
    cross = []
    for dz in a.dz:
        if dz:
            env.move_relative(np.array([0.0, 0.0, dz]))
        u, v = shoot(f"drop{dz:+.2f}")
        cross.append(
            [
                round(float(env.tcp_pos[2] - cube_z), 4),
                round(u, 1),
                round(v, 1),
                round(float(np.sqrt(areas[f"drop{dz:+.2f}"])), 1),
            ]
        )
        if dz:
            env.move_relative(np.array([0.0, 0.0, -dz]))
    cross.sort()

    # 2. image directions: move the gripper along world x and y and watch the cube move the other way
    u0, v0 = shoot("ref")
    J = np.zeros((2, 2))
    for i, d in enumerate(([a.jog, 0.0, 0.0], [0.0, a.jog, 0.0])):
        env.move_relative(np.array(d))
        u, v = shoot(f"jog{'xy'[i]}")
        env.move_relative(-np.array(d))
        J[:, i] = -np.array([u - u0, v - v0]) / a.jog  # pixel shift per metre of target offset
    # a target offset e (world xy) appears at cross + J e; image "up" is -v, "right" is +u
    up_w = np.linalg.solve(J, [0.0, -1.0])
    right_w = np.linalg.solve(J, [1.0, 0.0])
    import mujoco

    R = env.data.xmat[mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "link5")].reshape(3, 3)
    to_l5 = lambda w: (R.T @ np.array([*(w / np.linalg.norm(w)), 0.0])).round(4).tolist()  # noqa: E731
    calib = {
        "square_size": 320,
        "cross_px_by_drop": [c[:3] for c in cross],
        # apparent cube size (sqrt of the blob area, px on the 320 px square) per drop: sets the zoom crop
        "cube_px_by_drop": [[c[0], c[3]] for c in cross],
        "image_up_link5": to_l5(up_w),
        "image_right_link5": to_l5(right_w),
        "pixels_per_cm_at_home": round(float(np.linalg.norm(J, axis=0).mean()) / 100, 2),
        "angle_between_axes_deg": round(
            float(
                np.degrees(np.arccos(abs(up_w @ right_w) / np.linalg.norm(up_w) / np.linalg.norm(right_w)))
            ),
            1,
        ),
    }
    out.write_text(yaml.safe_dump(calib, sort_keys=False))
    print(yaml.safe_dump(calib, sort_keys=False))
    print("wrote", out, "(the axes should be ~90 degrees apart; debug images next to it)")
    env.close()


if __name__ == "__main__":
    main()
