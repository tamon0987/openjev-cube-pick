"""Saved poses and the table height for the real OMX-F, captured by hand (as in Show-Harness's go_begin.py).

    python scripts/real_poses.py --list               # saved poses with fingertip position and pitch
    python scripts/real_poses.py --go begin           # slow joint move to a saved pose
    python scripts/real_poses.py --capture begin      # torque off, pose the arm by hand, Enter -> saved
    python scripts/real_poses.py --capture rest
    python scripts/real_poses.py --z-floor            # torque off, rest the FINGERTIPS on the table, Enter -> saved

A begin pose should hold the gripper pitched down over the workspace with joint-limit headroom (>= 8 deg);
the script prints the margins. Values go into configs/robot/omx_f.yaml.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np


def write_value(path: Path, key: str, value: str) -> None:
    """Replace ``key: ...`` (top level or under poses:) in place, keeping the comments."""
    text = path.read_text()
    pat = re.compile(rf"^(\s*){re.escape(key)}:.*$", re.M)
    if not pat.search(text):
        raise SystemExit(f"{key} not found in {path}")
    path.write_text(pat.sub(lambda m: f"{m.group(1)}{key}: {value}", text, count=1))


def describe(kin, q) -> str:
    p, _ = kin.fk(np.asarray(q, float))
    return (
        f"fingertips ({p[0] * 100:.1f}, {p[1] * 100:.1f}, {p[2] * 100:.1f}) cm, pitch "
        f"{math.degrees(kin.tool_pitch_down_rad(q)):.0f} deg down, limit margins "
        f"{np.round(kin.limit_margins_deg(q), 0).tolist()} deg"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/robot/omx_f.yaml")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--list", action="store_true")
    g.add_argument("--go", metavar="NAME")
    g.add_argument("--capture", metavar="NAME")
    g.add_argument("--z-floor", action="store_true")
    a = ap.parse_args()

    from dlb.real.omx import RealOMX
    from dlb.real.omx_kinematics import OmxKinematics

    kin = OmxKinematics()
    if a.list:
        import yaml

        cfg = yaml.safe_load(Path(a.config).read_text())
        for name, q in (cfg.get("poses") or {}).items():
            print(f"  {name:<6} {q}  -> {describe(kin, q)}")
        print("  z_floor_m:", cfg.get("z_floor_m"))
        return
    env = RealOMX(a.config, camera_source="twin", render=False)
    try:
        if a.go:
            input(f"The arm moves slowly to '{a.go}'. Clear the area, then press Enter ")
            env.go_pose(a.go)
            print("at", a.go, "->", describe(kin, env.measured()[0]))
        elif a.capture:
            q = env.free_arm(
                f"Pose the arm as the '{a.capture}' pose (gripper pitched down over the workspace)."
            )
            print("captured", np.round(q, 4).tolist(), "->", describe(kin, q))
            if float(np.min(kin.limit_margins_deg(q))) < 8.0:
                print("WARNING: a joint is within 8 deg of its limit; moves that way will clamp early")
            write_value(Path(a.config), a.capture, "[" + ", ".join(f"{v:.4f}" for v in q) + "]")
            print(f"wrote poses.{a.capture} to {a.config}")
        else:
            q = env.free_arm("Rest the gripper FINGERTIPS on the table where the robot will work.")
            z = float(kin.fk(q)[0][2])
            print(f"fingertip height at contact: {z * 100:.2f} cm ->", describe(kin, q))
            if abs(z) > 0.06:
                print("WARNING: far from the base plane; is the arm really touching the table?")
            write_value(Path(a.config), "z_floor_m", f"{z:.4f}")
            print(f"wrote z_floor_m to {a.config}; lift the arm with --go begin")
    finally:
        env.close()


if __name__ == "__main__":
    main()
