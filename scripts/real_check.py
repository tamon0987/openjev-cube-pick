"""Step R1 of docs/real_robot.md: first contact with the real OMX-F (or the mock), one confirmed step at a time.

Start the follower first (real robot):
    ros2 launch open_manipulator_bringup omx_f_follower_ai.launch.py port_name:=/dev/ttyACM0
or without hardware:
    ros2 launch open_manipulator_bringup omx_f_follower_ai.launch.py use_mock_hardware:=true

Then, with ~/ros2_ws/install/setup.bash sourced:
    uv run python scripts/real_check.py            # read only: joint angles, twin tcp, gripper reading
    uv run python scripts/real_check.py --home     # + slow move to the tool-down home pose
    uv run python scripts/real_check.py --jog      # + 1 cm moves in each direction and back
    uv run python scripts/real_check.py --gripper  # + open / close, prints readings for configs/robot/omx_f.yaml

Every motion asks for Enter first. Ctrl+C stops; the arm holds its last target.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np


def confirm(what: str, yes: bool) -> None:
    if yes:
        print(f"-> {what}")
        return
    if input(f"{what}  [Enter = go, anything else = quit] ").strip():
        sys.exit("stopped by user")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/robot/omx_f.yaml")
    ap.add_argument("--home", action="store_true")
    ap.add_argument("--jog", action="store_true")
    ap.add_argument("--gripper", action="store_true")
    ap.add_argument("--step-cm", type=float, default=1.0)
    ap.add_argument("--yes", action="store_true", help="do not ask before each motion (mock only)")
    a = ap.parse_args()

    from dlb.real.omx import RealOMX

    # RealOMX.reset() would move home; construct without resetting and read first
    env = RealOMX(a.config, camera_source="twin", render=False)
    q, g = env.measured()
    env._sync_twin(settle=0.0)
    print("joints (rad):", np.round(q, 3).tolist(), " gripper:", None if g is None else round(g, 3))
    print(
        "twin tcp (cm):",
        np.round(env.tcp_pos * 100, 1).tolist(),
        "  URDF end_effector_link should read 2.0 cm lower in z (check with: ros2 run tf2_ros tf2_echo link0 end_effector_link)",
    )

    if a.home or a.jog or a.gripper:
        home = env.home_ctrl[env.arm_act]
        confirm(
            f"move slowly to home joints {np.round(home, 2).tolist()} (tool down, ~8 cm above the table)",
            a.yes,
        )
        env.reset(seed=0)
        print("at home, tcp (cm):", np.round(env.tcp_pos * 100, 1).tolist())

    if a.jog:
        s = a.step_cm / 100
        fwd, right = env.wrist_axes()
        for name, d in (("forward", [*(fwd * s), 0]), ("right", [*(right * s), 0]), ("up", [0, 0, s])):
            for sign in (1, -1):
                confirm(f"move {a.step_cm:g} cm {'+' if sign > 0 else '-'}{name}", a.yes)
                before = env.tcp_pos.copy()
                env.move_relative(np.array(d) * sign)
                print(f"   moved {np.round((env.tcp_pos - before) * 100, 2).tolist()} cm")

    if a.gripper:
        for what, fn in (
            ("open", env.open_gripper),
            ("close (nothing between the fingers)", env.close_gripper),
            ("open", env.open_gripper),
        ):
            confirm(f"gripper {what}", a.yes)
            fn()
            g = env.measured()[1]
            print("   gripper reading:", None if g is None else round(g, 3))
        print(
            "Put these readings into configs/robot/omx_f.yaml (gripper_open / gripper_closed). Then close on the"
            " cube by hand and note the reading: gripper_held_margin should sit between it and gripper_closed."
        )
    env.close()


if __name__ == "__main__":
    main()
