"""Bridge to the real OMX-F (or its ros2_control mock) with the same interface as ``PickPlaceEnv``.

``RealOMX`` subclasses the simulator env and keeps it as a *digital twin*:

* Motion goes to the robot. ``_move_tcp`` solves IK on the twin, sends a joint trajectory to the arm
  controller (``/leader/joint_trajectory`` -> the follower's JointTrajectoryController) and waits until the
  measured joints arrive. ``_set_jaw`` commands ``gripper_joint_1`` the same way.
* After every command the twin copies the measured joint angles, so ``tcp_pos``, ``wrist_axes``,
  ``wrist_pixel`` and the cross drawn on the wrist image come from the robot's own kinematics. The twin's
  MJCF and the OMX-F URDF agree: for the same joint angles the URDF's ``end_effector_link`` is 2.0 cm below
  the twin's tcp site, with the same xy.
* ``camera_source="twin"`` renders the twin, which also holds a virtual cube and bin. Together with
  ``use_mock_hardware:=true`` this tests the whole ROS path (and the decision layer) without the robot.
  ``camera_source="usb"`` (step R2 of docs/real_robot.md) will take frames from the calibrated USB cameras.

Safety: every target passes the twin's workspace clamp, IK solutions that jump more than
``max_joint_step`` are refused, moves are timed from ``max_tcp_speed`` and must arrive within a timeout.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from dlb.sim.env import JAW_CLOSED, JAW_OPEN, PickPlaceEnv


class MotionError(RuntimeError):
    pass


class RealOMX(PickPlaceEnv):
    def __init__(
        self, config: str | Path = "configs/robot/omx_f.yaml", camera_source: str = "twin", **env_kw: Any
    ):
        super().__init__(**env_kw)
        self.cfg = yaml.safe_load(Path(config).read_text())
        self.camera_mode = camera_source
        if camera_source not in ("twin", "usb"):
            raise ValueError(f"camera_source must be twin or usb, got {camera_source!r}")
        # simulator truth (cube / bin positions) exists only for the twin's virtual scene
        self.has_truth = camera_source == "twin"
        self.front_calibrated = camera_source == "twin"
        if camera_source == "usb":
            self._start_cameras()
        self._q_meas: np.ndarray | None = None
        # The trajectory controller replaces the running trajectory with every new message, so each command
        # carries both the arm and the gripper target.
        self._grip_target: float = float(self.cfg["gripper_open"])
        self._grip_meas: float | None = None
        self._stamp = 0.0
        self._lock = threading.Lock()
        from dlb.real.omx_kinematics import OmxKinematics

        self.kin = OmxKinematics()  # firmware joint limits, analytic IK (replaces the twin's damped IK here)
        # Heights in the harness are above the TABLE (z = 0 in simulation). z_floor_m is the fingertip height at
        # contact with the table in the robot base frame (scripts/real_poses.py --z-floor); unset = base plane.
        self.table_z = float(self.cfg.get("z_floor_m") or 0.0)
        # tcp height for reach/align of the cube with the real wrist camera (the servo climbs there first)
        self.observe_z = self.cfg.get("observe_z")
        self._start_ros()
        # hold whatever the gripper does now: a default "open" target dropped a held cube on start-up
        _, g = self.measured()
        if g is not None:
            self._grip_target = float(g)
            cfg = self.cfg
            if abs(g - cfg["gripper_open"]) > cfg["gripper_held_margin"]:
                # not open: it was commanded closed (resting on an object, or empty); keep squeezing
                self._grip_target = float(cfg["gripper_closed"])
                self.data.ctrl[self.jaw_act] = JAW_CLOSED

    # ------------------------------------------------------------------ #
    # ROS plumbing
    # ------------------------------------------------------------------ #
    def _start_ros(self) -> None:
        import os

        os.environ.setdefault("ROS_DOMAIN_ID", str(self.cfg.get("ros_domain_id", 0)))
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from sensor_msgs.msg import JointState
        from trajectory_msgs.msg import JointTrajectory

        if not rclpy.ok():
            rclpy.init()
        self._node = rclpy.create_node("dlb_real_omx")
        self._JointTrajectory = JointTrajectory
        self._pub = self._node.create_publisher(JointTrajectory, self.cfg["command_topic"], 10)
        names = self.cfg["arm_joints"]
        gname = self.cfg["gripper_joint"]

        def on_js(msg: JointState) -> None:
            idx = {n: i for i, n in enumerate(msg.name)}
            if not all(n in idx for n in names):
                return
            with self._lock:
                self._q_meas = np.array([msg.position[idx[n]] for n in names])
                if gname in idx:
                    self._grip_meas = float(msg.position[idx[gname]])
                self._stamp = time.monotonic()

        self._node.create_subscription(JointState, self.cfg["joint_states_topic"], on_js, 20)
        self._exec = SingleThreadedExecutor()
        self._exec.add_node(self._node)
        self._spin = threading.Thread(target=self._exec.spin, daemon=True)
        self._spin.start()
        t0 = time.monotonic()
        while self._q_meas is None:
            if time.monotonic() - t0 > 10:
                raise MotionError(
                    f"no {self.cfg['joint_states_topic']} within 10 s: is the robot (or mock) running?"
                )
            time.sleep(0.05)

    def close(self) -> None:
        self._exec.shutdown()
        self._node.destroy_node()

    def measured(self) -> tuple[np.ndarray, float | None]:
        with self._lock:
            if time.monotonic() - self._stamp > 1.0:
                raise MotionError("joint states are stale (> 1 s)")
            return self._q_meas.copy(), self._grip_meas

    def _send(self, points: list[tuple[list[float], float]], grip: float | None = None) -> None:
        """Send arm waypoints (joint angles, time from start) together with the gripper target."""
        from builtin_interfaces.msg import Duration
        from trajectory_msgs.msg import JointTrajectoryPoint

        if grip is not None:
            self._grip_target = float(grip)
        msg = self._JointTrajectory()
        msg.joint_names = [*self.cfg["arm_joints"], self.cfg["gripper_joint"]]
        for q, t in points:
            p = JointTrajectoryPoint()
            p.positions = [float(v) for v in q] + [self._grip_target]
            p.time_from_start = Duration(sec=int(t), nanosec=int((t % 1.0) * 1e9))
            msg.points.append(p)
        self._pub.publish(msg)

    # ------------------------------------------------------------------ #
    # twin synchronisation
    # ------------------------------------------------------------------ #
    def _sync_twin(self, settle: float = 0.15) -> None:
        """Copy the measured arm joints into the twin and let the virtual objects follow."""
        import mujoco

        q, _ = self.measured()
        d = self.data
        d.qpos[self.arm_qadr] = q
        d.qvel[
            [
                self.model.jnt_dofadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)]
                for j in ("Joint1", "Joint2", "Joint3", "Joint4", "Joint5")
            ]
        ] = 0.0
        d.ctrl[self.arm_act] = q
        mujoco.mj_forward(self.model, d)
        super()._settle(settle)

    # ------------------------------------------------------------------ #
    # motion (overrides of the simulator's execution layer)
    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    # table frame and tool pitch
    # ------------------------------------------------------------------ #
    @property
    def tcp_pos(self) -> np.ndarray:
        """tcp position with z above the table (the twin's site z minus the measured table height)."""
        return self.data.site_xpos[self.tcp_site] - np.array([0.0, 0.0, self.table_z])

    def tool_pitch_deg(self) -> float:
        """Pitch for every IK solve: the config value, or the begin pose's (Show-Harness keeps that pitch)."""
        v = self.cfg.get("tool_pitch_deg", "begin")
        if v == "begin" and self.cfg.get("poses", {}).get("begin"):
            q = self.cfg["poses"]["begin"]
            return float(np.degrees(q[1] + q[2] + q[3]))
        return 90.0 if v == "begin" else float(v)

    def _ik(self, tcp_target: np.ndarray, q_seed: np.ndarray) -> np.ndarray:
        """Analytic, position-exact IK (Show-Harness OMX port) for the tcp with the tool pitched ``tool_pitch_deg``
        down, checked against the firmware joint limits. The tcp sits 2 cm behind the fingertips on the tool."""
        tcp_target = np.asarray(tcp_target, float) + np.array([0.0, 0.0, self.table_z])  # table -> base frame
        heading = np.arctan2(tcp_target[1], tcp_target[0] + 0.01125)  # arm plane about the joint-1 axis
        base = self.tool_pitch_deg()
        # Far/high targets have no straight-down solution inside the firmware limits: tilt only as much as needed.
        for pitch_deg in (base, base - 10, base - 20, base - 30):
            pitch = np.deg2rad(pitch_deg)
            axis = np.array(
                [np.cos(pitch) * np.cos(heading), np.cos(pitch) * np.sin(heading), -np.sin(pitch)]
            )
            tip = np.asarray(tcp_target, float) + 0.02 * axis
            q = self.kin.ik_position(tip, np.asarray(q_seed, float), pitch_down=float(pitch))
            if q is not None:
                return q
        raise MotionError(
            f"no IK solution within the firmware joint limits for tcp {np.round(tcp_target, 3)}"
        )

    def _move_tcp(self, target: np.ndarray, seg: float = 0.01, timeout: float = 0.0) -> float:
        cfg = self.cfg
        self._sync_twin(settle=0.0)
        self._tcp_cmd = np.asarray(target, float).copy()
        start = self.tcp_pos.copy()
        dist = float(np.linalg.norm(target - start))
        n = max(1, int(np.ceil(dist / seg)))
        q_prev = self.data.qpos[self.arm_qadr].copy()
        fast = not getattr(self, "precise", True)
        speed = cfg.get("fast_tcp_speed", cfg["max_tcp_speed"]) if fast else cfg["max_tcp_speed"]
        duration = max(
            cfg.get("fast_min_move_time", cfg["min_move_time"]) if fast else cfg["min_move_time"],
            dist / speed,
        )
        points = []
        for i in range(1, n + 1):
            wp = start + (target - start) * (i / n)
            try:
                q = self._ik(wp, q_prev)
            except MotionError:
                if not points:
                    return float(np.linalg.norm(self.tcp_pos - target))  # pinned at the boundary: stay
                break  # stop at the last reachable waypoint (the workspace boundary)
            if np.max(np.abs(q - q_prev)) > cfg["max_joint_step"]:
                raise MotionError(f"IK jump {np.max(np.abs(q - q_prev)):.2f} rad toward {np.round(wp, 3)}")
            points.append((q.tolist(), duration * i / n))
            q_prev = q
        self._send(points)
        self._wait_arm(np.array(points[-1][0]), duration)
        self._sync_twin()
        return float(np.linalg.norm(self.tcp_pos - target))

    def _wait_arm(self, q_goal: np.ndarray, duration: float) -> None:
        """Wait until the joints come to rest after the planned duration.

        Real servos stop short of the goal under gravity (joint3 settled ~0.04 rad low at the look-down pose),
        so arrival means "at rest"; only a large remaining error counts as a failed move. The twin copies the
        measured joints afterwards, so the tcp stays exact and the image-based servo absorbs the shortfall.
        """
        cfg = self.cfg
        time.sleep(max(0.0, duration - 0.05))
        q = self._rest()
        # Gravity compensation by an outer integrator: the position loop settles short of the goal under load
        # (a +2 cm lift rested at +0.65 cm, joint3 ~0.04 rad low), so re-send the goal shifted by the residual.
        q_cmd = q_goal.copy()
        # Fast (intermediate) moves skip the correction (the servo looks again anyway) unless the sag is large: at
        # long reach with the cube held it reached 0.09 rad on joint2 (real_overhead4_blue, 27 cm), the height
        # drifted from 10 to 6.7 cm over the alignment and the lift failed. One correction then.
        precise = getattr(self, "precise", True)
        big = np.max(np.abs(q_goal - q)) > cfg.get("fast_sag_trigger", 0.02)
        for _ in range(cfg["sag_iterations"] if precise else (1 if big else 0)):
            err = q_goal - q
            if np.max(np.abs(err)) < cfg["sag_tolerance"]:
                break
            q_cmd = q_goal + np.clip(q_cmd - q_goal + err, -cfg["sag_max_offset"], cfg["sag_max_offset"])
            self._send([(q_cmd.tolist(), 0.4)])
            time.sleep(0.45)
            q = self._rest()
        if np.max(np.abs(q - q_goal)) > cfg["max_goal_error"] * (1.0 if precise else 1.5):
            raise MotionError(f"arm stopped far from the goal: joint error {np.round(q - q_goal, 3)} rad")

    def _rest(self) -> np.ndarray:
        """Block until the joints stop moving (or the timeout), return the measured joints."""
        t_end = time.monotonic() + self.cfg["arrive_timeout_extra"]
        q_prev, _ = self.measured()
        while time.monotonic() < t_end:
            time.sleep(0.1)
            q, _ = self.measured()
            if np.max(np.abs(q - q_prev)) < self.cfg["rest_tolerance"]:
                return q
            q_prev = q
        return self.measured()[0]

    def _set_jaw(self, ctrl: float, seconds: float = 0.4) -> None:
        cfg = self.cfg
        # map the twin's jaw command (JAW_CLOSED..JAW_OPEN) onto the real gripper joint range
        f = (ctrl - JAW_CLOSED) / (JAW_OPEN - JAW_CLOSED)
        g = cfg["gripper_closed"] + f * (cfg["gripper_open"] - cfg["gripper_closed"])
        q, _ = self.measured()
        self._send([(q.tolist(), max(0.5, seconds))], grip=g)  # hold the arm, move the gripper
        time.sleep(max(0.5, seconds) + 0.3)
        # the twin's jaw and weld follow the command (virtual cube); the real grasp is read in `held`
        self.data.ctrl[self.jaw_act] = ctrl
        self._sync_twin(settle=seconds)

    def _step_until(self, q_target: np.ndarray, timeout: float, tol: float = 0.01) -> None:
        # only the primitives' home move uses this; send it as a joint move
        self._send([(list(q_target), max(self.cfg["min_move_time"], 2.0))])
        self._wait_arm(np.asarray(q_target), 2.0)
        self._sync_twin()

    def go_pose(self, name: str, time_s: float | None = None) -> None:
        """Joint move to a saved pose (``poses.<name>`` in the config), slowly."""
        q_goal = np.asarray(self.cfg.get("poses", {}).get(name) or [], float)
        if q_goal.size != 5:
            raise MotionError(
                f"pose {name!r} is not set: capture it with scripts/real_poses.py --capture {name}"
            )
        if not self.kin.within_limits(q_goal):
            raise MotionError(
                f"pose {name!r} is outside the firmware joint limits: {np.round(q_goal, 3).tolist()}"
            )
        q, _ = self.measured()
        t = float(time_s or max(self.cfg.get("begin_time_s", 3.0), float(np.max(np.abs(q - q_goal))) / 0.4))
        self._send([(q_goal.tolist(), t)])
        self._wait_arm(q_goal, t)
        self._sync_twin()
        self._tcp_cmd = None

    def at_pose(self, name: str, tol: float = 0.03) -> bool:
        """Whether the measured joints are within ``tol`` rad of a saved pose."""
        q_goal = np.asarray(self.cfg.get("poses", {}).get(name) or [], float)
        return q_goal.size == 5 and float(np.max(np.abs(self.measured()[0] - q_goal))) < tol

    def reset(self, seed: int | None = None, on_begin: Any = None):
        # New virtual scene in the twin; the robot goes to the begin pose (as in Show-Harness: a pose over the
        # workspace with the gripper pitched down), then opens the gripper once it is there.
        # ``on_begin()`` runs as soon as the arm rests at the begin pose (the overhead guide takes its first image
        # then and marks it in the background): right away when it is there already, else after the move.
        obs = super().reset(seed)
        if self.cfg.get("poses", {}).get("begin"):
            self._sync_twin(settle=0.0)  # the twin's arm to where the robot is (super().reset put it at home)
            early = on_begin is not None and self.at_pose("begin")
            if early:
                on_begin()
            self.go_pose("begin")
            if on_begin is not None and not early:
                on_begin()
        else:  # fallback: the twin's home keyframe (tool straight down)
            home = self.home_ctrl[self.arm_act].copy()
            q, _ = self.measured()
            t = max(3.0, float(np.max(np.abs(q - home))) / 0.4)
            self._send([(home.tolist(), t)])
            self._wait_arm(home, t)
            self._sync_twin()
            self._tcp_cmd = None
            if on_begin is not None:
                on_begin()
        self.open_gripper()
        self.step_count = 0
        return obs

    # ------------------------------------------------------------------ #
    # torque off for posing by hand (begin / rest poses, z floor)
    # ------------------------------------------------------------------ #
    def set_torque(self, on: bool) -> None:
        from std_srvs.srv import SetBool

        cli = self._node.create_client(SetBool, "/dynamixel_hardware_interface/set_dxl_torque")
        if not cli.wait_for_service(timeout_sec=3.0):
            raise MotionError("set_dxl_torque service not available")
        if on:
            # The trajectory controller would drive back to its last setpoint the moment torque returns:
            # re-target it to where the arm is now first.
            q, g = self.measured()
            self._send([(q.tolist(), 0.3)], grip=g if g is not None else self._grip_target)
            time.sleep(0.5)
        fut = cli.call_async(SetBool.Request(data=bool(on)))
        t0 = time.monotonic()
        while not fut.done() and time.monotonic() - t0 < 3.0:
            time.sleep(0.02)
        time.sleep(0.5)
        self._sync_twin(settle=0.0)

    def free_arm(self, prompt: str) -> np.ndarray:
        """Torque off, let a person pose the arm, torque on where it was left. Returns the joints."""
        self.set_torque(False)
        try:
            input(f"TORQUE OFF (support the arm). {prompt}  Press Enter to lock it there ")
        finally:
            self.set_torque(True)
        return self.measured()[0]

    # ------------------------------------------------------------------ #
    # USB cameras (camera_source="usb")
    # ------------------------------------------------------------------ #
    def _start_cameras(self) -> None:
        import cv2

        self._frames: dict[str, np.ndarray] = {}
        self._caps = {}
        for name, c in self.cfg["cameras"].items():
            cap = cv2.VideoCapture(int(c["device"]), cv2.CAP_V4L2)
            # MJPG: two uncompressed 640x480 streams do not fit through the shared USB hub
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not cap.isOpened():
                raise RuntimeError(f"camera {name} (/dev/video{c['device']}) did not open")
            self._caps[name] = cap

            def reader(name: str = name, cap: Any = cap) -> None:
                while True:
                    ok, f = cap.read()
                    if ok:
                        self._frames[name] = f

            threading.Thread(target=reader, daemon=True).start()
        t0 = time.monotonic()
        while len(self._frames) < len(self._caps):
            if time.monotonic() - t0 > 5:
                raise RuntimeError("cameras produced no frames within 5 s")
            time.sleep(0.05)
        time.sleep(1.5)  # let auto exposure settle
        calib = Path(self.cfg["cameras"]["wrist"]["calibration"])
        self.wrist_calib = yaml.safe_load(calib.read_text()) if calib.exists() else None
        top = Path(self.cfg["cameras"]["front"]["calibration"])
        self.top_calib = yaml.safe_load(top.read_text()) if top.exists() else None
        # the "front" device now looks straight down from above (Show-Harness-style scene camera)
        # a calibration marked stale (the camera was moved) is ignored; the overhead guide needs none
        self.top_view = bool(
            self.top_calib and "affine" in self.top_calib and not self.top_calib.get("stale")
        )

    @staticmethod
    def square(frame_bgr: np.ndarray, size: int) -> np.ndarray:
        """Centre square crop of a camera frame, resized, as RGB (how every image reaches the harness)."""
        import cv2

        h, w = frame_bgr.shape[:2]
        s = min(h, w)
        y0, x0 = (h - s) // 2, (w - s) // 2
        sq = frame_bgr[y0 : y0 + s, x0 : x0 + s]
        return cv2.cvtColor(cv2.resize(sq, (size, size), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)

    def overhead_frame(self) -> np.ndarray:
        """The whole overhead ("front" device) frame as RGB, for marking objects (no square crop)."""
        import cv2

        if self.camera_mode == "twin":
            return super().render_all()["front"]
        time.sleep(0.25)
        return cv2.cvtColor(self._frames["front"], cv2.COLOR_BGR2RGB)

    def render_all(self) -> dict[str, np.ndarray]:
        if self.camera_mode == "twin":
            return super().render_all()
        time.sleep(0.25)  # a fresh frame after the last motion
        return {
            name: self.square(self._frames[name], self.image_size)
            for name in self.cameras
            if name in self._frames
        }

    @property
    def zoom_half(self) -> int:
        """Half size of the zoom crop: the cube should span ~29% of the crop, as in simulation (37 of 128 px)."""
        if self.camera_mode == "twin" or not self.wrist_calib:
            return 64
        if "model" in self.wrist_calib:
            # apparent size of a 3 cm cube below the gripper, from the model
            p0 = np.array([*self.tcp_pos[:2], 0.015])
            a, b = self._project_wrist(p0, 320), self._project_wrist(p0 + np.array([0.0, 0.03, 0.0]), 320)
            cube = float(np.hypot(a[0] - b[0], a[1] - b[1])) if a and b else 100.0
            return int(np.clip(cube / 0.29 / 2, 48, 159))
        if "cube_px_by_drop" not in self.wrist_calib:
            return 64
        tab = np.array(self.wrist_calib["cube_px_by_drop"])
        drop = float(self.tcp_pos[2] - 0.015)
        cube = float(np.interp(drop, tab[:, 0], tab[:, 1]))
        return int(np.clip(cube / 0.29 / 2, 48, 159))

    def _need_calib(self) -> dict:
        if not self.wrist_calib:
            raise RuntimeError("no wrist calibration: run scripts/calibrate_wrist_model.py first")
        return self.wrist_calib

    def camera_pixel(
        self, camera: str, point: np.ndarray, size: int | None = None
    ) -> tuple[float, float] | None:
        if self.camera_mode == "twin":
            return super().camera_pixel(camera, point, size)
        if camera == "front" and getattr(self, "top_view", False):
            # overhead camera: table-plane affine map (heights are ignored; the camera is far above)
            M = np.asarray(self.top_calib["affine"])
            uv = (
                np.r_[np.asarray(point, float)[:2], 1.0]
                @ M
                * ((size or self.image_size) / self.top_calib["square_size"])
            )
            return float(uv[0]), float(uv[1])
        if camera != "wrist":
            raise RuntimeError("the front camera is not calibrated")
        c = self._need_calib()
        if "model" in c:
            return self._project_wrist(np.asarray(point, float), size or self.image_size)
        # The calibration measured where a point directly below the tcp appears, per vertical distance.
        c = self._need_calib()
        size = size or self.image_size
        tab = np.array(c["cross_px_by_drop"])  # rows: drop (m, tcp z - point z), u, v (in a 320 px square)
        drop = float(self.tcp_pos[2] - np.asarray(point, float)[2])
        u = np.interp(drop, tab[:, 0], tab[:, 1]) * size / c["square_size"]
        v = np.interp(drop, tab[:, 0], tab[:, 2]) * size / c["square_size"]
        return float(u), float(v)

    def _project_wrist(self, point_table: np.ndarray, size: int) -> tuple[float, float] | None:
        """Pinhole model fitted by scripts/calibrate_wrist_model.py (camera rigid on link5)."""
        import mujoco

        m = self.wrist_calib["model"]
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "link5")
        p5, R5 = self.data.xpos[bid], self.data.xmat[bid].reshape(3, 3)
        X = point_table + np.array([0.0, 0.0, self.table_z])
        Xc = np.asarray(m["R_link5_cam"]).T @ (R5.T @ (X - p5) - np.asarray(m["t_link5_cam"]))
        if Xc[2] <= 1e-6:
            return None
        k = size / self.wrist_calib["square_size"]
        return float((m["cx"] + m["f"] * Xc[0] / Xc[2]) * k), float((m["cy"] + m["f"] * Xc[1] / Xc[2]) * k)

    def wrist_axes(self) -> tuple[np.ndarray, np.ndarray]:
        if self.camera_mode == "twin":
            return super().wrist_axes()
        import mujoco

        c = self._need_calib()
        if "model" in c:
            # which horizontal target offsets move the below-gripper point up / right in the image, at this pose
            p0 = np.array([*self.tcp_pos[:2], 0.015])
            u0 = np.array(self._project_wrist(p0, 320))
            J = np.column_stack(
                [
                    (np.array(self._project_wrist(p0 + d, 320)) - u0) / 0.01
                    for d in (np.array([0.01, 0, 0]), np.array([0, 0.01, 0]))
                ]
            )
            up, right = np.linalg.solve(J, [0.0, -1.0]), np.linalg.solve(J, [1.0, 0.0])
            return up / np.linalg.norm(up), right / np.linalg.norm(right)
        R = self.data.xmat[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "link5")].reshape(3, 3)
        out = []
        for key in ("image_up_link5", "image_right_link5"):
            v = R @ np.asarray(c[key], float)
            h = v[:2] / max(np.linalg.norm(v[:2]), 1e-9)
            out.append(h)
        return out[0], out[1]

    # ------------------------------------------------------------------ #
    # grasp state
    # ------------------------------------------------------------------ #
    @property
    def held(self) -> bool:
        if self.camera_mode == "twin":
            return super().held  # virtual cube: the twin's weld
        _, g = self.measured()
        cfg = self.cfg
        closing = abs(self._grip_target - cfg["gripper_closed"]) < 1e-6
        if not closing or g is None:
            return False
        # stopped between closed and open: the fingers rest on something (a jaw that never moved is not a grasp)
        lo, hi = sorted((cfg["gripper_closed"], cfg["gripper_open"]))
        return bool(lo + cfg["gripper_held_margin"] < g < hi - cfg["gripper_held_margin"])
