"""Leader -> follower relay for tele-operation: a slow approach first, then a velocity-limited stream.

    python -m dlb.johnny.teleop_relay --config configs/robot/omx_f.yaml --status results/johnny/teleop_status.json

Reads the leader's joint stream on ``/leader/joint_trajectory_raw`` (the leader launched through
``dlb/johnny/launch/omx_l_leader_relay.launch.py``) and the follower's ``/joint_states``, and publishes to the
follower's command topic (``command_topic`` in the config, ``/leader/joint_trajectory``):

1. ``waiting``: until both the leader's pose and the follower's joints are known.
2. ``approach``: refuse when the leader's pose is outside the firmware joint limits or would put the fingertips
   below ``--min-z``; otherwise ONE trajectory point to the leader's pose timed from the joint distance at
   ``--approach-speed`` (0.4 rad/s, as ``go_pose``), and wait for it.
3. ``following``: every leader message (rate-limited to ``--rate``) becomes a command that moves at most
   ``--max-step`` rad per joint from the previous command (a velocity limit: 0.02 rad x 50 Hz = 1 rad/s), with
   the gripper clamped to ``[gripper_closed, gripper_open]`` from the config. ``paused`` when the leader stream
   or the follower's joint states stop; the follower holds.

The phase and counters go to ``--status`` (JSON) for Johnny's ``teleop_status``. SIGINT stops the forwarding
and exits; exit code 3 means the approach was refused. ``Relay`` has no ROS in it (tests drive it directly).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ARM_LIMIT_MARGIN = 0.02  # rad inside the firmware limits for the approach check and the clamp


@dataclass
class Settings:
    arm_joints: list[str]
    gripper_joint: str
    grip_lo: float
    grip_hi: float
    approach_speed: float = 0.4  # rad/s for the first move
    approach_min_s: float = 2.0
    max_step: float = 0.02  # rad per forwarded message
    rate_hz: float = 50.0
    stale_s: float = 0.5  # no leader message for this long: paused
    follower_stale_s: float = 1.0
    min_z: float = 0.02  # m above the base plane the fingertips must keep at the approach

    @classmethod
    def from_config(cls, cfg: dict[str, Any], **over: Any) -> Settings:
        lo, hi = sorted((float(cfg["gripper_closed"]), float(cfg["gripper_open"])))
        return cls(list(cfg["arm_joints"]), str(cfg["gripper_joint"]), lo, hi, **over)


@dataclass
class Relay:
    """The logic: feed it the follower's joints and the leader's messages; it calls ``publish(q, grip, t)``."""

    s: Settings
    kin: Any  # OmxKinematics (within_limits, clamp, fk)
    publish: Callable[[list[float], float, float], None]
    now: Callable[[], float] = time.monotonic
    status_path: Path | None = None
    phase: str = "waiting"
    reason: str = ""
    q_meas: np.ndarray | None = None
    grip_meas: float | None = None
    t_meas: float = -1e9
    leader_q: np.ndarray | None = None
    leader_grip: float | None = None
    t_leader: float = -1e9
    cmd_q: np.ndarray | None = None
    cmd_grip: float | None = None
    t_pub: float = -1e9
    t_arrive: float = 0.0
    n_in: int = 0
    n_out: int = 0
    n_clamped: int = 0
    _t_status: float = field(default=-1e9, repr=False)

    # ---- inputs -------------------------------------------------------------
    def on_follower(self, names: list[str], positions: list[float]) -> None:
        idx = {n: i for i, n in enumerate(names)}
        if not all(j in idx for j in self.s.arm_joints):
            return
        self.q_meas = np.array([positions[idx[j]] for j in self.s.arm_joints], float)
        if self.s.gripper_joint in idx:
            self.grip_meas = float(positions[idx[self.s.gripper_joint]])
        self.t_meas = self.now()

    def on_leader(self, names: list[str], positions: list[float]) -> None:
        idx = {n: i for i, n in enumerate(names)}
        if not all(j in idx for j in self.s.arm_joints):
            return
        self.n_in += 1
        self.leader_q = np.array([positions[idx[j]] for j in self.s.arm_joints], float)
        if self.s.gripper_joint in idx:
            self.leader_grip = float(positions[idx[self.s.gripper_joint]])
        self.t_leader = self.now()
        if self.phase == "waiting":
            if self.q_meas is not None:
                self._approach()
        elif self.phase == "approach":
            if self.now() >= self.t_arrive:
                self.phase = "following"
                self._forward()
        elif self.phase == "paused":
            if self.now() - self.t_meas <= self.s.follower_stale_s:
                self.phase = "following"
                self._forward()
        elif self.phase == "following":
            self._forward()

    def tick(self) -> None:
        """Call at ~10 Hz: phase changes that need the clock, and the status file."""
        t = self.now()
        if self.phase == "approach" and t >= self.t_arrive:
            self.phase = "following"
        elif self.phase == "following" and (
            t - self.t_leader > self.s.stale_s or t - self.t_meas > self.s.follower_stale_s
        ):
            self.phase = "paused"
        if t - self._t_status >= 0.5 or self.phase in ("refused", "stopped"):
            self._t_status = t
            self.write_status()

    def stop(self) -> None:
        self.phase = "stopped"
        self.write_status()

    # ---- the two moves --------------------------------------------------------
    def _grip(self) -> float:
        g = self.leader_grip if self.leader_grip is not None else self.grip_meas
        if g is None:
            g = self.s.grip_hi
        return float(min(self.s.grip_hi, max(self.s.grip_lo, g)))

    def _approach(self) -> None:
        q = self.leader_q
        if not self.kin.within_limits(q, ARM_LIMIT_MARGIN):
            margins = np.round(self.kin.limit_margins_deg(q), 1).tolist()
            self._refuse(
                f"the leader's pose {np.round(q, 2).tolist()} rad is outside the follower's joint limits "
                f"(margins {margins} deg): move the leader nearer to the follower's pose and start again"
            )
            return
        z = float(self.kin.fk(q)[0][2])
        if z < self.s.min_z:
            self._refuse(
                f"the leader's pose would put the follower's fingertips {z * 100:.1f} cm above the base, below "
                f"the {self.s.min_z * 100:.0f} cm floor: raise the leader and start again"
            )
            return
        dist = float(np.max(np.abs(q - self.q_meas)))
        t = max(self.s.approach_min_s, dist / self.s.approach_speed)
        self.cmd_q, self.cmd_grip = q.copy(), self._grip()
        self.publish(self.cmd_q.tolist(), self.cmd_grip, t)
        self.n_out += 1
        self.t_pub = self.now()
        self.t_arrive = self.t_pub + t + 0.5
        self.phase = "approach"
        self.reason = f"moving {np.degrees(dist):.0f} deg at most in {t:.1f} s"
        self.write_status()

    def _refuse(self, why: str) -> None:
        self.phase, self.reason = "refused", why
        self.write_status()

    def _forward(self) -> None:
        if self.now() - self.t_pub < 1.0 / self.s.rate_hz - 1e-6:
            return  # rate limit: the leader publishes at 100 Hz
        step = np.clip(self.leader_q - self.cmd_q, -self.s.max_step, self.s.max_step)
        if np.any(np.abs(self.leader_q - self.cmd_q) > self.s.max_step + 1e-9):
            self.n_clamped += 1
        lo = self.kin.joint_limits[:, 0] + ARM_LIMIT_MARGIN
        hi = self.kin.joint_limits[:, 1] - ARM_LIMIT_MARGIN
        self.cmd_q = np.clip(self.cmd_q + step, lo, hi)
        self.cmd_grip = self._grip()
        self.publish(self.cmd_q.tolist(), self.cmd_grip, 1.0 / self.s.rate_hz)
        self.n_out += 1
        self.t_pub = self.now()

    # ---- status ---------------------------------------------------------------
    def status(self) -> dict[str, Any]:
        r = lambda a: None if a is None else np.round(a, 3).tolist()  # noqa: E731
        return {
            "phase": self.phase,
            "reason": self.reason,
            "messages_in": self.n_in,
            "messages_out": self.n_out,
            "clamped": self.n_clamped,
            "leader_q": r(self.leader_q),
            "cmd_q": r(self.cmd_q),
            "follower_q": r(self.q_meas),
            "gripper": None if self.cmd_grip is None else round(self.cmd_grip, 3),
            "pid": os.getpid(),
            "updated": time.time(),
        }

    def write_status(self) -> None:
        if self.status_path is None:
            return
        try:
            self.status_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.status_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.status()))
            os.replace(tmp, self.status_path)
        except OSError:
            pass


def read_status(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


# ------------------------------------------------------------------ #
# ROS
# ------------------------------------------------------------------ #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dlb.johnny.teleop_relay", description=__doc__.split("\n")[0])
    ap.add_argument("--config", default="configs/robot/omx_f.yaml")
    ap.add_argument("--raw", default="/leader/joint_trajectory_raw", help="the leader's parked stream")
    ap.add_argument("--status", default="results/johnny/teleop_status.json")
    ap.add_argument("--rate", type=float, default=Settings.rate_hz, help="forwarded messages per second")
    ap.add_argument("--max-step", type=float, default=Settings.max_step, help="rad per joint per message")
    ap.add_argument("--approach-speed", type=float, default=Settings.approach_speed, help="rad/s")
    ap.add_argument("--min-z", type=float, default=Settings.min_z, help="fingertip floor (m) at the approach")
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(a.config).read_text())
    os.environ.setdefault("ROS_DOMAIN_ID", str(cfg.get("ros_domain_id", 0)))

    import rclpy
    from rclpy.duration import Duration
    from rclpy.executors import ExternalShutdownException
    from sensor_msgs.msg import JointState
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

    from dlb.real.omx_kinematics import OmxKinematics

    settings = Settings.from_config(
        cfg, rate_hz=a.rate, max_step=a.max_step, approach_speed=a.approach_speed, min_z=a.min_z
    )
    rclpy.init()
    node = rclpy.create_node("johnny_teleop_relay")
    pub = node.create_publisher(JointTrajectory, cfg["command_topic"], 10)

    def publish(q: list[float], grip: float, t: float) -> None:
        msg = JointTrajectory()
        msg.joint_names = [*settings.arm_joints, settings.gripper_joint]
        p = JointTrajectoryPoint()
        p.positions = [float(v) for v in q] + [float(grip)]
        p.time_from_start = Duration(seconds=float(t)).to_msg()
        msg.points = [p]
        pub.publish(msg)

    relay = Relay(settings, OmxKinematics(), publish, status_path=Path(a.status))
    node.create_subscription(
        JointState,
        cfg.get("joint_states_topic", "/joint_states"),
        lambda m: relay.on_follower(list(m.name), list(m.position)),
        20,
    )
    node.create_subscription(
        JointTrajectory,
        a.raw,
        lambda m: relay.on_leader(list(m.joint_names), list(m.points[0].positions)) if m.points else None,
        10,
    )
    node.create_timer(0.1, relay.tick)
    print(f"relay: {a.raw} -> {cfg['command_topic']} (status {a.status})", flush=True)
    code = 0
    try:
        while rclpy.ok() and relay.phase != "refused":
            rclpy.spin_once(node, timeout_sec=0.2)
        if relay.phase == "refused":
            print("relay refused:", relay.reason, file=sys.stderr, flush=True)
            code = 3
    except (KeyboardInterrupt, ExternalShutdownException):  # SIGINT: rclpy's handler shuts the context down
        pass
    finally:
        if relay.phase != "refused":
            relay.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    print(f"relay: {relay.phase}, {relay.n_out} commands sent", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
