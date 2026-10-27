"""What the skills share about the rig: devices, health probes, ROS launches as child processes, ``rig.json``.

Nothing here is a skill. Everything that touches the machine goes through a small function (``list_cameras``,
``grab_frame``, ``http_ok``, ``run``, ``spawn``) so that tests replace it with a fake; every call has a timeout.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

import yaml

from dlb.harness.marking import load_env_file

V4L_BY_ID = Path("/dev/v4l/by-id")
SERIAL_BY_ID = Path("/dev/serial/by-id")
INTERNAL_CAMERA = "Integrated_Webcam"  # the laptop's own camera: never one of the rig's
ARM_USB = "OpenRB-150"  # the controller board of both OMX arms (leader and follower)
DEFAULT_FOLLOWER_PORT = "/dev/ttyACM0"  # the follower launch's default; this rig's leader is ttyACM1
DEFAULT_LEADER_PORT = "/dev/ttyACM1"
OPENJEV_URL = "http://127.0.0.1:8080"
STT_URL = "http://127.0.0.1:8010"
ROS_SETUP = "source /opt/ros/jazzy/setup.bash && source ~/ros2_ws/install/setup.bash"
FOLLOWER_LAUNCH = "ros2 launch open_manipulator_bringup omx_f_follower_ai.launch.py port_name:={port}"
LEADER_LAUNCH = "ros2 launch open_manipulator_bringup omx_l_leader_ai.launch.py port_name:={port}"
OUT_DIR = Path("results/johnny")
RIG_JSON = OUT_DIR / "rig.json"  # what identify_cameras / identify_arms found (by-id paths: stable)


def load_config(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text())


# ------------------------------------------------------------------ #
# devices
# ------------------------------------------------------------------ #
def list_cameras() -> list[Path]:
    """Capture nodes of the rig's USB cameras (by-id, stable across replugs); the laptop's webcam left out."""
    if not V4L_BY_ID.exists():
        return []
    return sorted(p for p in V4L_BY_ID.glob("*-video-index0") if INTERNAL_CAMERA not in p.name)


def camera_label(dev: str | Path) -> str:
    return Path(dev).name.removeprefix("usb-").removesuffix("-video-index0")


def list_arm_ports() -> list[Path]:
    """The arms' USB serial ports (by-id); ``resolve`` gives the /dev/ttyACMn of the moment."""
    if not SERIAL_BY_ID.exists():
        return []
    return sorted(p for p in SERIAL_BY_ID.iterdir() if ARM_USB in p.name)


def resolve(dev: str | Path) -> str:
    return os.path.realpath(str(dev))


def grab_frame(dev: str | Path, settle_s: float = 1.0) -> Any:
    """One frame (BGR) from a camera, or None when it gives none. Opens and releases the device."""
    try:
        import cv2

        cap = cv2.VideoCapture(str(dev), cv2.CAP_V4L2)
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            frame, t0 = None, time.monotonic()
            while time.monotonic() - t0 < settle_s:  # let auto exposure settle
                ok, f = cap.read()
                if ok:
                    frame = f
                elif frame is None and time.monotonic() - t0 > 0.5:
                    break  # not even one frame: do not sit on a dead device
            return frame
        finally:
            cap.release()
    except Exception:  # noqa: BLE001 - a camera that cannot be opened gives no frame
        return None


# ------------------------------------------------------------------ #
# probes
# ------------------------------------------------------------------ #
def http_ok(url: str, timeout: float = 2.0) -> bool:
    try:
        import httpx

        return httpx.get(url, timeout=timeout).status_code == 200
    except Exception:  # noqa: BLE001 - not reachable means not up
        return False


def run(cmd: list[str], timeout: float = 10.0, env: dict[str, str] | None = None) -> Any:
    """``subprocess.run`` with a timeout; None when the command could not run or timed out."""
    try:
        return subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, env=env, stdin=subprocess.DEVNULL
        )
    except (subprocess.TimeoutExpired, OSError):
        return None


def docker_state(name: str) -> str | None:
    """``running`` / ``exited`` / ... for a container, None when there is no such container (or no docker)."""
    r = run(["docker", "inspect", "-f", "{{.State.Status}}", name], timeout=10)
    return r.stdout.strip() if r is not None and r.returncode == 0 else None


def docker_start(name: str) -> bool:
    r = run(["docker", "start", name], timeout=30)
    return r is not None and r.returncode == 0


def docker_stop(names: list[str]) -> list[str]:
    """Stops the containers that exist; returns the names stopped."""
    todo = [n for n in names if docker_state(n) == "running"]
    if todo:
        run(["docker", "stop", *todo], timeout=60)
    return todo


def openai_key(env_file: str | Path = ".env") -> bool:
    load_env_file(env_file)
    return bool(os.environ.get("OPENAI_API_KEY"))


def openjev_ok() -> bool:
    """Healthy = one of the paths the Jev backend probes answers 200 (it takes 1-3 min after a start)."""
    return any(http_ok(f"{OPENJEV_URL}{path}") for path in ("/health", "/ready", "/v1/models"))


def stt_ok() -> bool:
    return http_ok(f"{STT_URL}/health")


# ------------------------------------------------------------------ #
# ROS: launches as child processes of Johnny
# ------------------------------------------------------------------ #
def ros_env(cfg: dict[str, Any]) -> dict[str, str]:
    env = dict(os.environ)
    env["ROS_DOMAIN_ID"] = str(cfg.get("ros_domain_id", 0))
    return env


def ros_shell(cmd: str) -> list[str]:
    """The command in a login shell with ROS 2 and the workspace sourced (``exec``: the launch gets the pid)."""
    return ["bash", "-lc", f"{ROS_SETUP} && exec {cmd}"]


def ros_topics(cfg: dict[str, Any], timeout: float = 10.0) -> list[str] | None:
    r = run(ros_shell("ros2 topic list"), timeout=timeout, env=ros_env(cfg))
    if r is None or r.returncode != 0:
        return None
    return [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]


def joint_states_up(cfg: dict[str, Any], timeout: float = 10.0) -> bool:
    """A follower driver (ours or one launched in a terminal) publishes the joint states.

    The topic alone is not enough: a robot_state_publisher left behind by a dead launch subscribes to it, so the
    topic stays listed with no publisher. Ask for a message instead."""
    topic = cfg.get("joint_states_topic", "/joint_states")
    r = run(
        ros_shell(f"ros2 topic echo --once --timeout {max(1, int(timeout))} {topic}"),
        timeout + 5,
        ros_env(cfg),
    )
    return r is not None and r.returncode == 0 and "position" in (r.stdout or "")


def spawn(ctx: Any, name: str, cmd: str, env: dict[str, str]) -> subprocess.Popen:
    """Start ``cmd`` (ROS sourced) in its own session, output to results/johnny/<name>.log; ``ctx.processes[name]``."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log = open(OUT_DIR / f"{name}.log", "ab")  # noqa: SIM115 - the child owns the handle
    p = subprocess.Popen(
        ros_shell(cmd),
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,  # the whole launch tree is one process group: SIGINT reaches every node
        env=env,
    )
    log.close()
    ctx.processes[name] = p
    ctx.log(f"started {name} (pid {p.pid}): {cmd}")
    return p


def stop(ctx: Any, name: str, timeout: float = 10.0) -> str:
    """SIGINT to the process group (a launch shuts its nodes down on it), then SIGTERM, then SIGKILL."""
    p = ctx.processes.pop(name, None)
    if p is None or p.poll() is not None:
        return "not running"
    for sig, wait in ((signal.SIGINT, timeout), (signal.SIGTERM, 5.0), (signal.SIGKILL, 5.0)):
        try:
            os.killpg(p.pid, sig)
        except ProcessLookupError:
            return "stopped"
        try:
            p.wait(timeout=wait)
            ctx.log(f"stopped {name} with {sig.name}")
            return "stopped" if sig == signal.SIGINT else "killed"
        except subprocess.TimeoutExpired:
            continue
    return "still running"  # not expected after SIGKILL


# ------------------------------------------------------------------ #
# rig.json and the arms' ports
# ------------------------------------------------------------------ #
def load_rig() -> dict[str, Any]:
    try:
        return json.loads(RIG_JSON.read_text())
    except (OSError, ValueError):
        return {}


def save_rig(**sections: Any) -> dict[str, Any]:
    rig = {**load_rig(), **sections}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RIG_JSON.write_text(json.dumps(rig, indent=2, ensure_ascii=False) + "\n")
    return rig


def arm_ports(ctx: Any, cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """``{"follower": "/dev/ttyACM0" | None, "leader": ..., "identified": bool, "found": [...]}``.

    Identified arms (``identify_arms``: by-id names in ``ctx.state["arms"]`` / rig.json) are resolved to the
    /dev/ttyACMn of the moment. Without that: the follower is the config's ``follower_port`` or the launch's
    default /dev/ttyACM0, the leader the other arm port (/dev/ttyACM1 on this rig).
    """
    ident = ctx.state.get("arms") or load_rig().get("arms") or {}
    devs = {p.name: resolve(p) for p in list_arm_ports()}
    out: dict[str, Any] = {
        "follower": None,
        "leader": None,
        "identified": False,
        "found": sorted(devs.values()),
    }
    for role in ("follower", "leader"):
        by_id = (ident.get(role) or {}).get("by_id")
        if by_id in devs:
            out[role] = devs[by_id]
    out["identified"] = bool(out["follower"] and out["leader"])
    if out["follower"] is None:
        want = str((cfg or {}).get("follower_port") or DEFAULT_FOLLOWER_PORT)
        out["follower"] = want if want in devs.values() else None
    if out["leader"] is None:
        others = [d for d in devs.values() if d != out["follower"]]
        if DEFAULT_LEADER_PORT in others:
            out["leader"] = DEFAULT_LEADER_PORT
        elif len(others) == 1:
            out["leader"] = others[0]
    return out


# ------------------------------------------------------------------ #
# the follower driver (shared by setup, teleop and the robot skills)
# ------------------------------------------------------------------ #
def follower_running(ctx: Any, cfg: dict[str, Any], probe: bool = True) -> bool:
    """Our launch is alive, the robot's own ROS node sees joint states, or (``probe``) the topic exists."""
    if ctx.running("follower"):
        return True
    env = getattr(ctx.robot, "env", None)
    if env is not None and time.monotonic() - getattr(env, "_stamp", 0.0) < 2.0:
        return True
    return probe and joint_states_up(cfg)


def start_follower(ctx: Any, cfg: dict[str, Any], wait_s: float = 25.0) -> tuple[bool, str]:
    """Launch the follower driver unless one runs already; wait for the joint states (``wait_s``)."""
    if follower_running(ctx, cfg):
        return True, "the follower driver is already running"
    port = arm_ports(ctx, cfg)["follower"]
    if port is None:
        return False, (
            "no follower arm port: connect the follower's USB and power it on (identify_arms tells which port is which)"
        )
    p = spawn(ctx, "follower", FOLLOWER_LAUNCH.format(port=port), ros_env(cfg))
    ctx.state["follower_port"] = port
    t0 = time.monotonic()
    while time.monotonic() - t0 < wait_s:
        if p.poll() is not None:
            return (
                False,
                f"the follower launch exited right away (code {p.returncode}); see {OUT_DIR}/follower.log",
            )
        if joint_states_up(cfg, timeout=5.0):
            return True, f"started the follower driver on {port} (torque on, the arm holds its pose)"
        time.sleep(1.0)
    return (
        False,
        f"the follower launch runs but no joint states after {wait_s:.0f} s; see {OUT_DIR}/follower.log",
    )
