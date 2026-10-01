"""Choose which USB camera is the wrist camera and which is the overhead (front) camera, and how the overhead
image is turned, and write the choice into configs/robot/omx_f.yaml. Run it after (re)attaching the cameras:
/dev/videoN numbers can swap after a replug or reboot, so the config stores the stable /dev/v4l/by-id/ paths.

    python scripts/setup_cameras.py                                   # grabs a frame from every camera, then asks
    python scripts/setup_cameras.py --wrist 2 --front 3 --rotate 180  # numbers as listed by the script

The frames go to results/cameras.jpg, numbered as in the list, to see which camera is which. The overhead
image must show the robot at the bottom (the overhead marking assumes it): results/cameras_front.jpg shows it
turned by 0 / 90 / 180 / 270 degrees clockwise to pick from.
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import cv2
import numpy as np

BY_ID = Path("/dev/v4l/by-id")


def cameras() -> list[Path]:
    """Capture nodes of the attached cameras (each camera also has a metadata node, video-index1)."""
    return sorted(BY_ID.glob("*-video-index0")) if BY_ID.exists() else []


def grab(dev: Path, settle_s: float = 1.5) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(dev), cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    frame, t0 = None, time.monotonic()
    while time.monotonic() - t0 < settle_s:  # let auto exposure settle
        ok, f = cap.read()
        if ok:
            frame = f
    cap.release()
    return frame


def sheet(frames: list[np.ndarray | None], names: list[str]) -> np.ndarray:
    tiles = []
    for i, (f, name) in enumerate(zip(frames, names, strict=True), 1):
        t = np.zeros((240, 320, 3), np.uint8) if f is None else cv2.resize(f, (320, 240))
        label = f"[{i}] {name[:34]}" + ("  (no frame)" if f is None else "")
        cv2.rectangle(t, (0, 0), (320, 22), (0, 0, 0), -1)
        cv2.putText(t, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(t)
    while len(tiles) % 2:
        tiles.append(np.zeros_like(tiles[0]))
    return np.vstack([np.hstack(tiles[i : i + 2]) for i in range(0, len(tiles), 2)])


ROTATIONS = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def rotations(frame: np.ndarray) -> np.ndarray:
    """The overhead frame turned by 0 / 90 / 180 / 270 degrees clockwise, side by side and labelled."""
    tiles = []
    for deg in (0, 90, 180, 270):
        t = frame if deg == 0 else cv2.rotate(frame, ROTATIONS[deg])
        t = cv2.resize(t, (240, 240))
        cv2.rectangle(t, (0, 0), (240, 22), (0, 0, 0), -1)
        cv2.putText(
            t, f"rotate {deg}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
        )
        tiles.append(t)
    return np.hstack(tiles)


def ask(what: str, choices: list[int]) -> int:
    while True:
        try:
            s = input(f"{what}: ").strip()
        except EOFError:
            raise SystemExit(
                "no answer; pass --wrist / --front / --rotate to run without questions"
            ) from None
        if s.isdigit() and int(s) in choices:
            return int(s)


def write_config(path: Path, values: dict[tuple[str, str], object]) -> None:
    """Set cameras.<camera>.<key> in place, keeping the comments of the file."""
    text = path.read_text()
    for (cam, key), value in values.items():
        text, n = re.subn(rf"(\n  {cam}:\n(?:    .*\n)*?    {key}: )\S+", rf"\g<1>{value}", text)
        if n != 1:
            raise SystemExit(
                f"cameras.{cam}.{key} not found in {path} (expected the block layout of the repo)"
            )
    path.write_text(text)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/robot/omx_f.yaml")
    ap.add_argument("--wrist", type=int, help="number of the wrist camera in the list")
    ap.add_argument("--front", type=int, help="number of the overhead camera in the list")
    ap.add_argument(
        "--rotate",
        type=int,
        choices=[0, 90, 180, 270],
        help="clockwise turn that puts the robot at the bottom",
    )
    ap.add_argument("--out", default="results/cameras.jpg")
    a = ap.parse_args()

    devs = cameras()
    if len(devs) < 2:
        raise SystemExit(
            f"found {len(devs)} camera(s) under {BY_ID}; the wrist and overhead cameras are both needed"
        )
    names = [d.name.removeprefix("usb-").removesuffix("-video-index0") for d in devs]
    frames = [grab(d) for d in devs]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), sheet(frames, names))
    for i, name in enumerate(names, 1):
        print(f"  [{i}] {name}")
    print(
        f"one frame per camera: {out}  (the wrist camera sees the table up close, the overhead one the whole rig)"
    )

    numbers = list(range(1, len(devs) + 1))
    wrist = a.wrist or ask("number of the wrist camera", numbers)
    front = a.front or ask("number of the overhead (front) camera", numbers)
    if wrist == front:
        raise SystemExit("the wrist and overhead cameras must differ")
    for k in (wrist, front):
        if k not in numbers:
            raise SystemExit(f"no camera [{k}]")
    rotate = a.rotate
    if rotate is None:
        if frames[front - 1] is None:
            raise SystemExit("the overhead camera gave no frame; check it and run again")
        rot_out = out.with_name(out.stem + "_front.jpg")
        cv2.imwrite(str(rot_out), rotations(frames[front - 1]))
        print(f"the overhead image turned 0 / 90 / 180 / 270 degrees clockwise: {rot_out}")
        rotate = ask("turn that puts the robot at the bottom of the image (0/90/180/270)", [0, 90, 180, 270])
    write_config(
        Path(a.config),
        {
            ("wrist", "device"): devs[wrist - 1],
            ("front", "device"): devs[front - 1],
            ("front", "rotate"): rotate,
        },
    )
    print(
        f"wrote {a.config}: wrist = [{wrist}] {names[wrist - 1]}, front = [{front}] {names[front - 1]},"
        f" overhead image turned {rotate} degrees"
    )


if __name__ == "__main__":
    main()
