"""Calibration-free overhead guidance: a VLM marks the gripper and the objects in the overhead image.

No camera calibration and no markers. The gripper's mark with the tcp from forward kinematics, and the cube's
mark with the tcp where the cube was later grasped, form table-to-pixel correspondences; ``TableMap`` fits a
similarity transform to them online. A coarse move to the cube or the bin is then "map the target's mark to
the table, move there", and the wrist-camera servo (or the release) takes over.

The map rests on two priors of the setup: the camera looks down with the robot's forward direction towards
the top of the image, and the cube's apparent size (3 cm edge) gives the scale. Marking takes ~7-9 s (gpt-5.5,
effort low; effort none misplaced the cube by 70 px). The first mark starts in the background as soon as the
arm rests at the begin pose (the episode reset calls ``start_background``), so it overlaps the reset; the next
runs in the background while the wrist servo aligns and grasps.
"""

from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

PROMPT_HEAD = """This is a top-down camera view of a table. A small robot arm stands at the bottom centre of the image.
Give, in normalised image coordinates (x from 0 at the left edge to 1000 at the right edge, y from 0 at the top
edge to 1000 at the bottom edge):
- "gripper": the point on the table directly below the gripper's fingertips (the end of the arm),"""
PROMPT_OBJECT = """
- "{key}": the centre of the {name},
- "{key}_box": the {name}'s bounding box [x_min, y_min, x_max, y_max],"""
PROMPT_TAIL = """
Give null for anything not visible."""

_POINT = {
    "anyOf": [{"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2}, {"type": "null"}]
}
_BOX = {
    "anyOf": [{"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4}, {"type": "null"}]
}


def key_of(name: str) -> str:
    """JSON key for an object name ("red cube" -> "red_cube")."""
    return "".join(c if c.isalnum() else "_" for c in name.strip().lower())


def prompt_for(objects: tuple[str, ...]) -> str:
    return PROMPT_HEAD + "".join(PROMPT_OBJECT.format(key=key_of(o), name=o) for o in objects) + PROMPT_TAIL


def schema_for(objects: tuple[str, ...]) -> dict[str, Any]:
    keys = [key_of(o) for o in objects]
    props = {"gripper": _POINT, **{k: _POINT for k in keys}, **{f"{k}_box": _BOX for k in keys}}
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


def is_cube(name: str) -> bool:
    return "cube" in name.lower() or "キューブ" in name


def load_env_file(path: str | Path = ".env") -> None:
    p = Path(path)
    for line in p.read_text().splitlines() if p.exists() else []:
        k, _, v = line.partition("=")
        if k.strip() and not k.lstrip().startswith("#"):
            os.environ.setdefault(k.strip(), v.strip().strip("'\""))


@dataclass
class Marks:
    points: dict[str, np.ndarray | None]  # pixel (u, v) in the marked image, or None when not seen
    cube_size_px: float | None
    spread_px: dict[str, float]
    latency_s: float
    raw: list[dict[str, Any]] = field(default_factory=list)
    boxes: dict[str, np.ndarray | None] = field(
        default_factory=dict
    )  # median boxes [x0, y0, x1, y1] (pixels)


def summarize(
    raw: list[dict[str, Any]],
    w: int,
    h: int,
    latency_s: float = 0.0,
    objects: tuple[str, ...] = ("cube", "bin"),
) -> Marks:
    """Per-coordinate medians of the answers (a majority must see a target; the median ignores the odd one out).

    Points and boxes are keyed by object name (plus "gripper"). A large object's centre (the bin) is the mean of
    its marked centre and the centre of its bounding box: its box is the steadier of the two when the arm covers
    part of it. A cube's box includes a visible side face, so its marked centre is kept; the median size of the
    cubes' boxes is the scale cue.
    """
    scale = np.array([w / 1000, h / 1000])
    pts, spread, boxes = {}, {}, {}
    for name, k in [("gripper", "gripper"), *((o, key_of(o)) for o in objects)]:
        P = np.array([r[k] for r in raw if r.get(k)], float).reshape(-1, 2) * scale
        if len(P) * 2 > len(raw):
            m = np.median(P, axis=0)
            pts[name], spread[name] = m, float(np.median(np.linalg.norm(P - m, axis=1)))
        else:
            pts[name] = None
    for o in objects:
        B = np.array([r[f"{key_of(o)}_box"] for r in raw if r.get(f"{key_of(o)}_box")], float).reshape(-1, 4)
        boxes[o] = np.median(B, axis=0) * np.r_[scale, scale] if len(B) * 2 > len(raw) else None
    sizes = [
        float(np.mean([b[2] - b[0], b[3] - b[1]])) for o, b in boxes.items() if b is not None and is_cube(o)
    ]
    size = float(np.median(sizes)) if sizes else None
    for o, bb in boxes.items():
        if bb is None or is_cube(o):
            continue
        c = np.array([(bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2])
        # a centre outside its own box is a confused answer: keep the box's centre then
        inside = pts[o] is not None and bb[0] <= pts[o][0] <= bb[2] and bb[1] <= pts[o][1] <= bb[3]
        pts[o] = (pts[o] + c) / 2 if inside else c
    return Marks(pts, size, spread, latency_s, raw, boxes)


class OverheadMarker:
    """Asks a vision model ``n`` times in parallel and keeps the per-coordinate median (one call's latency)."""

    def __init__(
        self,
        model: str = "gpt-5.5",
        n: int = 5,
        effort: str | None = "low",
        object_names: tuple[str, ...] = ("orange cube", "black bin"),
        timeout_s: float = 120.0,
    ):
        import httpx

        load_env_file()
        self.model, self.n, self.effort, self.names = model, n, effort, tuple(object_names)
        self.client = httpx.Client(
            base_url="https://api.openai.com/v1",
            timeout=timeout_s,
            headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
        )

    def _ask(self, data_url: str) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt_for(self.names)},
                        {"type": "input_image", "image_url": data_url, "detail": "high"},
                    ],
                }
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "marks",
                    "schema": schema_for(self.names),
                    "strict": True,
                }
            },
        }
        if self.effort:
            body["reasoning"] = {"effort": self.effort}
        r = self.client.post("/responses", json=body)
        if r.status_code != 200:
            raise RuntimeError(f"{self.model}: {r.status_code} {r.text[:300]}")
        text = "".join(
            c.get("text", "")
            for o in r.json().get("output", [])
            if o.get("type") == "message"
            for c in o.get("content", [])
            if c.get("type") == "output_text"
        )
        return json.loads(text)

    def mark(self, img: np.ndarray) -> Marks:
        from dlb.contract import image_to_data_url

        h, w = img.shape[:2]
        url = image_to_data_url(img, fmt="JPEG")
        t0 = time.perf_counter()
        with ThreadPoolExecutor(self.n) as ex:
            futs = [ex.submit(self._ask, url) for _ in range(self.n)]
        raw = []
        for f in futs:
            try:
                raw.append(f.result())
            except Exception as e:  # noqa: BLE001 - one failed query should not stop the others
                print("  marking query failed:", e, flush=True)
        if not raw:
            raise RuntimeError("every marking query failed")
        return summarize(raw, w, h, time.perf_counter() - t0, self.names)


class TableMap:
    """Similarity transform table xy (m) -> overhead pixel, fitted to (table xy, pixel) pairs.

    Pairs come from the marked gripper (with the tcp from forward kinematics) and from the grasp (the cube's
    mark and the tcp where it was grasped). Each pair carries a pixel uncertainty: the gripper's mark is poor
    (the fingertips are 8-12 cm above the table, and the model's point drifts towards a nearby cube: a mark over
    the cube sat 20 px from where the grasp later found it), the grasp pair is exact up to the cube's mark.

    The scale is a weighted least-squares fit with the prior (from the cube's apparent size) as one more
    observation: ``a = (sum w_i conj(z_i) w_i + s0 / sd0^2) / (sum w_i |z_i|^2 + 1 / sd0^2)``. With pairs a few
    cm apart the prior dominates (three pairs within 9 cm fitted 5.5 px/cm against a true ~6.6 and put the bin
    6 cm too far left); pairs spread over the table take over. The rotation stays at the prior (robot forward =
    image up, robot left = image left) until the pairs span ``free_rotation_m``: a rotation fitted to pairs a few
    cm apart turns 5 px of marking noise into tens of degrees (the first real run fitted -46 degrees).
    """

    def __init__(self, px_per_m: float | None = None, free_rotation_m: float = 0.08, prior_sd: float = 0.15):
        self.xy: list[np.ndarray] = []
        self.uv: list[np.ndarray] = []
        self.sd: list[float] = []
        self.prior_scale = px_per_m
        self.prior_sd = prior_sd  # relative standard deviation of the prior scale
        self.free_rotation_m = free_rotation_m
        self._lock = threading.Lock()  # background marks add pairs while the main thread adds the grasp pair

    def add(self, xy: np.ndarray, uv: np.ndarray, sd_px: float = 10.0) -> None:
        with self._lock:
            self.xy.append(np.asarray(xy, float)[:2].copy())
            self.uv.append(np.asarray(uv, float)[:2].copy())
            self.sd.append(float(sd_px))

    @property
    def ready(self) -> bool:
        return (len(self.xy) >= 2 and self._spread() > 0.02) or (
            len(self.xy) >= 1 and self.prior_scale is not None
        )

    def _spread(self) -> float:
        X = np.array(self.xy)
        return float(np.linalg.norm(X - X.mean(0), axis=1).max()) if len(X) else 0.0

    def params(self) -> tuple[complex, complex]:
        """(a, b) with pixel u + iv = a * z + b for the table point z = -y - ix (x forward, y left).

        In these coordinates the prior (forward = image up, left = image left) is a real a (the scale); a
        rotated camera makes a complex.
        """
        with self._lock:
            xy, uv, sd = list(self.xy), list(self.uv), list(self.sd)
        if not xy:
            raise RuntimeError("the table map needs a mark of the gripper")
        Z = np.array([complex(-p[1], -p[0]) for p in xy])
        W = np.array([complex(q[0], q[1]) for q in uv])
        wt = 1.0 / np.array(sd) ** 2
        zm, wm = complex(np.sum(wt * Z) / wt.sum()), complex(np.sum(wt * W) / wt.sum())
        Zc, Wc = Z - zm, W - wm
        s0 = self.prior_scale
        lam = 1.0 / (self.prior_sd * s0) ** 2 if s0 else 0.0
        X = np.array(xy)
        spread = float(np.linalg.norm(X - X.mean(0), axis=1).max())
        fit = len(Z) >= 2 and spread > 0.02
        if not fit and s0 is None:
            raise RuntimeError("the table map needs two marks, or one mark and a scale")
        num = complex(np.sum(wt * np.conj(Zc) * Wc)) + lam * (s0 or 0.0) if fit else complex(s0, 0.0)
        den = float(np.sum(wt * np.abs(Zc) ** 2)) + lam if fit else 1.0
        a = num / den
        if spread < self.free_rotation_m:
            a = complex(max(1e-6, a.real), 0.0)
        return a, wm - a * zm

    def to_px(self, xy: np.ndarray) -> np.ndarray:
        a, b = self.params()
        w = a * complex(-xy[1], -xy[0]) + b
        return np.array([w.real, w.imag])

    def to_table(self, uv: np.ndarray) -> np.ndarray:
        a, b = self.params()
        z = (complex(uv[0], uv[1]) - b) / a
        return np.array([-z.imag, -z.real])

    def residual_px(self) -> float:
        if len(self.xy) < 3:
            return float("nan")
        return float(
            np.mean([np.linalg.norm(self.to_px(x) - u) for x, u in zip(self.xy, self.uv, strict=True)])
        )

    def describe(self) -> dict[str, Any]:
        a, _ = self.params()
        return {
            "pairs": len(self.xy),
            "px_per_cm": round(abs(a) / 100, 2),
            "prior_px_per_cm": None if self.prior_scale is None else round(self.prior_scale / 100, 2),
            "rotation_deg": round(float(np.degrees(np.angle(a))), 1),
            "residual_px": round(self.residual_px(), 1),
        }


@dataclass
class GotoResult:
    ok: bool
    target_xy: np.ndarray | None
    iterations: int
    error_cm: float
    reason: str
    log: list[dict[str, Any]]


@dataclass
class _Shot:
    """A mark with the tcp at the moment its image was taken (the arm at rest)."""

    kind: str  # "begin": the first mark of an episode, for goto_cube; "refine": map refinement only
    tcp: np.ndarray
    marks: Marks
    wait_s: float = 0.0  # time the caller waited for it (critical path)


class OverheadGuide:
    """Coarse moves to the marked cube or bin, with as few marks in the critical path as possible.

    * The episode reset starts the first mark in the background (``start_background(kind="begin")``) once the
      arm rests at the begin pose; it runs while the reset finishes and the planner starts.
    * ``goto_cube``: waits for that mark (or marks now when there is none or it failed), one move by the
      difference of the cube's and the gripper's marks in the same image. The wrist servo absorbs what is left.
    * right after the move a second mark runs in the background (image and tcp captured at rest), and the grasp
      adds the pair (grasp tcp, cube mark): by ``goto_bin`` the map has scale and offset from three pairs.
    * ``goto_bin``: no new mark; the average of the bin's marks so far and the map give the goal.

    A stationary re-mark just before the release (gripper and bin in one image, as for the cube) is left out:
    it would add a whole mark (~8 s) to the critical path, since the image must be taken at rest over the bin,
    and the gripper's mark drifts towards whatever object lies below it (20 px, ~3 cm, over the cube), so it
    would rarely beat the map. The scale prior and the grasp pair bring the bin within ~3 cm instead.
    """

    CUBE_EDGE_M = 0.03
    # the cube's box spans its top face and a visible side (7.6 and 8.5 px/cm from the box on the real table,
    # ~6.6 measured): the box edge is ~1.2 cube edges
    BOX_PER_EDGE = 1.2
    GRIPPER_SD_PX = 20.0  # pixel uncertainty of a gripper mark (see TableMap)
    CUBE_SD_PX = 5.0  # pixel uncertainty of the cube's mark (the grasp pair)

    def __init__(
        self,
        marker: OverheadMarker,
        table_map: TableMap | None = None,
        tol_m: float = 0.012,
        max_step_m: float = 0.25,
        bounds: tuple[tuple[float, float], tuple[float, float]] = ((0.10, 0.30), (-0.20, 0.20)),
        log_dir: str | Path | None = None,
        roles: dict[str, str] | None = None,
    ):
        self.marker, self.map = marker, table_map or TableMap()
        # role -> object name: "cube" is the object to pick, "bin" the place to put it (the bin, or another object
        # to stack on). Marks are keyed by object name; with the defaults the names are the roles themselves.
        names = tuple(getattr(marker, "names", ()) or ())
        self.roles = dict(
            roles
            or ({"cube": names[0], "bin": names[1]} if len(names) >= 2 else {"cube": "cube", "bin": "bin"})
        )
        self.seen: dict[str, np.ndarray] = {}  # latest mark of every object name
        self._static: dict[str, list[np.ndarray]] = {}  # marks of objects the robot does not move (averaged)
        self.tol, self.max_step, self.bounds = tol_m, max_step_m, bounds
        self.log_dir = Path(log_dir) if log_dir else None
        self._k = 0
        self._pool = ThreadPoolExecutor(1)
        self._bg: Future | None = None
        self._bg_kind = ""
        self._t_bg = 0.0
        self._cube_sizes: list[float] = []
        self._bins: list[np.ndarray] = []
        self.shots: list[dict[str, Any]] = []  # every mark of the episode, for the log
        self.last: dict[
            str, np.ndarray
        ] = {}  # latest mark of each target (pixels); the bin: mean of its marks
        self.cube_mark: np.ndarray | None = None  # the cube's mark that the current approach used

    def reset(self) -> None:
        self.join()
        self.last, self.cube_mark, self._bins, self.shots = {}, None, [], []
        self.seen, self._static = {}, {}

    def set_targets(self, pick: str | None = None, place: str | None = None) -> None:
        """Choose the object to pick and the place (object) to put it on; earlier marks of them are reused."""
        if pick and pick != self.roles["cube"]:
            self.roles["cube"], self.cube_mark = pick, None
            self.last.pop("cube", None)
            if pick in self.seen:
                self.last["cube"] = self.seen[pick]
        if place and place != self.roles["bin"]:
            self.roles["bin"] = place
            self._bins = list(self._static.get(place, []))
            self.last.pop("bin", None)
            if self._bins:
                B = np.array(self._bins)
                self.last["bin"] = np.median(B, axis=0) if len(B) >= 3 else B.mean(axis=0)
            elif place in self.seen:
                self.last["bin"] = self.seen[place]

    def _pt(self, m: Marks, role: str) -> np.ndarray | None:
        """A role's point in a mark (by the object's name, or the role itself for marks keyed by role)."""
        name = self.roles.get(role, role)
        return m.points.get(name) if name in m.points else m.points.get(role)

    def _use(self, img: np.ndarray | None, tcp: np.ndarray, m: Marks, kind: str = "") -> None:
        g, c = m.points.get("gripper"), self._pt(m, "cube")
        for name, p in m.points.items():
            if p is None or name == "gripper":
                continue
            self.seen[name] = p
            if not is_cube(name) and name not in ("cube",):
                self._static.setdefault(name, []).append(p)
        if g is not None and self._plausible_gripper(tcp, g):
            self.map.add(tcp, g, sd_px=float(np.hypot(self.GRIPPER_SD_PX, m.spread_px.get("gripper", 0.0))))
        elif g is not None:
            print(
                f"  gripper mark {np.round(g).tolist()} px is far from where the map puts the tcp: ignored",
                flush=True,
            )
        # the cube's box, unless the gripper hides part of it; the median of all boxes seen sets the scale prior
        if m.cube_size_px and (g is None or c is None or np.linalg.norm(g - c) > 1.5 * m.cube_size_px):
            self._cube_sizes.append(m.cube_size_px)
            self.map.prior_scale = float(np.median(self._cube_sizes)) / (self.CUBE_EDGE_M * self.BOX_PER_EDGE)
        elif self.map.prior_scale is None and m.cube_size_px:
            self.map.prior_scale = m.cube_size_px / (self.CUBE_EDGE_M * self.BOX_PER_EDGE)
        if c is not None:
            self.last["cube"] = c
        b = self._pt(m, "bin")
        if b is not None:
            if is_cube(self.roles["bin"]):
                self.last["bin"] = b  # stacking on a cube: its latest mark
            else:
                self._bins.append(b)  # the bin does not move: average its marks
                B = np.array(self._bins)
                self.last["bin"] = np.median(B, axis=0) if len(B) >= 3 else B.mean(axis=0)
        self.shots.append(
            {
                "kind": kind,
                "tcp": np.round(tcp, 4).tolist(),
                "latency_s": round(m.latency_s, 2),
                "marks": {k: None if v is None else np.round(v, 1).tolist() for k, v in m.points.items()},
                "spread_px": {k: round(v, 1) for k, v in m.spread_px.items()},
                "boxes": {k: None if v is None else np.round(v, 1).tolist() for k, v in m.boxes.items()},
                "cube_size_px": None if m.cube_size_px is None else round(m.cube_size_px, 1),
            }
        )
        if self.log_dir and img is not None:
            self._save(img, m)

    GRIPPER_OUTLIER_PX = 40.0

    def _plausible_gripper(self, tcp: np.ndarray, g: np.ndarray) -> bool:
        """Once the map rests on two pairs or more, a gripper mark far from the tcp's mapped pixel is a wrong mark
        (the model once marked the robot's base, 175 px off, and the map's residual jumped to 68 px)."""
        if len(self.map.xy) < 2 or not self.map.ready:
            return True
        return float(np.linalg.norm(self.map.to_px(tcp) - g)) <= self.GRIPPER_OUTLIER_PX

    def _mark_now(self, env: Any) -> _Shot:
        img, tcp = env.overhead_frame(), env.tcp_pos.copy()
        m = self.marker.mark(img)
        self._use(img, tcp, m, "foreground")
        return _Shot("foreground", tcp, m)

    def start_background(self, env: Any, kind: str = "refine") -> None:
        """Mark the current view in the background (the arm must be at rest; image and tcp are taken now).

        ``kind="begin"``: the episode's first mark, which ``goto_cube`` then waits for instead of marking.
        """
        self.join()
        img, tcp = env.overhead_frame(), env.tcp_pos.copy()

        def job() -> _Shot:
            m = self.marker.mark(img)
            self._use(img, tcp, m, kind)
            return _Shot(kind, tcp, m)

        self._bg, self._bg_kind, self._t_bg = self._pool.submit(job), kind, time.perf_counter()

    @property
    def pending(self) -> str:
        """The kind of the background mark not joined yet ("" when there is none)."""
        return self._bg_kind if self._bg is not None else ""

    def join(self) -> _Shot | None:
        """Wait for the background mark; returns it, or None when there was none or it failed."""
        if self._bg is None:
            return None
        t0 = time.perf_counter()
        try:
            shot = self._bg.result()
            shot.wait_s = time.perf_counter() - t0
        except Exception as e:  # noqa: BLE001 - a lost background mark only costs accuracy (or a foreground mark)
            print("  background mark failed:", e, flush=True)
            shot = None
        self._bg, self._bg_kind = None, ""
        return shot

    def on_grasp(self, env: Any) -> None:
        """The cube is between the fingers: its earlier mark is where the tcp is now (one exact pair)."""
        if self.cube_mark is not None:
            self.map.add(env.tcp_pos, self.cube_mark, sd_px=self.CUBE_SD_PX)
            self.cube_mark = None

    def _save(self, img: np.ndarray, m: Marks) -> None:
        from PIL import Image, ImageDraw

        self.log_dir.mkdir(parents=True, exist_ok=True)
        im = Image.fromarray(img)
        d = ImageDraw.Draw(im)
        palette = [(0, 255, 0), (255, 0, 255), (255, 200, 0), (255, 80, 80), (80, 160, 255)]
        col = {"gripper": (0, 200, 255)}
        for i, k in enumerate(n for n in m.points if n != "gripper"):
            col[k] = palette[i % len(palette)]
        for k, b in m.boxes.items():
            if b is not None:
                d.rectangle([float(v) for v in b], outline=col.get(k, (255, 255, 255)), width=1)
        for k, p in m.points.items():
            if p is not None:
                c = col.get(k, (255, 255, 255))
                d.ellipse([p[0] - 5, p[1] - 5, p[0] + 5, p[1] + 5], outline=c, width=2)
                d.text((p[0] + 7, p[1] - 7), k, fill=c)
        self._k += 1
        im.save(self.log_dir / f"mark_{self._k:03d}.jpg", quality=85)

    def goto(self, env: Any, target: str, z: float) -> GotoResult:
        rec: dict[str, Any] = {"tcp": np.round(env.tcp_pos, 4).tolist()}
        shot = None
        if target == "cube":
            # the episode's first mark, started at the begin pose; a refinement mark is only joined (the cube may
            # have moved since, e.g. after a missed grasp)
            if self.pending == "begin":
                shot = self.join()
                ok = (
                    shot is not None
                    and shot.marks.points.get("gripper") is not None
                    and self._pt(shot.marks, "cube") is not None
                )
                rec.update(
                    begin_mark="used" if ok else "failed", wait_s=round(shot.wait_s, 2) if shot else None
                )
                if not ok:
                    shot = None
            else:
                self.join()
        else:
            # the background mark refines the map and the place's average; one taken at rest at the begin pose
            # (idle, before a place-only instruction) also serves as the fresh image for this move
            got = self.join()
            if (
                got is not None
                and got.kind == "begin"
                and got.marks.points.get("gripper") is not None
                and self._pt(got.marks, target) is not None
            ):
                shot = got
                rec.update(begin_mark="used", wait_s=round(got.wait_s, 2))
        if shot is None and (target == "cube" or target not in self.last):
            shot = self._mark_now(env)
        if shot is not None:
            fresh = shot.marks
            rec.update(
                mark_kind=shot.kind,
                mark_tcp=np.round(shot.tcp, 4).tolist(),
                latency_s=round(fresh.latency_s, 2),
                marks={k: None if v is None else np.round(v, 1).tolist() for k, v in fresh.points.items()},
                spread_px={k: round(v, 1) for k, v in fresh.spread_px.items()},
            )
            if self._pt(fresh, target) is None:
                return GotoResult(
                    False,
                    None,
                    1,
                    float("nan"),
                    f"{self.roles.get(target, target)} not visible in the overhead image",
                    [rec],
                )
            if target == "cube":
                # an object already inside a container (the bin): reaching in would hit its rim
                c = self._pt(fresh, "cube")
                for name, b in fresh.boxes.items():
                    if (
                        b is not None
                        and not is_cube(name)
                        and name != self.roles["cube"]
                        and b[0] <= c[0] <= b[2]
                        and b[1] <= c[1] <= b[3]
                    ):
                        return GotoResult(
                            False, None, 1, float("nan"), f"{self.roles['cube']} is inside the {name}", [rec]
                        )
        if not self.map.ready:
            return GotoResult(
                False, None, 1, float("nan"), "no table map (gripper or cube size not marked)", [rec]
            )
        px = self._pt(shot.marks, target) if shot is not None else self.last[target]
        if target == "bin" and "bin" in self.last:
            px = self.last["bin"]
            rec["bin_marks"] = len(self._bins)
        goal = self.map.to_table(px)
        here = env.tcp_pos[:2].copy()
        if shot is not None and shot.marks.points.get("gripper") is not None:
            # both marks in the same image: move by their difference from where that image was taken (the map's
            # offset error cancels)
            dest0 = shot.tcp[:2] + goal - self.map.to_table(shot.marks.points["gripper"])
        else:
            dest0 = goal
        step = dest0 - here
        if np.linalg.norm(step) > self.max_step:
            step *= self.max_step / np.linalg.norm(step)
        dest = np.clip(here + step, [b[0] for b in self.bounds], [b[1] for b in self.bounds])
        err = float(np.linalg.norm(step))
        rec.update(
            map=self.map.describe(),
            goal_xy=np.round(goal, 4).tolist(),
            move_to=np.round(dest, 4).tolist(),
            error_cm=round(err * 100, 1),
            shots=self.shots,
        )
        wait = (
            f", waited {shot.wait_s:.1f} s for the begin mark"
            if shot is not None and shot.kind == "begin"
            else ""
        )
        print(
            f"  overhead: {target} -> table ({goal[0] * 100:.1f}, {goal[1] * 100:.1f}) cm, move "
            f"{err * 100:.1f} cm, map {rec['map']}"
            + (f", mark {shot.marks.latency_s:.1f} s" if shot is not None else "")
            + wait,
            flush=True,
        )
        if target == "cube":
            self.cube_mark = px.copy()
        cur_z = float(env.tcp_pos[2])
        if err > self.tol or abs(cur_z - z) > 0.01:
            if cur_z < z - 0.01 and err > 0.03:
                env._move_tcp(np.array([*here, min(z, cur_z + 0.05)]))  # rise clear of the table first
            env._move_tcp(np.array([*dest, z]))
        if target == "cube":
            self.start_background(env)
        return GotoResult(True, goal, 1, err, "moved", [rec])
