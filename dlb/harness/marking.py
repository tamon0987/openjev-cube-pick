"""Calibration-free overhead guidance: a VLM marks the gripper and the objects in the overhead image.

No camera calibration file and no markers. The gripper's mark with the tcp from forward kinematics, and an
object's mark with the tcp where the object was later grasped, form table-to-pixel correspondences; ``TableMap``
fits a similarity transform to them online. A coarse move to the object or the bin is then "map the target's
mark to the table, move there", and the wrist-camera servo (or the release) takes over.

The map rests on two priors: the camera looks down with the robot's forward direction towards the top of the
image, and a scale (pixels per metre). The robot measures the scale itself at start-up
(``OverheadGuide.calibrate``): the arm visits a few poses around the begin pose, the gripper is marked in each
overhead image, and a robust fit (``fit_scale``) of the (tcp, pixel) pairs gives the scale and the first pairs
of the map. Nothing about the objects (name, size) enters, and nothing is stored between sessions.

Marking takes ~7-9 s (gpt-5.5, effort low; effort none misplaced the cube by 70 px). The first mark starts in
the background as soon as the arm rests at the begin pose (the episode reset calls ``start_background``), so it
overlaps the reset; the next runs in the background while the wrist servo aligns and grasps.
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
    """The marking prompt; with no objects it asks for the gripper alone (the start-up calibration)."""
    return PROMPT_HEAD + "".join(PROMPT_OBJECT.format(key=key_of(o), name=o) for o in objects) + PROMPT_TAIL


def schema_for(objects: tuple[str, ...]) -> dict[str, Any]:
    keys = [key_of(o) for o in objects]
    props = {"gripper": _POINT, **{k: _POINT for k in keys}, **{f"{k}_box": _BOX for k in keys}}
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


def load_env_file(path: str | Path = ".env") -> None:
    p = Path(path)
    for line in p.read_text().splitlines() if p.exists() else []:
        k, _, v = line.partition("=")
        if k.strip() and not k.lstrip().startswith("#"):
            os.environ.setdefault(k.strip(), v.strip().strip("'\""))


@dataclass
class Marks:
    points: dict[str, np.ndarray | None]  # pixel (u, v) in the marked image, or None when not seen
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

    Points and boxes are keyed by object name (plus "gripper"). Every point is the marked centre; the guide
    steadies the bin's centre with its box (``box_centred``).
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
    return Marks(pts, spread, latency_s, raw, boxes)


def box_centred(m: Marks, name: str) -> None:
    """Steady the centre of the container ``name`` (the bin, which the robot does not move) with its bounding box:
    the mean of the marked centre and the box's centre, the box being the steadier of the two when the arm covers
    part of it. An object to pick keeps its marked centre (its box includes a visible side face)."""
    bb = m.boxes.get(name)
    if bb is None:
        return
    c, p = np.array([(bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2]), m.points.get(name)
    # a centre outside its own box is a confused answer: keep the box's centre then
    inside = p is not None and bb[0] <= p[0] <= bb[2] and bb[1] <= p[1] <= bb[3]
    m.points[name] = (p + c) / 2 if inside else c


class OverheadMarker:
    """Asks a vision model ``n`` times in parallel and keeps the per-coordinate median (one call's latency)."""

    def __init__(
        self,
        model: str = "gpt-5.5",
        n: int = 5,
        effort: str | None = "low",
        object_names: tuple[str, ...] = (),
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

    def _ask(self, data_url: str, names: tuple[str, ...]) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt_for(names)},
                        {"type": "input_image", "image_url": data_url, "detail": "high"},
                    ],
                }
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "marks",
                    "schema": schema_for(names),
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

    def list_objects(self, img: np.ndarray, flat: bool = False) -> tuple[list[str], str | None]:
        """The loose objects on the table and the container to put things into, named by the vision model (one call).

        Used when no object names are given: the names then come from what is on the table, not from a fixed list.
        ``flat``: the place is a flat mark on the table (e.g. a cross of tape), not a container.
        """
        from dlb.contract import image_to_data_url

        place = (
            "flat mark on the table that things are to be put on, such as a cross of tape"
            if flat
            else "container that things can be put into"
        )
        prompt = (
            "This is a top-down camera view of a table. A small robot arm stands at the bottom centre of the image.\n"
            'List in "objects" the loose objects on the table that the arm could pick up, and give in "container" the '
            f"{place} (null if there is none). Leave out the robot itself, its cables, "
            "the cameras and their stands, and the walls. Name each one in 2-4 English words that tell it apart from "
            "the others (its colour and what it is)."
        )
        schema = {
            "type": "object",
            "properties": {
                "objects": {"type": "array", "items": {"type": "string"}},
                "container": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            },
            "required": ["objects", "container"],
            "additionalProperties": False,
        }
        body: dict[str, Any] = {
            "model": self.model,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {
                            "type": "input_image",
                            "image_url": image_to_data_url(img, fmt="JPEG"),
                            "detail": "high",
                        },
                    ],
                }
            ],
            "text": {"format": {"type": "json_schema", "name": "objects", "schema": schema, "strict": True}},
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
        out = json.loads(text)
        names = [n.strip().lower() for n in out["objects"] if n.strip()]
        container = (out["container"] or "").strip().lower() or None
        return [n for n in dict.fromkeys(names) if n != container], container

    def mark_gripper(self, img: np.ndarray) -> Marks:
        """The gripper alone, with the same ``n`` parallel queries and medians (the start-up calibration)."""
        return self.mark(img, names=())

    def mark(self, img: np.ndarray, names: tuple[str, ...] | None = None) -> Marks:
        from dlb.contract import image_to_data_url

        names = self.names if names is None else tuple(names)
        h, w = img.shape[:2]
        url = image_to_data_url(img, fmt="JPEG")
        t0 = time.perf_counter()
        with ThreadPoolExecutor(self.n) as ex:
            futs = [ex.submit(self._ask, url, names) for _ in range(self.n)]
        raw = []
        for f in futs:
            try:
                raw.append(f.result())
            except Exception as e:  # noqa: BLE001 - one failed query should not stop the others
                print("  marking query failed:", e, flush=True)
        if not raw:
            raise RuntimeError("every marking query failed")
        return summarize(raw, w, h, time.perf_counter() - t0, names)


class TableMap:
    """Similarity transform table xy (m) -> overhead pixel, fitted to (table xy, pixel) pairs.

    Pairs come from the marked gripper (with the tcp from forward kinematics) and from the grasp (the object's
    mark and the tcp where it was grasped). Each pair carries a pixel uncertainty: the gripper's mark is poor
    (the fingertips are 8-12 cm above the table, and the model's point drifts towards a nearby object: a mark over
    a cube sat 20 px from where the grasp later found it), the grasp pair is exact up to the object's mark.

    The scale is a weighted least-squares fit with the prior (from the start-up calibration) as one more
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


# Start-up calibration: what a believable fit of the gripper's marks looks like. Pixel values are for a 640 px
# wide image and scale with the width.
CAL_PX_PER_CM = (5.0, 30.0)  # the image shows between ~20 cm and ~1.3 m of table across its width
CAL_MAX_ROTATION_DEG = 30.0  # from the prior: robot forward = image up, robot left = image left
CAL_INLIER_PX = 20.0  # a mark further than this from the fit is a wrong mark (gripper marks scatter 10-20 px)
CAL_MIN_INLIERS = 6  # of the 9 poses (5 let a chance agreement of wrong marks through too often)
CAL_MIN_SPAN_M = 0.05  # the agreeing poses must spread this far in x and in y
CAL_MAX_RESIDUAL_PX = 15.0  # rms distance of the agreeing marks to the fit


@dataclass
class ScaleFit:
    """Result of ``fit_scale``: pixel = a * z + b for z = -y - ix (as ``TableMap.params``)."""

    ok: bool
    reason: str  # why the fit is not believable ("" when ok)
    a: complex
    b: complex
    inlier: np.ndarray  # per pair: agrees with the fit
    residual_px: np.ndarray  # per pair: distance of its mark to the fit (nan without a fit)

    @property
    def px_per_m(self) -> float:
        return abs(self.a)

    @property
    def rotation_deg(self) -> float:
        return float(np.degrees(np.angle(self.a))) if self.a else 0.0

    @property
    def rms_px(self) -> float:
        r = self.residual_px[self.inlier]
        return float(np.sqrt(np.mean(r**2))) if len(r) else float("nan")

    def describe(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "reason": self.reason,
            "px_per_cm": round(self.px_per_m / 100, 2),
            "rotation_deg": round(self.rotation_deg, 1),
            "inliers": int(self.inlier.sum()),
            "pairs": len(self.inlier),
            "rms_px": None if not self.inlier.any() else round(self.rms_px, 1),
        }


def fit_scale(xy: Any, uv: Any, width: int = 640, min_inliers: int = CAL_MIN_INLIERS) -> ScaleFit:
    """Robust similarity fit of table positions ``xy`` (m) to the gripper's marks ``uv`` (pixels).

    One wrong mark must not set the scale (after a 6 cm sideways step the gripper was once marked 70 px straight
    "down" the image, which gave 4.4 px/cm instead of ~11), so:

    * every two poses at least 4 cm apart propose a transform; a proposal is dropped unless the pixel displacement
      points the way the arm moved under the prior (forward = image up, left = image left, within
      ``CAL_MAX_ROTATION_DEG``) and its scale is plausible (``CAL_PX_PER_CM``);
    * the proposal that the most marks agree with (within ``CAL_INLIER_PX``) wins, and is refitted by least
      squares to those marks alone; the others are rejected;
    * the result counts only when ``min_inliers`` marks and more than half of all agree, they spread in both x
      and y, the scale and rotation are plausible, and the residual is small.
    """
    xy, uv = np.asarray(xy, float).reshape(-1, 2), np.asarray(uv, float).reshape(-1, 2)
    n, k = len(xy), width / 640.0
    Z, W = -xy[:, 1] - 1j * xy[:, 0], uv[:, 0] + 1j * uv[:, 1]
    lo, hi = 100 * CAL_PX_PER_CM[0] * k, 100 * CAL_PX_PER_CM[1] * k

    def plausible(a: complex) -> bool:
        return lo <= abs(a) <= hi and abs(np.degrees(np.angle(a))) <= CAL_MAX_ROTATION_DEG

    def residuals(a: complex, b: complex) -> np.ndarray:
        return np.abs(W - a * Z - b)

    best: tuple[tuple[int, float], np.ndarray] | None = None
    for i in range(n):
        for j in range(i + 1, n):
            dz = Z[j] - Z[i]
            if abs(dz) < 0.04:
                continue
            a = (W[j] - W[i]) / dz
            if not plausible(a):
                continue
            r = W - a * Z
            res = residuals(a, complex(np.median(r.real), np.median(r.imag)))
            inl = res <= CAL_INLIER_PX * k
            score = (int(inl.sum()), -float(np.median(res[inl])) if inl.any() else 0.0)
            if best is None or score > best[0]:
                best = (score, inl)
    none = np.zeros(n, bool)
    if best is None:
        return ScaleFit(False, "no two marks agree with the arm's moves", 0j, 0j, none, np.full(n, np.nan))
    inl, a, b = best[1], 0j, 0j
    for _ in range(5):  # least squares on the agreeing marks; marks that agree with the refit join
        if inl.sum() < 2:
            break
        zm, wm = Z[inl].mean(), W[inl].mean()
        den = float(np.sum(np.abs(Z[inl] - zm) ** 2))
        if den < 1e-9:
            break
        a = complex(np.sum(np.conj(Z[inl] - zm) * (W[inl] - wm)) / den)
        b = complex(wm - a * zm)
        new = residuals(a, b) <= CAL_INLIER_PX * k
        if (new == inl).all():
            break
        inl = new
    fit = ScaleFit(True, "", a, b, inl, residuals(a, b))
    span = np.ptp(xy[inl], axis=0) if inl.any() else np.zeros(2)
    need = max(
        min_inliers, n // 2 + 1
    )  # and a majority: a few wrong marks can agree with each other by chance
    if inl.sum() < need:
        fit.reason = f"only {int(inl.sum())} of {n} marks agree (need {need})"
    elif span.min() < CAL_MIN_SPAN_M:
        fit.reason = f"the agreeing poses span only {span[0] * 100:.0f} x {span[1] * 100:.0f} cm"
    elif not plausible(a):
        fit.reason = f"implausible fit: {abs(a) / 100:.1f} px/cm, rotated {fit.rotation_deg:.0f} deg"
    elif fit.rms_px > CAL_MAX_RESIDUAL_PX * k:
        fit.reason = f"residual {fit.rms_px:.0f} px"
    fit.ok = not fit.reason
    return fit


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
    """Coarse moves to the marked object ("cube" role) or bin, with as few marks in the critical path as possible.

    * At start-up ``calibrate`` measures the map's scale with the arm itself (no object's size is assumed).

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

    GRIPPER_SD_PX = 20.0  # pixel uncertainty of a gripper mark (see TableMap)
    CUBE_SD_PX = 5.0  # pixel uncertainty of the picked object's mark (the grasp pair)
    # Start-up calibration: tcp offsets (forward, left; m) from the begin pose, at the begin pose's height. A ring
    # of ~6-10 cm in every direction (reachable with the begin pitch, inside the workspace clamp), visited in
    # order round the ring so that the moves are short.
    CALIB_OFFSETS = (
        (0.0, 0.0),
        (0.06, 0.0),
        (0.05, 0.07),
        (0.0, 0.10),
        (-0.04, 0.07),
        (-0.05, 0.0),
        (-0.04, -0.07),
        (0.0, -0.10),
        (0.05, -0.07),
    )
    # added when too few marks agree: between the ring's poses, as far out (poses near the middle say little
    # about the scale)
    CALIB_EXTRA = ((0.03, 0.10), (-0.06, 0.04), (-0.06, -0.04), (0.03, -0.10))

    def __init__(
        self,
        marker: OverheadMarker,
        table_map: TableMap | None = None,
        tol_m: float = 0.012,
        max_step_m: float = 0.25,
        bounds: tuple[tuple[float, float], tuple[float, float]] = ((0.10, 0.30), (-0.20, 0.20)),
        log_dir: str | Path | None = None,
        roles: dict[str, str] | None = None,
        container: bool = True,
    ):
        self.marker, self.map = marker, table_map or TableMap()
        # the place ("bin") is a container with a rim; False: a flat mark on the table (objects on it can be picked
        # again, and the held object is set down on it instead of dropped)
        self.container = container
        # role -> object name: "cube" is the object to pick, "bin" the place to put it (the bin, or another object
        # to stack on). Marks are keyed by object name; with the defaults the names are the roles themselves.
        names = tuple(getattr(marker, "names", ()) or ())
        self.roles = dict(
            roles
            or ({"cube": names[0], "bin": names[1]} if len(names) >= 2 else {"cube": "cube", "bin": "bin"})
        )
        self.bin_name = self.roles["bin"]  # where "bin" places go (roles["bin"] changes when stacking)
        self.seen: dict[str, np.ndarray] = {}  # latest mark of every object name
        self._static: dict[str, list[np.ndarray]] = {}  # marks of the bin, which does not move (averaged)
        self.tol, self.max_step, self.bounds = tol_m, max_step_m, bounds
        self.log_dir = Path(log_dir) if log_dir else None
        self._k = 0
        self._pool = ThreadPoolExecutor(1)
        self._bg: Future | None = None
        self._bg_kind = ""
        self._t_bg = 0.0
        self.calibration: dict[str, Any] | None = None  # the start-up calibration's record
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
        box_centred(m, self.bin_name)
        g, c = m.points.get("gripper"), self._pt(m, "cube")
        for name, p in m.points.items():
            if p is None or name == "gripper":
                continue
            self.seen[name] = p
            if name == self.bin_name:
                self._static.setdefault(name, []).append(p)
        if g is not None and self._plausible_gripper(tcp, g):
            self.map.add(tcp, g, sd_px=float(np.hypot(self.GRIPPER_SD_PX, m.spread_px.get("gripper", 0.0))))
        elif g is not None:
            print(
                f"  gripper mark {np.round(g).tolist()} px is far from where the map puts the tcp: ignored",
                flush=True,
            )
        if c is not None:
            self.last["cube"] = c
        b = self._pt(m, "bin")
        if b is not None:
            if self.roles["bin"] != self.bin_name:
                self.last["bin"] = b  # stacking on another object: its latest mark
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
            }
        )
        if self.log_dir and img is not None:
            self._save(img, m)

    @property
    def needs_calibration(self) -> bool:
        """No scale yet: neither given with the map nor measured by ``calibrate``."""
        return self.map.prior_scale is None

    def calibrate(self, env: Any) -> dict[str, Any]:
        """Measure the map's scale with the arm itself (start-up; nothing about the objects is assumed).

        From the begin pose (nothing held) the arm visits ``CALIB_OFFSETS`` at the begin pose's height and takes
        an overhead image at rest in each; every image is marked for the gripper alone (the marker's ``n`` parallel
        queries and medians) while the arm moves on, and the arm returns to the begin pose. ``fit_scale`` fits the
        (tcp, mark) pairs and rejects marks that disagree; rejected images are marked once more, and if still too
        few agree the arm visits ``CALIB_EXTRA``. The fit's scale becomes the map's prior and the agreeing pairs
        its first pairs. Raises ``RuntimeError`` when no believable fit comes out. Nothing is stored between
        sessions; the images with the marks drawn and ``calib.jsonl`` go to the log directory.

        The gripper is marked ~9 cm above the table, where things look larger than on the table: on the real rig
        the gripper-level scale was 11.5-13.5 px/cm against ~10 after grasps. The poses stay at the begin pose's
        height all the same (lower, the arm could meet objects it knows nothing of yet), the scale enters as a
        prior with ``TableMap.prior_sd``, and the grasp pairs (table level) refine the map as before.
        """
        self.join()
        t0 = time.perf_counter()
        home = env.tcp_pos.copy()
        mark = getattr(self.marker, "mark_gripper", None) or self.marker.mark
        pool = ThreadPoolExecutor(len(self.CALIB_OFFSETS) + len(self.CALIB_EXTRA))
        shots: list[dict[str, Any]] = []
        lo, hi = [b[0] for b in self.bounds], [b[1] for b in self.bounds]

        def visit(offsets: tuple[tuple[float, float], ...]) -> None:
            precise = getattr(env, "precise", True)
            env.precise = False  # travel moves: the tcp is read from the joints at rest
            try:
                for d in offsets:
                    dest = np.clip(home[:2] + d, lo, hi)
                    try:
                        if np.linalg.norm(dest - env.tcp_pos[:2]) > 1e-3:
                            env._move_tcp(np.array([*dest, home[2]]))
                    except Exception as e:  # noqa: BLE001 - an unreachable pose is left out
                        print(
                            f"  calibration: pose {np.round(dest * 100, 1).tolist()} cm skipped: {e}",
                            flush=True,
                        )
                        continue
                    tcp = env.tcp_pos.copy()
                    if any(np.linalg.norm(tcp[:2] - s["tcp"][:2]) < 0.015 for s in shots):
                        continue  # stopped short, next to a pose already taken: nothing new
                    img = env.overhead_frame()
                    shots.append({"tcp": tcp, "img": img, "job": pool.submit(mark, img), "tries": 1})
                if hasattr(env, "go_pose"):
                    env.go_pose("begin")
                else:
                    env._move_tcp(home)
            finally:
                env.precise = precise

        def fit() -> ScaleFit:
            for s in shots:
                if "job" in s:
                    try:
                        m = s.pop("job").result()
                        s["mark"], s["spread"] = m.points.get("gripper"), m.spread_px.get("gripper", 0.0)
                        s["raw"] = [r.get("gripper") for r in m.raw]  # each query's answer (0-1000)
                    except Exception as e:  # noqa: BLE001 - one lost mark only costs a pose
                        print("  calibration: marking failed:", e, flush=True)
                        s["mark"] = None
            seen = [s for s in shots if s["mark"] is not None]
            width = shots[0]["img"].shape[1] if shots else 640
            f = fit_scale([s["tcp"][:2] for s in seen], [s["mark"] for s in seen], width)
            for s in shots:
                s["inlier"], s["residual_px"] = False, None
            for s, inl, res in zip(seen, f.inlier, f.residual_px, strict=True):
                s["inlier"], s["residual_px"] = bool(inl), None if np.isnan(res) else round(float(res), 1)
            return f

        def report(f: ScaleFit, what: str) -> None:
            bad = [
                f"pose {i} at {np.round(s['tcp'][:2] * 100, 1).tolist()} cm: "
                + (
                    "gripper not marked"
                    if s["mark"] is None
                    else f"mark {np.round(s['mark']).astype(int).tolist()} px rejected"
                    + ("" if s["residual_px"] is None else f", {s['residual_px']:.0f} px from the fit")
                )
                for i, s in enumerate(shots)
                if not s["inlier"]
            ]
            print(
                f"  calibration ({what}): {f.px_per_m / 100:.2f} px/cm, rotation {f.rotation_deg:.1f} deg, "
                f"{int(f.inlier.sum())} of {len(shots)} poses agree"
                + (f", residual {f.rms_px:.1f} px" if f.inlier.any() else "")
                + ("" if f.ok else f" -- not accepted: {f.reason}"),
                flush=True,
            )
            for line in bad:
                print("    " + line, flush=True)

        try:
            visit(self.CALIB_OFFSETS)
            f = fit()
            report(f, f"{len(shots)} poses")
            again = [s for s in shots if not s["inlier"]]
            if not f.ok and again:  # a wrong mark is often a one-off: mark the same images once more
                for s in again:
                    s["job"], s["tries"] = pool.submit(mark, s["img"]), s["tries"] + 1
                f = fit()
                report(f, f"{len(again)} marked again")
            if not f.ok:
                visit(self.CALIB_EXTRA)
                f = fit()
                report(f, f"{len(shots)} poses")
        finally:
            pool.shutdown(wait=False)
        rec = {
            **f.describe(),
            "seconds": round(time.perf_counter() - t0, 1),
            "poses": [
                {
                    "tcp": np.round(s["tcp"], 4).tolist(),
                    "mark": None if s["mark"] is None else np.round(s["mark"], 1).tolist(),
                    "spread_px": round(float(s.get("spread", 0.0)), 1),
                    "inlier": s["inlier"],
                    "residual_px": s["residual_px"],
                    "marked": s["tries"],
                    "raw_1000": s.get("raw"),
                }
                for s in shots
            ],
        }
        self.calibration = rec
        if self.log_dir:
            self._save_calibration(shots, rec)
        if not f.ok:
            raise RuntimeError(
                f"overhead calibration failed ({f.reason}): check that the overhead camera sees the gripper "
                "around the begin pose, with the robot at the bottom of the image (cameras.front.rotate)"
            )
        self.map.prior_scale = f.px_per_m
        for s in shots:
            if s["inlier"]:
                self.map.add(s["tcp"], s["mark"], sd_px=float(np.hypot(self.GRIPPER_SD_PX, s["spread"])))
        print(f"  calibration done in {rec['seconds']:.0f} s: map {self.map.describe()}", flush=True)
        return rec

    def _save_calibration(self, shots: list[dict[str, Any]], rec: dict[str, Any]) -> None:
        from PIL import Image, ImageDraw

        self.log_dir.mkdir(parents=True, exist_ok=True)
        for i, s in enumerate(shots):
            im = Image.fromarray(s["img"])
            d = ImageDraw.Draw(im)
            p, c = s["mark"], (0, 255, 0) if s["inlier"] else (255, 0, 0)
            if p is not None:
                d.ellipse([p[0] - 6, p[1] - 6, p[0] + 6, p[1] + 6], outline=c, width=2)
            text = f"pose {i} tcp {np.round(s['tcp'] * 100, 1).tolist()} cm: " + (
                "not marked" if p is None else "ok" if s["inlier"] else "rejected"
            )
            d.text((5, 5), text, fill=c)
            im.save(self.log_dir / f"calib_{i:02d}.jpg", quality=85)
        with open(self.log_dir / "calib.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")

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

    def goto(self, env: Any, target: str, z: float, travel_z: float | None = None) -> GotoResult:
        """Move above ``target`` by its overhead mark and end at height ``z``. With ``travel_z`` (carrying an
        object) the arm rises straight up to that height first, travels level, and only then goes to ``z``."""
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
                c, name = self._pt(fresh, "cube"), self.bin_name
                b = fresh.boxes.get(name)
                if (
                    self.container
                    and b is not None
                    and name != self.roles["cube"]
                    and b[0] <= c[0] <= b[2]
                    and b[1] <= c[1] <= b[3]
                ):
                    return GotoResult(
                        False, None, 1, float("nan"), f"{self.roles['cube']} is inside the {name}", [rec]
                    )
        if not self.map.ready:
            return GotoResult(
                False,
                None,
                1,
                float("nan"),
                "no table map (no scale: the start-up calibration did not run)",
                [rec],
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
            if travel_z is not None and err > 0.03:
                # a held object hangs below the fingers: a diagonal climb from the grasp dragged a carrot over
                # the table, so climb first, then travel level
                top = max(travel_z, z)
                if cur_z < top - 0.01:
                    env._move_tcp(np.array([*here, top]))
                env._move_tcp(np.array([*dest, top]))
            elif cur_z < z - 0.01 and err > 0.03:
                env._move_tcp(np.array([*here, min(z, cur_z + 0.05)]))  # rise clear of the table first
            env._move_tcp(np.array([*dest, z]))
        if target == "cube":
            self.start_background(env)
        return GotoResult(True, goal, 1, err, "moved", [rec])
