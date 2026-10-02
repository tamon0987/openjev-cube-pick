"""Voice (or typed) instructions to the real robot: listen, parse into tasks, run them, wait for the next one.

    python -m dlb.voice.agent                      # microphone (the STT server must be up: scripts/stt_server.sh)
    python -m dlb.voice.agent --text               # type instructions instead
    python -m dlb.voice.agent --dry-run --text     # parse only, no robot

While idle at the begin pose the overhead view of every known object is marked in the background, so a new
instruction starts from marks that are at most a few seconds old instead of waiting ~8 s for one. Interrupting a
running task is not wired yet: instructions heard while the robot works are queued and run afterwards.
"""

from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
from pathlib import Path

from dlb.voice.intent import Intent, IntentParser, RobotState, apply, check_sequence


def _speech(args: argparse.Namespace, out: queue.Queue) -> None:
    """Transcripts from the microphone (or stdin with --text) into ``out``; None at the end."""
    try:
        if args.text:
            print("type an instruction (empty line or Ctrl-D to quit)", file=sys.stderr, flush=True)
            for line in sys.stdin:
                if not line.strip():
                    break
                out.put(line.strip())
            return
        from dlb.voice.listen import SILERO_MODEL, Segmenter, SileroSegmenter, Transcriber, listen, mic_frames

        tr = Transcriber(args.stt_url, args.stt_model) if args.stt_model else Transcriber(args.stt_url)
        if not tr.health():
            print(
                f"transcription server not reachable at {args.stt_url} (bash scripts/stt_server.sh)",
                file=sys.stderr,
            )
            return
        seg = SileroSegmenter() if SILERO_MODEL.exists() else Segmenter()
        print("listening...", file=sys.stderr, flush=True)
        for u in listen(mic_frames(args.device), tr, seg):
            print(f"[heard {u.audio_s:.1f} s, text after {u.eou_s:.1f} s] {u.text}", flush=True)
            out.put(u.text)
    finally:
        out.put(None)


class Agent:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        # with no --objects the names come from the overhead image when the robot starts (start_robot)
        self.objects = [o.strip() for o in (args.objects or "").split(",") if o.strip()]
        self.state = RobotState(None, [*self.objects, "bin"], [])
        self.parser = IntentParser(args.intent_model, args.effort or None)
        self.env = self.runner = self.guide = None
        self.episode = 0
        self.log = Path(args.out) / "logs" / args.run_name
        self._last_idle_mark = 0.0
        self._ref = None  # small grey overhead image at the last idle mark (scene-change check)
        self._prev = None
        self._dirty = False  # the scene changed while a mark was running: mark again when it ends

    def start_robot(self) -> None:
        from dlb.backends import build_backend
        from dlb.harness.marking import OverheadGuide, OverheadMarker
        from dlb.harness.twotier import JevBisectServoPolicy, TaskPlanner, TwoTierRunner
        from dlb.real.omx import RealOMX

        self.env = RealOMX(
            self.args.robot_config, camera_source="usb", cameras=("front", "wrist"), image_size=320
        )
        if not self.objects or not self.args.bin_name:
            self.discover()
        marker = OverheadMarker(
            model=self.args.mark_model, n=self.args.mark_n, object_names=(*self.objects, self.args.bin_name)
        )
        self.guide = OverheadGuide(
            marker, log_dir=self.log / "marks", roles={"cube": self.objects[0], "bin": self.args.bin_name}
        )
        policy = JevBisectServoPolicy(build_backend("openjev"), cameras=("wrist",))
        self.runner = TwoTierRunner(
            self.env,
            TaskPlanner([]),
            policy,
            max_decisions=self.args.max_decisions,
            log_dir=self.log,
            guide=self.guide,
        )
        self.env.reset(seed=0)  # begin pose, gripper open
        # the overhead map's scale, measured with the arm itself (~30 s; nothing is kept between sessions)
        print(
            "  calibrating the overhead view (the arm visits a few poses around the begin pose)", flush=True
        )
        self.guide.calibrate(self.env)
        self.idle_mark()

    def discover(self) -> None:
        """Name the objects on the table and the container from the overhead image (the ones not given)."""
        from dlb.harness.marking import OverheadMarker

        objects, container = OverheadMarker(model=self.args.mark_model).list_objects(
            self.env.overhead_frame()
        )
        self.objects = self.objects or objects
        self.args.bin_name = self.args.bin_name or container
        print(
            f"  on the table: {', '.join(self.objects) or '-'}; container: {self.args.bin_name or '-'}",
            flush=True,
        )
        if not self.objects or not self.args.bin_name:
            raise SystemExit("no objects or no container found on the table: pass --objects / --bin-name")
        self.state = RobotState(self.state.held, [*self.objects, "bin"], [])

    def idle_mark(self) -> None:
        """Mark everything now (background) so the next instruction starts from fresh marks."""
        if self.guide is None or self.guide.pending:
            return
        try:
            self._ref = self._prev = self._small()
            self.guide.start_background(self.env, kind="begin")
            self._last_idle_mark = time.monotonic()
        except Exception as e:  # noqa: BLE001 - the next task then marks in the foreground
            print("  idle mark failed:", e, file=sys.stderr)

    def _small(self):
        """The overhead view as a small grey image (cheap to compare)."""
        import cv2

        g = cv2.cvtColor(self.env.overhead_frame(), cv2.COLOR_RGB2GRAY)
        return cv2.GaussianBlur(cv2.resize(g, (160, 120), interpolation=cv2.INTER_AREA), (3, 3), 0).astype(
            int
        )

    @staticmethod
    def _changed(a, b, level: int = 30, pixels: int = 12) -> bool:
        # an object of ~3 cm covers ~25 px of the 160x120 view; camera noise and light flicker stay under
        # ``level`` (a smaller object that is moved is picked up by the periodic re-mark, --remark-s)
        return a is not None and b is not None and int((abs(a - b) > level).sum()) >= pixels

    def refresh_if_stale(self) -> None:
        """Keep the marks fresh while idle: re-mark once the scene has changed and settled (a person moved an
        object and took the hand away), or after ``remark_s``. A stale idle mark sent the arm 6 cm off (the
        blue cube had been moved after it)."""
        if self.guide is None:
            return
        cur = self._small()
        settled = not self._changed(cur, self._prev)
        self._prev = cur
        if self._changed(cur, self._ref):
            self._dirty = True
        if self.guide.pending or not settled:
            return
        if self._dirty or time.monotonic() - self._last_idle_mark > self.args.remark_s:
            if self._dirty:
                print("  the scene changed: marking again", flush=True)
            self._dirty = False
            self.guide.reset()  # forget the old marks (the map is kept)
            self.idle_mark()

    def parse(self, text: str) -> Intent | None:
        try:
            intent = self.parser.parse(text, self.state)
        except Exception as e:  # noqa: BLE001 - a failed request should not end the session
            print(f"  could not parse: {e}", file=sys.stderr)
            return None
        for w in check_sequence(intent, self.state):
            print(f"  warning: {w}", file=sys.stderr)
        print(json.dumps(intent.to_json(), ensure_ascii=False), flush=True)
        if intent.clarify:
            print(f"  ? {intent.clarify}", flush=True)
        return intent

    def execute(self, tasks: list[dict[str, str]]) -> None:
        from dlb.harness.twotier import TaskPlanner

        if not tasks:
            return
        if self.runner is None:  # dry run
            self.state = apply(Intent(tasks, True), self.state)
            return
        planner = TaskPlanner(tasks, bin_name=self.args.bin_name)
        self.runner.planner = planner
        t0 = time.perf_counter()
        r = self.runner.run(self.episode, seed=self.episode, reset=False, keep_marks=True)
        self.episode += 1
        # what the robot holds now: the last picked object if the gripper is closed on something
        held = None
        if self.env.held:
            picks = [t["object"] for t in tasks if t["op"] == "pick"]
            held = picks[-1] if picks else self.state.held
        self.state = RobotState(held, self.state.objects, [])
        print(
            f"  done in {time.perf_counter() - t0:.1f} s ({r.stop_reason}, decisions {r.decisions}, "
            f"missed grasps {r.missed_grasps}); holding {held}",
            flush=True,
        )
        self.idle_mark()

    def run(self) -> int:
        if not self.args.dry_run:
            try:
                self.start_robot()
            except BaseException:
                if self.env is not None:
                    self.env.close()
                raise
        heard: queue.Queue = queue.Queue()
        threading.Thread(target=_speech, args=(self.args, heard), daemon=True).start()
        try:
            while True:
                try:
                    text = heard.get(timeout=1.0)
                except queue.Empty:
                    try:
                        self.refresh_if_stale()
                    except Exception as e:  # noqa: BLE001 - a missed check only risks a stale mark
                        print("  scene check failed:", e, file=sys.stderr)
                    continue
                if text is None:
                    break
                intent = self.parse(text)
                if intent is not None and not intent.clarify:
                    self.execute(intent.tasks)
        except KeyboardInterrupt:
            pass
        finally:
            if self.env is not None:
                self.env.close()
        return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dlb.voice.agent", description=__doc__.split("\n")[0])
    ap.add_argument("--text", action="store_true", help="type instructions instead of speaking")
    ap.add_argument("--dry-run", action="store_true", help="parse only; no robot")
    ap.add_argument(
        "--objects",
        default="",
        help="objects to pick/stack (comma-separated); default: found in the overhead image",
    )
    ap.add_argument(
        "--bin-name",
        default="",
        help="how the bin looks (for the overhead marks); default: found in the image",
    )
    ap.add_argument("--robot-config", default="configs/robot/omx_f.yaml")
    ap.add_argument("--device", default="default", help="ALSA capture device")
    ap.add_argument("--stt-url", default="http://127.0.0.1:8010/v1/")
    ap.add_argument("--stt-model", default=None)
    ap.add_argument("--intent-model", default="gpt-5.5")
    ap.add_argument("--effort", default="low")
    ap.add_argument("--mark-model", default="gpt-5.5")
    ap.add_argument("--mark-n", type=int, default=5)
    ap.add_argument("--remark-s", type=float, default=30.0, help="re-mark the scene when idle this long")
    ap.add_argument("--max-decisions", type=int, default=40)
    ap.add_argument("--out", default="results/voice")
    ap.add_argument("--run-name", default=time.strftime("session_%Y%m%d_%H%M%S"))
    return Agent(ap.parse_args(argv)).run()


if __name__ == "__main__":
    raise SystemExit(main())
