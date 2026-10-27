"""Johnny: listen -> wake word -> LLM with the chat history and the skills as tools -> reply.

    python -m dlb.johnny                 # microphone (the STT server must be up: scripts/stt_server.sh)
    python -m dlb.johnny --text          # type instead of speaking (the wake word is still needed)
    python -m dlb.johnny --file x.wav    # a recording instead of the microphone

Utterances without the wake word never leave the laptop. A request runs in a worker thread (the LLM call, the
tools it picks, the reply), so a stop word heard meanwhile cancels the running skill at once. One request at a
time: a second wake-word request while one runs is answered with 「作業中です」.
"""

from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any

from dlb.johnny.history import History
from dlb.johnny.llm import LLM, Reply
from dlb.johnny.skills.base import Context, Skill, SkillResult
from dlb.johnny.skills.registry import Registry
from dlb.johnny.speak import Speaker
from dlb.johnny.wake import WakeGate, WakeWord

BUSY_REPLY = "作業中です。止めるときは「止まって」と言ってください。"
WAKE_REPLY = "はい、どうぞ。"
STOPPED_REPLY = "止めました。"
STOP_USER_TEXT = "止まって"
CANCELLED_OUTPUT = "[cancelled] the user said stop; this call was not run"


def result_output(r: SkillResult) -> str:
    """A skill result as the text the model sees."""
    text = r.text if r.ok else f"[failed] {r.text}"
    if r.data and r.data.get("ask_user"):
        text += "\n[waiting for the user's answer; then call this tool again with the next step"
        text += f": {r.data['next']}]" if r.data.get("next") else "]"
    return text


class Johnny:
    def __init__(
        self,
        skills: Registry | dict[str, Skill],
        llm: LLM,
        speaker: Speaker | None = None,
        history: History | None = None,
        ctx: Context | None = None,
        wake: WakeWord | None = None,
        wake_window_s: float = 8.0,
        ask_window_s: float = 20.0,
        max_rounds: int = 8,
        log=None,
        debug: bool = False,
    ):
        self.skills = Registry(skills)
        self.llm, self.speaker = llm, speaker or Speaker()
        self.history = history or History()
        self.log = log or (lambda m: print(m, file=sys.stderr, flush=True))
        self.debug = debug
        self.gate = WakeGate(wake, wake_window_s)
        self.ask_window_s = ask_window_s
        self.max_rounds = max_rounds
        self.ctx = ctx or Context()
        self.ctx.say, self.ctx.log = self.speaker.say, self.log
        self._worker: threading.Thread | None = None
        self._running: Skill | None = None  # the skill whose run() is executing
        self._asking: Skill | None = None  # the skill waiting for the user's answer (ask_user)
        self._cancelled = threading.Event()
        self._lock = threading.Lock()

    # ------------------------------------------------------------ input

    @property
    def busy(self) -> bool:
        return self._worker is not None and self._worker.is_alive()

    def hear(self, text: str, now: float | None = None) -> str:
        """One transcript. Returns what was done with it: stop / command / wake / busy / ignored."""
        now = time.monotonic() if now is None else now
        m = self.gate.feed(text, now)
        if m.kind == "stop":
            self.log(f"stop word: {text!r}")
            self.stop()
            return "stop"
        if m.kind == "wake":
            self.log(f"wake word ({m.matched}): {text!r}")
            self.speaker.say(WAKE_REPLY)
            return "wake"
        if m.kind == "command":
            self.log(f"command ({m.matched}): {m.command!r}")
            if self.busy:
                self.speaker.say(BUSY_REPLY)
                return "busy"
            self.request(m.command)
            return "command"
        if self.debug:
            self.log(f"ignored (no wake word): {text!r}")
        return "ignored"

    # ------------------------------------------------------------ requests

    def request(self, text: str) -> threading.Thread:
        """Run one request in a worker thread (returned so callers and tests can join it)."""
        self._cancelled.clear()
        self._worker = threading.Thread(target=self._run, args=(text,), daemon=True, name="johnny-request")
        self._worker.start()
        return self._worker

    def _run(self, text: str) -> None:
        self.history.user(text)
        try:
            reply = self._loop()
        except Exception as e:  # noqa: BLE001 - a failed request must not end the session
            self.log(f"request failed: {type(e).__name__}: {e}\n{traceback.format_exc()}")
            reply = "すみません、処理に失敗しました。もう一度お願いします。"
        self.history.assistant(reply)
        self.speaker.say(reply)
        # the user answers without the wake word: a longer window when a skill waits for a physical step
        if self._asking is not None:
            self._arm(self.ask_window_s)
        elif reply.rstrip().endswith(("?", "？")):
            self._arm(self.gate.window_s)

    def _arm(self, seconds: float) -> None:
        self.gate.armed_until = time.monotonic() + seconds

    def _loop(self) -> str:
        tools = self.skills.tool_specs()
        for _ in range(self.max_rounds):
            reply: Reply = self.llm.respond(self.history.snapshot(), tools)
            self.log(
                f"llm {reply.latency_s:.1f} s: text={reply.text!r} calls={[c.name for c in reply.calls]}"
            )
            if not reply.calls:
                return reply.text or "はい。"
            for call in reply.calls:
                self.history.call(call.call_id, call.name, call.raw_arguments or json.dumps(call.arguments))
                out = self._call(call.name, call.arguments)
                self.history.output(call.call_id, out)
            if self._cancelled.is_set():
                break
        else:
            return reply.text or "ここまでで一度止めます。続けるならもう一度言ってください。"
        reply = self.llm.respond(self.history.snapshot(), [])  # after a stop: just tell the user
        return reply.text or STOPPED_REPLY

    def _call(self, name: str, args: dict[str, Any]) -> str:
        skill = self.skills.get(name)
        if skill is None:
            return f"[failed] no such tool: {name}"
        if self._cancelled.is_set():
            return CANCELLED_OUTPUT
        self.log(f"skill {name} {json.dumps(args, ensure_ascii=False)}")
        with self._lock:
            self._running, self._asking = skill, None
        try:
            r = skill.run(self.ctx, **args)
        except TypeError as e:  # wrong arguments from the model
            r = SkillResult(f"bad arguments for {name}: {e}", ok=False)
        except Exception as e:  # noqa: BLE001
            self.log(f"skill {name} raised: {traceback.format_exc()}")
            r = SkillResult(f"{name} failed: {type(e).__name__}: {e}", ok=False)
        finally:
            with self._lock:
                self._running = None
        if r.data and r.data.get("ask_user") and not self._cancelled.is_set():
            with self._lock:
                self._asking = skill  # the window opens once the reply has been spoken (_run)
        out = result_output(r)
        self.log(f"skill {name} -> ok={r.ok} {out!r}")
        return out

    # ------------------------------------------------------------ stop

    def stop(self) -> None:
        """A stop word: cancel the running skill (or the one waiting for an answer), run the stop skill."""
        self._cancelled.set()
        self.gate.armed_until = 0.0
        with self._lock:
            running, asking = self._running, self._asking
            self._asking = None
        for s in (running, asking):
            if s is not None:
                try:
                    s.cancel(self.ctx)
                except Exception as e:  # noqa: BLE001
                    self.log(f"cancel {s.name} failed: {e}")
        stop_skill = self.skills.get("stop_all") or self.skills.get("stop")  # control.py names it stop_all
        text = STOPPED_REPLY
        if stop_skill is not None and stop_skill not in (running, asking):
            try:
                r = stop_skill.run(self.ctx)
                text = r.text or text
            except Exception as e:  # noqa: BLE001
                self.log(f"stop skill failed: {e}")
        if not self.busy:
            # no request in flight: nothing else tells the model, so the history records the stop here
            self.history.user(STOP_USER_TEXT)
            self.history.assistant(text)
            self.speaker.say(text)
        else:
            # the cancelled call's result reaches the model and the worker replies; acknowledge now anyway
            self.speaker.say("止めます。")

    def close(self, timeout_s: float = 10.0) -> None:
        if self.busy:
            self._cancelled.set()
            with self._lock:
                running = self._running
            if running is not None:
                try:
                    running.cancel(self.ctx)
                except Exception as e:  # noqa: BLE001
                    self.log(f"cancel {running.name} failed: {e}")
            assert self._worker is not None
            self._worker.join(timeout_s)
        robot = self.ctx.robot
        env = getattr(robot, "env", None)
        if env is not None and hasattr(env, "close"):
            try:
                env.close()
            except Exception as e:  # noqa: BLE001
                self.log(f"robot close failed: {e}")
        for name, p in list(self.ctx.processes.items()):
            if p.poll() is None:
                self.log(f"terminating {name}")
                p.terminate()


# ---------------------------------------------------------------- speech in


def _speech(args: argparse.Namespace, out: queue.Queue, log) -> None:
    """Transcripts into ``out`` as ``(text, t_start, t_end)`` (monotonic); None at the end."""
    try:
        if args.text:
            print("type (wake word first, e.g. ジョニー、…; empty line or Ctrl-D to quit)", file=sys.stderr)
            for line in sys.stdin:
                if not line.strip():
                    break
                now = time.monotonic()
                out.put((line.strip(), now, now))
            return
        from dlb.voice.listen import (
            SILERO_MODEL,
            Segmenter,
            SileroSegmenter,
            Transcriber,
            file_frames,
            listen,
            mic_frames,
        )

        tr = Transcriber(args.stt_url, args.stt_model) if args.stt_model else Transcriber(args.stt_url)
        if not tr.health():
            print(
                f"transcription server not reachable at {args.stt_url} (bash scripts/stt_server.sh)",
                file=sys.stderr,
            )
            return
        seg = SileroSegmenter() if SILERO_MODEL.exists() else Segmenter()
        frames = file_frames(args.file, realtime=args.realtime) if args.file else mic_frames(args.device)
        print("listening... (say the wake word first)", file=sys.stderr, flush=True)
        for u in listen(frames, tr, seg):
            t_end = time.monotonic() - u.eou_s
            print(f"[heard {u.audio_s:.1f} s, text after {u.eou_s:.1f} s] {u.text}", flush=True)
            out.put((u.text, t_end - u.audio_s, t_end))
    except Exception as e:  # noqa: BLE001 - reported, then the main loop ends
        log(f"speech input ended: {type(e).__name__}: {e}")
    finally:
        out.put(None)


class FileLog:
    """Lines to a log file, and to stderr with --debug."""

    def __init__(self, path: Path, echo: bool):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = path.open("a")
        self.echo = echo
        self._lock = threading.Lock()

    def __call__(self, msg: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        with self._lock:
            self.f.write(line + "\n")
            self.f.flush()
        if self.echo:
            print(f"  {line}", file=sys.stderr, flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dlb.johnny", description=__doc__.split("\n")[0])
    ap.add_argument("--text", action="store_true", help="type instead of speaking")
    ap.add_argument("--file", help="a recording instead of the microphone (needs the STT server)")
    ap.add_argument("--realtime", action="store_true", help="pace --file like a live microphone")
    ap.add_argument("--device", default="default", help="ALSA capture device (arecord -L)")
    ap.add_argument("--stt-url", default="http://127.0.0.1:8010/v1/")
    ap.add_argument("--stt-model", default=None)
    ap.add_argument("--model", default="gpt-5.5")
    ap.add_argument("--effort", default="low", help="reasoning effort ('' for none)")
    ap.add_argument("--history", default="results/johnny/history.jsonl", help="chat history (jsonl)")
    ap.add_argument("--fresh", action="store_true", help="do not load the history file (still appended)")
    ap.add_argument("--max-turns", type=int, default=40, help="user turns kept in the history sent")
    ap.add_argument("--tts", choices=["none", "openai"], default="none")
    ap.add_argument("--voice", default="alloy")
    ap.add_argument("--wake-window", type=float, default=12.0, help="s to speak after a bare wake word")
    ap.add_argument("--ask-window", type=float, default=20.0, help="s to answer when a skill asks something")
    ap.add_argument("--skills", default=None, help="only these skills (comma-separated)")
    ap.add_argument("--robot-config", default="configs/robot/omx_f.yaml")
    ap.add_argument("--env-file", default=".env")
    ap.add_argument("--out", default="results/johnny")
    ap.add_argument("--debug", action="store_true", help="also log dropped utterances and show the log")
    args = ap.parse_args(argv)

    from dlb.johnny.skills.registry import load
    from dlb.johnny.speak import make_speaker

    log = FileLog(Path(args.out) / time.strftime("johnny_%Y%m%d_%H%M%S.log"), args.debug)
    skills = load(args.skills.split(",") if args.skills else None, log=log)
    print(f"skills: {', '.join(skills) or '(none)'}", file=sys.stderr)
    log(f"skills: {list(skills)}")
    history = History(args.history, max_turns=args.max_turns)
    if not args.fresh and args.history:
        history = History.load(args.history, max_turns=args.max_turns)
        if history.items:
            print(f"history: {history.turns()} turns from {args.history}", file=sys.stderr)
    speaker = make_speaker(args.tts, voice=args.voice, env_file=args.env_file, log=log)
    johnny = Johnny(
        skills,
        LLM(args.model, args.effort or None, env_file=args.env_file),
        speaker,
        history,
        Context(robot_config=args.robot_config, env_file=args.env_file),
        wake_window_s=args.wake_window,
        ask_window_s=args.ask_window,
        log=log,
        debug=args.debug,
    )
    heard: queue.Queue = queue.Queue()
    threading.Thread(target=_speech, args=(args, heard, log), daemon=True).start()
    try:
        while True:
            try:
                item = heard.get(timeout=0.5)
            except queue.Empty:
                continue
            if item is None:
                if args.file or args.text:
                    while johnny.busy:  # let the last request finish
                        time.sleep(0.2)
                break
            text, t_start, t_end = item
            if speaker.overlaps(t_start, t_end):
                log(f"ignored (Johnny was speaking): {text!r}")
                continue
            johnny.hear(text)
    except KeyboardInterrupt:
        print("\nstopping", file=sys.stderr)
    finally:
        johnny.close()
    return 0
