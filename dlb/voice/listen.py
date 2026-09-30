"""Microphone -> utterances -> transcripts.

Audio comes from ``arecord`` (alsa-utils; ``-D default`` goes through PipeWire/PulseAudio to the default
source), so no Python audio package is needed. An energy VAD with an adaptive noise floor cuts it into
utterances, and each utterance goes to an OpenAI-compatible ``/v1/audio/transcriptions`` endpoint: the local
speaches server (``scripts/stt_server.sh``) by default, or OpenAI with ``--openai``.

    python -m dlb.voice.listen                       # default mic, local server
    python -m dlb.voice.listen --file x.wav          # a recording instead (any format ffmpeg reads)
    python -m dlb.voice.listen --file x.wav --realtime --intent --holding "red cube"

End-of-utterance-to-text latency = the VAD's trailing silence (``--hangover``, 0.5 s) + the transcription.
"""

from __future__ import annotations

import argparse
import io
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import wave
from collections import deque
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SR = 16000
FRAME = 480  # 30 ms at 16 kHz
LOCAL_URL = os.environ.get("STT_URL", "http://127.0.0.1:8010/v1")
LOCAL_MODEL = os.environ.get("STT_MODEL", "deepdml/faster-whisper-large-v3-turbo-ct2")
OPENAI_URL, OPENAI_MODEL = "https://api.openai.com/v1", "gpt-4o-mini-transcribe"
# vocabulary hint: without it whisper writes the bin (ビン) as 瓶 (bottle) and small models hear チューブ
# No vocabulary prompt by default: on noisy audio Whisper echoed it ("赤いキューブ、青いキューブ、青いキューブ、...",
# 2026-09-28, 15 s segments), while the same audio without it came out right. Mishearings (瓶, チューブ) are left to
# the intent parser, which maps them onto the known objects.
PROMPT = None
# what whisper says to silence and noise (trained on subtitled videos)
HALLUCINATIONS = (
    "ご視聴ありがとうございました",
    "ご視聴ありがとうございます",
    "チャンネル登録",
    "おやすみなさい",
)


class MicUnavailable(RuntimeError):
    pass


# ---------------------------------------------------------------- audio sources


def mic_frames(device: str = "default", frame: int = FRAME) -> Iterator[np.ndarray]:
    """16 kHz mono int16 frames from ``arecord``. A reader thread drains the pipe, so a slow consumer (the
    transcription runs in the caller's loop) queues audio instead of making arecord overrun."""
    if shutil.which("arecord") is None:
        raise MicUnavailable("arecord not found (sudo apt install alsa-utils)")
    p = subprocess.Popen(
        ["arecord", "-q", "-D", device, "-f", "S16_LE", "-r", str(SR), "-c", "1", "-t", "raw"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    q: queue.Queue[bytes | None] = queue.Queue()

    def reader() -> None:
        assert p.stdout is not None
        while (b := p.stdout.read(frame * 2)) and len(b) == frame * 2:
            q.put(b)
        q.put(None)

    threading.Thread(target=reader, daemon=True).start()
    try:
        while (b := q.get()) is not None:
            yield np.frombuffer(b, np.int16)
        err = p.stderr.read().decode(errors="replace").strip() if p.stderr else ""
        raise MicUnavailable(f"arecord -D {device} stopped: {err or 'no audio'} (see `arecord -l`)")
    finally:
        p.terminate()
        p.wait()


def read_audio(path: str | Path) -> np.ndarray:
    """A recording as 16 kHz mono int16: WAV by the standard library, anything else through ffmpeg."""
    path = Path(path)
    if path.suffix.lower() != ".wav":
        if shutil.which("ffmpeg") is None:
            raise RuntimeError(f"{path}: only .wav without ffmpeg")
        out = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(path), "-f", "s16le", "-ac", "1", "-ar", str(SR), "-"],
            capture_output=True,
            check=True,
        ).stdout
        return np.frombuffer(out, np.int16)
    with wave.open(str(path)) as w:
        if w.getsampwidth() != 2:
            raise RuntimeError(f"{path}: 16-bit WAV expected")
        a = np.frombuffer(w.readframes(w.getnframes()), np.int16).reshape(-1, w.getnchannels()).mean(axis=1)
        sr = w.getframerate()
    if sr != SR:  # linear interpolation is enough for speech recognition
        a = np.interp(np.arange(0, len(a) * SR / sr) * sr / SR, np.arange(len(a)), a)
    return a.astype(np.int16)


def file_frames(
    path: str | Path, realtime: bool = False, frame: int = FRAME, tail_s: float = 1.0
) -> Iterator[np.ndarray]:
    """A recording as frames, with ``tail_s`` of silence appended so the VAD closes the last utterance.
    ``realtime`` paces the frames like a microphone, which makes the end-of-utterance latency meaningful."""
    a = np.concatenate([read_audio(path), np.zeros(int(tail_s * SR), np.int16)])
    t0 = time.perf_counter()
    for i in range(0, len(a) - frame + 1, frame):
        if realtime:
            time.sleep(max(0.0, t0 + (i + frame) / SR - time.perf_counter()))
        yield a[i : i + frame]


# ---------------------------------------------------------------- VAD


@dataclass
class Segmenter:
    """Energy VAD: speech starts ``start_db`` above the noise floor (for ``onset_s``) and ends after
    ``hangover_s`` below ``stop_db`` above it. The floor follows the quiet frames."""

    start_db: float = 12.0
    stop_db: float = 7.0
    onset_s: float = 0.09
    hangover_s: float = 0.5
    min_speech_s: float = 0.25
    pre_roll_s: float = 0.3
    max_utterance_s: float = 8.0
    noise_window_s: float = 3.0
    drop_db: float = 15.0  # also silence: this far below the utterance's median voiced level (loud rooms)  # the floor is a low percentile of this window, updated during speech too
    floor_db: float = -70.0
    frame: int = FRAME
    noise_db: float | None = None
    _pre: deque = field(default_factory=deque, repr=False)
    _buf: list = field(default_factory=list, repr=False)
    _onset: int = 0
    _silence: int = 0
    _voiced: int = 0
    in_speech: bool = False
    _hist: deque = field(default_factory=deque, repr=False)
    _speech_db: list = field(default_factory=list, repr=False)

    def _n(self, s: float) -> int:
        return max(1, round(s * SR / self.frame))

    @staticmethod
    def level_db(x: np.ndarray) -> float:
        return float(20 * np.log10(np.sqrt(np.mean(x.astype(np.float64) ** 2)) / 32768 + 1e-9))

    def push(self, x: np.ndarray) -> np.ndarray | None:
        """Feed one frame; returns an utterance's audio when one has just ended."""
        db = self.level_db(x)
        if self.noise_db is None:
            self.noise_db = db
        # Minimum statistics: a floor frozen during speech let steady room noise 7 dB over a quiet moment keep an
        # "utterance" open until the length cap. The 10th percentile of the last few seconds follows the room.
        self._hist.append(db)
        while len(self._hist) > self._n(self.noise_window_s):
            self._hist.popleft()
        if len(self._hist) >= self._n(1.0):
            self.noise_db = (
                min(self.noise_db, float(np.percentile(self._hist, 10)))
                if self.in_speech
                else float(np.percentile(self._hist, 10))
            )
        noise = max(self.noise_db, self.floor_db)
        if not self.in_speech:
            self._pre.append(x)
            if db > noise + self.start_db:
                self._onset += 1
                if self._onset >= self._n(self.onset_s):
                    self.in_speech, self._buf, self._silence = True, list(self._pre), 0
                    self._voiced = self._onset
                    self._pre.clear()
            else:
                self._onset = 0
            while len(self._pre) > self._n(self.pre_roll_s) + self._onset:
                self._pre.popleft()
            return None
        self._buf.append(x)
        # In a loud room the level never falls to floor + stop_db (utterances ran to the length cap): a drop well
        # below the speaker's own level ends it too.
        level = float(np.median(self._speech_db)) if len(self._speech_db) >= self._n(0.2) else None
        quiet = db < noise + self.stop_db or (level is not None and db < level - self.drop_db)
        if quiet:
            self._silence += 1
        else:
            self._silence, self._voiced = 0, self._voiced + 1
            self._speech_db.append(db)
        if self._silence < self._n(self.hangover_s) and len(self._buf) < self._n(self.max_utterance_s):
            return None
        self.in_speech, self._onset = False, 0
        self._speech_db = []
        keep = len(self._buf) - max(0, self._silence - self._n(0.2))  # keep 0.2 s of the trailing silence
        audio = np.concatenate(self._buf[:keep])
        self._buf = []
        return audio if self._voiced >= self._n(self.min_speech_s) else None


SILERO_MODEL = Path(__file__).resolve().parents[2] / "models" / "silero_vad.onnx"


class SileroSegmenter:
    """Speech/non-speech from the Silero VAD model (v5, onnx), same interface as ``Segmenter``.

    The energy VAD could not find the ends of utterances in a noisy room (every one ran to the length cap, one
    was cut after 「やっぱり」); Silero tells speech from noise. Speech starts after ``onset_s`` of probability
    over ``start_p`` and ends after ``hangover_s`` under ``stop_p``. scripts/stt_server.sh downloads the model
    (github.com/snakers4/silero-vad v5.1.2, MIT) to models/.
    """

    CHUNK, CONTEXT = 512, 64  # 32 ms at 16 kHz; v5 wants the previous 64 samples prepended

    def __init__(
        self,
        model: str | Path = SILERO_MODEL,
        start_p: float = 0.5,
        stop_p: float = 0.35,
        onset_s: float = 0.064,
        hangover_s: float = 0.5,
        min_speech_s: float = 0.25,
        pre_roll_s: float = 0.3,
        max_utterance_s: float = 15.0,
    ):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.inter_op_num_threads = so.intra_op_num_threads = 1
        self.sess = ort.InferenceSession(str(model), sess_options=so, providers=["CPUExecutionProvider"])
        self.start_p, self.stop_p = start_p, stop_p
        self.onset_s, self.hangover_s, self.min_speech_s = onset_s, hangover_s, min_speech_s
        self.pre_roll_s, self.max_utterance_s = pre_roll_s, max_utterance_s
        self._state = np.zeros((2, 1, 128), np.float32)
        self._ctx = np.zeros(self.CONTEXT, np.float32)
        self._pending = np.zeros(0, np.float32)
        self.prob = 0.0
        self.in_speech = False
        self._pre: deque = deque()
        self._buf: list = []
        self._onset = self._silence = self._voiced = 0

    def _n(self, s: float, frame: int) -> int:
        return max(1, int(round(s * SR / frame)))

    def _update_prob(self, x: np.ndarray) -> None:
        f = x.astype(np.float32) / 32768.0 if x.dtype == np.int16 else x.astype(np.float32)
        self._pending = np.concatenate([self._pending, f])
        while len(self._pending) >= self.CHUNK:
            chunk, self._pending = self._pending[: self.CHUNK], self._pending[self.CHUNK :]
            inp = np.concatenate([self._ctx, chunk])[None, :]
            out, self._state = self.sess.run(
                None, {"input": inp, "state": self._state, "sr": np.array(SR, np.int64)}
            )
            self._ctx = chunk[-self.CONTEXT :]
            self.prob = float(out.reshape(-1)[0])

    def push(self, x: np.ndarray) -> np.ndarray | None:
        self._update_prob(x)
        n = len(x)
        if not self.in_speech:
            self._pre.append(x)
            if self.prob > self.start_p:
                self._onset += 1
                if self._onset >= self._n(self.onset_s, n):
                    self.in_speech, self._buf, self._silence, self._voiced = (
                        True,
                        list(self._pre),
                        0,
                        self._onset,
                    )
                    self._pre.clear()
            else:
                self._onset = 0
            while len(self._pre) > self._n(self.pre_roll_s, n) + self._onset:
                self._pre.popleft()
            return None
        self._buf.append(x)
        if self.prob < self.stop_p:
            self._silence += 1
        else:
            self._silence, self._voiced = 0, self._voiced + 1
        if self._silence < self._n(self.hangover_s, n) and len(self._buf) < self._n(self.max_utterance_s, n):
            return None
        self.in_speech, self._onset = False, 0
        keep = len(self._buf) - max(0, self._silence - self._n(0.2, n))
        audio = np.concatenate(self._buf[:keep])
        self._buf = []
        return audio if self._voiced >= self._n(self.min_speech_s, n) else None


# ---------------------------------------------------------------- transcription


def wav_bytes(audio: np.ndarray, sr: int = SR) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(np.asarray(audio, np.int16).tobytes())
    return buf.getvalue()


class Transcriber:
    """Client of an OpenAI-compatible ``/audio/transcriptions`` endpoint (speaches or OpenAI)."""

    def __init__(
        self,
        base_url: str = LOCAL_URL,
        model: str = LOCAL_MODEL,
        language: str = "ja",
        prompt: str | None = PROMPT,
        api_key: str | None = None,
        timeout_s: float = 30.0,
        client=None,
    ):
        import httpx

        self.model, self.language, self.prompt = model, language, prompt
        self.local = "api.openai.com" not in base_url
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.client = client or httpx.Client(base_url=base_url, timeout=timeout_s, headers=headers)

    def transcribe(self, audio: np.ndarray, sr: int = SR) -> str:
        data = {"model": self.model, "language": self.language, "response_format": "json"}
        if self.prompt:
            data["prompt"] = self.prompt
        if self.local:
            data["vad_filter"] = "false"  # already segmented; speaches' silero pass only adds time
        r = self.client.post(
            "/audio/transcriptions",
            data=data,
            files={"file": ("utterance.wav", wav_bytes(audio, sr), "audio/wav")},
        )
        if r.status_code != 200:
            raise RuntimeError(f"transcription {self.model}: {r.status_code} {r.text[:300]}")
        return r.json().get("text", "").strip()

    def health(self) -> bool:
        try:
            return self.client.get("/models").status_code == 200
        except Exception:  # noqa: BLE001 - any connection problem means "not up"
            return False


@dataclass
class Utterance:
    text: str
    audio_s: float
    stt_s: float  # request round trip
    eou_s: float  # last voiced frame received -> text (hangover + stt); only meaningful for live audio


def repetitive(text: str, n: int = 3) -> bool:
    """Whisper's loop on noise: the same phrase (split at 、。, or spaces) three or more times."""
    parts = [x for x in re.split(r"[、。,.!?！？\s]+", text) if x]
    return any(parts.count(x) >= n for x in set(parts))


def clean(text: str) -> str:
    t = text.strip()
    return "" if not t or any(h in t for h in HALLUCINATIONS) or repetitive(t) else t


def listen(
    frames: Iterable[np.ndarray],
    transcriber: Transcriber,
    segmenter: Segmenter | SileroSegmenter | None = None,
) -> Iterator[Utterance]:
    """Transcripts of the utterances in ``frames`` (empty ones and whisper's stock hallucinations dropped)."""
    seg = segmenter or Segmenter()
    last_voiced = time.perf_counter()
    for x in frames:
        audio = seg.push(x)
        if seg.in_speech and seg._silence == 0:
            last_voiced = time.perf_counter()
        if audio is None:
            continue
        t0 = time.perf_counter()
        text = clean(transcriber.transcribe(audio))
        t1 = time.perf_counter()
        if text:
            yield Utterance(text, len(audio) / SR, t1 - t0, t1 - last_voiced)


def make_transcriber(args: argparse.Namespace) -> Transcriber:
    if args.openai:
        from dlb.harness.marking import load_env_file

        load_env_file(args.env_file)
        return Transcriber(OPENAI_URL, args.model or OPENAI_MODEL, api_key=os.environ["OPENAI_API_KEY"])
    return Transcriber(args.url, args.model or LOCAL_MODEL)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dlb.voice.listen", description=__doc__.split("\n")[0])
    ap.add_argument("--file", help="transcribe a recording instead of the microphone")
    ap.add_argument("--realtime", action="store_true", help="pace --file like a live microphone")
    ap.add_argument("--device", default="default", help="ALSA capture device (arecord -L), e.g. plughw:2,0")
    ap.add_argument("--url", default=LOCAL_URL, help="OpenAI-compatible base URL (scripts/stt_server.sh)")
    ap.add_argument("--model", default=None)
    ap.add_argument(
        "--openai", action="store_true", help=f"use OpenAI ({OPENAI_MODEL}) instead of the local server"
    )
    ap.add_argument("--hangover", type=float, default=0.5, help="trailing silence that ends an utterance (s)")
    ap.add_argument(
        "--vad",
        choices=["silero", "energy"],
        default="silero" if SILERO_MODEL.exists() else "energy",
        help="utterance segmentation (silero needs models/silero_vad.onnx and onnxruntime)",
    )
    ap.add_argument("--env-file", default=".env")
    ap.add_argument("--intent", action="store_true", help="also parse each transcript into tasks (gpt-5.5)")
    from dlb.voice.console import add_state_args

    add_state_args(ap)
    args = ap.parse_args(argv)

    tr = make_transcriber(args)
    if not tr.health():
        print(
            f"transcription server not reachable at {tr.client.base_url} (bash scripts/stt_server.sh)",
            file=sys.stderr,
        )
        return 2
    session = None
    if args.intent:
        from dlb.voice.console import Session

        session = Session.from_args(args)
    frames = file_frames(args.file, args.realtime) if args.file else mic_frames(args.device)
    print(
        f"listening ({args.file or 'mic ' + args.device}) -> {tr.client.base_url} {tr.model}", file=sys.stderr
    )
    try:
        seg = (
            SileroSegmenter(hangover_s=args.hangover)
            if args.vad == "silero"
            else Segmenter(hangover_s=args.hangover)
        )
        for u in listen(frames, tr, seg):
            eou = f", eou {u.eou_s:.2f} s" if not args.file or args.realtime else ""
            print(f"[{u.audio_s:.1f} s audio, stt {u.stt_s:.2f} s{eou}] {u.text}", flush=True)
            if session:
                session.handle(u.text)
    except MicUnavailable as e:
        print(
            f"no microphone: {e}\n  use --file <recording>, or python -m dlb.voice.console to type instead",
            file=sys.stderr,
        )
        return 1
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
