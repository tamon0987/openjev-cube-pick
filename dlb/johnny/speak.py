"""How Johnny talks back: printed (default) or spoken through OpenAI text-to-speech and ``aplay``/``paplay``.

The speaker remembers when it was playing so the agent can drop utterances the microphone picked up from the
loudspeaker (see ``overlaps``). Speech is best effort: when the request or the player fails the text is still
printed and the agent goes on.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Callable


class Speaker:
    """Prints. ``say`` may be called from any thread."""

    def __init__(self, out=None, grace_s: float = 0.8):
        self.out = out or sys.stdout
        self.grace_s = grace_s  # after playback the room still echoes; ignore utterances ending this soon
        self._lock = threading.Lock()
        self._spans: deque[tuple[float, float]] = deque(maxlen=8)  # (start, end) of recent playbacks
        self.speaking_since: float | None = None

    def say(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        with self._lock:
            print(f"Johnny: {text}", file=self.out, flush=True)
            self.speaking_since = time.monotonic()
            played = False
            try:
                played = self._play(text)
            except Exception as e:  # noqa: BLE001 - speech is optional
                print(f"  (speech failed: {e})", file=sys.stderr)
            finally:
                if played:  # only real playback can be picked up by the microphone
                    self._spans.append((self.speaking_since, time.monotonic()))
                self.speaking_since = None

    def _play(self, text: str) -> bool:
        """Play ``text`` aloud; return True when audio was actually played."""
        return False

    def overlaps(self, t_start: float, t_end: float) -> bool:
        """Whether an utterance heard between these monotonic times coincides with Johnny's own speech."""
        since = self.speaking_since
        if since is not None and t_end >= since:
            return True
        return any(t_start <= e + self.grace_s and t_end >= s for s, e in self._spans)


class OpenAISpeaker(Speaker):
    """``/audio/speech`` -> WAV -> ``aplay`` (or ``paplay``)."""

    def __init__(
        self,
        model: str = "gpt-4o-mini-tts",
        voice: str = "alloy",
        env_file: str = ".env",
        timeout_s: float = 30.0,
        client=None,
        player: list[str] | None = None,
        log: Callable[[str], None] | None = None,
        **kw,
    ):
        super().__init__(**kw)
        self.model, self.voice, self.log = model, voice, log
        if client is None:
            import httpx

            from dlb.harness.marking import load_env_file

            load_env_file(env_file)
            client = httpx.Client(
                base_url="https://api.openai.com/v1",
                timeout=timeout_s,
                headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
            )
        self.client = client
        self.player = player or find_player()

    def _play(self, text: str) -> bool:
        if self.player is None:
            raise RuntimeError("no audio player (aplay or paplay)")
        r = self.client.post(
            "/audio/speech",
            json={"model": self.model, "voice": self.voice, "input": text, "response_format": "wav"},
        )
        if r.status_code != 200:
            raise RuntimeError(f"tts {self.model}: {r.status_code} {r.text[:200]}")
        p = subprocess.run(self.player, input=r.content, capture_output=True)
        if p.returncode != 0:
            raise RuntimeError(f"{self.player[0]}: {p.stderr.decode(errors='replace').strip()[:200]}")
        return True


def find_player() -> list[str] | None:
    if shutil.which("aplay"):
        return ["aplay", "-q", "-"]
    if shutil.which("paplay"):
        return ["paplay"]
    return None


def make_speaker(kind: str, **kw) -> Speaker:
    if kind == "openai":
        return OpenAISpeaker(**kw)
    return Speaker()
