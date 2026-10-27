"""The wake word ("Hi, Johnny") and the stop words, matched on Whisper transcripts.

Every request begins with the wake word. Whisper hears it in many ways ("Hi Johnny", "ハイ、ジョニー", "ジョニ",
"ジャニー", "John", ...), so the transcript is normalized (NFKC, lower case, hiragana -> katakana, punctuation
and spaces dropped) and compared with a list of forms; a form that is one edit away also counts. What follows
the wake word is the command. A bare stop word (止まって / ストップ / やめて / stop) needs no wake word.

``WakeGate`` adds the listening window: after the wake word alone, the next utterance within ``window_s`` is
the command (the agent also opens the window when it asks the user something).
"""

from __future__ import annotations

import time
import unicodedata
from dataclasses import dataclass

# How Whisper writes "Hi, Johnny" (both languages) and its mishearings. Normalized before use.
WAKE_FORMS = (
    "hi johnny",
    "hi jonny",
    "hi johny",
    "hi jhonny",
    "hey johnny",
    "johnny",
    "jonny",
    "johny",
    "johnnie",
    "john",
    "ハイジョニー",
    "ハイジョーニー",
    "ハイジャニー",
    # what Whisper wrote for a Japanese-accented "Hi, Johnny" on the rig (kanji, so no fuzzy katakana match)
    "排除に",
    "排除にー",
    "排除",
    "廃城",
    "配所に",
    "はいじょに",
    "ハイジョニ",
    "ジョニーさん",
    "ジョニーくん",
    "ジョニー",
    "ジョーニー",
    "ジョニィ",
    "ジョーニ",
    "ジョニ",
    "ジャニー",
    "ジョン",
)
# Said before the wake word without meaning anything ("ねえ、ジョニー", "OK Johnny").
FILLERS = (
    "ねえ",
    "ねぇ",
    "ね",
    "あのー",
    "あの",
    "えっと",
    "えー",
    "え",
    "あー",
    "あ",
    "うん",
    "おい",
    "ok",
    "okay",
    "hey",
)
STOP_WORDS = ("止まって", "止めて", "とまって", "ストップ", "やめて", "停止", "待って", "stop")
TAIL = "ーィ"  # a long vowel or small i after the name is part of how it was heard
# What may follow a stop word and still leave it a bare stop ("止まってください", "ストップして").
STOP_SLACK = 6


def _norm_char(c: str) -> str:
    """One transcript character as compared: NFKC, lower case, hiragana as katakana, punctuation gone."""
    out = []
    for d in unicodedata.normalize("NFKC", c).lower():
        if "ぁ" <= d <= "ゖ":  # hiragana -> katakana
            d = chr(ord(d) + 0x60)
        if unicodedata.category(d)[0] in "PZC":  # punctuation, spaces, controls
            continue
        out.append(d)
    return "".join(out)


def normalize(text: str) -> str:
    return "".join(_norm_char(c) for c in text)


def _norm_map(text: str) -> tuple[str, list[int]]:
    """The normalized text and, per normalized character, the index after its source character."""
    chars, ends = [], []
    for i, c in enumerate(text):
        for d in _norm_char(c):
            chars.append(d)
            ends.append(i + 1)
    return "".join(chars), ends


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


@dataclass
class Match:
    kind: str  # "stop" | "command" | "wake" | "none"
    command: str = ""  # what follows the wake word (kind "command"), or the whole text in a listening window
    matched: str = ""  # the wake / stop form that matched


class WakeWord:
    """Matches one transcript against the wake word and the stop words."""

    def __init__(
        self,
        forms: tuple[str, ...] | list[str] = WAKE_FORMS,
        stop_words: tuple[str, ...] | list[str] = STOP_WORDS,
        fillers: tuple[str, ...] | list[str] = FILLERS,
        fuzzy: bool = True,
    ):
        self.forms = sorted({normalize(f) for f in forms if normalize(f)}, key=len, reverse=True)
        self.stop_words = sorted({normalize(s) for s in stop_words if normalize(s)}, key=len, reverse=True)
        self.fillers = sorted({normalize(f) for f in fillers if normalize(f)}, key=len, reverse=True)
        self.fuzzy = fuzzy

    def _skip_fillers(self, text: str, pos: int) -> int:
        while True:
            for f in self.fillers:
                if text.startswith(f, pos) and len(text) > pos + len(f):
                    pos += len(f)
                    break
            else:
                return pos

    def _wake_end(self, text: str, pos: int) -> tuple[int, str] | None:
        """Where the wake word ends in the normalized text starting at ``pos``, and which form it was."""
        hit = None
        for f in self.forms:
            if text.startswith(f, pos):
                hit = (pos + len(f), f)
                break
        if hit is None and self.fuzzy:
            best = None
            for f in self.forms:
                if len(f) < 4:
                    continue
                allowed = 1 if len(f) < 8 else 2
                for n in range(max(2, len(f) - allowed), len(f) + allowed + 1):
                    d = edit_distance(text[pos : pos + n], f)
                    if d <= allowed and (best is None or (d, -n) < (best[0], -best[1])):
                        best = (d, n, f)
            hit = (pos + best[1], best[2]) if best else None
        if hit is None:
            return None
        end, form = hit
        while end < len(text) and text[end] in TAIL:  # "ジョニィ", "ジョニーー": still the name
            end += 1
        return end, form

    def is_stop(self, text: str) -> str | None:
        """The stop word when ``text`` is a bare stop (with or without the wake word in front)."""
        norm, _ = _norm_map(text)
        pos = self._skip_fillers(norm, 0)
        if (hit := self._wake_end(norm, pos)) is not None:
            pos = hit[0]
        for s in self.stop_words:
            if norm.startswith(s, pos) and len(norm) - pos <= len(s) + STOP_SLACK:
                return s
        return None

    def match(self, text: str) -> Match:
        if (s := self.is_stop(text)) is not None:
            return Match("stop", matched=s)
        norm, ends = _norm_map(text)
        pos = self._skip_fillers(norm, 0)
        hit = self._wake_end(norm, pos)
        if hit is None:
            return Match("none")
        end, form = hit
        rest = text[ends[min(end, len(ends)) - 1] :] if end else text  # end may run past a short transcript
        # leading punctuation and spaces between the wake word and the command ("ジョニー、取って")
        rest = rest.lstrip(" \t　、。,.!?！？:：;；・…ー-")
        return Match("command" if rest else "wake", rest, form)


class WakeGate:
    """``WakeWord`` plus the listening window after a bare wake word (or after the agent asked back).
    ``log`` (optional) is told when the window opens, closes and expires: with --debug this shows why an
    utterance was taken as a command or dropped."""

    def __init__(self, wake: WakeWord | None = None, window_s: float = 8.0, log=None):
        self.wake = wake or WakeWord()
        self.window_s = window_s
        self.armed_until = 0.0
        self.log = log

    def arm(self, now: float | None = None, seconds: float | None = None, why: str = "wake word") -> None:
        now = time.monotonic() if now is None else now
        seconds = self.window_s if seconds is None else seconds
        self.armed_until = now + seconds
        if self.log:
            self.log(f"wake window open for {seconds:.0f} s ({why})")

    def disarm(self, why: str, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        if self.log and self.armed_until:
            state = "closed" if now < self.armed_until else f"expired {now - self.armed_until:.1f} s ago"
            self.log(f"wake window {state} ({why})")
        self.armed_until = 0.0

    def armed(self, now: float | None = None) -> bool:
        return (time.monotonic() if now is None else now) < self.armed_until

    def feed(self, text: str, now: float | None = None) -> Match:
        now = time.monotonic() if now is None else now
        m = self.wake.match(text)
        if m.kind == "stop":
            self.disarm("stop word", now)
            return m
        if m.kind == "command":
            self.disarm("command with the wake word", now)
            return m
        if m.kind == "wake":
            self.arm(now)
            return m
        if self.armed(now):  # the utterance after a bare wake word (or an answer to a question)
            self.disarm("command in the window", now)
            return Match("command", text.strip(), "window")
        if self.armed_until:  # it was open once: say when it ran out
            self.disarm("utterance after the window", now)
        return m
