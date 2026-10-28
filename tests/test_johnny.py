import json
import subprocess
import sys
import threading
import time

import httpx
import pytest

from dlb.johnny.agent import (
    BUSY_REPLY,
    CANCELLED_OUTPUT,
    STOPPED_REPLY,
    WAKE_REPLY,
    Johnny,
    Listener,
    Shutdown,
)
from dlb.johnny.history import History
from dlb.johnny.llm import LLM
from dlb.johnny.skills.base import Context, Skill, SkillResult
from dlb.johnny.skills.registry import Registry, load
from dlb.johnny.speak import Speaker
from dlb.johnny.wake import WakeGate, WakeWord, normalize

# ---------------------------------------------------------------- wake word

# (transcript, kind, command) as Whisper writes them
WAKE_TABLE = [
    ("Hi, Johnny", "wake", ""),
    ("hi johnny", "wake", ""),
    ("Hi Johnny.", "wake", ""),
    ("Hey Johnny!", "wake", ""),
    ("Johnny", "wake", ""),
    ("Jonny", "wake", ""),
    ("John", "wake", ""),
    ("ハイ、ジョニー", "wake", ""),
    ("ハイジョニー", "wake", ""),
    ("はいジョニー", "wake", ""),
    ("ジョニー", "wake", ""),
    ("ジョニ", "wake", ""),
    ("ジョーニー", "wake", ""),
    ("ハイ、ジョーニー", "wake", ""),
    ("ジャニー", "wake", ""),
    ("ジョニーさん", "wake", ""),
    ("ジョニー。", "wake", ""),
    ("ねえ、ジョニー", "wake", ""),
    ("ハイジョニィ", "wake", ""),
    ("Hi Johnny, put the red cube in the bin", "command", "put the red cube in the bin"),
    ("Hi, Johnny. Set up the rig.", "command", "Set up the rig."),
    ("ジョニー、赤いキューブをビンに入れて", "command", "赤いキューブをビンに入れて"),
    ("ジョニー赤いキューブをビンに入れて", "command", "赤いキューブをビンに入れて"),
    ("ハイ、ジョニー。セットアップして。", "command", "セットアップして。"),
    ("はい、ジョニー、テレオペを始めて", "command", "テレオペを始めて"),
    ("ジョニーさん、状態を教えて", "command", "状態を教えて"),
    ("ジャニー、ビンの位置は？", "command", "ビンの位置は？"),
    ("あ、ジョニー、抜きました", "command", "抜きました"),
    ("介助に人参をトレイに入れて、", "command", "人参をトレイに入れて、"),
    ("介助の仕事について", "none", ""),
    # bare stop words, with or without the wake word
    ("止まって", "stop", ""),
    ("止まって！", "stop", ""),
    ("止まってください", "stop", ""),
    ("ストップ", "stop", ""),
    ("ストップ。", "stop", ""),
    ("やめて", "stop", ""),
    ("Stop", "stop", ""),
    ("stop!", "stop", ""),
    ("ジョニー、止まって", "stop", ""),
    ("Johnny stop", "stop", ""),
    # negatives: never sent anywhere
    ("17、19時から20時", "none", ""),
    ("お疲れ様でした", "none", ""),
    ("赤いキューブをビンに入れて", "none", ""),
    ("ご視聴ありがとうございました", "none", ""),
    ("今日はいい天気ですね", "none", ""),
    ("ジョブを確認して", "none", ""),
    ("ジャンプして", "none", ""),
    ("ハイジャンプ", "none", ""),
    ("止まってから右に動いて", "none", ""),
    ("ちょっと待ってから続けて", "none", ""),
    ("", "none", ""),
    ("。", "none", ""),
]


@pytest.mark.parametrize("text,kind,command", WAKE_TABLE)
def test_wake_word_table(text, kind, command):
    m = WakeWord().match(text)
    assert (m.kind, m.command) == (kind, command), f"{text!r} -> {m}"


def test_wake_word_forms_configurable_and_fuzzy_optional():
    strict = WakeWord(forms=["Hi Johnny"], fuzzy=False)
    assert strict.match("Hi Johnny, go").kind == "command"
    assert strict.match("Hi Jonny, go").kind == "none"
    assert WakeWord(forms=["Hi Johnny"]).match("Hi Jonny, go").kind == "command"  # one edit away
    assert normalize("ハイ、ジョニー！") == normalize("はい じょにー") == "ハイジョニー"


def test_wake_gate_window():
    g = WakeGate(window_s=8.0)
    assert g.feed("赤いキューブを取って", now=0.0).kind == "none"  # no wake word: dropped
    assert g.feed("ジョニー", now=1.0).kind == "wake"
    m = g.feed("赤いキューブを取って", now=5.0)  # within the window: the command
    assert (m.kind, m.command) == ("command", "赤いキューブを取って")
    assert g.feed("青いキューブも", now=6.0).kind == "none"  # the window closed with the command
    g.feed("ジョニー", now=10.0)
    assert g.feed("赤いキューブを取って", now=18.5).kind == "none"  # too late
    g.feed("ジョニー", now=20.0)
    assert g.feed("止まって", now=21.0).kind == "stop"
    assert g.feed("取って", now=22.0).kind == "none"  # the stop closed the window
    g.arm(now=30.0)  # the agent asked something
    assert g.feed("はい、抜きました", now=35.0).command == "はい、抜きました"
    g.arm(now=40.0)
    assert g.feed("ジョニー、抜きました", now=41.0).command == "抜きました"  # the wake form still works


# ---------------------------------------------------------------- history


def test_history_trims_whole_turns_and_persists(tmp_path):
    path = tmp_path / "h.jsonl"
    h = History(path, max_turns=2)
    h.user("one")
    h.call("c1", "fake", "{}")
    h.output("c1", "done")
    h.assistant("ok one")
    h.user("two")
    h.assistant("ok two")
    assert h.turns() == 2 and len(h.items) == 6
    h.user("three")
    h.assistant("ok three")
    assert [it.get("role") or it["type"] for it in h.items] == ["user", "assistant", "user", "assistant"]
    assert h.items[0]["content"][0]["text"] == "two"
    # the file keeps everything; loading trims the same way
    assert len(path.read_text().splitlines()) == 8
    again = History.load(path, max_turns=2)
    assert again.items == h.items
    assert History.load(path, max_turns=40).turns() == 3
    # a char budget drops the oldest turns too, but keeps the last one
    small = History.load(path, max_turns=40, max_chars=10)
    assert small.turns() == 1 and small.items[0]["content"][0]["text"] == "three"
    assert History.load(tmp_path / "missing.jsonl").items == []


def test_history_never_starts_with_a_tool_item():
    h = History(max_turns=40)
    h.call("c0", "fake", "{}")  # e.g. a file cut at the wrong line
    h.output("c0", "x")
    h.user("hello")
    assert h.items[0]["role"] == "user"


# ---------------------------------------------------------------- registry


def test_registry_skips_broken_modules(tmp_path, monkeypatch):
    pkg = tmp_path / "fakeskills"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "base.py").write_text("from dlb.johnny.skills.base import Skill\n")
    (pkg / "good.py").write_text(
        "from dlb.johnny.skills.base import Skill, SkillResult\n"
        "class Hello(Skill):\n"
        "    name = 'hello'\n"
        "    description = 'says hello'\n"
        "    def run(self, ctx, **kw):\n"
        "        return SkillResult('hello')\n"
        "class Abstract(Skill):\n"
        "    pass\n"
    )
    (pkg / "broken.py").write_text("import module_that_does_not_exist\n")
    (pkg / "_private.py").write_text("raise RuntimeError('never imported')\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    logs: list[str] = []
    reg = load(package="fakeskills", log=logs.append)
    assert list(reg) == ["hello"] and reg.tool_specs()[0]["name"] == "hello"
    assert any("broken" in m and "ModuleNotFoundError" in m for m in logs)
    assert list(load(["hello", "nope"], package="fakeskills", log=logs.append)) == ["hello"]
    assert any("nope: not found" in m for m in logs)
    for m in list(sys.modules):
        if m.startswith("fakeskills"):
            del sys.modules[m]


# ---------------------------------------------------------------- the agent with a stubbed model


class FakeSkill(Skill):
    """Records its calls; blocks until cancelled when ``block`` is set."""

    name = "fake"
    description = "a test skill"
    parameters = {
        "type": "object",
        "properties": {"what": {"type": "string"}},
        "required": ["what"],
        "additionalProperties": False,
    }

    def __init__(self, block: bool = False):
        self.calls: list[dict] = []
        self.cancelled = 0
        self.block = block
        self.started = threading.Event()
        self._stop = threading.Event()

    def run(self, ctx: Context, what: str = "") -> SkillResult:
        self.calls.append({"what": what})
        if self.block:
            self.started.set()
            self._stop.wait(5.0)
            return SkillResult("stopped by the user" if self._stop.is_set() else "timed out", ok=False)
        return SkillResult(f"did {what}", data={"what": what})

    def cancel(self, ctx: Context) -> None:
        self.cancelled += 1
        self._stop.set()


class TwoStepSkill(Skill):
    """An interactive check: step 1 asks the user to do something, step 2 continues after the answer."""

    name = "check_camera"
    description = "checks the wrist camera (two steps: start, then after the user unplugged the cable)"
    parameters = {
        "type": "object",
        "properties": {"step": {"type": "string", "enum": ["start", "unplugged"]}},
        "required": ["step"],
        "additionalProperties": False,
    }

    def __init__(self):
        self.steps: list[str] = []
        self.cancelled = 0

    def run(self, ctx: Context, step: str = "start") -> SkillResult:
        self.steps.append(step)
        if step == "start":
            return SkillResult(
                "フォロワーアームの手首カメラのUSBを抜いてください。",
                data={"ask_user": True, "next": "unplugged"},
            )
        return SkillResult("the wrist camera is /dev/video2", data={"ask_user": False})

    def cancel(self, ctx: Context) -> None:
        self.cancelled += 1


class StopSkill(Skill):
    name = "stop"
    description = "halts the rig"

    def __init__(self):
        self.calls = 0

    def run(self, ctx: Context, **kw) -> SkillResult:
        self.calls += 1
        return SkillResult("halted")


def _message(text: str) -> dict:
    return {"type": "message", "content": [{"type": "output_text", "text": text}]}


def _call(name: str, args: dict, call_id: str = "call_1") -> dict:
    return {"type": "function_call", "call_id": call_id, "name": name, "arguments": json.dumps(args)}


def scripted(outputs: list[list[dict]], seen: list | None = None) -> LLM:
    """An LLM whose n-th call returns the n-th output list; the request bodies go to ``seen``."""
    it = iter(outputs)

    def handler(req: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(json.loads(req.content))
        return httpx.Response(200, json={"output": [{"type": "reasoning", "summary": []}, *next(it)]})

    return LLM(
        client=httpx.Client(base_url="https://api.openai.com/v1", transport=httpx.MockTransport(handler))
    )


class Spoken(Speaker):
    def __init__(self):
        super().__init__()
        self.lines: list[str] = []

    def say(self, text: str) -> None:
        self.lines.append(text)


def _johnny(skills, llm, **kw) -> tuple[Johnny, Spoken]:
    sp = Spoken()
    j = Johnny(Registry((s.name, s) for s in skills), llm, sp, log=lambda m: None, **kw)
    return j, sp


def _wait(j: Johnny, timeout_s: float = 5.0) -> None:
    assert j._worker is not None
    j._worker.join(timeout_s)
    assert not j.busy


def test_tool_loop_runs_the_skill_and_sends_the_history():
    seen: list = []
    llm = scripted([[_call("fake", {"what": "x"})], [_message("x をやりました。")]], seen)
    fake = FakeSkill()
    j, sp = _johnny([fake], llm)
    assert j.hear("赤いキューブ取って") == "ignored"  # no wake word: nothing sent
    assert seen == []
    assert j.hear("ジョニー、x をやって") == "command"
    _wait(j)
    assert fake.calls == [{"what": "x"}]
    assert sp.lines == ["x をやりました。"]
    first, second = seen
    assert first["model"] == "gpt-5.5" and first["tools"][0]["name"] == "fake" and first["store"] is False
    assert first["input"][-1] == {"role": "user", "content": [{"type": "input_text", "text": "x をやって"}]}
    types = [it.get("type") or it["role"] for it in second["input"]]
    assert types == ["user", "function_call", "function_call_output"]
    assert second["input"][2]["output"] == "did x" and "id" not in second["input"][1]
    assert [it.get("type") or it["role"] for it in j.history.items] == types + ["assistant"]
    # the next request carries the whole history
    j.hear("ジョニー", now=100.0)
    assert sp.lines[-1] == WAKE_REPLY


def test_question_opens_the_window_and_the_answer_needs_no_wake_word():
    llm = scripted([[_message("どのキューブですか？")], [_message("赤ですね。")]])
    j, sp = _johnny([FakeSkill()], llm)
    j.hear("ジョニー、キューブを取って")
    _wait(j)
    assert sp.lines == ["どのキューブですか？"] and j.gate.armed()
    assert j.hear("赤いの") == "command"
    _wait(j)
    assert sp.lines[-1] == "赤ですね。"
    assert j.history.items[2]["content"][0]["text"] == "赤いの"


def test_stop_word_cancels_the_running_skill():
    seen: list = []
    llm = scripted([[_call("fake", {"what": "long"})], [_message("止めました。")]], seen)
    fake, stop = FakeSkill(block=True), StopSkill()
    j, sp = _johnny([fake, stop], llm)
    j.hear("ジョニー、長い作業をして")
    assert fake.started.wait(5.0)
    assert j.busy
    assert j.hear("ジョニー、何か") == "busy" and sp.lines[-1] == BUSY_REPLY
    assert j.hear("止まって") == "stop"  # no wake word needed
    _wait(j)
    assert fake.cancelled == 1 and stop.calls == 1
    assert sp.lines[-1] == "止めました。"
    out = seen[1]["input"][-1]
    assert out["type"] == "function_call_output" and out["output"] == "[failed] stopped by the user"
    assert seen[1]["tools"] == []  # after a stop the model only talks


def test_stop_when_idle_runs_the_stop_skill_and_is_recorded():
    llm = scripted([])
    stop = StopSkill()
    j, sp = _johnny([stop], llm)
    assert j.hear("ストップ") == "stop"
    assert stop.calls == 1 and sp.lines == ["halted"]
    assert [it["role"] for it in j.history.items] == ["user", "assistant"]
    j2, sp2 = _johnny([FakeSkill()], llm)  # no stop skill
    j2.hear("stop")
    assert sp2.lines == [STOPPED_REPLY]


def test_tool_calls_after_a_stop_are_not_run():
    seen: list = []
    llm = scripted(
        [
            [_call("fake", {"what": "a"}, "c1"), _call("fake", {"what": "b"}, "c2")],
            [_message("止めました。")],
        ],
        seen,
    )
    fake = FakeSkill(block=True)
    j, _ = _johnny([fake], llm)
    j.hear("ジョニー、a と b")
    assert fake.started.wait(5.0)
    j.hear("やめて")
    _wait(j)
    assert fake.calls == [{"what": "a"}]  # b was not run
    outs = [it["output"] for it in seen[1]["input"] if it.get("type") == "function_call_output"]
    assert outs == ["[failed] stopped by the user", CANCELLED_OUTPUT]


def test_interactive_two_step_skill():
    seen: list = []
    llm = scripted(
        [
            [_call("check_camera", {"step": "start"}, "c1")],
            [_message("フォロワーアームの手首カメラのUSBを抜いてください。")],
            [_call("check_camera", {"step": "unplugged"}, "c2")],
            [_message("手首カメラは /dev/video2 でした。")],
        ],
        seen,
    )
    skill = TwoStepSkill()
    j, sp = _johnny([skill], llm, wake_window_s=8.0, ask_window_s=20.0)
    j.hear("ジョニー、手首カメラを確認して", now=0.0)
    _wait(j)
    assert skill.steps == ["start"]
    assert sp.lines == ["フォロワーアームの手首カメラのUSBを抜いてください。"]
    # the result told the model which step comes next
    out = seen[1]["input"][-1]["output"]
    assert out.startswith("フォロワーアーム") and "next step: unplugged" in out
    # the window is the longer ask window, and the answer needs no wake word
    assert 15.0 < j.gate.armed_until - time.monotonic() <= 20.0
    assert j.hear("抜きました") == "command"
    _wait(j)
    assert skill.steps == ["start", "unplugged"]
    assert sp.lines[-1] == "手首カメラは /dev/video2 でした。"
    # the third call saw the whole dialogue: question, instruction, answer
    types = [it.get("type") or it["role"] for it in seen[2]["input"]]
    assert types == ["user", "function_call", "function_call_output", "assistant", "user"]
    assert not j.gate.armed()


def test_stop_during_an_interactive_step_cancels_the_skill():
    llm = scripted(
        [
            [_call("check_camera", {"step": "start"}, "c1")],
            [_message("USBを抜いてください。")],
        ]
    )
    skill = TwoStepSkill()
    j, sp = _johnny([skill], llm)
    j.hear("ジョニー、カメラを確認して")
    _wait(j)
    assert j.gate.armed()
    assert j.hear("止まって") == "stop"
    assert skill.cancelled == 1 and not j.gate.armed()
    assert sp.lines[-1] == STOPPED_REPLY
    assert j.history.items[-2:] == [
        {"role": "user", "content": [{"type": "input_text", "text": "止まって"}]},
        {"role": "assistant", "content": [{"type": "output_text", "text": STOPPED_REPLY}]},
    ]
    assert j.hear("抜きました") == "ignored"  # the window is closed


def test_skill_errors_become_tool_results():
    seen: list = []
    llm = scripted([[_call("fake", {"bogus": 1}), _call("nope", {})], [_message("できませんでした。")]], seen)
    j, sp = _johnny([FakeSkill()], llm)
    j.hear("Hi Johnny, do something")
    _wait(j)
    outs = [it["output"] for it in seen[1]["input"] if it.get("type") == "function_call_output"]
    assert outs[0].startswith("[failed] bad arguments for fake") and outs[1] == "[failed] no such tool: nope"


def test_llm_failure_is_reported_not_fatal():
    err = httpx.Client(
        base_url="https://api.openai.com/v1",
        transport=httpx.MockTransport(lambda r: httpx.Response(500, text="boom")),
    )
    j, sp = _johnny([FakeSkill()], LLM(client=err))
    j.hear("ジョニー、何かして")
    _wait(j)
    assert "失敗" in sp.lines[-1]
    assert j.history.items[-1]["role"] == "assistant"


def test_close_closes_the_robot_env():
    class Env:
        closed = 0

        def close(self):
            self.closed += 1

    class Robot:
        env = Env()

    j, _ = _johnny([FakeSkill()], scripted([]))
    j.ctx.robot = Robot()
    j.close()
    assert Robot.env.closed == 1


# ---------------------------------------------------------------- shutdown (SIGINT / SIGTERM path)


class FakeListener(Listener):
    def __init__(self):
        super().__init__()
        self.stops = 0

    def stop(self):
        self.stops += 1
        self.stopped.set()


def test_shutdown_order_and_timeouts():
    """A blocking robot close, a skill that runs until cancelled, a launch that ignores SIGINT: each step is
    time-boxed, the order is listen -> skill -> robot -> processes, and the whole thing stays short."""

    class SlowEnv:
        def __init__(self):
            self.closing = threading.Event()

        def close(self):
            self.closing.set()
            time.sleep(30)  # e.g. RealOMX.close stuck in an executor shutdown

    class Robot:
        env = SlowEnv()

    logs: list[str] = []
    fake = FakeSkill(block=True)
    listener = FakeListener()
    sp = Spoken()
    j = Johnny(
        Registry([(fake.name, fake)]),
        scripted([[_call("fake", {"what": "long"})], [_message("止めました。")]]),
        sp,
        log=logs.append,
        listener=listener,
    )
    j.ctx.robot = Robot()
    polite = subprocess.Popen(["sleep", "100"], start_new_session=True)  # dies on SIGINT
    stubborn = subprocess.Popen(
        ["bash", "-c", "trap '' INT; sleep 100"], start_new_session=True
    )  # needs SIGKILL
    j.ctx.processes = {"polite": polite, "stubborn": stubborn}
    j.hear("ジョニー、長い作業をして")
    assert fake.started.wait(5.0) and j.busy
    try:
        took = j.close(worker_s=1.0, robot_s=0.3, process_s=0.5)
    finally:
        for p in (polite, stubborn):
            if p.poll() is None:
                p.kill()
    assert took < 4.0
    assert listener.stops == 1
    assert fake.cancelled == 1 and not j.busy  # the skill returned after cancel; the worker finished
    assert Robot.env.closing.is_set()
    assert polite.poll() is not None and stubborn.poll() is not None
    steps = [m for m in logs if m.startswith("shutdown")]
    assert [m.split(":")[0] for m in steps] == [
        "shutdown 1/4",
        "shutdown 2/4",
        "shutdown 3/4",
        "shutdown 4/4",
        "shutdown done in " + steps[-1].split(":")[0].split("in ")[1],
    ]
    assert "  fake cancelled" in logs
    assert any("robot close still busy after 0 s" in m for m in logs)
    assert any("SIGINT polite" in m for m in logs) and "  polite stopped" in logs
    assert any("stubborn still running" in m and "SIGKILL" in m for m in logs)


def test_close_with_nothing_running_is_quick():
    logs: list[str] = []
    j, _ = _johnny([FakeSkill()], scripted([]))
    j.log = logs.append
    assert j.close() < 0.5
    assert "  nothing running" in logs and "  no robot" in logs and "  no processes" in logs


def test_listener_stops_arecord_like_process():
    lst = Listener()
    p = subprocess.Popen(["sleep", "100"])
    lst.attach(p)
    lst.stop()
    assert p.poll() is not None and lst.stopped.is_set()
    late = Listener()
    late.stop()  # before the process exists
    q = subprocess.Popen(["sleep", "100"])
    late.attach(q)  # terminated on arrival
    q.wait(timeout=2.0)
    assert q.poll() is not None


def test_signal_handling_first_requests_second_exits_and_watchdog_forces():
    logs: list[str] = []
    exits: list[int] = []
    sd = Shutdown(logs.append, watchdog_s=0.2, exit=exits.append)
    sd.request("SIGINT")
    assert sd.requested.is_set() and exits == []
    sd.done.set()  # a clean shutdown: the watchdog stays quiet
    time.sleep(0.4)
    assert exits == []
    sd.request("SIGINT")
    assert exits == [130] and any("second SIGINT" in m for m in logs)
    slow = Shutdown(logs.append, watchdog_s=0.2, exit=exits.append)
    slow.request("SIGTERM")
    time.sleep(0.5)  # never done: the watchdog force-exits
    assert exits == [130, 3] and any("watchdog" in m for m in logs)


def test_debug_logs_the_wake_window():
    logs: list[str] = []
    j, _ = _johnny([FakeSkill()], scripted([[_message("どれですか？")]]), debug=True)
    j.log = j.gate.log = logs.append
    j.hear("ジョニー", now=0.0)
    j.hear("あれ", now=20.0)  # too late
    assert any("wake window open for 8 s (wake word)" in m for m in logs)
    assert any("wake window expired 12.0 s ago (utterance after the window)" in m for m in logs)
    j.hear("ジョニー、どれか取って")
    _wait(j)
    assert any("open for 8 s (Johnny asked a question)" in m for m in logs)
    j.hear("赤いの")
    assert any("closed (command in the window)" in m for m in logs)


# ---------------------------------------------------------------- speaker


def test_speaker_marks_its_own_speech():
    class Played(Speaker):  # a speaker that really plays audio marks the span; the text-only one does not
        def _play(self, text):
            return True

    silent = Speaker(out=open("/dev/null", "w"), grace_s=0.5)
    silent.say("はい")
    assert not silent._spans
    sp = Played(out=open("/dev/null", "w"), grace_s=0.5)
    sp.say("はい")
    (s, e), *_ = sp._spans
    assert sp.overlaps(s - 1.0, s + 0.001) and sp.overlaps(e, e + 0.3)
    assert not sp.overlaps(e + 1.0, e + 2.0) and not sp.overlaps(s - 3.0, s - 1.0)
