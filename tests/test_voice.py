import io
import json
import wave

import httpx
import numpy as np
import pytest

from dlb.voice.console import Session
from dlb.voice.intent import Intent, IntentParser, RobotState, apply, check_sequence, schema
from dlb.voice.listen import SR, Segmenter, Transcriber, file_frames, listen

OBJS = ["red cube", "blue cube", "bin"]  # example table (the state lists what is on the table)


def _responses_client(answer: dict, seen: list) -> httpx.Client:
    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(json.loads(req.content))
        return httpx.Response(
            200,
            json={
                "output": [
                    {"type": "reasoning", "summary": []},
                    {"type": "message", "content": [{"type": "output_text", "text": json.dumps(answer)}]},
                ]
            },
        )

    return httpx.Client(base_url="https://api.openai.com/v1", transport=httpx.MockTransport(handler))


def test_parse_builds_request_and_normalizes():
    seen: list = []
    answer = {
        "tasks": [
            {"op": "place", "object": None, "where": "here"},
            {"op": "pick", "object": "blue cube", "where": None},
            {"op": "place", "object": None, "where": "bin"},
        ],
        "replace_queue": True,
        "clarify": None,
    }
    state = RobotState(held="red cube", queue=[{"op": "place", "where": "bin"}], objects=OBJS)
    intent = IntentParser(client=_responses_client(answer, seen)).parse(
        "それをその場に置いて青いキューブをビンに入れて", state
    )
    assert intent.tasks == [
        {"op": "place", "where": "here"},
        {"op": "pick", "object": "blue cube"},
        {"op": "place", "where": "bin"},
    ]
    assert intent.replace_queue and intent.clarify is None
    body = seen[0]
    assert body["model"] == "gpt-5.5" and body["reasoning"] == {"effort": "low"}
    fmt = body["text"]["format"]
    assert fmt["strict"] and fmt["type"] == "json_schema"
    where = fmt["schema"]["properties"]["tasks"]["items"]["properties"]["where"]["enum"]
    assert set(where) == {"bin", "here", "on:red cube", "on:blue cube", None}
    prompt = body["input"][0]["content"][0]["text"]
    assert '"red cube"' in prompt and "それをその場に置いて" in prompt
    assert check_sequence(intent, state) == []


def test_parse_rejects_unknown_names_and_http_errors():
    state = RobotState(objects=OBJS)
    bad = {
        "tasks": [{"op": "pick", "object": "green cube", "where": None}],
        "replace_queue": True,
        "clarify": None,
    }
    with pytest.raises(ValueError):
        IntentParser(client=_responses_client(bad, [])).parse("緑のキューブを取って", state)
    err = httpx.Client(
        base_url="https://api.openai.com/v1",
        transport=httpx.MockTransport(lambda r: httpx.Response(500, text="boom")),
    )
    with pytest.raises(RuntimeError):
        IntentParser(client=err).parse("止まって", state)


def test_schema_is_strict_everywhere():
    s = schema(RobotState(objects=["red cube", "bin"]))
    item = s["properties"]["tasks"]["items"]
    for obj in (s, item):
        assert obj["additionalProperties"] is False and set(obj["required"]) == set(obj["properties"])
    assert item["properties"]["object"]["enum"] == [
        "red cube",
        None,
    ]  # the bin is a place, not an object to pick


def test_check_sequence_and_apply():
    state = RobotState(held="red cube", queue=[{"op": "place", "where": "bin"}], objects=OBJS)
    grab = Intent([{"op": "pick", "object": "blue cube"}], replace_queue=True)
    assert check_sequence(grab, state) == ["task 0: pick blue cube while holding red cube"]
    after = Intent([{"op": "pick", "object": "blue cube"}, {"op": "place", "where": "on:red cube"}], False)
    assert check_sequence(after, state) == []  # appended: the queue empties the gripper first
    assert apply(after, state).queue == [{"op": "place", "where": "bin"}, *after.tasks]
    assert apply(grab, state).queue == grab.tasks
    assert apply(Intent([], True, clarify="どの赤？"), state).queue == state.queue
    assert apply(Intent([], True), state).queue == []  # stop


def test_session_updates_queue_and_handles_commands(capsys):
    answers = iter(
        [
            {
                "tasks": [{"op": "place", "object": None, "where": "here"}],
                "replace_queue": True,
                "clarify": None,
            },
            {
                "tasks": [
                    {"op": "pick", "object": "blue cube", "where": None},
                    {"op": "place", "object": None, "where": "bin"},
                ],
                "replace_queue": False,
                "clarify": None,
            },
        ]
    )

    def handler(req):
        return httpx.Response(
            200,
            json={
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": json.dumps(next(answers))}],
                    }
                ]
            },
        )

    parser = IntentParser(
        client=httpx.Client(base_url="https://x/v1", transport=httpx.MockTransport(handler))
    )
    session = Session(parser, RobotState(held="red cube", objects=OBJS))
    session.run(["それを置いて", "", ":hold", "それから青いキューブをビンに入れて", ":queue"])
    assert session.state.held is None
    assert [t["op"] for t in session.state.queue] == ["place", "pick", "place"]
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert len(lines) == 2 and lines[1]["replace_queue"] is False


def _tone(s: float, amp: float = 3000.0, f: float = 220.0) -> np.ndarray:
    t = np.arange(int(s * SR)) / SR
    return (amp * np.sin(2 * np.pi * f * t)).astype(np.int16)


def _noise(s: float, amp: float = 30.0) -> np.ndarray:
    return np.random.default_rng(0).normal(0, amp, int(s * SR)).astype(np.int16)


def test_segmenter_splits_utterances_and_ignores_clicks():
    audio = np.concatenate(
        [_noise(1.0), _tone(1.2), _noise(1.0), _tone(0.06), _noise(1.0), _tone(0.8), _noise(1.0)]
    )
    seg = Segmenter()
    utts = [u for i in range(0, len(audio) - 480 + 1, 480) if (u := seg.push(audio[i : i + 480])) is not None]
    assert len(utts) == 2  # the 60 ms click is too short
    assert 1.2 < len(utts[0]) / SR < 1.2 + 0.3 + 0.25 and 0.8 < len(utts[1]) / SR < 0.8 + 0.3 + 0.25


def test_listen_from_file_with_mock_server(tmp_path):
    path = tmp_path / "x.wav"
    with wave.open(str(path), "wb") as w:  # 24 kHz like the TTS samples: exercises the resampling
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes(
            np.concatenate([_noise(0.5), _tone(1.0), _noise(1.0), _tone(1.0)]).repeat(3)[::2].tobytes()
        )
    got: list = []

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/v1/audio/transcriptions"
        body = req.content
        wav = body[body.index(b"RIFF") :]
        with wave.open(io.BytesIO(wav)) as w:
            got.append(w.getnframes() / w.getframerate())
        assert b'name="language"' in body and b"vad_filter" in body
        return httpx.Response(
            200, json={"text": ["赤いキューブ", "ご視聴ありがとうございました"][len(got) - 1]}
        )

    tr = Transcriber(
        "http://127.0.0.1:8010/v1",
        client=httpx.Client(base_url="http://127.0.0.1:8010/v1", transport=httpx.MockTransport(handler)),
    )
    utts = list(listen(file_frames(path), tr))
    assert len(got) == 2  # two utterances sent
    assert [u.text for u in utts] == ["赤いキューブ"]  # the stock hallucination is dropped
