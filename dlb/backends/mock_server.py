"""A tiny Jev-wire-compatible fake server (stdlib only).

Used by the test-suite and handy for developing the harness on a machine
without a GPU or API key::

    python -m dlb.backends.mock_server --port 8099
    dlb smoke --backend typesafe --base-url http://127.0.0.1:8099

It validates the request shape, echoes usage numbers proportional to the
state size, and answers with a deterministic pseudo-random distribution
(seeded by the state) so runs are repeatable. If a question's state contains
``{"_mock_answer": {...}}`` those answers are returned instead.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

ENDPOINTS = {"/v1/systemone", "/v1/request"}


def _answers(body: dict[str, Any]) -> dict[str, Any]:
    state = body.get("state")
    seed = int(hashlib.sha1(json.dumps(state, sort_keys=True, default=str).encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)
    forced = state.get("_mock_answer", {}) if isinstance(state, dict) else {}
    out: dict[str, Any] = {}
    for key, q in body["questions"].items():
        t = q["type"]
        if t == "noul":
            p = forced.get(key, rng.random())
            out[key] = {"type": "noul", "noul": float(p)}
        elif t == "choice":
            opts = list(q["criteria"].keys())
            w = [rng.random() for _ in opts]
            if key in forced:
                w = [1.0 if o == forced[key] else 0.05 for o in opts]
            s = sum(w)
            probs = {o: x / s for o, x in zip(opts, w, strict=True)}
            top = sorted(probs.values(), reverse=True)
            best = max(probs, key=probs.get)
            out[key] = {
                "type": "choice",
                "choice": best,
                "probabilities": probs,
                "confidence": top[0] - (top[1] if len(top) > 1 else 0),
            }
        elif t == "score":
            n = len(q["criteria"])
            w = [rng.random() for _ in range(n)]
            if key in forced:
                w = [1.0 if i == int(forced[key]) else 0.05 for i in range(n)]
            s = sum(w)
            probs = {str(i): x / s for i, x in enumerate(w)}
            top = sorted(probs.values(), reverse=True)
            out[key] = {
                "type": "score",
                "score": sum(i * probs[str(i)] for i in range(n)),
                "legend": {str(i): c for i, c in enumerate(q["criteria"])},
                "probabilities": probs,
                "confidence": top[0] - (top[1] if len(top) > 1 else 0),
            }
        else:
            raise ValueError(f"bad question type {t}")
    return out


class Handler(BaseHTTPRequestHandler):
    accept_images = True

    def log_message(self, *a: Any) -> None:  # silence
        pass

    def _send(self, code: int, obj: dict[str, Any]) -> None:
        raw = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        if self.path in ("/health", "/ready"):
            self._send(200, {"ok": True})
        elif self.path == "/v1/models":
            self._send(200, {"data": [{"id": "mock-jev"}]})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path not in ENDPOINTS:
            self._send(404, {"error": f"unknown endpoint {self.path}"})
            return
        n = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(n))
            if "questions" not in body or "state" not in body:
                raise ValueError("state and questions are required")
            if body.get("images") and not self.accept_images:
                self._send(422, {"error": "images not supported"})
                return
            ans = _answers(body)
        except Exception as e:  # noqa: BLE001
            self._send(422, {"error": str(e)})
            return
        n_img = len(body.get("images") or [])
        tokens = len(json.dumps(body["state"])) // 4 + len(json.dumps(body["questions"])) // 4 + 280 * n_img
        self._send(
            200, {"model": "mock-jev", "answers": ans, "usage": {"input_tokens": tokens, "output_tokens": 0}}
        )


def serve(port: int = 8099, accept_images: bool = True, background: bool = False) -> HTTPServer:
    Handler.accept_images = accept_images
    srv = HTTPServer(("127.0.0.1", port), Handler)
    if background:
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        return srv
    print(f"mock Jev server on http://127.0.0.1:{port}  (POST /v1/systemone | /v1/request)")
    srv.serve_forever()
    return srv


if __name__ == "__main__":  # pragma: no cover
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--no-images", action="store_true")
    a = ap.parse_args()
    serve(a.port, accept_images=not a.no_images)
