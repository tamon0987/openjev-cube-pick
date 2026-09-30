"""HTTP backend speaking the Jev / System-One wire format.

Covers three servers with one class, differing only in URL, endpoint path and
whether images are accepted:

    profile     base_url                    endpoint          images
    ---------   -------------------------   ---------------   ------
    typesafe    https://api.typesafe.ai     /v1/systemone     no (state JSON only)
    openjev     http://127.0.0.1:8080       /v1/systemone     yes (per README, up to 8, ~280 tok each)
    djev        http://127.0.0.1:8000       /v1/request       yes (data URLs in ``images``)

Everything is configured from YAML (see ``configs/backends``).
"""

from __future__ import annotations

import os
import time
from typing import Any

import httpx

from dlb.backends.base import BackendInfo, DecisionBackend, ImagePolicy
from dlb.contract import DecisionRequest, DecisionResponse


class JevHTTPBackend(DecisionBackend):
    def __init__(
        self,
        name: str,
        base_url: str,
        endpoint: str = "/v1/systemone",
        api_key_env: str | None = None,
        api_key: str | None = None,
        model: str | None = "jev-latest",
        send_model_field: bool = True,
        image_policy: ImagePolicy = "drop",
        image_field: str = "images",
        timeout_s: float = 30.0,
        price_per_m_input_tokens_usd: float = 0.0,
        max_retries: int = 2,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.endpoint = endpoint
        self.model = model
        self.send_model_field = send_model_field
        self.image_policy = image_policy
        self.image_field = image_field
        self.max_retries = max_retries
        key = api_key or (os.environ.get(api_key_env) if api_key_env else None)
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        if extra_headers:
            headers.update(extra_headers)
        self._client = httpx.Client(base_url=self.base_url, headers=headers, timeout=timeout_s)
        self.info = BackendInfo(
            name=name,
            modality="image" if image_policy == "send" else "text",
            model=model or "",
            price_per_m_input_tokens_usd=price_per_m_input_tokens_usd,
            extra={"base_url": self.base_url, "endpoint": endpoint},
        )

    # ------------------------------------------------------------------ #
    def _body(self, req: DecisionRequest) -> dict[str, Any]:
        if req.images and self.image_policy == "error":
            raise ValueError(f"backend {self.info.name} does not accept images")
        body = req.to_wire(
            model=self.model if self.send_model_field else None,
            include_images=False,
        )
        if req.images and self.image_policy == "send":
            # image_field may be a dotted path, e.g. "state.images" if a server wants
            # the attachments nested inside the state object instead of top-level.
            parts = self.image_field.split(".")
            target: dict[str, Any] = body
            for p in parts[:-1]:
                nxt = target.get(p)
                if not isinstance(nxt, dict):
                    raise ValueError(f"cannot nest images under {self.image_field!r}: {p!r} is not an object")
                target = nxt
            target[parts[-1]] = list(req.images)
        return body

    def decide(self, req: DecisionRequest) -> DecisionResponse:
        body = self._body(req)
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            t0 = time.perf_counter()
            try:
                r = self._client.post(self.endpoint, json=body)
                dt = time.perf_counter() - t0
                if r.status_code >= 500 and attempt < self.max_retries:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                r.raise_for_status()
                resp = DecisionResponse.from_wire(r.json(), latency_s=dt)
                # sanity: every question answered
                missing = set(req.questions) - set(resp.answers)
                if missing:
                    raise RuntimeError(f"{self.info.name}: answers missing for {sorted(missing)}")
                return resp
            except (httpx.TransportError, httpx.HTTPStatusError) as e:  # pragma: no cover - network
                last_exc = e
                if isinstance(e, httpx.HTTPStatusError) and e.response.status_code < 500:
                    # 4xx: do not retry; surface server message
                    raise RuntimeError(
                        f"{self.info.name} {e.response.status_code}: {e.response.text[:500]}"
                    ) from e
                if attempt >= self.max_retries:
                    raise
                time.sleep(0.5 * (attempt + 1))
        raise RuntimeError(f"{self.info.name}: request failed") from last_exc

    def health(self) -> dict[str, Any]:
        out: dict[str, Any] = {"ok": False, "base_url": self.base_url}
        for path in ("/health", "/ready", "/v1/models"):
            try:
                r = self._client.get(path, timeout=5.0)
                out[path] = r.status_code
                if r.status_code == 200:
                    out["ok"] = True
            except httpx.TransportError as e:  # pragma: no cover - network
                out[path] = f"error: {e.__class__.__name__}"
        return out

    def close(self) -> None:
        self._client.close()
