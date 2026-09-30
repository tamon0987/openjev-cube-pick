from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

from dlb.contract import DecisionRequest, DecisionResponse

ImagePolicy = Literal["send", "drop", "error"]


@dataclass
class BackendInfo:
    name: str
    modality: Literal["text", "image", "local"]
    model: str = ""
    price_per_m_input_tokens_usd: float = 0.0  # 0 for self-hosted
    extra: dict[str, Any] = field(default_factory=dict)


class DecisionBackend(ABC):
    """A thing that answers typed questions about a state (and optionally images)."""

    info: BackendInfo
    image_policy: ImagePolicy = "drop"

    @abstractmethod
    def decide(self, req: DecisionRequest) -> DecisionResponse: ...

    def health(self) -> dict[str, Any]:
        """Cheap liveness probe. Override for HTTP backends."""
        return {"ok": True}

    def close(self) -> None:  # noqa: B027 - optional hook for HTTP backends
        """Release resources (no-op by default)."""

    # ---- convenience for gating ----------------------------------------- #
    def wants_images(self) -> bool:
        return self.image_policy == "send"
