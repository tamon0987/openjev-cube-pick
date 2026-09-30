"""Build backends from YAML profiles in ``configs/backends``.

A profile looks like::

    kind: jev_http           # jev_http | oracle | random
    name: typesafe
    base_url: https://api.typesafe.ai
    endpoint: /v1/systemone
    api_key_env: TYPESAFE_API_KEY
    model: jev-latest
    image_policy: drop       # send | drop | error
    price_per_m_input_tokens_usd: 0.042
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from dlb.backends.base import DecisionBackend
from dlb.backends.jev_http import JevHTTPBackend
from dlb.backends.local import OracleBackend, RandomBackend

ROOT = Path(__file__).resolve().parents[2]
PROFILE_DIR = ROOT / "configs" / "backends"


def _expand_env(v: Any) -> Any:
    if isinstance(v, str):
        return os.path.expandvars(v)
    if isinstance(v, dict):
        return {k: _expand_env(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_expand_env(x) for x in v]
    return v


def load_profile(name_or_path: str) -> dict[str, Any]:
    p = Path(name_or_path)
    if not p.exists():
        p = PROFILE_DIR / f"{name_or_path}.yaml"
    if not p.exists():
        raise FileNotFoundError(f"backend profile not found: {name_or_path} (looked in {PROFILE_DIR})")
    with open(p, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg = _expand_env(cfg)
    cfg.setdefault("name", p.stem)
    return cfg


def build_backend(profile: str | dict[str, Any], **overrides: Any) -> DecisionBackend:
    cfg = dict(load_profile(profile)) if isinstance(profile, str) else dict(profile)
    cfg.update(overrides)
    kind = cfg.pop("kind", "jev_http")
    if kind == "jev_http":
        return JevHTTPBackend(**cfg)
    if kind == "oracle":
        return OracleBackend(**cfg)
    if kind == "random":
        return RandomBackend(**cfg)
    raise ValueError(f"unknown backend kind: {kind}")


def list_profiles() -> list[str]:
    return sorted(p.stem for p in PROFILE_DIR.glob("*.yaml"))
