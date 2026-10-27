"""Finds the skills: every module in ``dlb.johnny.skills`` (except this one and ``base``) is imported and its
``Skill`` subclasses are instantiated with no arguments. A module that fails to import is logged and skipped,
so the agent still runs with the skills that load.
"""

from __future__ import annotations

import importlib
import pkgutil
import sys
from collections.abc import Callable, Iterable
from typing import Any

from dlb.johnny.skills.base import Skill

SKIP = {"base", "registry"}


class Registry(dict):
    """Skills by name, in load order."""

    def tool_specs(self) -> list[dict[str, Any]]:
        return [s.tool_spec() for s in self.values()]


def load(
    names: Iterable[str] | None = None,
    package: str = "dlb.johnny.skills",
    log: Callable[[str], None] = lambda m: print(m, file=sys.stderr),
) -> Registry:
    """``names`` restricts to those skills (an unknown name is reported, not fatal)."""
    pkg = importlib.import_module(package)
    reg = Registry()
    for info in sorted(pkgutil.iter_modules(pkg.__path__), key=lambda i: i.name):
        if info.name in SKIP or info.name.startswith("_"):
            continue
        try:
            mod = importlib.import_module(f"{package}.{info.name}")
        except Exception as e:  # noqa: BLE001 - one broken skill module must not take the agent down
            log(f"skill module {info.name} not loaded: {type(e).__name__}: {e}")
            continue
        for obj in vars(mod).values():
            if not (isinstance(obj, type) and issubclass(obj, Skill) and obj is not Skill):
                continue
            if obj.__module__ != mod.__name__ or not obj.name:
                continue  # imported from elsewhere, or an abstract base
            try:
                skill = obj()
            except Exception as e:  # noqa: BLE001
                log(f"skill {obj.name} not loaded: {type(e).__name__}: {e}")
                continue
            if skill.name in reg:
                log(f"skill {skill.name}: duplicate name, {info.name} replaces the earlier one")
            reg[skill.name] = skill
    if names is not None:
        wanted = [n.strip() for n in names if n.strip()]
        for n in wanted:
            if n not in reg:
                log(f"skill {n}: not found")
        reg = Registry((n, reg[n]) for n in wanted if n in reg)
    return reg
