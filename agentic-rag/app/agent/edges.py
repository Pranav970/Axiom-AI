"""Conditional routing — pure functions, no side effects."""
from __future__ import annotations

from typing import Any, Dict, Literal

from app.config import get_settings


def route_after_grade(
    state: Dict[str, Any],
) -> Literal["generate", "rewrite", "websearch"]:
    """CRAG decision point.

    - relevant            -> generate
    - irrelevant & rewrites_left -> rewrite (and re-retrieve)
    - irrelevant & exhausted     -> websearch
    """
    settings = get_settings()
    verdict = state.get("retrieval_verdict", "unknown")
    rewrites = state.get("rewrite_count", 0)

    if verdict == "relevant":
        return "generate"
    if rewrites < settings.max_rewrite_attempts:
        return "rewrite"
    return "websearch"
