""" Async SQLite checkpointer — provides per-session_id persistent memory.

LangGraph automatically stores the full GraphState on every node transition
keyed by `thread_id` (== our session_id), enabling resumable, multi-turn
conversations across separate API calls and process restarts.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from app.config import get_settings


@asynccontextmanager
async def lifespan_checkpointer() -> AsyncIterator[AsyncSqliteSaver]:
    """Yield a ready-to-use checkpointer for the application lifespan."""
    settings = get_settings()
    conn = await aiosqlite.connect(
        settings.memory_db_path,
        check_same_thread=False,
    )
    try:
        saver = AsyncSqliteSaver(conn)
        await saver.setup()
        yield saver
    finally:
        await conn.close()


# Module-level handle populated by FastAPI lifespan, read by the graph layer
_GLOBAL_CHECKPOINTER: AsyncSqliteSaver | None = None


def set_checkpointer(cp: AsyncSqliteSaver | None) -> None:
    global _GLOBAL_CHECKPOINTER
    _GLOBAL_CHECKPOINTER = cp


def get_checkpointer() -> AsyncSqliteSaver:
    if _GLOBAL_CHECKPOINTER is None:
        raise RuntimeError(
            "Checkpointer not initialised. Ensure FastAPI lifespan ran."
        )
    return _GLOBAL_CHECKPOINTER
