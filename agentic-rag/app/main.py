"""FastAPI entrypoint."""
from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.agent.graph import build_graph
from app.config import get_settings
from app.memory.checkpointer import lifespan_checkpointer, set_checkpointer


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    session_id: Optional[str] = Field(
        default=None,
        description="Persistent conversation id. Omit to start a fresh session.",
    )


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    confidence: str
    sources: List[str]
    retrieval_verdict: str
    relevance_score: float
    used_web_search: bool
    rewrite_count: int


class HealthResponse(BaseModel):
    status: str
    model: str
    collection: str


# ---------------------------------------------------------------------------
# Lifespan: bring up checkpointer + compiled graph once per process
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    async with lifespan_checkpointer() as cp:
        set_checkpointer(cp)
        app.state.graph = build_graph(cp)
        yield
        app.state.graph = None
        set_checkpointer(None)


app = FastAPI(
    title="Production Agentic RAG",
    version="1.0.0",
    description="CRAG-style LangGraph agent with persistent per-session memory.",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    s = get_settings()
    return HealthResponse(
        status="ok",
        model=s.claude_model,
        collection=s.chroma_collection,
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    graph = app.state.graph
    if graph is None:
        raise HTTPException(status_code=503, detail="Graph not initialised")

    session_id = req.session_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": session_id}}

    initial_state = {
        "question": req.question,
        "rewrite_count": 0,
    }

    try:
        final_state = await graph.ainvoke(initial_state, config=config)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {e}") from e

    return ChatResponse(
        session_id=session_id,
        answer=final_state.get("answer", ""),
        confidence=final_state.get("confidence", "low"),
        sources=final_state.get("sources", []) or [],
        retrieval_verdict=final_state.get("retrieval_verdict", "unknown"),
        relevance_score=float(final_state.get("relevance_score", 0.0)),
        used_web_search=bool(final_state.get("web_results")),
        rewrite_count=int(final_state.get("rewrite_count", 0)),
    )


@app.get("/session/{session_id}/history")
async def session_history(session_id: str):
    """Inspect persisted state for a session (debug/observability)."""
    graph = app.state.graph
    if graph is None:
        raise HTTPException(status_code=503, detail="Graph not initialised")

    config = {"configurable": {"thread_id": session_id}}
    snapshot = await graph.aget_state(config)
    if snapshot is None or not snapshot.values:
        raise HTTPException(status_code=404, detail="Session not found")

    msgs = snapshot.values.get("messages", []) or []
    return {
        "session_id": session_id,
        "turns": [
            {"role": m.__class__.__name__.replace("Message", "").lower(), "content": m.content}
            for m in msgs
        ],
    }
