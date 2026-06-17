"""GraphState — the single object that flows through every node."""
from __future__ import annotations

from operator import add
from typing import Annotated, List, Literal, Sequence, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage


class GraphState(TypedDict, total=False):
    # --- Conversation memory (accumulated across turns via Annotated reducer) ---
    messages: Annotated[Sequence[BaseMessage], add]

    # --- Per-turn working memory ---
    question: str
    rewritten_question: str
    documents: List[Document]
    web_results: List[dict]

    # --- Routing signals ---
    retrieval_verdict: Literal["relevant", "irrelevant", "unknown"]
    relevance_score: float
    rewrite_count: int
    route: Literal["generate", "rewrite", "websearch"]

    # --- Final outputs ---
    answer: str
    sources: List[str]
    confidence: Literal["high", "medium", "low"]
