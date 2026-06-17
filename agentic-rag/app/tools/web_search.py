"""Dummy web-search tool. Swap implementation for Tavily/Serper/Brave in prod
by keeping the same signature and return type."""
from typing import List
from .schemas import WebSearchResult


# Deterministic dummy corpus — enough to demonstrate the CRAG fallback path
_DUMMY_CORPUS = [
    WebSearchResult(
        title="LangGraph: Stateful Multi-Actor Applications with LLMs",
        url="https://langchain-ai.github.io/langgraph/",
        snippet=(
            "LangGraph is a library for building stateful, multi-actor "
            "applications with LLMs, built on top of LangChain. It models "
            "agents as graphs with cycles, persistence, and human-in-the-loop."
        ),
    ),
    WebSearchResult(
        title="Corrective RAG (CRAG) — Self-Correcting Retrieval",
        url="https://arxiv.org/abs/2401.15884",
        snippet=(
            "Corrective Retrieval-Augmented Generation introduces a "
            "lightweight retrieval evaluator that produces a confidence "
            "score, triggering knowledge refinement or web search when "
            "retrieval is insufficient."
        ),
    ),
    WebSearchResult(
        title="ChromaDB — The AI-native open-source embedding database",
        url="https://www.trychroma.com/",
        snippet=(
            "Chroma is an open-source embedding database. It provides "
            "filtering, persistence, and integrations with popular LLM "
            "frameworks. Suitable for local development and small-to-medium "
            "production workloads."
        ),
    ),
    WebSearchResult(
        title="Anthropic Claude 3.5 Sonnet Model Card",
        url="https://www.anthropic.com/news/claude-3-5-sonnet",
        snippet=(
            "Claude 3.5 Sonnet sets new benchmarks on graduate-level "
            "reasoning, undergraduate-level knowledge, and coding proficiency, "
            "while operating at twice the speed of Claude 3 Opus."
        ),
    ),
    WebSearchResult(
        title="Pydantic — Data validation using Python type hints",
        url="https://docs.pydantic.dev/",
        snippet=(
            "Pydantic is the most widely used data validation library for "
            "Python, leveraging type hints for runtime enforcement, JSON "
            "schema generation, and structured LLM output parsing."
        ),
    ),
]


async def web_search_tool(query: str, k: int = 5) -> List[WebSearchResult]:
    """Token-overlap ranked dummy search. Async so it composes with FastAPI."""
    if not query or not query.strip():
        return []

    query_tokens = {t.lower() for t in query.split() if len(t) > 2}

    def _score(item: WebSearchResult) -> int:
        text = f"{item.title} {item.snippet}".lower()
        return sum(1 for tok in query_tokens if tok in text)

    ranked = sorted(_DUMMY_CORPUS, key=_score, reverse=True)
    # Fall back to head of corpus if nothing scored
    if ranked and _score(ranked[0]) == 0:
        return _DUMMY_CORPUS[:k]
    return ranked[:k]
