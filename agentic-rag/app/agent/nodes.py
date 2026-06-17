"""All LangGraph nodes — each is a pure async function (state -> state-delta)."""
from __future__ import annotations

from typing import Any, Dict, List

from langchain_anthropic import ChatAnthropic
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from app.config import get_settings
from app.tools.schemas import FinalAnswer, RetrievalGrade, RewrittenQuery
from app.tools.web_search import web_search_tool
from app.vectorstore.store import get_retriever


# ---------------------------------------------------------------------------
# Shared LLM factory
# ---------------------------------------------------------------------------
def _llm(temperature: float = 0.0) -> ChatAnthropic:
    settings = get_settings()
    return ChatAnthropic(
        model=settings.claude_model,
        api_key=settings.anthropic_api_key,
        temperature=temperature,
        max_tokens=2048,
        timeout=60,
        max_retries=2,
    )


def _format_docs(docs: List[Document]) -> str:
    if not docs:
        return "(no documents)"
    blocks = []
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("chunk_id") or d.metadata.get("source") or f"doc-{i}"
        blocks.append(f"[{i}] source={src}\n{d.page_content.strip()}")
    return "\n\n".join(blocks)


def _format_web(results: List[Dict[str, Any]]) -> str:
    if not results:
        return "(no web results)"
    blocks = []
    for i, r in enumerate(results, 1):
        blocks.append(
            f"[W{i}] {r.get('title','')}\nURL: {r.get('url','')}\n{r.get('snippet','')}"
        )
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# 1. RETRIEVE
# ---------------------------------------------------------------------------
async def retrieve_node(state: Dict[str, Any]) -> Dict[str, Any]:
    settings = get_settings()
    query = state.get("rewritten_question") or state["question"]
    retriever = get_retriever(k=settings.retrieval_k)
    docs = await retriever.ainvoke(query)
    return {
        "documents": docs,
        "rewrite_count": state.get("rewrite_count", 0),
    }


# ---------------------------------------------------------------------------
# 2. GRADE (CRAG evaluator with strict Pydantic output)
# ---------------------------------------------------------------------------
_GRADER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict retrieval-quality grader for a RAG system.\n"
            "Given a USER QUESTION and a set of RETRIEVED DOCUMENTS, decide "
            "whether the documents contain information sufficient to answer "
            "the question.\n\n"
            "Rules:\n"
            "- Score in [0,1]. >= {threshold} => verdict='relevant'.\n"
            "- Be conservative: vague or tangential matches are 'irrelevant'.\n"
            "- 'reasoning' must be 1-3 sentences.\n",
        ),
        (
            "human",
            "USER QUESTION:\n{question}\n\nRETRIEVED DOCUMENTS:\n{documents}\n\n"
            "Return ONLY the structured grade.",
        ),
    ]
)


async def grade_node(state: Dict[str, Any]) -> Dict[str, Any]:
    settings = get_settings()
    docs: List[Document] = state.get("documents", [])

    # Short-circuit: nothing retrieved -> irrelevant
    if not docs:
        return {
            "retrieval_verdict": "irrelevant",
            "relevance_score": 0.0,
        }

    structured = _llm(0.0).with_structured_output(RetrievalGrade)
    chain = _GRADER_PROMPT | structured
    grade: RetrievalGrade = await chain.ainvoke(
        {
            "question": state["question"],
            "documents": _format_docs(docs),
            "threshold": settings.grade_threshold,
        }
    )

    verdict = grade.verdict
    # Enforce threshold consistency even if the LLM picks the wrong label
    if grade.relevance_score >= settings.grade_threshold:
        verdict = "relevant"
    else:
        verdict = "irrelevant"

    return {
        "retrieval_verdict": verdict,
        "relevance_score": grade.relevance_score,
    }


# ---------------------------------------------------------------------------
# 3. REWRITE (corrective step before falling back to web)
# ---------------------------------------------------------------------------
_REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You rewrite user questions to maximize vector-search recall. "
            "Expand acronyms, add likely keywords, remove filler. "
            "Return only the structured object.",
        ),
        ("human", "Original question: {question}\nRewrite it."),
    ]
)


async def rewrite_node(state: Dict[str, Any]) -> Dict[str, Any]:
    structured = _llm(0.2).with_structured_output(RewrittenQuery)
    chain = _REWRITE_PROMPT | structured
    out: RewrittenQuery = await chain.ainvoke({"question": state["question"]})
    return {
        "rewritten_question": out.rewritten_query,
        "rewrite_count": state.get("rewrite_count", 0) + 1,
    }


# ---------------------------------------------------------------------------
# 4. WEB SEARCH (CRAG fallback)
# ---------------------------------------------------------------------------
async def web_search_node(state: Dict[str, Any]) -> Dict[str, Any]:
    settings = get_settings()
    query = state.get("rewritten_question") or state["question"]
    hits = await web_search_tool(query, k=settings.max_search_results)
    web_results = [h.model_dump() for h in hits]

    # Surface web hits into the documents list so the generator has one source
    web_docs = [
        Document(
            page_content=f"{h['title']}\n{h['snippet']}",
            metadata={"source": h["url"], "chunk_id": h["url"], "origin": "web"},
        )
        for h in web_results
    ]
    merged_docs = list(state.get("documents", [])) + web_docs
    return {"web_results": web_results, "documents": merged_docs}


# ---------------------------------------------------------------------------
# 5. GENERATE (final structured answer with conversation memory)
# ---------------------------------------------------------------------------
_GENERATE_SYSTEM = (
    "You are a precise assistant powering a production RAG system.\n"
    "- Answer ONLY from the provided CONTEXT (vector store + web results).\n"
    "- If the context is insufficient, say so explicitly — do not fabricate.\n"
    "- Cite sources by their source/url string in `sources_used`.\n"
    "- Calibrate `confidence`: high (well-supported), medium (partial), "
    "low (weakly supported or speculative).\n"
)


async def generate_node(state: Dict[str, Any]) -> Dict[str, Any]:
    docs: List[Document] = state.get("documents", [])
    web: List[Dict[str, Any]] = state.get("web_results", [])

    context_block = (
        f"VECTOR STORE CONTEXT:\n{_format_docs([d for d in docs if d.metadata.get('origin') != 'web'])}\n\n"
        f"WEB CONTEXT:\n{_format_web(web)}"
    )

    # Carry prior conversation turns (LangGraph checkpointer hydrates this)
    history = list(state.get("messages", []))
    history_msgs = [
        SystemMessage(content=_GENERATE_SYSTEM),
        *history,
        HumanMessage(
            content=(
                f"CONTEXT:\n{context_block}\n\n"
                f"QUESTION: {state['question']}\n\n"
                "Return ONLY the structured FinalAnswer."
            )
        ),
    ]

    structured = _llm(0.1).with_structured_output(FinalAnswer)
    final: FinalAnswer = await structured.ainvoke(history_msgs)

    # Persist this turn into long-term conversation memory
    new_messages = [
        HumanMessage(content=state["question"]),
        AIMessage(content=final.answer),
    ]

    return {
        "answer": final.answer,
        "sources": final.sources_used,
        "confidence": final.confidence,
        "messages": new_messages,
    }
