"""Pydantic schemas — strict contracts for every LLM structured output."""
from typing import List, Literal
from pydantic import BaseModel, Field, field_validator


class RetrievalGrade(BaseModel):
    """LLM-as-judge verdict on retrieved context quality."""

    relevance_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Continuous relevance score in [0, 1].",
    )
    verdict: Literal["relevant", "irrelevant"] = Field(
        ...,
        description="Hard binary verdict used for routing.",
    )
    reasoning: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Brief justification (1-3 sentences).",
    )

    @field_validator("verdict", mode="before")
    @classmethod
    def _normalize_verdict(cls, v: str) -> str:
        if isinstance(v, str):
            v = v.strip().lower()
            if v in ("yes", "true", "relevant"):
                return "relevant"
            if v in ("no", "false", "irrelevant"):
                return "irrelevant"
        return v


class RewrittenQuery(BaseModel):
    """Query reformulation for retry attempts."""

    rewritten_query: str = Field(..., min_length=3, max_length=500)
    rationale: str = Field(..., min_length=1, max_length=300)


class FinalAnswer(BaseModel):
    """Schema-enforced final response to the user."""

    answer: str = Field(..., min_length=1)
    sources_used: List[str] = Field(
        default_factory=list,
        description="Source identifiers (doc ids or URLs) cited in the answer.",
    )
    confidence: Literal["high", "medium", "low"] = Field(...)


class WebSearchResult(BaseModel):
    """Single web search hit."""

    title: str
    url: str
    snippet: str
