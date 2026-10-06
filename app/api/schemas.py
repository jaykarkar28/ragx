from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request body for the RAG query endpoint."""

    query: str = Field(
        min_length=1,
        description="User question to answer using the RAG pipeline.",
    )


class SourceDocument(BaseModel):
    """A retrieved source document."""

    id: str
    text: str
    score: float | None = None


class QueryResponse(BaseModel):
    """Response returned by the RAG query endpoint."""

    query: str
    answer: str
    source_ids: list[str]
    sources: list[SourceDocument]
    model: str