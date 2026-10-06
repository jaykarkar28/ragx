from __future__ import annotations

from functools import lru_cache

from fastapi import APIRouter, HTTPException

from app.api.schemas import QueryRequest, QueryResponse, SourceDocument
from app.pipeline import RAGPipeline


router = APIRouter(
    prefix="/api",
    tags=["RAG"],
)


@lru_cache(maxsize=1)
def get_pipeline() -> RAGPipeline:
    """Load and cache the production RAG pipeline."""

    return RAGPipeline()


@router.get("/health")
def health_check() -> dict[str, str]:
    """Return API health status."""

    return {
        "status": "ok",
        "service": "RAGX",
    }


@router.post(
    "/query",
    response_model=QueryResponse,
)
def query_rag(request: QueryRequest) -> QueryResponse:
    """Run the production RAG pipeline for a user query."""

    try:
        pipeline = get_pipeline()
        result = pipeline.query(request.query)

        sources = [
            SourceDocument(
                id=str(document["id"]),
                text=document["text"],
                score=(
                    document.get("rerank_score")
                    if "rerank_score" in document
                    else document.get("score")
                ),
            )
            for document in result["sources"]
        ]

        return QueryResponse(
            query=result["query"],
            answer=result["answer"],
            source_ids=[
                str(source_id)
                for source_id in result["source_ids"]
            ],
            sources=sources,
            model=result["model"],
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"RAG pipeline error: {exc}",
        ) from exc
