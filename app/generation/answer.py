from __future__ import annotations

from typing import Any

from langchain_groq import ChatGroq

from app.config import settings
from app.generation.prompts import (
    SYSTEM_PROMPT,
    build_user_prompt,
)


class RAGAnswerGenerator:
    """Generate grounded answers from reranked RAG context."""

    def __init__(
        self,
        model_name: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        if not settings.groq_api_key:
            raise ValueError(
                "GROQ_API_KEY is not configured."
            )

        self.model_name = model_name or settings.llm_model
        self.temperature = (
            settings.llm_temperature
            if temperature is None
            else temperature
        )
        self.max_tokens = (
            settings.llm_max_tokens
            if max_tokens is None
            else max_tokens
        )

        self.llm = ChatGroq(
            api_key=settings.groq_api_key,
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

    def generate(
        self,
        query: str,
        documents: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Generate a grounded answer from retrieved documents."""
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        prompt = build_user_prompt(
            query=query,
            documents=documents,
        )

        response = self.llm.invoke(
            [
                ("system", SYSTEM_PROMPT),
                ("human", prompt),
            ]
        )

        answer = response.content

        if isinstance(answer, list):
            answer = "".join(
                item.get("text", "")
                for item in answer
                if isinstance(item, dict)
            )

        answer = str(answer).strip()

        return {
            "query": query.strip(),
            "answer": answer,
            "model": self.model_name,
            "source_ids": [
                document.get("id")
                for document in documents
            ],
        }
