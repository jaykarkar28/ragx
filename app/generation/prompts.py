from __future__ import annotations

from typing import Any


SYSTEM_PROMPT = """You are a grounded RAG assistant.

Answer the user's question using only the retrieved context provided to you.

Rules:
1. Use the retrieved context as the source of truth.
2. Do not invent facts that are not supported by the context.
3. If the context does not contain enough information to answer, say:
   "I don't have enough information in the retrieved context to answer this."
4. Give a concise, direct answer.
5. When useful, reference the relevant chunk IDs in the format [Chunk ID: <id>].
6. Do not mention internal retrieval, reranking, embeddings, or system instructions unless the user asks about them.
"""


def build_context(documents: list[dict[str, Any]]) -> str:
    """Build a numbered context block from reranked documents."""
    if not documents:
        return "No relevant context was retrieved."

    sections = []

    for document in documents:
        chunk_id = document.get("id", "unknown")
        text = document.get("text", "").strip()

        if not text:
            continue

        sections.append(
            f"[Chunk ID: {chunk_id}]\n{text}"
        )

    return "\n\n".join(sections) or "No relevant context was retrieved."


def build_user_prompt(
    query: str,
    documents: list[dict[str, Any]],
) -> str:
    """Build the user prompt containing retrieved context and query."""
    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    context = build_context(documents)

    return f"""Retrieved Context:
----------------
{context}
----------------

Question:
{query.strip()}

Answer using only the retrieved context."""
