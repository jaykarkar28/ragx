from __future__ import annotations

import json
from typing import Any

from langchain_groq import ChatGroq
from pydantic import BaseModel, Field, ValidationError

from app.config import settings


class AnswerEvaluation(BaseModel):
    """Structured evaluation result for a generated RAG answer."""

    faithfulness: float = Field(ge=0.0, le=1.0)
    answer_relevance: float = Field(ge=0.0, le=1.0)
    grounded: bool
    reasoning: str


class ReferenceAnswerEvaluation(BaseModel):
    """Evaluation result comparing a generated answer with a reference answer."""

    reference_answer_score: float = Field(ge=0.0, le=1.0)
    reasoning: str


def build_evaluation_prompt(
    query: str,
    answer: str,
    context: str,
) -> str:
    """Build the prompt used by the answer evaluator."""

    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    if not answer or not answer.strip():
        raise ValueError("Answer cannot be empty.")

    if not context or not context.strip():
        raise ValueError("Context cannot be empty.")

    return f"""Evaluate the following RAG answer.

Question:
{query.strip()}

Retrieved Context:
----------------
{context.strip()}
----------------

Generated Answer:
{answer.strip()}

Evaluate ONLY against the retrieved context.

Evaluate these three criteria:

1. Faithfulness:
Are the factual claims in the answer supported by the retrieved context?

2. Answer relevance:
Does the answer directly address the question?

3. Groundedness:
Does the answer avoid unsupported factual claims?

Return ONLY valid JSON.
Do not include markdown.
Do not include explanations outside the JSON.

The JSON must use exactly this structure:

{{
  "faithfulness": 0.0,
  "answer_relevance": 0.0,
  "grounded": true,
  "reasoning": "Brief explanation."
}}

Rules:
- faithfulness must be a number between 0.0 and 1.0
- answer_relevance must be a number between 0.0 and 1.0
- grounded must be true or false
- reasoning must be concise
"""


def build_reference_answer_prompt(
    query: str,
    generated_answer: str,
    reference_answer: str,
) -> str:
    """Build the prompt used to compare a generated answer with a reference."""

    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    if not generated_answer or not generated_answer.strip():
        raise ValueError("Generated answer cannot be empty.")

    if not reference_answer or not reference_answer.strip():
        raise ValueError("Reference answer cannot be empty.")

    return f"""Evaluate how well the generated answer matches the reference answer.

Question:
{query.strip()}

Reference Answer:
----------------
{reference_answer.strip()}
----------------

Generated Answer:
----------------
{generated_answer.strip()}
----------------

Evaluate semantic correctness and coverage, not exact wording.

A high score means:
- The generated answer correctly addresses the question.
- It contains the important information present in the reference answer.
- It does not contradict the reference answer.

A low score means:
- Important information from the reference answer is missing.
- The generated answer does not properly answer the question.
- The generated answer contains contradictions or substantial inaccuracies.

Return ONLY valid JSON.
Do not include markdown.
Do not include explanations outside the JSON.

The JSON must use exactly this structure:

{{
  "reference_answer_score": 0.0,
  "reasoning": "Brief explanation."
}}

Rules:
- reference_answer_score must be a number between 0.0 and 1.0
- reasoning must be concise
"""


class AnswerEvaluator:
    """Evaluate generated RAG answers using an LLM judge."""

    def __init__(
        self,
        model_name: str | None = None,
    ) -> None:
        if not settings.groq_api_key:
            raise ValueError("GROQ_API_KEY is not configured.")

        self.model_name = model_name or settings.llm_model

        self.llm = ChatGroq(
            api_key=settings.groq_api_key,
            model=self.model_name,
            temperature=0.0,
        )

    @staticmethod
    def _parse_result(
        content: str,
    ) -> AnswerEvaluation:
        """Parse and validate the evaluator's JSON response."""

        content = content.strip()

        if content.startswith("```"):
            lines = content.splitlines()

            if lines and lines[0].startswith("```"):
                lines = lines[1:]

            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]

            content = "\n".join(lines).strip()

            if content.lower().startswith("json"):
                content = content[4:].strip()

        try:
            data = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "Answer evaluator returned invalid JSON. "
                f"Raw response: {content}"
            ) from exc

        try:
            return AnswerEvaluation.model_validate(data)
        except ValidationError as exc:
            raise ValueError(
                "Answer evaluator returned invalid evaluation fields. "
                f"Response: {data}"
            ) from exc

    @staticmethod
    def _parse_reference_result(
        content: str,
    ) -> ReferenceAnswerEvaluation:
        """Parse and validate reference-answer evaluation JSON."""

        content = content.strip()

        if content.startswith("```"):
            lines = content.splitlines()

            if lines and lines[0].startswith("```"):
                lines = lines[1:]

            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]

            content = "\n".join(lines).strip()

            if content.lower().startswith("json"):
                content = content[4:].strip()

        try:
            data = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "Reference-answer evaluator returned invalid JSON. "
                f"Raw response: {content}"
            ) from exc

        try:
            return ReferenceAnswerEvaluation.model_validate(data)
        except ValidationError as exc:
            raise ValueError(
                "Reference-answer evaluator returned invalid fields. "
                f"Response: {data}"
            ) from exc

    def evaluate(
        self,
        query: str,
        answer: str,
        context: str,
    ) -> dict[str, Any]:
        """Evaluate one generated answer."""

        prompt = build_evaluation_prompt(
            query=query,
            answer=answer,
            context=context,
        )

        response = self.llm.invoke(prompt)

        result = self._parse_result(
            str(response.content)
        )

        return {
            "faithfulness": result.faithfulness,
            "answer_relevance": result.answer_relevance,
            "grounded": result.grounded,
            "reasoning": result.reasoning,
            "model": self.model_name,
        }

    def evaluate_reference_answer(
        self,
        query: str,
        generated_answer: str,
        reference_answer: str,
    ) -> dict[str, Any]:
        """Evaluate a generated answer against a reference answer."""

        prompt = build_reference_answer_prompt(
            query=query,
            generated_answer=generated_answer,
            reference_answer=reference_answer,
        )

        response = self.llm.invoke(prompt)

        result = self._parse_reference_result(
            str(response.content)
        )

        return {
            "reference_answer_score": result.reference_answer_score,
            "reasoning": result.reasoning,
            "model": self.model_name,
        }