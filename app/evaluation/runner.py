from __future__ import annotations

from typing import Any

from app.evaluation.answer_metrics import AnswerEvaluator
from app.evaluation.dataset import EvaluationDataset
from app.evaluation.retrieval_metrics import evaluate_retrieval
from app.generation.answer import RAGAnswerGenerator
from app.reranking.cross_encoder import CrossEncoderReranker
from app.retrieval.fusion import ReciprocalRankFusion
from app.retrieval.hybrid import HybridRetriever


class EvaluationRunner:
    """Evaluate retrieval stages and optionally generated RAG answers."""

    def __init__(
        self,
        dataset: EvaluationDataset,
        hybrid_retriever: HybridRetriever,
        fusion: ReciprocalRankFusion,
        reranker: CrossEncoderReranker | None = None,
        k: int = 5,
        candidate_k: int = 10,
        evaluate_answers: bool = True,
    ) -> None:
        if k <= 0 or candidate_k <= 0:
            raise ValueError("k and candidate_k must be positive.")

        if len(dataset) == 0:
            raise ValueError("Evaluation dataset cannot be empty.")

        self.dataset = dataset
        self.hybrid_retriever = hybrid_retriever
        self.fusion = fusion
        self.reranker = reranker
        self.k = k
        self.candidate_k = candidate_k
        self.evaluate_answers = evaluate_answers

        self.answer_generator = None
        self.answer_evaluator = None

        if self.evaluate_answers:
            self.answer_generator = RAGAnswerGenerator()
            self.answer_evaluator = AnswerEvaluator()

    @staticmethod
    def _extract_ids(
        documents: list[dict[str, Any]],
    ) -> list[str]:
        """Extract document IDs from retrieval results."""

        ids = []

        for document in documents:
            doc_id = document.get("id")

            if doc_id is None:
                raise ValueError(
                    "A retrieved document is missing its 'id' field."
                )

            ids.append(str(doc_id))

        return ids

    @staticmethod
    def _build_context(
        documents: list[dict[str, Any]],
    ) -> str:
        """Build evaluation context from retrieved documents."""

        sections = []

        for document in documents:
            chunk_id = document.get("id", "unknown")
            text = document.get("text", "").strip()

            if not text:
                continue

            sections.append(
                f"[Chunk ID: {chunk_id}]\n{text}"
            )

        return "\n\n".join(sections)

    def run(self) -> dict[str, Any]:
        """Run retrieval evaluation and optionally answer evaluation."""

        per_query: list[dict[str, Any]] = []

        stage_metrics: dict[
            str,
            list[dict[str, float]],
        ] = {}

        answer_metrics: list[dict[str, float]] = []

        for record in self.dataset:

            # ---------------------------------------------------------
            # 1. Retrieve candidates
            # ---------------------------------------------------------
            candidates = self.hybrid_retriever.retrieve_candidates(
                query=record.query,
                dense_top_k=self.candidate_k,
                keyword_top_k=self.candidate_k,
            )

            # ---------------------------------------------------------
            # 2. Fuse dense + keyword results
            # ---------------------------------------------------------
            fused_results = self.fusion.fuse(
                candidates,
                top_k=self.candidate_k,
            )

            stages = {
                "dense": candidates["dense"],
                "keyword": candidates["keyword"],
                "fused": fused_results,
            }

            # ---------------------------------------------------------
            # 3. Rerank fused results
            # ---------------------------------------------------------
            if self.reranker is not None:
                stages["reranked"] = self.reranker.rerank(
                    query=record.query,
                    documents=fused_results,
                    top_k=self.candidate_k,
                )

            # ---------------------------------------------------------
            # 4. Retrieval evaluation
            # ---------------------------------------------------------
            relevant = set(record.relevant_docs)

            query_result: dict[str, Any] = {
                "query": record.query,
                "stages": {},
            }

            for stage_name, documents in stages.items():

                retrieved_ids = self._extract_ids(documents)

                metrics = evaluate_retrieval(
                    retrieved=retrieved_ids,
                    relevant=relevant,
                    k=self.k,
                )

                query_result["stages"][stage_name] = {
                    "retrieved_ids": retrieved_ids,
                    **metrics,
                }

                stage_metrics.setdefault(
                    stage_name,
                    [],
                ).append(metrics)

            # ---------------------------------------------------------
            # 5-9. Optional answer generation + evaluation
            # ---------------------------------------------------------
            if self.evaluate_answers:

                generation_documents = (
                    stages["reranked"][:5]
                    if self.reranker is not None
                    else fused_results[:5]
                )

                # Generate answer.
                generated = self.answer_generator.generate(
                    query=record.query,
                    documents=generation_documents,
                )

                # Build context.
                context = self._build_context(
                    generation_documents
                )

                # Evaluate answer against retrieved context.
                evaluation = self.answer_evaluator.evaluate(
                    query=record.query,
                    answer=generated["answer"],
                    context=context,
                )

                answer_result: dict[str, Any] = {
                    "answer": generated["answer"],
                    "model": generated["model"],
                    "source_ids": generated["source_ids"],
                    "evaluation": evaluation,
                }

                # -----------------------------------------------------
                # Reference-answer evaluation
                # -----------------------------------------------------
                if record.reference_answer:
                    reference_evaluation = (
                        self.answer_evaluator.evaluate_reference_answer(
                            query=record.query,
                            generated_answer=generated["answer"],
                            reference_answer=record.reference_answer,
                        )
                    )

                    answer_result["reference_answer"] = {
                        "reference_answer": record.reference_answer,
                        "evaluation": reference_evaluation,
                    }

                    answer_metrics.append(
                        {
                            "faithfulness": float(
                                evaluation["faithfulness"]
                            ),
                            "answer_relevance": float(
                                evaluation["answer_relevance"]
                            ),
                            "grounded": float(
                                evaluation["grounded"]
                            ),
                            "reference_answer_score": float(
                                reference_evaluation[
                                    "reference_answer_score"
                                ]
                            ),
                        }
                    )

                else:
                    answer_metrics.append(
                        {
                            "faithfulness": float(
                                evaluation["faithfulness"]
                            ),
                            "answer_relevance": float(
                                evaluation["answer_relevance"]
                            ),
                            "grounded": float(
                                evaluation["grounded"]
                            ),
                        }
                    )

                query_result["answer"] = answer_result

            per_query.append(query_result)

        # -------------------------------------------------------------
        # 10. Calculate retrieval averages
        # -------------------------------------------------------------
        averages: dict[str, Any] = {}

        for stage_name, records in stage_metrics.items():

            metric_names = records[0].keys()

            averages[stage_name] = {
                metric: sum(
                    row[metric]
                    for row in records
                ) / len(records)
                for metric in metric_names
            }

        # -------------------------------------------------------------
        # 11. Calculate answer evaluation averages
        # -------------------------------------------------------------
        if answer_metrics:

            averages["answer_evaluation"] = {
                "faithfulness": sum(
                    row["faithfulness"]
                    for row in answer_metrics
                ) / len(answer_metrics),

                "answer_relevance": sum(
                    row["answer_relevance"]
                    for row in answer_metrics
                ) / len(answer_metrics),

                "grounded_rate": sum(
                    row["grounded"]
                    for row in answer_metrics
                ) / len(answer_metrics),
            }

            reference_scores = [
                row["reference_answer_score"]
                for row in answer_metrics
                if "reference_answer_score" in row
            ]

            if reference_scores:
                averages["answer_evaluation"][
                    "reference_answer_score"
                ] = (
                    sum(reference_scores)
                    / len(reference_scores)
                )

        # -------------------------------------------------------------
        # 12. Return complete evaluation result
        # -------------------------------------------------------------
        return {
            "queries_evaluated": len(per_query),
            "k": self.k,
            "candidate_k": self.candidate_k,
            "evaluate_answers": self.evaluate_answers,
            "average_metrics": averages,
            "per_query": per_query,
        }