from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# Project root
BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Central configuration for the RAGX framework."""

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Groq LLM
    groq_api_key: str = Field(default="", repr=False)
    llm_provider: Literal["groq"] = "groq"
    llm_model: str = "llama-3.3-70b-versatile"
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=1024, gt=0)

    # Embeddings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_batch_size: int = Field(default=32, gt=0)

    # Vector store
    vector_store_type: Literal["faiss"] = "faiss"
    vector_store_path: Path = BASE_DIR / "experiments" / "vector_store"

    # BM25
    bm25_index_path: Path = BASE_DIR / "experiments" / "bm25_index"

    # Retrieval
    dense_top_k: int = Field(default=10, gt=0)
    keyword_top_k: int = Field(default=10, gt=0)
    hybrid_top_k: int = Field(default=10, gt=0)
    fusion_method: Literal["rrf"] = "rrf"
    rrf_k: int = Field(default=60, gt=0)

    # Reranking
    reranker_model: str = (
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    reranker_top_k: int = Field(default=5, gt=0)
    reranker_enabled: bool = True

    # Generation
    generation_top_k: int = Field(default=5, gt=0)

    # Chunking
    chunk_size: int = Field(default=500, gt=0)
    chunk_overlap: int = Field(default=100, ge=0)

    # Evaluation
    evaluation_k_values: list[int] = [1, 3, 5, 10]

    # Retrieval ground truth
    evaluation_dataset_path: Path = (
        BASE_DIR / "datasets" / "ground_truth.jsonl"
    )

    # Reference-answer ground truth
    answer_evaluation_dataset_path: Path = (
        BASE_DIR / "datasets" / "answer_ground_truth.jsonl"
    )

    # Experiments
    experiment_config_dir: Path = (
        BASE_DIR / "experiments" / "configs"
    )
    experiment_report_dir: Path = (
        BASE_DIR / "experiments" / "reports"
    )

    # Application
    log_level: str = "INFO"

    @field_validator("chunk_overlap")
    @classmethod
    def validate_chunk_overlap(cls, value: int, info):
        chunk_size = info.data.get("chunk_size", 500)

        if value >= chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size"
            )

        return value

    @field_validator("evaluation_k_values")
    @classmethod
    def validate_k_values(cls, values: list[int]):
        if not values or any(k <= 0 for k in values):
            raise ValueError(
                "evaluation_k_values must contain positive integers"
            )

        return sorted(set(values))

    @property
    def documents_dir(self) -> Path:
        return BASE_DIR / "datasets" / "documents"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Get a cached settings instance."""

    return Settings()


settings = get_settings()