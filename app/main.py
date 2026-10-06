from fastapi import FastAPI

from app.api.routes import router


app = FastAPI(
    title="RAGX API",
    description="Production RAG API with hybrid retrieval and reranking.",
    version="1.0.0",
)

app.include_router(router)


@app.get("/")
def root() -> dict[str, str]:
    """Return basic API information."""

    return {
        "service": "RAGX API",
        "status": "running",
    }