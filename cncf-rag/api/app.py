"""FastAPI application: exactly three endpoints — /query, /health, /metrics."""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager

import structlog
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

from cncf_rag.observability.metrics import metrics
from cncf_rag.observability.tracing import init_tracing

logger = structlog.get_logger(__name__)

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv()
    # Heavy clients constructed once at startup, not per-request.
    from cncf_rag.embedding.openai_provider import OpenAIEmbeddingService
    from cncf_rag.generation.service import GenerationService
    from cncf_rag.retrieval.pipeline import RetrievalPipeline
    from cncf_rag.retrieval.reranker import CohereReranker
    from cncf_rag.vectorstore.qdrant_store import QdrantVectorStore

    embedder = OpenAIEmbeddingService()
    store = QdrantVectorStore()
    _state["store"] = store
    _state["pipeline"] = RetrievalPipeline(embedder, store, reranker=CohereReranker())
    # GenerationService reads GENERATION_PROVIDER env var: "openai" or "anthropic".
    _state["generator"] = GenerationService()
    _state["tracer"] = init_tracing()
    yield
    _state.clear()


app = FastAPI(
    title="CNCF RAG API",
    version="0.1.0",
    description="RAG-powered Q&A over Kubernetes, Helm, Argo CD, and Prometheus documentation.",
    lifespan=lifespan,
)


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, description="Natural-language question")
    project_filter: str | None = Field(None, description="kubernetes|helm|argocd|prometheus")
    version_filter: str | None = Field(None, description="e.g. v1.29")


class Citation(BaseModel):
    project: str | None = None
    doc_type: str | None = None
    version: str | None = None
    source_path: str | None = None


class QueryResponse(BaseModel):
    query_id: str
    answer: str
    citations: list[dict]
    confidence: float
    cannot_answer: bool
    version_warning: str | None
    tokens_input: int
    tokens_output: int
    cost_usd: float
    model_used: str
    strategy_used: str
    total_ms: float


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    query_id = str(uuid.uuid4())
    start = time.perf_counter()
    tracer = _state["tracer"]
    # One span covers the FULL pipeline so retrieval and generation latency
    # are attributable within a single trace.
    with tracer.start_as_current_span("rag_query") as span:
        span.set_attribute("query_id", query_id)
        # Build explicit user filters. These are passed into the retrieval pipeline
        # so Qdrant applies them server-side before scoring — not as Python post-filters.
        # Post-filtering was wrong: if Qdrant returned top-5 from the wrong project and
        # Python filtered them all out, the answer was "cannot_answer" even when relevant
        # docs existed at rank 6+.
        filter_override: dict | None = None
        if request.project_filter or request.version_filter:
            filter_override = {}
            if request.project_filter:
                filter_override["project"] = request.project_filter
            if request.version_filter:
                filter_override["version_tag"] = request.version_filter

        retrieval = await _state["pipeline"].retrieve(request.query, filter_override=filter_override)
        chunks = retrieval.chunks

        generation = await _state["generator"].generate(request.query, chunks)
        total_ms = (time.perf_counter() - start) * 1000

        span.set_attribute("query_type", retrieval.analysis.query_type.value)
        span.set_attribute("cannot_answer", generation.cannot_answer)

    metrics.record_request(
        latency_seconds=total_ms / 1000,
        tokens_in=generation.tokens_input,
        tokens_out=generation.tokens_output,
        cannot_answer=generation.cannot_answer,
    )
    logger.info(
        "query_handled",
        query_id=query_id,
        query_type=retrieval.analysis.query_type.value,
        strategy_used=retrieval.strategy_used,
        total_ms=round(total_ms, 1),
        cannot_answer=generation.cannot_answer,
    )
    return QueryResponse(
        query_id=query_id,
        answer=generation.answer,
        citations=generation.citations,
        confidence=generation.confidence,
        cannot_answer=generation.cannot_answer,
        version_warning=generation.version_warning,
        tokens_input=generation.tokens_input,
        tokens_output=generation.tokens_output,
        cost_usd=generation.cost_usd,
        model_used=generation.model_used,
        strategy_used=retrieval.strategy_used,
        total_ms=round(total_ms, 1),
    )


@app.get("/health")
async def health() -> dict:
    qdrant_ok = await _state["store"].healthcheck() if "store" in _state else False
    return {"status": "ok", "qdrant": "connected" if qdrant_ok else "error", "version": "0.1.0"}


@app.get("/metrics")
async def prometheus_metrics() -> PlainTextResponse:
    return PlainTextResponse(metrics.render(), media_type="text/plain; version=0.0.4")


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Stack traces go to logs, never to clients — a leaked trace reveals
    internal paths and library versions to anyone who can trigger an error."""
    query_id = str(uuid.uuid4())
    logger.error("unhandled_exception", query_id=query_id, error=str(exc), exc_info=True)
    return JSONResponse(status_code=500, content={"error": "internal error", "query_id": query_id})
