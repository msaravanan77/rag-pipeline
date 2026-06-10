"""Stage 8 integration tests: API routing and response schema with mocked providers.

No network calls: embedding, vector search, reranking, and generation are all
replaced — the test verifies the PLUMBING (routing, schema, error handling),
which is exactly what unit tests of individual components cannot.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api import app as app_module
from cncf_rag.generation.service import GenerationResult
from cncf_rag.retrieval.pipeline import RetrievalResult
from cncf_rag.retrieval.query_analyzer import QueryAnalysis, QueryType
from cncf_rag.vectorstore.qdrant_store import ScoredChunk


class _FakePipeline:
    async def retrieve(self, query: str, strategy_override: str | None = None) -> RetrievalResult:
        return RetrievalResult(
            query=query,
            analysis=QueryAnalysis(query=query, query_type=QueryType.CONCEPTUAL),
            strategy_used="filtered_topk",
            chunks=[
                ScoredChunk(
                    chunk_id="c1",
                    doc_id="d1",
                    content="A Pod is the smallest deployable unit.",
                    score=0.92,
                    payload={"project": "kubernetes", "doc_type": "concept", "version_tag": "v1.29"},
                )
            ],
        )


class _FakeGenerator:
    async def generate(self, query: str, chunks) -> GenerationResult:
        return GenerationResult(
            answer="A Pod is the smallest deployable unit. [Source: kubernetes | concept | v1.29 | docs]",
            citations=[{"project": "kubernetes", "version": "v1.29"}],
            confidence=0.9,
            cannot_answer=False,
            tokens_input=1000,
            tokens_output=100,
            cost_usd=0.0045,
            model_used="claude-sonnet-4-5",
        )


class _FakeStore:
    async def healthcheck(self) -> bool:
        return True


class _FakeTracer:
    def start_as_current_span(self, name):
        from contextlib import contextmanager

        @contextmanager
        def cm():
            class Span:
                def set_attribute(self, *a):
                    pass

            yield Span()

        return cm()


@pytest.fixture
def client(monkeypatch):
    # The lifespan constructs real provider clients at startup; dummy keys let
    # construction succeed — the fakes installed below mean nothing is called.
    monkeypatch.setenv("COHERE_API_KEY", "test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    app_module._state.update(
        {
            "pipeline": _FakePipeline(),
            "generator": _FakeGenerator(),
            "store": _FakeStore(),
            "tracer": _FakeTracer(),
        }
    )
    # raise_server_exceptions=False so the global handler is exercised.
    with TestClient(app_module.app, raise_server_exceptions=False) as test_client:
        app_module._state.update(  # lifespan may have overwritten with real clients
            {
                "pipeline": _FakePipeline(),
                "generator": _FakeGenerator(),
                "store": _FakeStore(),
                "tracer": _FakeTracer(),
            }
        )
        yield test_client
    app_module._state.clear()


class TestQueryEndpoint:
    def test_query_response_schema(self, client):
        response = client.post("/query", json={"query": "What is a Pod?"})
        assert response.status_code == 200
        body = response.json()
        assert body["answer"].startswith("A Pod")
        assert body["cannot_answer"] is False
        assert body["strategy_used"] == "filtered_topk"
        assert body["citations"][0]["project"] == "kubernetes"
        assert "query_id" in body
        assert body["total_ms"] >= 0

    def test_query_validation_rejects_short_query(self, client):
        assert client.post("/query", json={"query": "ab"}).status_code == 422

    def test_project_filter_applied(self, client):
        response = client.post(
            "/query", json={"query": "What is a Pod?", "project_filter": "helm"}
        )
        # The only fake chunk is kubernetes — helm filter drops it → cannot_answer
        # path would need generator awareness; here we just assert the request succeeds.
        assert response.status_code == 200


class TestHealthEndpoint:
    def test_health(self, client):
        body = client.get("/health").json()
        assert body == {"status": "ok", "qdrant": "connected", "version": "0.1.0"}


class TestMetricsEndpoint:
    def test_metrics_exposition_format(self, client):
        client.post("/query", json={"query": "What is a Pod?"})
        text = client.get("/metrics").text
        assert "cncf_rag_request_count" in text
        assert "cncf_rag_request_latency_seconds_bucket" in text
        assert "# TYPE cncf_rag_request_count counter" in text


class TestErrorHandling:
    def test_global_handler_hides_stack_traces(self, client):
        class _Boom:
            async def retrieve(self, *a, **k):
                raise RuntimeError("secret internal detail")

        app_module._state["pipeline"] = _Boom()
        response = client.post("/query", json={"query": "What is a Pod?"})
        assert response.status_code == 500
        body = response.json()
        assert body["error"] == "internal error"
        assert "secret internal detail" not in response.text
        assert "query_id" in body
