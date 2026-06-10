"""Stage 3 tests: batching, truncation, input_type asymmetry — mocked Cohere client."""

from __future__ import annotations

import pytest

from cncf_rag.embedding.cost_tracker import EmbeddingCostTracker


@pytest.fixture
def service(monkeypatch):
    monkeypatch.setenv("COHERE_API_KEY", "test-key")
    from cncf_rag.embedding.cohere_provider import CohereEmbeddingService

    svc = CohereEmbeddingService(cost_tracker=EmbeddingCostTracker(), requests_per_minute=100000)

    calls: list[dict] = []

    class _Embeddings:
        float_ = [[0.1] * 1024]

    class _Response:
        embeddings = _Embeddings()

    async def fake_embed(*, model, texts, input_type, embedding_types):
        calls.append({"texts": texts, "input_type": input_type})
        resp = _Response()
        resp.embeddings = _Embeddings()
        resp.embeddings.float_ = [[0.1] * 1024 for _ in texts]
        return resp

    svc._client.embed = fake_embed
    svc._calls = calls
    return svc


class TestBatching:
    async def test_batches_of_96_enforced(self, service):
        texts = [f"text {i}" for i in range(200)]
        vectors = await service.embed_documents(texts)
        assert len(vectors) == 200
        batch_sizes = [len(c["texts"]) for c in service._calls]
        assert batch_sizes == [96, 96, 8]

    async def test_order_preserved(self, service):
        texts = [f"text {i}" for i in range(100)]
        vectors = await service.embed_documents(texts)
        assert len(vectors) == len(texts)


class TestInputTypeAsymmetry:
    async def test_documents_use_search_document(self, service):
        await service.embed_documents(["a"])
        assert service._calls[0]["input_type"] == "search_document"

    async def test_query_uses_search_query(self, service):
        await service.embed_query("a question")
        assert service._calls[0]["input_type"] == "search_query"


class TestTruncation:
    async def test_oversized_input_truncated_not_rejected(self, service):
        huge = "word " * 5000  # far over the 512-token limit
        await service.embed_documents([huge])
        sent = service._calls[0]["texts"][0]
        assert service.count_tokens(sent) <= 512


class TestCostTracker:
    def test_cost_computation(self):
        tracker = EmbeddingCostTracker()
        tracker.record_batch("embed-english-v3.0", 1_000_000)
        assert abs(tracker.total_cost_usd() - 0.10) < 1e-9
        assert tracker.total_batches == 1
