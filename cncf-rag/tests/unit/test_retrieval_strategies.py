"""Stage 5 tests: query routing rules, filter building, MMR diversity, RRF merge."""

from __future__ import annotations

import numpy as np
import pytest

from cncf_rag.retrieval.filters import FilterBuilder
from cncf_rag.retrieval.query_analyzer import QueryAnalyzer, QueryType
from cncf_rag.retrieval.strategies.mmr import MMRRetriever
from cncf_rag.retrieval.strategies.multi_query import MultiQueryRetriever
from cncf_rag.vectorstore.qdrant_store import ScoredChunk


def chunk(cid: str, content: str = "x", score: float = 0.9) -> ScoredChunk:
    return ScoredChunk(chunk_id=cid, doc_id="d", content=content, score=score, payload={})


class TestQueryAnalyzer:
    @pytest.fixture
    def analyzer(self):
        return QueryAnalyzer()

    def test_version_specific(self, analyzer):
        analysis = analyzer.analyze("How does Ingress work in v1.22?")
        assert analysis.query_type == QueryType.VERSION_SPECIFIC
        assert analysis.detected_version == "v1.22"

    def test_cross_project(self, analyzer):
        analysis = analyzer.analyze("How do Helm and Argo CD handle rollbacks?")
        assert analysis.query_type == QueryType.CROSS_PROJECT
        assert set(analysis.detected_projects) >= {"helm", "argocd"}

    def test_procedural(self, analyzer):
        assert analyzer.analyze("How do I configure liveness probes?").query_type == QueryType.PROCEDURAL

    def test_conceptual(self, analyzer):
        assert analyzer.analyze("What is a Kubernetes Service?").query_type == QueryType.CONCEPTUAL

    def test_exploratory(self, analyzer):
        assert analyzer.analyze("service mesh").query_type == QueryType.EXPLORATORY

    def test_factual_default(self, analyzer):
        assert (
            analyzer.analyze("kubectl command that shows pod resource usage in a namespace").query_type
            == QueryType.FACTUAL
        )


class TestFilterBuilder:
    @pytest.fixture
    def build(self):
        analyzer = QueryAnalyzer()
        builder = FilterBuilder()
        return lambda q: builder.build_from_analysis(analyzer.analyze(q))

    def test_procedural_forces_task(self, build):
        assert build("How do I install Prometheus?")["doc_type"] == "task"

    def test_conceptual_forces_concept(self, build):
        assert build("What is a kubernetes pod?")["doc_type"] == "concept"

    def test_version_filter(self, build):
        assert build("Ingress API in kubernetes v1.22")["version_tag"] == "v1.22"

    def test_single_project_filter(self, build):
        assert build("How do I install Prometheus?")["project"] == "prometheus"

    def test_factual_no_doc_type(self, build):
        assert "doc_type" not in build("the kubectl command that shows resource usage for nodes")


class _FakeEmbedder:
    """Deterministic embedder: vector encodes which 'topic' a text belongs to."""

    async def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Geometry chosen so topicA and topicB are EQUALLY query-relevant
        # (both 0.9 cosine to the query) but mutually dissimilar — the only
        # way topicB gets selected is through MMR's redundancy penalty.
        out = []
        for t in texts:
            if "topicB" in t:
                out.append([0.9, -0.436, 0.0])
            else:  # topicA texts: identical to each other
                out.append([0.9, 0.436, 0.0])
        return out

    def count_tokens(self, text: str) -> int:
        return len(text.split())


class _FakeStore:
    def __init__(self, chunks):
        self._chunks = chunks

    async def search(self, query_vector, top_k=5, filters=None, ef=50):
        return self._chunks[:top_k]


class TestMMR:
    async def test_mmr_more_diverse_than_topk(self):
        # 5 near-duplicate topicA chunks + 1 topicB chunk ranked last by similarity.
        chunks = [chunk(f"a{i}", f"topicA text {i}", 0.95) for i in range(5)]
        chunks.append(chunk("b1", "topicB text", 0.70))
        store = _FakeStore(chunks)
        mmr = MMRRetriever(_FakeEmbedder(), store, lambda_param=0.5, fetch_k=6)
        selected = await mmr.retrieve("query", top_k=3)
        # Pure top-k would pick a0,a1,a2. MMR's diversity term must pull in topicB.
        assert any(c.chunk_id == "b1" for c in selected)


class TestRRF:
    def test_rrf_merge_correctness(self):
        retriever = MultiQueryRetriever.__new__(MultiQueryRetriever)  # skip __init__ (no API client)
        retriever.rrf_k = 60
        list1 = [chunk("shared"), chunk("only1")]
        list2 = [chunk("shared"), chunk("only2")]
        merged = retriever._rrf_merge([list1, list2])
        # "shared" appears rank 1 in both lists → 2/(60+1) — must rank first.
        assert merged[0].chunk_id == "shared"
        assert abs(merged[0].score - 2 / 61) < 1e-9
        assert {c.chunk_id for c in merged} == {"shared", "only1", "only2"}

    def test_rrf_position_beats_single_list_presence(self):
        retriever = MultiQueryRetriever.__new__(MultiQueryRetriever)
        retriever.rrf_k = 60
        list1 = [chunk("x"), chunk("y")]
        list2 = [chunk("y")]
        merged = retriever._rrf_merge([list1, list2])
        # y: 1/62 + 1/61 > x: 1/61 → y wins despite x topping list1.
        assert merged[0].chunk_id == "y"
