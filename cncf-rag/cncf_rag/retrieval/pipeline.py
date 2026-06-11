"""Retrieval orchestration: analyzer → filters → strategy router → optional rerank.

The routing table is DECISIONS.md 5.2 made executable.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import structlog

from cncf_rag.embedding.service import EmbeddingService
from cncf_rag.retrieval.filters import FilterBuilder
from cncf_rag.retrieval.query_analyzer import QueryAnalysis, QueryAnalyzer, QueryType
from cncf_rag.retrieval.reranker import CohereReranker
from cncf_rag.retrieval.strategies.mmr import MMRRetriever
from cncf_rag.retrieval.strategies.multi_query import MultiQueryRetriever
from cncf_rag.retrieval.strategies.naive_topk import NaiveTopKRetriever
from cncf_rag.vectorstore.qdrant_store import QdrantVectorStore, ScoredChunk

logger = structlog.get_logger(__name__)


@dataclass
class RetrievalResult:
    query: str
    analysis: QueryAnalysis
    strategy_used: str
    chunks: list[ScoredChunk] = field(default_factory=list)
    reranked: bool = False
    latency_ms: float = 0.0


class RetrievalPipeline:
    def __init__(
        self,
        embedder: EmbeddingService,
        store: QdrantVectorStore,
        reranker: CohereReranker | None = None,
    ) -> None:
        self._analyzer = QueryAnalyzer()
        self._filter_builder = FilterBuilder()
        self._topk = NaiveTopKRetriever(embedder, store)
        self._mmr = MMRRetriever(embedder, store, lambda_param=0.6)
        self._multi_query = MultiQueryRetriever(embedder, store)
        self._reranker = reranker

    async def retrieve(
        self,
        query: str,
        strategy_override: str | None = None,
        filter_override: dict | None = None,
    ) -> RetrievalResult:
        start = time.perf_counter()
        analysis = self._analyzer.analyze(query)
        filters = self._filter_builder.build_from_analysis(analysis)
        # User-supplied explicit filters (project_filter / version_filter from the API
        # request) take precedence over anything the analyzer inferred from the query text.
        if filter_override:
            filters.update(filter_override)

        strategy = strategy_override or self._route(analysis.query_type)
        match strategy:
            case "mmr":
                chunks = await self._mmr.retrieve(query, top_k=8, filters=filters)
            case "multi_query":
                chunks = await self._multi_query.retrieve(query, top_k_per_project=3, filters=filters)
            case _:
                chunks = await self._topk.retrieve(query, top_k=5, filters=filters)

        # Selective reranking (5.3): only the two noisy-ranking query types.
        reranked = False
        if self._reranker is not None and analysis.query_type in (
            QueryType.EXPLORATORY,
            QueryType.CROSS_PROJECT,
        ):
            chunks = await self._reranker.rerank(query, chunks)
            reranked = True

        result = RetrievalResult(
            query=query,
            analysis=analysis,
            strategy_used=strategy,
            chunks=chunks,
            reranked=reranked,
            latency_ms=(time.perf_counter() - start) * 1000,
        )
        logger.info(
            "retrieval_complete",
            query=query,
            query_type=analysis.query_type.value,
            strategy=strategy,
            filters=filters,
            chunk_count=len(chunks),
            reranked=reranked,
            top_score=chunks[0].score if chunks else None,
            latency_ms=round(result.latency_ms, 1),
        )
        return result

    @staticmethod
    def _route(query_type: QueryType) -> str:
        match query_type:
            case QueryType.EXPLORATORY:
                return "mmr"
            case QueryType.CROSS_PROJECT:
                return "multi_query"
            case _:
                return "filtered_topk"
