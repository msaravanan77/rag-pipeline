"""Cohere cross-encoder reranking (DECISIONS.md 5.3).

Applied ONLY to EXPLORATORY and CROSS_PROJECT queries: those are where initial
ranking is noisy enough that a cross-encoder (which reads the full
query+chunk pair, unlike the bi-encoder used for retrieval) changes outcomes.
"""

from __future__ import annotations

import os

import cohere
import structlog

from cncf_rag.vectorstore.qdrant_store import ScoredChunk

logger = structlog.get_logger(__name__)


class CohereReranker:
    def __init__(self, model: str = "rerank-english-v3.0", top_n: int = 5) -> None:
        self._client = cohere.AsyncClientV2(api_key=os.environ["COHERE_API_KEY"])
        self.model = model
        self.top_n = top_n

    async def rerank(self, query: str, chunks: list[ScoredChunk]) -> list[ScoredChunk]:
        if not chunks:
            return chunks
        response = await self._client.rerank(
            model=self.model,
            query=query,
            documents=[c.content for c in chunks],
            top_n=min(self.top_n, len(chunks)),
        )
        reranked = []
        for result in response.results:
            chunk = chunks[result.index]
            chunk.score = result.relevance_score  # replace cosine score with cross-encoder score
            reranked.append(chunk)
        logger.info("reranked", input_count=len(chunks), output_count=len(reranked))
        return reranked
