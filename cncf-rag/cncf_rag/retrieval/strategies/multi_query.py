"""Multi-query retrieval with Reciprocal Rank Fusion — for CROSS_PROJECT queries.

"How do Helm and Argo CD handle rollbacks?" needs guaranteed coverage from
BOTH projects; a single embedding inevitably favors whichever project the
phrasing leans toward.
"""

from __future__ import annotations

import asyncio
import json
import os

import structlog
from anthropic import AsyncAnthropic

from cncf_rag.embedding.service import EmbeddingService
from cncf_rag.vectorstore.qdrant_store import QdrantVectorStore, ScoredChunk

logger = structlog.get_logger(__name__)

_DECOMPOSE_PROMPT = """Decompose this question about CNCF tools into one focused sub-question \
per project it mentions. Return a JSON array of objects: [{{"project": "...", "question": "..."}}].
Projects must be one of: kubernetes, helm, argocd, prometheus.
Return ONLY the JSON array.

Question: {query}"""


class MultiQueryRetriever:
    def __init__(
        self,
        embedder: EmbeddingService,
        store: QdrantVectorStore,
        anthropic_client: AsyncAnthropic | None = None,
        model: str = "claude-haiku-4-5-20251001",
        rrf_k: int = 60,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._client = anthropic_client or AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.model = model
        self.rrf_k = rrf_k

    async def retrieve(
        self, query: str, top_k_per_project: int = 3, filters: dict | None = None
    ) -> list[ScoredChunk]:
        sub_queries = await self._decompose(query)
        if not sub_queries:
            sub_queries = [{"project": None, "question": query}]

        # Run all sub-retrievals concurrently — they are independent.
        result_lists = await asyncio.gather(
            *(
                self._retrieve_one(sq["question"], sq.get("project"), top_k_per_project, filters)
                for sq in sub_queries
            )
        )
        return self._rrf_merge(result_lists)

    async def _decompose(self, query: str) -> list[dict]:
        response = await self._client.messages.create(
            model=self.model,
            max_tokens=500,
            messages=[{"role": "user", "content": _DECOMPOSE_PROMPT.format(query=query)}],
        )
        try:
            parsed = json.loads(response.content[0].text.strip())
            return parsed if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            logger.warning("decompose_parse_failed", raw=response.content[0].text[:200])
            return []

    async def _retrieve_one(
        self, question: str, project: str | None, top_k: int, base_filters: dict | None
    ) -> list[ScoredChunk]:
        filters = dict(base_filters or {})
        if project:
            filters["project"] = project
        vector = await self._embedder.embed_query(question)
        return await self._store.search(vector, top_k=top_k, filters=filters)

    def _rrf_merge(self, result_lists: list[list[ScoredChunk]]) -> list[ScoredChunk]:
        """Reciprocal Rank Fusion:  score(d) = Σ_lists 1 / (k + rank_d).

        RRF merges ranked lists using only positions, never raw scores — vital
        here because each sub-query's cosine scores live on slightly different
        scales (different query embeddings), making raw-score merging biased.
        k=60 is the standard from the original Cormack et al. paper: large enough
        that rank 1 vs rank 2 (1/61 vs 1/62) doesn't dominate, small enough that
        order still matters. It has proven robust across domains and is rarely tuned.
        """
        scores: dict[str, float] = {}
        chunk_by_id: dict[str, ScoredChunk] = {}
        for results in result_lists:
            for rank, chunk in enumerate(results, start=1):
                scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (self.rrf_k + rank)
                chunk_by_id.setdefault(chunk.chunk_id, chunk)
        merged = sorted(chunk_by_id.values(), key=lambda c: scores[c.chunk_id], reverse=True)
        for chunk in merged:
            chunk.score = scores[chunk.chunk_id]  # expose fused score downstream
        return merged
