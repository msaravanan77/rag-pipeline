"""Plain filtered top-k retrieval — the baseline every other strategy is judged against."""

from __future__ import annotations

from cncf_rag.embedding.service import EmbeddingService
from cncf_rag.vectorstore.qdrant_store import QdrantVectorStore, ScoredChunk


class NaiveTopKRetriever:
    def __init__(self, embedder: EmbeddingService, store: QdrantVectorStore) -> None:
        self._embedder = embedder
        self._store = store

    async def retrieve(
        self, query: str, top_k: int = 5, filters: dict | None = None
    ) -> list[ScoredChunk]:
        query_vector = await self._embedder.embed_query(query)
        return await self._store.search(query_vector, top_k=top_k, filters=filters)
