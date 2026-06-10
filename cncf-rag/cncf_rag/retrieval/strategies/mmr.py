"""Maximal Marginal Relevance, implemented from scratch — no framework black box.

The MMR objective, selected greedily one document at a time:

    MMR = argmax_{d in candidates \\ selected} [
        lambda * sim(d, query)  -  (1 - lambda) * max_{s in selected} sim(d, s)
    ]

lambda controls the relevance/diversity tradeoff:
  lambda = 1.0 → pure similarity ranking (identical to top-k)
  lambda = 0.0 → pure diversity (ignores the query entirely)
  lambda = 0.6 (ours) → mostly relevance, with enough diversity pressure that
  near-duplicate chunks (common in CNCF docs: the same concept re-explained in
  three pages) don't crowd out a second subtopic the explorer wants to see.
"""

from __future__ import annotations

import numpy as np

from cncf_rag.embedding.service import EmbeddingService
from cncf_rag.vectorstore.qdrant_store import QdrantVectorStore, ScoredChunk


class MMRRetriever:
    def __init__(
        self,
        embedder: EmbeddingService,
        store: QdrantVectorStore,
        lambda_param: float = 0.6,
        fetch_k: int = 30,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self.lambda_param = lambda_param
        self.fetch_k = fetch_k  # over-fetch pool size: MMR needs candidates to choose among

    async def retrieve(
        self, query: str, top_k: int = 8, filters: dict | None = None
    ) -> list[ScoredChunk]:
        query_vector = await self._embedder.embed_query(query)
        # Step 1: over-fetch fetch_k candidates by plain similarity. MMR can only
        # diversify within the pool it is given — a pool of top_k would leave
        # nothing to choose between.
        candidates = await self._store.search(query_vector, top_k=self.fetch_k, filters=filters)
        if len(candidates) <= top_k:
            return candidates

        # Step 2: re-embed candidate contents to get vectors for pairwise
        # similarity (Qdrant search results don't return stored vectors by default).
        cand_vectors = np.array(await self._embedder.embed_documents([c.content for c in candidates]))
        cand_vectors = cand_vectors / np.linalg.norm(cand_vectors, axis=1, keepdims=True)
        q = np.array(query_vector)
        q = q / np.linalg.norm(q)
        query_sims = cand_vectors @ q  # sim(d, query) for every candidate

        # Step 3: greedy selection loop — the heart of MMR.
        selected: list[int] = [int(np.argmax(query_sims))]  # seed with the single most relevant
        while len(selected) < top_k:
            best_idx, best_score = -1, -np.inf
            for i in range(len(candidates)):
                if i in selected:
                    continue
                # Redundancy penalty: similarity to the CLOSEST already-selected
                # doc (max, not mean — one near-duplicate is enough to penalize).
                redundancy = float(np.max(cand_vectors[selected] @ cand_vectors[i]))
                score = self.lambda_param * float(query_sims[i]) - (1 - self.lambda_param) * redundancy
                if score > best_score:
                    best_idx, best_score = i, score
            selected.append(best_idx)

        return [candidates[i] for i in selected]
