"""HyDE — Hypothetical Document Embeddings.

Generate a hypothetical answer with an LLM, embed THAT, and search with it.
"""

from __future__ import annotations

import os

import structlog
from anthropic import AsyncAnthropic

from cncf_rag.embedding.service import EmbeddingService
from cncf_rag.vectorstore.qdrant_store import QdrantVectorStore, ScoredChunk

logger = structlog.get_logger(__name__)

_HYDE_PROMPT = """Write a short technical documentation paragraph (3-5 sentences) that would \
answer this question about the CNCF ecosystem. Write it in the style of official Kubernetes/Helm \
documentation. Do not say you are unsure — write a plausible documentation passage.

Question: {query}"""


class HyDERetriever:
    def __init__(
        self,
        embedder: EmbeddingService,
        store: QdrantVectorStore,
        anthropic_client: AsyncAnthropic | None = None,
        # haiku, not sonnet: HyDE needs plausible-sounding doc prose, not
        # correctness — the fast cheap model is the right tool.
        model: str = "claude-haiku-4-5-20251001",
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._client = anthropic_client or AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.model = model

    async def retrieve(
        self, query: str, top_k: int = 5, filters: dict | None = None
    ) -> list[ScoredChunk]:
        response = await self._client.messages.create(
            model=self.model,
            max_tokens=300,
            messages=[{"role": "user", "content": _HYDE_PROMPT.format(query=query)}],
        )
        hypothetical = response.content[0].text

        # Why embedding a hypothetical ANSWER helps: queries and documents live
        # in different linguistic registers — "how do I stop pods restarting?"
        # shares few terms with "the restartPolicy field controls...". A fake
        # answer written in documentation style lands in the same embedding
        # neighborhood as the real documentation, bridging that register gap.
        #
        # When HyDE FAILS: when the LLM doesn't know the domain and hallucinates
        # terminology that doesn't exist in the corpus — the embedding then drifts
        # TOWARD the hallucination and away from the real docs. Also fails on
        # version-specific queries: the fake answer will confidently describe the
        # wrong version's API. This is why HyDE is not in the default routing map.
        hyde_vector = await self._embedder.embed_query(hypothetical)
        logger.info("hyde_generated", query=query, hypothetical_preview=hypothetical[:120])
        return await self._store.search(hyde_vector, top_k=top_k, filters=filters)
