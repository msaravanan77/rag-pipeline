"""OpenAI text-embedding-3-small provider.

Switched from Cohere for ingestion speed: paid tier allows 1M tokens/min
vs Cohere trial's 100k/min — full-corpus embedding drops from ~40 min to ~6 min.
Cohere reranking is kept unchanged; per-query batch sizes never approach trial limits.
"""

from __future__ import annotations

import os

import structlog
import tiktoken
from openai import AsyncOpenAI, RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from cncf_rag.embedding.cost_tracker import EmbeddingCostTracker
from cncf_rag.embedding.service import EmbeddingService

logger = structlog.get_logger(__name__)

_MAX_INPUT_TOKENS = 8191  # text-embedding-3-small context window


class OpenAIEmbeddingService(EmbeddingService):
    # batch_size=512: well under OpenAI's 2048-text limit per request, keeps
    # request bodies manageable while still batching aggressively.
    def __init__(
        self,
        model: str = "text-embedding-3-small",
        batch_size: int = 512,
        cost_tracker: EmbeddingCostTracker | None = None,
    ) -> None:
        self._client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model = model
        self.batch_size = batch_size
        self._tracker = cost_tracker
        self._encoding = tiktoken.get_encoding("cl100k_base")

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = [self._truncate_if_needed(t) for t in texts[start : start + self.batch_size]]
            batch_vectors = await self._embed_batch(batch)
            vectors.extend(batch_vectors)
            if self._tracker is not None:
                self._tracker.record_batch(self.model, sum(self.count_tokens(t) for t in batch))
        return vectors

    async def embed_query(self, text: str) -> list[float]:
        result = await self._embed_batch([self._truncate_if_needed(text)])
        return result[0]

    def count_tokens(self, text: str) -> int:
        return len(self._encoding.encode(text))

    def _truncate_if_needed(self, text: str) -> str:
        tokens = self._encoding.encode(text)
        if len(tokens) <= _MAX_INPUT_TOKENS:
            return text
        logger.warning("embedding_input_truncated", tokens=len(tokens), limit=_MAX_INPUT_TOKENS)
        return self._encoding.decode(tokens[:_MAX_INPUT_TOKENS])

    @retry(
        retry=retry_if_exception_type(RateLimitError),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        response = await self._client.embeddings.create(
            model=self.model,
            input=texts,
            encoding_format="float",
        )
        # OpenAI guarantees response order matches input order.
        return [item.embedding for item in response.data]
