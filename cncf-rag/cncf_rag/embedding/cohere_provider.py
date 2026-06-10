"""Cohere embed-english-v3.0 provider (DECISIONS.md 3.1).

Chosen over OpenAI because the project's Cohere trial key makes embeddings
free to start, with a HIGHER MTEB retrieval score (64.5 vs 62.3) than
text-embedding-3-small.
"""

from __future__ import annotations

import asyncio
import os

import cohere
from cohere.core.api_error import ApiError as CohereApiError
import structlog
import tiktoken
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from cncf_rag.embedding.cost_tracker import EmbeddingCostTracker
from cncf_rag.embedding.service import EmbeddingService

logger = structlog.get_logger(__name__)

_MAX_INPUT_TOKENS = 512  # embed-english-v3.0 context limit


class CohereEmbeddingService(EmbeddingService):
    # batch_size=96: the Cohere API hard maximum per request. Larger batches
    # would error; smaller ones waste the trial key's 100 req/min quota —
    # at 96/request, full corpus ingestion fits in ~25 minutes of paced calls.
    def __init__(
        self,
        model: str = "embed-english-v3.0",
        batch_size: int = 96,
        cost_tracker: EmbeddingCostTracker | None = None,
        requests_per_minute: int = 100,
    ) -> None:
        self._client = cohere.AsyncClientV2(api_key=os.environ["COHERE_API_KEY"])
        self.model = model
        self.batch_size = batch_size
        self._tracker = cost_tracker
        # cl100k_base is an estimator for Cohere's tokenizer — close enough for
        # truncation warnings; exact counts come back in API responses.
        self._encoding = tiktoken.get_encoding("cl100k_base")
        self._min_request_interval = 60.0 / requests_per_minute
        self._last_request_at = 0.0
        # Trial keys also cap TOKENS at 100k/minute — a single 96-text batch of
        # 364-token chunks is ~35k tokens, so two fast batches can trip it even
        # under the request-rate cap. Track a rolling 60s token window.
        self._token_budget_per_minute = 90_000  # 10% headroom under the 100k cap
        self._token_window: list[tuple[float, int]] = []

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = [self._truncate_if_needed(t) for t in texts[start : start + self.batch_size]]
            response = await self._embed_batch(batch, input_type="search_document")
            vectors.extend(response)
            if self._tracker is not None:
                self._tracker.record_batch(self.model, sum(self.count_tokens(t) for t in batch))
        return vectors

    async def embed_query(self, text: str) -> list[float]:
        # input_type="search_query" is the critical asymmetry — see service.py.
        result = await self._embed_batch([self._truncate_if_needed(text)], input_type="search_query")
        return result[0]

    def count_tokens(self, text: str) -> int:
        return len(self._encoding.encode(text))

    def _truncate_if_needed(self, text: str) -> str:
        tokens = self._encoding.encode(text)
        if len(tokens) <= _MAX_INPUT_TOKENS:
            return text
        # Explicit truncation with a warning beats silent server-side truncation:
        # the log tells us chunking produced an oversized chunk (a chunking bug).
        logger.warning("embedding_input_truncated", tokens=len(tokens), limit=_MAX_INPUT_TOKENS)
        return self._encoding.decode(tokens[:_MAX_INPUT_TOKENS])

    @retry(
        # Retry only HTTP 429 (rate limit) — other API errors (bad key, bad
        # request) will never succeed on retry and should surface immediately.
        retry=retry_if_exception(
            lambda e: isinstance(e, CohereApiError) and e.status_code == 429
        ),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    async def _embed_batch(self, texts: list[str], input_type: str) -> list[list[float]]:
        await self._pace(sum(self.count_tokens(t) for t in texts))
        response = await self._client.embed(
            model=self.model,
            texts=texts,
            input_type=input_type,
            embedding_types=["float"],
        )
        return [list(vec) for vec in response.embeddings.float_]

    async def _pace(self, batch_tokens: int) -> None:
        """Sleep to stay under BOTH trial caps: requests/minute and tokens/minute.

        Proactive pacing beats reactive retries: a 429 from Cohere can penalize
        subsequent requests, so never triggering one is faster overall.
        """
        loop = asyncio.get_event_loop()
        # Request-rate pacing.
        wait = self._min_request_interval - (loop.time() - self._last_request_at)
        if wait > 0:
            await asyncio.sleep(wait)
        # Token-rate pacing: if this batch would push the rolling 60s window
        # over budget, sleep until the oldest entry ages out.
        while True:
            now = loop.time()
            self._token_window = [(t, n) for t, n in self._token_window if now - t < 60]
            used = sum(n for _, n in self._token_window)
            if used + batch_tokens <= self._token_budget_per_minute or not self._token_window:
                break
            oldest = self._token_window[0][0]
            await asyncio.sleep(max(0.5, 60 - (now - oldest)))
        self._token_window.append((loop.time(), batch_tokens))
        self._last_request_at = loop.time()
