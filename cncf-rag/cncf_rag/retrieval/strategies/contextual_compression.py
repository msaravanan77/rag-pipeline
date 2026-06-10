"""Contextual compression: trim retrieved chunks to query-relevant sentences.

Available as an opt-in post-processor (not in default routing): it spends an
LLM call per query to save generation tokens — worthwhile only when retrieved
chunks are large and the generation model is expensive relative to haiku.
"""

from __future__ import annotations

import os

import structlog
from anthropic import AsyncAnthropic

from cncf_rag.vectorstore.qdrant_store import ScoredChunk

logger = structlog.get_logger(__name__)

_COMPRESS_PROMPT = """Extract only the sentences from this documentation excerpt that are \
relevant to answering the question. Reproduce them EXACTLY — do not paraphrase, do not \
summarize. Keep code blocks intact if relevant. If nothing is relevant, respond with NONE.

Question: {query}

Excerpt:
{content}"""


class ContextualCompressor:
    def __init__(
        self,
        anthropic_client: AsyncAnthropic | None = None,
        model: str = "claude-haiku-4-5-20251001",
    ) -> None:
        self._client = anthropic_client or AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.model = model

    async def compress(self, query: str, chunks: list[ScoredChunk]) -> list[ScoredChunk]:
        compressed: list[ScoredChunk] = []
        for chunk in chunks:
            response = await self._client.messages.create(
                model=self.model,
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": _COMPRESS_PROMPT.format(query=query, content=chunk.content),
                    }
                ],
            )
            text = response.content[0].text.strip()
            if text == "NONE":
                continue  # chunk contributed nothing — drop it entirely
            compressed.append(
                ScoredChunk(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    content=text,
                    score=chunk.score,
                    payload=chunk.payload,
                )
            )
        logger.info("compression_done", before=len(chunks), after=len(compressed))
        return compressed
