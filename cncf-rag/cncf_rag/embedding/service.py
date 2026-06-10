"""Embedding service contract.

Why embed_documents and embed_query are SEPARATE methods (DECISIONS.md 3.1):
Cohere v3 embeddings require input_type="search_document" at index time and
input_type="search_query" at query time — mixing them measurably degrades
retrieval. Two methods put that asymmetry inside the provider; call sites
never need to know it exists. The same split keeps the door open for OpenAI
(where both map to one call) without touching callers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class EmbeddingService(ABC):
    @abstractmethod
    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed corpus texts for INDEXING.

        Contract: returns one vector per input, in input order, always the full
        list (no partial results — a failed batch raises after retries).
        """

    @abstractmethod
    async def embed_query(self, text: str) -> list[float]:
        """Embed a user query for SEARCH. Single text — queries never batch."""

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Token count for budgeting. Exists on the service (not a util module)
        because the correct tokenizer is a property of the chosen model."""
