"""Chunker contract shared by every strategy.

The contract (enforced, not aspirational):
1. No chunk may split a fenced code block — a half manifest is corrupt data.
2. chunk_index is contiguous from 0 within a document.
3. No chunk exceeds max_tokens (after heading-path prefixing).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import tiktoken

from cncf_rag.chunking.models import Chunk
from cncf_rag.ingestion.models import Document

# cl100k_base is shared across modern OpenAI/Anthropic-adjacent tokenizers and
# is a good neutral token estimator for Cohere too — exact parity is not needed,
# only consistent budgeting.
_ENCODING = tiktoken.get_encoding("cl100k_base")


class BaseChunker(ABC):
    def __init__(
        self, max_tokens: int = 512, overlap_tokens: int = 0, validate_code_blocks: bool = True
    ) -> None:
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        # Disabled only for the factory's last-resort fallback on documents
        # whose fence syntax is itself unbalanced (4-backtick nesting etc.) —
        # the odd-count check would be meaningless there.
        self.validate_code_blocks = validate_code_blocks

    @abstractmethod
    def chunk(self, document: Document) -> list[Chunk]:
        """Split a document into chunks honoring the three-part contract above."""

    @staticmethod
    def _count_tokens(text: str) -> int:
        return len(_ENCODING.encode(text))

    @staticmethod
    def _encode(text: str) -> list[int]:
        return _ENCODING.encode(text)

    @staticmethod
    def _decode(tokens: list[int]) -> str:
        return _ENCODING.decode(tokens)

    def _assert_no_split_code_blocks(self, chunks: list[Chunk]) -> None:
        """Raise if any chunk contains an odd number of code fences.

        An odd fence count means a chunk boundary landed inside a fenced block —
        the one corruption no downstream component can repair.
        """
        if not self.validate_code_blocks:
            return
        for chunk in chunks:
            if chunk.content.count("```") % 2 != 0:
                raise ValueError(
                    f"Chunk {chunk.chunk_id} (doc {chunk.doc_id}, index {chunk.chunk_index}) "
                    "splits a fenced code block — chunking contract violated"
                )
