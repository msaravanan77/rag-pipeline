"""Recursive character chunker: separator-hierarchy splitting.

Implemented for completeness of the strategy comparison (and used by the
evaluator for baseline comparisons), not in the production strategy map —
heading-aware beats it on this corpus because it uses real document structure
instead of separator heuristics.
"""

from __future__ import annotations

import hashlib

from cncf_rag.chunking.models import Chunk
from cncf_rag.chunking.strategies.base import BaseChunker
from cncf_rag.ingestion.models import Document

# Try the most structure-preserving separator first; degrade gracefully.
_SEPARATORS = ["\n\n", "\n", ". ", " "]


class RecursiveChunker(BaseChunker):
    def chunk(self, document: Document) -> list[Chunk]:
        pieces = self._split(document.content, 0)
        chunks = [
            Chunk(
                chunk_id=hashlib.sha256(f"{document.doc_id}:{i}".encode()).hexdigest()[:32],
                doc_id=document.doc_id,
                content=piece,
                token_count=self._count_tokens(piece),
                chunk_index=i,
                chunking_strategy="recursive",
                has_code_blocks="```" in piece,
            )
            for i, piece in enumerate(pieces)
        ]
        return chunks

    def _split(self, text: str, sep_index: int) -> list[str]:
        if self._count_tokens(text) <= self.max_tokens or sep_index >= len(_SEPARATORS):
            return [text] if text.strip() else []
        parts = text.split(_SEPARATORS[sep_index])
        results: list[str] = []
        current = ""
        for part in parts:
            candidate = current + _SEPARATORS[sep_index] + part if current else part
            if self._count_tokens(candidate) <= self.max_tokens:
                current = candidate
            else:
                if current.strip():
                    results.append(current)
                # Part alone may still exceed budget — recurse with next separator.
                if self._count_tokens(part) > self.max_tokens:
                    results.extend(self._split(part, sep_index + 1))
                    current = ""
                else:
                    current = part
        if current.strip():
            results.append(current)
        return results
