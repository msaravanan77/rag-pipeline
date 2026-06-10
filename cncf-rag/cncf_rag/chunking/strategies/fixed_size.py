"""Fixed-size chunker: paragraph-first splitting with token-overlap.

Used directly only for UNKNOWN doc types, and as the fallback delegate for
overlong heading sections (API_REF / DSL_REF).
"""

from __future__ import annotations

import hashlib
import re

from cncf_rag.chunking.models import Chunk
from cncf_rag.chunking.strategies.base import BaseChunker
from cncf_rag.ingestion.models import Document

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_FENCE_RE = re.compile(r"(```.*?```)", re.DOTALL)


class FixedSizeChunker(BaseChunker):
    def chunk(self, document: Document) -> list[Chunk]:
        pieces = self.split_text(document.content)
        chunks: list[Chunk] = []
        for i, (text, overlap) in enumerate(pieces):
            chunks.append(
                Chunk(
                    chunk_id=hashlib.sha256(f"{document.doc_id}:{i}".encode()).hexdigest()[:32],
                    doc_id=document.doc_id,
                    content=text,
                    token_count=self._count_tokens(text),
                    chunk_index=i,
                    chunking_strategy="fixed_size",
                    has_code_blocks="```" in text,
                    overlap_with_prev=overlap,
                )
            )
        self._assert_no_split_code_blocks(chunks)
        return chunks

    def split_text(self, text: str) -> list[tuple[str, int]]:
        """Split into (chunk_text, overlap_token_count) pairs within max_tokens.

        Split priority: paragraph boundaries (\\n\\n) first because paragraphs are
        the smallest complete thought; sentence boundaries only when a single
        paragraph exceeds the budget. Fenced code blocks are atomic units — they
        are never split even if oversized (the contract beats the budget).
        """
        units: list[str] = []
        for i, segment in enumerate(_FENCE_RE.split(text)):
            if i % 2 == 1:  # a complete fenced block — atomic
                units.append(segment)
                continue
            for para in segment.split("\n\n"):
                para = para.strip()
                if not para:
                    continue
                if self._count_tokens(para) <= self.max_tokens:
                    units.append(para)
                else:
                    units.extend(s for s in _SENTENCE_RE.split(para) if s.strip())

        results: list[tuple[str, int]] = []
        current: list[str] = []
        current_tokens = 0
        prev_tail = ""
        for unit in units:
            unit_tokens = self._count_tokens(unit)
            if current and current_tokens + unit_tokens > self.max_tokens:
                chunk_text = "\n\n".join(current)
                # Overlap: chunk N+1 starts with the last overlap_tokens of chunk N.
                # Purpose: a sentence referring to "the previous step" still has that
                # step in-context. 64/512 ≈ 12.5% — enough to carry one referent
                # sentence without doubling storage like a 50% overlap would.
                overlap = 0
                if prev_tail:
                    chunk_text = prev_tail + "\n\n" + chunk_text
                    overlap = self._count_tokens(prev_tail)
                results.append((chunk_text, overlap))
                if self.overlap_tokens > 0:
                    tail_tokens = self._encode("\n\n".join(current))[-self.overlap_tokens:]
                    prev_tail = self._decode(tail_tokens)
                    # Never let the overlap tail end mid-code-fence.
                    if prev_tail.count("```") % 2 != 0:
                        prev_tail = ""
                else:
                    prev_tail = ""
                current = []
                current_tokens = 0
            current.append(unit)
            current_tokens += unit_tokens
        if current:
            chunk_text = "\n\n".join(current)
            overlap = 0
            if prev_tail:
                chunk_text = prev_tail + "\n\n" + chunk_text
                overlap = self._count_tokens(prev_tail)
            results.append((chunk_text, overlap))
        return results
