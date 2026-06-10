"""Hybrid chunker: heading-aware with a fixed-size fallback for overlong sections.

This is just configuration glue — HeadingAwareChunker already accepts a
fallback delegate; this class names the combination used by API_REF/DSL_REF
so the factory reads declaratively.
"""

from __future__ import annotations

from cncf_rag.chunking.strategies.fixed_size import FixedSizeChunker
from cncf_rag.chunking.strategies.heading_aware import HeadingAwareChunker


class HybridChunker(HeadingAwareChunker):
    def __init__(self, max_tokens: int = 384, overlap_tokens: int = 48) -> None:
        super().__init__(
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
            fallback_chunker=FixedSizeChunker(max_tokens=max_tokens, overlap_tokens=overlap_tokens),
        )
