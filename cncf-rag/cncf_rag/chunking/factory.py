"""Strategy dispatch: DocType → configured chunker (DECISIONS.md 2.1 strategy map)."""

from __future__ import annotations

import os

import structlog

from cncf_rag.chunking.models import Chunk
from cncf_rag.chunking.strategies.base import BaseChunker
from cncf_rag.chunking.strategies.fixed_size import FixedSizeChunker
from cncf_rag.chunking.strategies.heading_aware import HeadingAwareChunker
from cncf_rag.chunking.strategies.hybrid import HybridChunker
from cncf_rag.chunking.strategies.semantic import SemanticChunker
from cncf_rag.embedding.service import EmbeddingService
from cncf_rag.ingestion.models import Document, DocType

logger = structlog.get_logger(__name__)


class ChunkerFactory:
    def __init__(self, embedding_service: EmbeddingService | None = None) -> None:
        # Semantic chunking needs an embedder; without one (dry runs), BLOG
        # falls back to heading-aware rather than failing the whole ingestion.
        self._embedder = embedding_service

    def get_chunker_for(self, doc_type: DocType) -> BaseChunker:
        match doc_type:
            case DocType.CONCEPT:
                return HeadingAwareChunker(max_tokens=512, overlap_tokens=0)
            case DocType.TASK:
                return HeadingAwareChunker(
                    max_tokens=400,
                    overlap_tokens=64,
                    fallback_chunker=FixedSizeChunker(max_tokens=400, overlap_tokens=64),
                )
            case DocType.API_REF:
                return HybridChunker(max_tokens=384, overlap_tokens=48)
            case DocType.DSL_REF:
                return HybridChunker(max_tokens=256, overlap_tokens=32)
            case DocType.BLOG:
                # SEMANTIC_CHUNKING=off downgrades blogs to heading-aware.
                # Needed on Cohere TRIAL keys: semantic chunking embeds every
                # sentence of ~700 blog posts — that alone would exhaust the
                # trial quota that real chunk embedding needs.
                semantic_enabled = os.environ.get("SEMANTIC_CHUNKING", "on") != "off"
                if self._embedder is not None and semantic_enabled:
                    return SemanticChunker(self._embedder, target_tokens=300)
                logger.warning("semantic_chunking_disabled_using_heading_aware")
                return HeadingAwareChunker(max_tokens=512, overlap_tokens=0)
            case DocType.RUNBOOK:
                return HeadingAwareChunker(
                    max_tokens=512,
                    overlap_tokens=64,
                    fallback_chunker=FixedSizeChunker(max_tokens=512, overlap_tokens=64),
                )
            case _:
                logger.warning("unknown_doc_type_using_fixed_size")
                return FixedSizeChunker(max_tokens=512, overlap_tokens=64)

    async def chunk_document(self, document: Document) -> list[Chunk]:
        chunker = self.get_chunker_for(document.doc_type)
        try:
            if isinstance(chunker, SemanticChunker):
                chunks = await chunker.achunk(document)
            else:
                chunks = chunker.chunk(document)
        except ValueError as exc:
            # The no-split-code-block contract fired: this document's fence
            # syntax is unbalanced (4-backtick nesting, unterminated fence).
            # One pathological file must not abort a full corpus run — fall
            # back to fixed-size with validation off and log it for review.
            logger.warning(
                "chunking_contract_violation_fallback",
                doc=document.metadata.source_path,
                error=str(exc)[:200],
            )
            fallback = FixedSizeChunker(
                max_tokens=512, overlap_tokens=0, validate_code_blocks=False
            )
            chunks = fallback.chunk(document)
        # Stamp shared payload metadata once, here, so individual chunkers
        # don't each need to know about projects/versions/URLs.
        for chunk in chunks:
            chunk.metadata.update(
                {
                    "project": document.project.value,
                    "doc_type": document.doc_type.value,
                    "version_tag": document.metadata.version_tag,
                    "source_path": document.metadata.source_path,
                    "title": document.metadata.title,
                }
            )
        return chunks
