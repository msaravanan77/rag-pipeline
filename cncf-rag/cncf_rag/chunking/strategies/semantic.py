"""Semantic chunker: embedding-similarity boundary detection.

Used ONLY for BLOG documents (DECISIONS.md 2.1). Two reasons:
1. Cost: it embeds every sentence at index time. Affordable for the small blog
   corpus; wasteful for the 10x larger docs corpus where headings already mark
   boundaries for free.
2. Reliability: blog posts often have no headings, or headings that are jokes
   or puns ("The Pod-father Part II") — heading-aware chunking misfires on them.
   Narrative prose needs boundaries detected from the text itself.
"""

from __future__ import annotations

import hashlib

import numpy as np
import structlog

from cncf_rag.chunking.models import Chunk
from cncf_rag.chunking.strategies.base import BaseChunker
from cncf_rag.embedding.service import EmbeddingService
from cncf_rag.ingestion.models import Document

logger = structlog.get_logger(__name__)


def _split_sentences(text: str) -> list[str]:
    """Sentence splitting via spaCy if available; regex fallback otherwise.

    The fallback keeps unit tests and dry runs free of the 40MB model download.
    """
    try:
        import spacy

        try:
            nlp = spacy.load("en_core_web_sm", disable=["ner", "tagger", "lemmatizer"])
        except OSError:
            raise ImportError
        return [s.text.strip() for s in nlp(text).sents if s.text.strip()]
    except ImportError:
        import re

        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


class SemanticChunker(BaseChunker):
    # 0.85: empirically, consecutive sentences within one topic in technical
    # writing score 0.88–0.95 cosine similarity; topic shifts drop to 0.70–0.84.
    # 0.85 sits in the gap — low enough to ignore stylistic variation, high
    # enough to catch genuine subject changes.
    def __init__(
        self,
        embedding_service: EmbeddingService,
        target_tokens: int = 300,
        similarity_threshold: float = 0.85,
    ) -> None:
        super().__init__(max_tokens=target_tokens * 2, overlap_tokens=0)
        self._embedder = embedding_service
        self.target_tokens = target_tokens
        self.similarity_threshold = similarity_threshold

    def chunk(self, document: Document) -> list[Chunk]:
        raise NotImplementedError("SemanticChunker is async — use achunk()")

    async def achunk(self, document: Document) -> list[Chunk]:
        sentences = _split_sentences(document.content)
        if len(sentences) < 2:
            return self._wrap([document.content], document)

        # Batch-embed all sentences in one pass (not per-sentence calls) —
        # 100 sentences = ~2 API requests instead of 100.
        vectors = await self._embedder.embed_documents(sentences)
        arr = np.array(vectors)
        norms = arr / np.linalg.norm(arr, axis=1, keepdims=True)
        # Cosine similarity of each sentence with its successor.
        sims = np.sum(norms[:-1] * norms[1:], axis=1)

        groups: list[list[str]] = [[sentences[0]]]
        group_tokens = self._count_tokens(sentences[0])
        for sentence, sim in zip(sentences[1:], sims):
            tokens = self._count_tokens(sentence)
            # Boundary on topic shift, or when the group passes target size
            # (prevents one long single-topic post becoming one giant chunk).
            if sim < self.similarity_threshold or group_tokens + tokens > self.target_tokens:
                groups.append([sentence])
                group_tokens = tokens
            else:
                groups[-1].append(sentence)
                group_tokens += tokens
        return self._wrap([" ".join(g) for g in groups], document)

    def _wrap(self, texts: list[str], document: Document) -> list[Chunk]:
        chunks = [
            Chunk(
                chunk_id=hashlib.sha256(f"{document.doc_id}:{i}".encode()).hexdigest()[:32],
                doc_id=document.doc_id,
                content=text,
                token_count=self._count_tokens(text),
                chunk_index=i,
                chunking_strategy="semantic",
                has_code_blocks="```" in text,
            )
            for i, text in enumerate(texts)
        ]
        self._assert_no_split_code_blocks(chunks)
        return chunks
