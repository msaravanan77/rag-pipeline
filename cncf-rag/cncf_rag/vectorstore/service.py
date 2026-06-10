"""Vector store facade — re-exports the Qdrant implementation.

Exists so call sites import `vectorstore.service` rather than a concrete
backend; swapping Qdrant later means changing this one module.
"""

from __future__ import annotations

from cncf_rag.vectorstore.qdrant_store import QdrantVectorStore, ScoredChunk

VectorStore = QdrantVectorStore

__all__ = ["VectorStore", "QdrantVectorStore", "ScoredChunk"]
