"""Qdrant store wrapper: collection lifecycle, upsert, filtered search."""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass

import structlog
from qdrant_client import AsyncQdrantClient
from qdrant_client import models as qm

from cncf_rag.chunking.models import Chunk
from cncf_rag.vectorstore.schema import COLLECTION_NAME, HNSW_CONFIG, PAYLOAD_INDEXES, VECTOR_CONFIG

logger = structlog.get_logger(__name__)

_UPSERT_BATCH = 100  # keeps request bodies ~1.5MB at 1024-dim float32 — well under limits


@dataclass
class ScoredChunk:
    chunk_id: str
    doc_id: str
    content: str
    score: float
    payload: dict


class QdrantVectorStore:
    def __init__(self, host: str | None = None, port: int | None = None) -> None:
        self._client = AsyncQdrantClient(
            host=host or os.environ.get("QDRANT_HOST", "localhost"),
            port=port or int(os.environ.get("QDRANT_PORT", "6333")),
        )
        self.collection_name = COLLECTION_NAME

    async def ensure_collection(self) -> None:
        """Idempotent: creates collection + payload indexes only if missing."""
        existing = {c.name for c in (await self._client.get_collections()).collections}
        if self.collection_name not in existing:
            await self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VECTOR_CONFIG,
                hnsw_config=HNSW_CONFIG,
            )
            logger.info("collection_created", name=self.collection_name)
        for field_name, schema_type in PAYLOAD_INDEXES.items():
            # create_payload_index is itself idempotent in Qdrant — safe to repeat.
            await self._client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field_name,
                field_schema=schema_type,
            )

    async def upsert_chunks(self, chunks: list[Chunk]) -> None:
        """Upsert (not insert): re-ingesting a changed document writes the same
        deterministic point IDs (derived from doc_id+chunk_index), so updated
        content REPLACES stale vectors instead of accumulating duplicates —
        this is what makes checksum-based incremental ingestion correct."""
        for start in range(0, len(chunks), _UPSERT_BATCH):
            batch = chunks[start : start + _UPSERT_BATCH]
            points = [
                qm.PointStruct(
                    # Qdrant point IDs must be UUIDs or ints; uuid5 makes the
                    # chunk_id → point ID mapping deterministic.
                    id=str(uuid.uuid5(uuid.NAMESPACE_OID, chunk.chunk_id)),
                    vector=chunk.embedding,
                    payload={
                        "chunk_id": chunk.chunk_id,
                        "doc_id": chunk.doc_id,
                        "content": chunk.content,
                        "token_count": chunk.token_count,
                        "chunk_index": chunk.chunk_index,
                        "chunking_strategy": chunk.chunking_strategy,
                        "heading_path": chunk.heading_path,
                        "has_code_blocks": chunk.has_code_blocks,
                        **chunk.metadata,
                    },
                )
                for chunk in batch
                if chunk.embedding is not None
            ]
            await self._client.upsert(collection_name=self.collection_name, points=points)

    async def delete_by_doc_id(self, doc_id: str) -> None:
        """Remove all chunks of one document — used when a source file is deleted
        or its chunk count shrinks (upsert alone would leave orphan tail chunks)."""
        await self._client.delete(
            collection_name=self.collection_name,
            points_selector=qm.FilterSelector(
                filter=qm.Filter(
                    must=[qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc_id))]
                )
            ),
        )

    async def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        filters: dict | None = None,
        ef: int = 50,
    ) -> list[ScoredChunk]:
        qdrant_filter = self._build_filter(filters) if filters else None
        results = await self._client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
            query_filter=qdrant_filter,
            search_params=qm.SearchParams(hnsw_ef=ef),
            with_payload=True,
        )
        return [
            ScoredChunk(
                chunk_id=hit.payload.get("chunk_id", str(hit.id)),
                doc_id=hit.payload.get("doc_id", ""),
                content=hit.payload.get("content", ""),
                score=hit.score,
                payload=hit.payload or {},
            )
            for hit in results
        ]

    @staticmethod
    def _build_filter(filters: dict) -> qm.Filter:
        """Translate {"project": "kubernetes", "version_tag": ["v1.29", "v1.30"]}
        into Qdrant must-conditions. Lists become MatchAny (OR within a field);
        separate keys are AND-ed — matching SQL WHERE intuition."""
        conditions: list[qm.FieldCondition] = []
        for key, value in filters.items():
            if value is None:
                continue
            if isinstance(value, list):
                conditions.append(qm.FieldCondition(key=key, match=qm.MatchAny(any=value)))
            else:
                conditions.append(qm.FieldCondition(key=key, match=qm.MatchValue(value=value)))
        return qm.Filter(must=conditions)

    async def healthcheck(self) -> bool:
        try:
            await self._client.get_collections()
            return True
        except Exception:
            return False
