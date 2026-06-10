"""Qdrant collection schema constants (DECISIONS.md 4.1, 4.2)."""

from __future__ import annotations

import os

from qdrant_client import models as qm

COLLECTION_NAME: str = os.environ.get("QDRANT_COLLECTION_NAME", "cncf_docs")

# 1024 = Cohere embed-english-v3.0 native dimensionality.
# COSINE because Cohere v3 vectors are not pre-normalized.
VECTOR_CONFIG = qm.VectorParams(size=1024, distance=qm.Distance.COSINE)

# m=16 / ef_construct=100: defaults, >0.95 recall at 200k vectors — see 4.2.
HNSW_CONFIG = qm.HnswConfigDiff(m=16, ef_construct=100)

# Every field the retrieval layer filters on gets a payload index — without
# one, Qdrant falls back to full payload scans, erasing the entire benefit
# of server-side filtering.
PAYLOAD_INDEXES: dict[str, qm.PayloadSchemaType] = {
    "project": qm.PayloadSchemaType.KEYWORD,
    "doc_type": qm.PayloadSchemaType.KEYWORD,
    "version_tag": qm.PayloadSchemaType.KEYWORD,
    "doc_id": qm.PayloadSchemaType.KEYWORD,
    "chunking_strategy": qm.PayloadSchemaType.KEYWORD,
    "has_code_blocks": qm.PayloadSchemaType.BOOL,
}
