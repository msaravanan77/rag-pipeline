"""Chunk model — the retrieval unit. Every field here becomes a Qdrant payload field."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    content: str
    token_count: int
    chunk_index: int
    chunking_strategy: str
    parent_heading: str | None = None
    parent_heading_level: int | None = None
    heading_path: str | None = None  # "Concepts > Workloads > Pods" breadcrumb
    has_code_blocks: bool = False
    overlap_with_prev: int = 0
    metadata: dict = field(default_factory=dict)
    embedding: list[float] | None = None


@dataclass
class ChunkingReport:
    """Aggregate quality stats — the evaluator's output for a chunking run."""

    total_chunks: int
    total_tokens: int
    mean_token_count: float
    p50_token_count: int
    p95_token_count: int
    max_token_count: int
    chunks_under_40_tokens: int
    split_code_block_count: int
    chunks_by_strategy: dict[str, int] = field(default_factory=dict)
