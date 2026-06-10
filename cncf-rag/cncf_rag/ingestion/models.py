"""Core document models for the ingestion pipeline.

These dataclasses are the contract between ingestion, chunking, and indexing.
Everything downstream (chunkers, embedders, the vector store payload) consumes
these types — change them here and the whole pipeline sees it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class DocType(str, Enum):
    """The six structural document types in the CNCF corpus, plus UNKNOWN.

    Each type gets its own chunking strategy (see DECISIONS.md 2.1) because
    their internal structure differs fundamentally: a concept guide is prose
    under headings, a DSL reference is dense parameter tables with no sentences.
    """

    CONCEPT = "concept"
    TASK = "task"
    API_REF = "api_ref"
    DSL_REF = "dsl_ref"
    BLOG = "blog"
    RUNBOOK = "runbook"
    UNKNOWN = "unknown"


class Project(str, Enum):
    """The four CNCF projects whose docs form the corpus."""

    KUBERNETES = "kubernetes"
    HELM = "helm"
    ARGOCD = "argocd"
    PROMETHEUS = "prometheus"


@dataclass
class Heading:
    """A single heading extracted from the Markdown AST.

    Level and text are separate typed fields (not rendered HTML) because
    heading-aware chunking needs to compare levels to find section boundaries.
    char_offset locates the heading in the cleaned body so chunkers can slice
    section content without re-parsing.
    """

    level: int
    text: str
    char_offset: int


@dataclass
class CodeBlock:
    """A fenced code block with its declared language.

    Kept as a distinct unit because the chunking contract forbids splitting
    code blocks — a half YAML manifest is worse than no manifest.
    """

    language: str
    content: str
    char_offset: int


@dataclass
class DocumentMetadata:
    """Frontmatter-derived and path-derived metadata carried into every chunk payload."""

    title: str
    source_path: str
    source_url: str | None = None
    version_tag: str | None = None
    last_modified: datetime | None = None
    description: str | None = None
    weight: int | None = None
    raw_frontmatter: dict = field(default_factory=dict)


@dataclass
class Document:
    """A fully parsed, classified, preprocessed document ready for chunking."""

    doc_id: str  # SHA-256 of source_path — stable across re-ingestion runs
    project: Project
    doc_type: DocType
    doc_type_confidence: float
    content: str  # cleaned body, frontmatter stripped, shortcodes removed
    headings: list[Heading]
    code_blocks: list[CodeBlock]
    metadata: DocumentMetadata
    checksum: str  # SHA-256 of raw file content — drives incremental ingestion


@dataclass
class ParsedDocument:
    """Intermediate parser output: structure extracted, not yet classified."""

    content: str
    headings: list[Heading]
    code_blocks: list[CodeBlock]
    frontmatter: dict
    source_path: str
