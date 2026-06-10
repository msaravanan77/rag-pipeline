"""Corpus loading with checksum-based incremental ingestion.

See DECISIONS.md 1.3: SHA-256 per file stored in SQLite — works identically for
git clones and S3-downloaded corpora, with zero server overhead.
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from pathlib import Path

import structlog

try:
    import git as gitpython
except ImportError:  # gitpython is optional at runtime (S3-sourced corpora)
    gitpython = None  # type: ignore[assignment]

from cncf_rag.ingestion.classifier import DocTypeClassifier
from cncf_rag.ingestion.models import Document, DocumentMetadata, Project
from cncf_rag.ingestion.parsers.markdown_parser import MarkdownParser
from cncf_rag.ingestion.preprocessor import Preprocessor

logger = structlog.get_logger(__name__)

# Kubernetes website docs encode the version in URL paths like
# /docs/reference/generated/kubernetes-api/v1.29/.
_URL_VERSION_RE = re.compile(r"/(v\d+\.\d+)/")

# Non-English content lives under language-coded directories in kubernetes/website
# (content/zh-cn, content/ja, ...). Only content/en is ingested.
_NON_ENGLISH_RE = re.compile(r"/content/(?!en/)[a-z]{2}(-[a-z]{2,4})?/")

_MAX_FILE_BYTES = 500_000  # auto-generated API dumps exceed this; they pollute retrieval

_SCHEMA = """
CREATE TABLE IF NOT EXISTS ingested_files (
    source_path TEXT PRIMARY KEY,
    checksum TEXT NOT NULL,
    ingested_at TEXT NOT NULL
)
"""


class IngestIndex:
    """SQLite-backed checksum index. One row per successfully ingested file."""

    def __init__(self, db_path: Path) -> None:
        self._conn = sqlite3.connect(db_path)
        self._conn.execute(_SCHEMA)
        self._conn.commit()

    def is_unchanged(self, source_path: str, checksum: str) -> bool:
        row = self._conn.execute(
            "SELECT checksum FROM ingested_files WHERE source_path = ?", (source_path,)
        ).fetchone()
        return row is not None and row[0] == checksum

    def record(self, source_path: str, checksum: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO ingested_files VALUES (?, ?, ?)",
            (source_path, checksum, datetime.now(timezone.utc).isoformat()),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()


class CorpusLoader:
    """Walks corpus directories yielding parsed, classified Documents."""

    def __init__(
        self,
        parser: MarkdownParser | None = None,
        preprocessor: Preprocessor | None = None,
        classifier: DocTypeClassifier | None = None,
        skip_unchanged: bool = True,
    ) -> None:
        self._parser = parser or MarkdownParser()
        self._preprocessor = preprocessor or Preprocessor()
        self._classifier = classifier or DocTypeClassifier()
        self._skip_unchanged = skip_unchanged

    async def load_all(
        self, corpus_dir: Path, projects: list[Project]
    ) -> AsyncIterator[Document]:
        """Yield Documents for every new-or-changed Markdown file under corpus_dir/<project>/."""
        index = IngestIndex(corpus_dir / ".ingest_index.db")
        try:
            for project in projects:
                project_dir = corpus_dir / project.value
                if not project_dir.exists():
                    logger.warning("project_dir_missing", project=project.value, dir=str(project_dir))
                    continue
                repo = self._open_repo(project_dir)
                for md_path in sorted(project_dir.rglob("*.md")):
                    doc = await self._load_one(md_path, project, project_dir, repo, index)
                    if doc is not None:
                        yield doc
        finally:
            index.close()

    async def _load_one(
        self,
        md_path: Path,
        project: Project,
        project_dir: Path,
        repo: object,
        index: IngestIndex,
    ) -> Document | None:
        rel_path = str(md_path.relative_to(project_dir.parent))
        if _NON_ENGLISH_RE.search("/" + rel_path):
            return None
        if md_path.stat().st_size > _MAX_FILE_BYTES:
            logger.info("skip_oversized_file", path=rel_path, bytes=md_path.stat().st_size)
            return None

        raw = md_path.read_bytes()
        checksum = hashlib.sha256(raw).hexdigest()
        if self._skip_unchanged and index.is_unchanged(rel_path, checksum):
            return None

        try:
            parsed = self._parser.parse(raw.decode("utf-8", errors="replace"), rel_path)
        except Exception as exc:
            # Real-world corpora contain malformed frontmatter (e.g. unescaped
            # colons in titles). One bad file must not abort a 2,650-file run.
            logger.warning("skip_unparseable_file", path=rel_path, error=str(exc))
            return None
        content = self._preprocessor.clean(parsed.content)
        if content != parsed.content:
            # Cleaning shifted character positions — re-extract structure from the
            # cleaned text so heading/code-block offsets match what chunkers slice.
            reparsed = self._parser.parse(content, rel_path)
            parsed.headings = reparsed.headings
            parsed.code_blocks = reparsed.code_blocks
        doc_type, confidence = await self._classifier.classify(
            rel_path, parsed.frontmatter, content, project
        )

        metadata = DocumentMetadata(
            title=str(parsed.frontmatter.get("title", md_path.stem)),
            source_path=rel_path,
            version_tag=self._extract_version(parsed.frontmatter, rel_path),
            last_modified=self._git_last_modified(repo, md_path),
            description=parsed.frontmatter.get("description"),
            weight=parsed.frontmatter.get("weight"),
            raw_frontmatter=parsed.frontmatter,
        )
        index.record(rel_path, checksum)
        return Document(
            # doc_id from the path (not content) so re-ingesting a changed file
            # upserts the same logical document instead of duplicating it.
            doc_id=hashlib.sha256(rel_path.encode()).hexdigest(),
            project=project,
            doc_type=doc_type,
            doc_type_confidence=confidence,
            content=content,
            headings=parsed.headings,
            code_blocks=parsed.code_blocks,
            metadata=metadata,
            checksum=checksum,
        )

    @staticmethod
    def _extract_version(frontmatter: dict, rel_path: str) -> str | None:
        """Frontmatter version field first; URL pattern (/v1.29/) as fallback."""
        for key in ("min-kubernetes-server-version", "version", "k8s_version"):
            if key in frontmatter:
                return str(frontmatter[key])
        match = _URL_VERSION_RE.search(rel_path)
        return match.group(1) if match else None

    @staticmethod
    def _open_repo(project_dir: Path) -> object:
        if gitpython is None:
            return None
        try:
            return gitpython.Repo(project_dir, search_parent_directories=True)
        except Exception:
            return None  # corpus came from S3/zip — git metadata simply unavailable

    @staticmethod
    def _git_last_modified(repo: object, md_path: Path) -> datetime | None:
        if repo is None:
            return None
        try:
            commit = next(repo.iter_commits(paths=str(md_path), max_count=1))  # type: ignore[attr-defined]
            return commit.committed_datetime
        except Exception:
            return None
