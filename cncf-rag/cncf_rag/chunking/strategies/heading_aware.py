"""Heading-aware chunker: one chunk per heading section.

The primary strategy for 5 of 7 doc types (DECISIONS.md 2.1) because CNCF docs
are written section-per-topic: an H2/H3 section is the natural complete answer
unit for documentation queries.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import structlog

from cncf_rag.chunking.models import Chunk
from cncf_rag.chunking.strategies.base import BaseChunker
from cncf_rag.chunking.strategies.fixed_size import FixedSizeChunker
from cncf_rag.ingestion.models import Document, Heading

logger = structlog.get_logger(__name__)

_MIN_SECTION_TOKENS = 40  # below this a section is noise alone; merge forward


@dataclass
class _Section:
    heading: Heading | None  # None = preamble before the first heading
    heading_path: str
    content: str


class HeadingAwareChunker(BaseChunker):
    def __init__(
        self,
        max_tokens: int = 512,
        overlap_tokens: int = 0,
        fallback_chunker: FixedSizeChunker | None = None,
    ) -> None:
        super().__init__(max_tokens=max_tokens, overlap_tokens=overlap_tokens)
        self._fallback = fallback_chunker

    def chunk(self, document: Document) -> list[Chunk]:
        sections = self._split_sections(document)
        sections = self._merge_short_sections(sections)

        chunks: list[Chunk] = []
        index = 0
        for section in sections:
            # Prepend the breadcrumb ("Concepts > Pods > Pod Lifecycle") to the
            # content that gets embedded: a section saying only "Set this field
            # to true" embeds uselessly without knowing WHAT it configures. The
            # breadcrumb injects that document-level context into the vector.
            prefixed = (
                f"{section.heading_path}\n\n{section.content}"
                if section.heading_path
                else section.content
            )
            if self._count_tokens(prefixed) <= self.max_tokens or self._fallback is None:
                chunks.append(self._make_chunk(document, section, prefixed, index))
                index += 1
            else:
                # Section exceeds budget (e.g. PodSpec's 50-field table): delegate
                # the section body to the fixed-size fallback, keeping breadcrumbs.
                for piece, overlap in self._fallback.split_text(section.content):
                    prefixed_piece = (
                        f"{section.heading_path}\n\n{piece}" if section.heading_path else piece
                    )
                    chunk = self._make_chunk(document, section, prefixed_piece, index)
                    chunk.overlap_with_prev = overlap
                    chunk.chunking_strategy = "heading_aware+fixed_fallback"
                    chunks.append(chunk)
                    index += 1

        self._assert_no_split_code_blocks(chunks)
        return chunks

    def _make_chunk(self, document: Document, section: _Section, content: str, index: int) -> Chunk:
        return Chunk(
            chunk_id=hashlib.sha256(f"{document.doc_id}:{index}".encode()).hexdigest()[:32],
            doc_id=document.doc_id,
            content=content,
            token_count=self._count_tokens(content),
            chunk_index=index,
            chunking_strategy="heading_aware",
            parent_heading=section.heading.text if section.heading else None,
            parent_heading_level=section.heading.level if section.heading else None,
            heading_path=section.heading_path or None,
            has_code_blocks="```" in content,
        )

    def _split_sections(self, document: Document) -> list[_Section]:
        """Slice content at heading offsets; a section runs until the next
        same-or-higher-level heading (lower level = nested subsection stays in)."""
        headings = sorted(document.headings, key=lambda h: h.char_offset)
        content = document.content
        sections: list[_Section] = []

        # Preamble before the first heading (k8s docs often open with a summary).
        first_offset = headings[0].char_offset if headings else len(content)
        preamble = content[:first_offset].strip()
        if preamble:
            sections.append(_Section(heading=None, heading_path=document.metadata.title, content=preamble))

        breadcrumb: list[Heading] = []
        for i, heading in enumerate(headings):
            end = len(content)
            for later in headings[i + 1:]:
                if later.level <= heading.level:
                    end = later.char_offset
                    break
            else:
                end = headings[i + 1].char_offset if i + 1 < len(headings) else len(content)
            # Maintain the breadcrumb stack: pop anything at this level or deeper.
            while breadcrumb and breadcrumb[-1].level >= heading.level:
                breadcrumb.pop()
            breadcrumb.append(heading)
            path = " > ".join([document.metadata.title] + [h.text for h in breadcrumb])
            # Section content: this heading's slice MINUS nested subsections that
            # become their own sections (next heading of any level ends the body).
            body_end = headings[i + 1].char_offset if i + 1 < len(headings) else len(content)
            body = content[heading.char_offset:min(end, body_end)]
            # Strip the heading line itself — it lives in heading_path instead.
            body = "\n".join(body.split("\n")[1:]).strip()
            if body or True:  # keep empty-bodied headings; merge pass handles them
                sections.append(_Section(heading=heading, heading_path=path, content=body))
        return sections

    def _merge_short_sections(self, sections: list[_Section]) -> list[_Section]:
        """Merge sections under 40 tokens into the following sibling.

        A 15-token section ("See the configuration guide.") embedded alone
        matches almost any query weakly and none well — merging keeps signal density up.
        """
        merged: list[_Section] = []
        pending: _Section | None = None
        for section in sections:
            if pending is not None:
                section = _Section(
                    heading=section.heading,
                    heading_path=section.heading_path,
                    content=f"{pending.heading_path}\n{pending.content}\n\n{section.content}".strip(),
                )
                pending = None
            if self._count_tokens(section.content) < _MIN_SECTION_TOKENS:
                pending = section
            else:
                merged.append(section)
        if pending is not None:  # trailing short section: append to last or keep alone
            if merged:
                merged[-1].content += f"\n\n{pending.heading_path}\n{pending.content}"
            else:
                merged.append(pending)
        return merged
