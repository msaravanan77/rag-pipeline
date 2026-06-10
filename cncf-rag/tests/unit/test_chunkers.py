"""Stage 2 tests: heading boundaries, code preservation, overlap, merging, dispatch."""

from __future__ import annotations

from cncf_rag.chunking.factory import ChunkerFactory
from cncf_rag.chunking.strategies.fixed_size import FixedSizeChunker
from cncf_rag.chunking.strategies.heading_aware import HeadingAwareChunker
from cncf_rag.chunking.strategies.hybrid import HybridChunker
from cncf_rag.ingestion.models import DocType
from tests.conftest import make_document


class TestHeadingAwareChunker:
    def test_one_chunk_per_section(self, concept_document):
        chunks = HeadingAwareChunker(max_tokens=512).chunk(concept_document)
        headings = {c.parent_heading for c in chunks if c.parent_heading}
        assert "Pod phase" in headings
        assert "Pod conditions" in headings

    def test_heading_path_breadcrumb(self, concept_document):
        chunks = HeadingAwareChunker(max_tokens=512).chunk(concept_document)
        container_chunk = next(
            (c for c in chunks if c.parent_heading == "Container states"), None
        )
        if container_chunk is not None:  # may be merged into sibling if short
            assert "Pod Lifecycle" in container_chunk.heading_path
            assert "Pod phase" in container_chunk.heading_path

    def test_no_code_block_splits(self, concept_document):
        chunks = HeadingAwareChunker(max_tokens=512).chunk(concept_document)
        for chunk in chunks:
            assert chunk.content.count("```") % 2 == 0

    def test_contiguous_indexes(self, concept_document):
        chunks = HeadingAwareChunker(max_tokens=512).chunk(concept_document)
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))

    def test_short_sections_merged(self):
        content = """---
title: Doc
---
## Tiny
One line.

## Big Section
""" + ("This is a long sentence about Kubernetes behavior in production clusters. " * 20)
        doc = make_document(content, title="Doc")
        chunks = HeadingAwareChunker(max_tokens=512).chunk(doc)
        # "Tiny" is under 40 tokens — must not appear as its own chunk.
        assert all(c.parent_heading != "Tiny" for c in chunks)
        merged = next(c for c in chunks if "One line." in c.content)
        assert "long sentence about Kubernetes" in merged.content


class TestFixedSizeChunker:
    def test_respects_max_tokens(self):
        doc = make_document(
            "---\ntitle: X\n---\n" + ("Sentence about pods. " * 500), headings=[], title="X"
        )
        chunker = FixedSizeChunker(max_tokens=100, overlap_tokens=0)
        for chunk in chunker.chunk(doc):
            assert chunk.token_count <= 110  # small tolerance for join chars

    def test_overlap_correctness(self):
        doc = make_document(
            "---\ntitle: X\n---\n" + ("Unique sentence number marker. " * 200), headings=[], title="X"
        )
        chunker = FixedSizeChunker(max_tokens=100, overlap_tokens=20)
        chunks = chunker.chunk(doc)
        assert len(chunks) >= 2
        for chunk in chunks[1:]:
            assert chunk.overlap_with_prev > 0
            # The overlapping prefix of chunk N+1 must appear at the end of chunk N's source.
            assert chunk.content.split("\n\n")[0] in chunks[chunks.index(chunk) - 1].content

    def test_atomic_code_blocks(self):
        big_code = "```yaml\n" + ("key: value\n" * 150) + "```"
        doc = make_document(f"---\ntitle: X\n---\nIntro.\n\n{big_code}\n\nOutro.", headings=[], title="X")
        chunks = FixedSizeChunker(max_tokens=100).chunk(doc)
        for chunk in chunks:
            assert chunk.content.count("```") % 2 == 0


class TestHybridFallback:
    def test_overlong_section_delegates_to_fixed(self):
        content = "---\ntitle: Big\n---\n## Huge Section\n" + ("Field description text here. " * 400)
        doc = make_document(content, title="Big")
        chunks = HybridChunker(max_tokens=200, overlap_tokens=20).chunk(doc)
        assert len(chunks) > 1
        assert any(c.chunking_strategy == "heading_aware+fixed_fallback" for c in chunks)
        # Breadcrumb context survives the fallback split.
        assert all("Huge Section" in (c.heading_path or c.content) for c in chunks)


class TestFactoryDispatch:
    def test_dispatch_per_doc_type(self):
        factory = ChunkerFactory(embedding_service=None)
        assert isinstance(factory.get_chunker_for(DocType.CONCEPT), HeadingAwareChunker)
        assert isinstance(factory.get_chunker_for(DocType.TASK), HeadingAwareChunker)
        assert isinstance(factory.get_chunker_for(DocType.API_REF), HybridChunker)
        assert isinstance(factory.get_chunker_for(DocType.DSL_REF), HybridChunker)
        # BLOG without an embedder falls back to heading-aware (no crash).
        assert isinstance(factory.get_chunker_for(DocType.BLOG), HeadingAwareChunker)
        assert isinstance(factory.get_chunker_for(DocType.UNKNOWN), FixedSizeChunker)

    def test_factory_params_match_decision_2_1(self):
        factory = ChunkerFactory()
        assert factory.get_chunker_for(DocType.CONCEPT).max_tokens == 512
        assert factory.get_chunker_for(DocType.TASK).max_tokens == 400
        assert factory.get_chunker_for(DocType.TASK).overlap_tokens == 64
        assert factory.get_chunker_for(DocType.API_REF).max_tokens == 384
        assert factory.get_chunker_for(DocType.DSL_REF).max_tokens == 256

    async def test_chunk_document_stamps_metadata(self, concept_document):
        factory = ChunkerFactory()
        chunks = await factory.chunk_document(concept_document)
        for chunk in chunks:
            assert chunk.metadata["project"] == "kubernetes"
            assert chunk.metadata["doc_type"] == "concept"
