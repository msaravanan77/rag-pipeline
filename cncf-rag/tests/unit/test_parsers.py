"""Stage 1 tests: parsing, cleaning, and classification rules."""

from __future__ import annotations

import pytest

from cncf_rag.ingestion.classifier import DocTypeClassifier
from cncf_rag.ingestion.models import DocType, Project
from cncf_rag.ingestion.parsers.markdown_parser import MarkdownParser
from cncf_rag.ingestion.preprocessor import Preprocessor


class TestMarkdownParser:
    def test_parses_real_k8s_snippet(self, parsed_k8s_doc):
        assert parsed_k8s_doc.frontmatter["title"] == "Pod Lifecycle"
        assert parsed_k8s_doc.frontmatter["content_type"] == "concept"
        heading_texts = [h.text for h in parsed_k8s_doc.headings]
        assert "Pod phase" in heading_texts
        assert "Container states" in heading_texts
        assert "Pod conditions" in heading_texts

    def test_heading_levels_preserved(self, parsed_k8s_doc):
        by_text = {h.text: h.level for h in parsed_k8s_doc.headings}
        assert by_text["Pod phase"] == 2
        assert by_text["Container states"] == 3

    def test_code_block_extracted_with_language(self, parsed_k8s_doc):
        assert len(parsed_k8s_doc.code_blocks) == 1
        assert parsed_k8s_doc.code_blocks[0].language == "yaml"
        assert "kind: Pod" in parsed_k8s_doc.code_blocks[0].content

    def test_hugo_shortcodes_removed(self, parsed_k8s_doc):
        assert "{{<" not in parsed_k8s_doc.content
        assert "{{%" not in parsed_k8s_doc.content

    def test_html_comments_removed(self, parsed_k8s_doc):
        assert "<!--" not in parsed_k8s_doc.content

    def test_heading_offsets_point_at_headings(self, parsed_k8s_doc):
        for heading in parsed_k8s_doc.headings:
            at_offset = parsed_k8s_doc.content[heading.char_offset:heading.char_offset + 10]
            assert at_offset.startswith("#")


class TestPreprocessor:
    def test_collapses_blank_lines(self):
        assert "\n\n\n" not in Preprocessor().clean("a\n\n\n\n\nb")

    def test_normalizes_crlf(self):
        assert "\r" not in Preprocessor().clean("line one\r\nline two")

    def test_preserves_code_blocks_verbatim(self):
        code = "```yaml\nkey:   value   \n\n\n\nother: thing\n```"
        assert code in Preprocessor().clean(f"prose\n\n{code}\n\nmore prose   ")

    def test_strips_trailing_whitespace_in_prose(self):
        result = Preprocessor().clean("hello   \nworld  ")
        assert "hello\nworld" in result


class TestClassifierRules:
    @pytest.fixture
    def classifier(self) -> DocTypeClassifier:
        return DocTypeClassifier(anthropic_client=None)

    async def test_concept_frontmatter(self, classifier):
        doc_type, conf = await classifier.classify(
            "k8s/doc.md", {"content_type": "concept"}, "text", Project.KUBERNETES
        )
        assert (doc_type, conf) == (DocType.CONCEPT, 1.0)

    async def test_task_frontmatter(self, classifier):
        doc_type, conf = await classifier.classify(
            "k8s/doc.md", {"content_type": "task"}, "text", Project.KUBERNETES
        )
        assert (doc_type, conf) == (DocType.TASK, 1.0)

    async def test_reference_frontmatter(self, classifier):
        doc_type, conf = await classifier.classify(
            "k8s/doc.md", {"content_type": "reference"}, "text", Project.KUBERNETES
        )
        assert (doc_type, conf) == (DocType.API_REF, 1.0)

    async def test_blog_path(self, classifier):
        doc_type, conf = await classifier.classify(
            "kubernetes/content/en/blog/2024/post.md", {}, "Announcing...", Project.KUBERNETES
        )
        assert (doc_type, conf) == (DocType.BLOG, 0.95)

    async def test_argocd_runbook_path(self, classifier):
        doc_type, conf = await classifier.classify(
            "argocd/docs/operator-manual/disaster_recovery.md", {}, "Steps...", Project.ARGOCD
        )
        assert (doc_type, conf) == (DocType.RUNBOOK, 0.90)

    async def test_dsl_ref_by_code_ratio(self, classifier):
        # Mostly code, terse prose — a Helm template reference shape.
        content = "Func ref.\n\n" + "```\n" + ("{{ .Values.x }}\n" * 100) + "```\n"
        doc_type, conf = await classifier.classify("helm/docs/chart_template_guide/functions.md", {}, content, Project.HELM)
        assert (doc_type, conf) == (DocType.DSL_REF, 0.85)

    async def test_llm_fallback_triggers_below_threshold(self, classifier, monkeypatch):
        """Ambiguous doc + no frontmatter → rules return < 0.80 → fallback path."""
        called = {}

        async def fake_llm(source_path, content):
            called["yes"] = True
            return DocType.CONCEPT, 0.85

        classifier._client = object()  # non-None so fallback path is taken
        monkeypatch.setattr(classifier, "_classify_by_llm", fake_llm)
        prose = "This document discusses various topics. " * 30
        doc_type, conf = await classifier.classify("helm/docs/intro.md", {}, prose, Project.HELM)
        assert called.get("yes")
        assert doc_type == DocType.CONCEPT

    async def test_no_llm_returns_rule_guess(self, classifier):
        prose = "This document discusses various topics. " * 30
        doc_type, conf = await classifier.classify("helm/docs/intro.md", {}, prose, Project.HELM)
        assert conf < 0.80  # honest low confidence, no crash
