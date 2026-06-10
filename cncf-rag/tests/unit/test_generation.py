"""Stage 6 tests: prompt assembly and generation short-circuit / parsing."""

from __future__ import annotations

import pytest

from cncf_rag.generation.prompt_builder import SYSTEM_PROMPT, PromptBuilder
from cncf_rag.generation.service import GenerationParseError, GenerationService
from cncf_rag.vectorstore.qdrant_store import ScoredChunk


def chunk(cid: str, content: str, score: float, payload: dict | None = None) -> ScoredChunk:
    return ScoredChunk(chunk_id=cid, doc_id="d", content=content, score=score, payload=payload or {})


class TestPromptBuilder:
    def test_system_prompt_contains_strict_rules_verbatim(self):
        system, _ = PromptBuilder().build("q", [chunk("a", "text", 0.9)])
        assert system == SYSTEM_PROMPT
        assert "Answer ONLY from the CONTEXT" in system
        assert "cannot_answer" in system

    def test_context_blocks_numbered_and_score_ordered(self):
        chunks = [
            chunk("low", "low scoring content", 0.71, {"project": "helm"}),
            chunk("high", "high scoring content", 0.95, {"project": "kubernetes"}),
        ]
        _, user = PromptBuilder().build("my question", chunks)
        assert user.index("CONTEXT 1") < user.index("CONTEXT 2")
        # Highest score must be CONTEXT 1 (lost-in-the-middle mitigation).
        assert user.index("high scoring content") < user.index("low scoring content")
        assert "QUESTION: my question" in user

    def test_heading_path_prepended(self):
        chunks = [chunk("a", "body text", 0.9, {"heading_path": "Concepts > Pods"})]
        _, user = PromptBuilder().build("q", chunks)
        assert "Concepts > Pods" in user


class TestGenerationService:
    @pytest.fixture
    def service(self):
        # No API call happens in these tests — pass a placeholder client.
        return GenerationService(anthropic_client=object())

    async def test_cannot_answer_on_low_similarity(self, service):
        result = await service.generate("q", [chunk("a", "text", 0.5)])
        assert result.cannot_answer is True
        assert result.cost_usd == 0.0

    async def test_cannot_answer_on_empty_chunks(self, service):
        result = await service.generate("q", [])
        assert result.cannot_answer is True

    def test_parse_valid_json(self):
        parsed = GenerationService._parse_response('{"answer": "A pod is...", "confidence": 0.9}')
        assert parsed["answer"] == "A pod is..."

    def test_parse_strips_markdown_fences(self):
        parsed = GenerationService._parse_response('```json\n{"answer": "x"}\n```')
        assert parsed["answer"] == "x"

    def test_parse_raises_on_non_json(self):
        with pytest.raises(GenerationParseError):
            GenerationService._parse_response("Sure! Here is the answer: pods are great.")

    def test_parse_raises_on_missing_answer(self):
        with pytest.raises(GenerationParseError):
            GenerationService._parse_response('{"confidence": 0.4}')
