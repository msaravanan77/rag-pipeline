"""Document type classification: rules first, LLM fallback below 0.80 confidence.

See DECISIONS.md 1.2. Kubernetes frontmatter carries explicit content_type so
rules classify it perfectly and free; Helm/Argo CD lack it, so ~20% of files
fall through to a cheap LLM call (claude-haiku — the Anthropic key is already
required for generation, so no extra provider is introduced).
"""

from __future__ import annotations

import os
import re

import structlog
from anthropic import AsyncAnthropic

from cncf_rag.ingestion.models import DocType, Project

logger = structlog.get_logger(__name__)

_CONFIDENCE_THRESHOLD = 0.80

_FENCED_BLOCK_RE = re.compile(r"```.*?```", re.DOTALL)
_SENTENCE_SPLIT_RE = re.compile(r"[.!?]+\s")

_LLM_CLASSIFY_PROMPT = """You are classifying a technical documentation page into exactly one category.

Categories:
- concept: explains what something is and why it exists
- task: step-by-step instructions to accomplish a goal
- api_ref: API/resource field reference (fields, types, descriptions)
- dsl_ref: template/configuration language reference (functions, syntax)
- blog: announcement, release note, or narrative post
- runbook: operational procedures for running/troubleshooting a system

Respond with ONLY the category name, lowercase, nothing else.

Document path: {path}
Document content (truncated):
{content}"""


class DocTypeClassifier:
    """Hybrid rule-based + LLM-fallback classifier.

    The LLM prompt (above, verbatim in `_LLM_CLASSIFY_PROMPT`) is structured as:
    (1) role framing, (2) closed category list with one-line definitions so the
    model cannot invent labels, (3) a strict output format instruction making the
    response machine-parseable, (4) path before content because the URL path is
    often the strongest single classification signal.
    """

    def __init__(self, anthropic_client: AsyncAnthropic | None = None) -> None:
        self._client = anthropic_client

    async def classify(
        self,
        source_path: str,
        frontmatter: dict,
        content: str,
        project: Project,
    ) -> tuple[DocType, float]:
        """Rule-based first pass; LLM fallback when rule confidence < 0.80."""
        doc_type, confidence = self._classify_by_rules(source_path, frontmatter, content, project)
        if confidence >= _CONFIDENCE_THRESHOLD:
            return doc_type, confidence
        if self._client is None:
            # No LLM available (e.g. unit tests, dry runs): keep the rule guess
            # but report its honest low confidence rather than failing.
            logger.warning("classifier_no_llm_fallback", path=source_path, rule_guess=doc_type.value)
            return doc_type, confidence
        return await self._classify_by_llm(source_path, content)

    def _classify_by_rules(
        self, source_path: str, frontmatter: dict, content: str, project: Project
    ) -> tuple[DocType, float]:
        content_type = str(frontmatter.get("content_type", "")).lower()
        # Kubernetes frontmatter is authoritative — confidence 1.0.
        if content_type == "concept":
            return DocType.CONCEPT, 1.0
        if content_type == "task":
            return DocType.TASK, 1.0
        if content_type == "reference":
            return DocType.API_REF, 1.0
        if "/blog/" in source_path:
            return DocType.BLOG, 0.95
        if project == Project.ARGOCD and "/operator-manual/" in source_path:
            return DocType.RUNBOOK, 0.90
        # Dense code + terse lines = DSL/parameter reference (Helm templates etc.)
        if self._compute_code_block_ratio(content) > 0.6 and self._avg_sentence_len(content) < 8:
            return DocType.DSL_REF, 0.85
        return DocType.UNKNOWN, 0.0

    @staticmethod
    def _compute_code_block_ratio(content: str) -> float:
        """Fraction of characters inside fenced code blocks."""
        if not content:
            return 0.0
        code_chars = sum(len(m.group(0)) for m in _FENCED_BLOCK_RE.finditer(content))
        return code_chars / len(content)

    @staticmethod
    def _avg_sentence_len(content: str) -> float:
        """Mean words per sentence in prose (code blocks excluded)."""
        prose = _FENCED_BLOCK_RE.sub("", content)
        sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(prose) if s.strip()]
        if not sentences:
            return 0.0
        return sum(len(s.split()) for s in sentences) / len(sentences)

    async def _classify_by_llm(self, source_path: str, content: str) -> tuple[DocType, float]:
        # DISABLE_LLM_CLASSIFICATION=true skips the API call entirely — use when
        # the Anthropic key is unavailable or to avoid burning the monthly budget.
        if os.environ.get("DISABLE_LLM_CLASSIFICATION", "").lower() in ("1", "true", "yes"):
            return DocType.UNKNOWN, 0.5
        # 2000 chars is enough signal for classification; sending whole files
        # would multiply token cost ~10x for no accuracy gain.
        prompt = _LLM_CLASSIFY_PROMPT.format(path=source_path, content=content[:2000])
        assert self._client is not None
        try:
            response = await self._client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=10,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as exc:
            # API unavailable (rate limit, spend cap, network) — don't crash
            # ingestion; use UNKNOWN and chunker falls back to fixed_size.
            logger.warning("llm_classifier_api_error_using_unknown", path=source_path, error=str(exc))
            return DocType.UNKNOWN, 0.5
        label = response.content[0].text.strip().lower()
        try:
            doc_type = DocType(label)
        except ValueError:
            logger.warning("llm_classifier_bad_label", path=source_path, label=label)
            return DocType.UNKNOWN, 0.5
        # 0.85, not 1.0: the LLM is accurate but unverified — keeps the door
        # open for a future confidence-weighted re-classification pass.
        return doc_type, 0.85
