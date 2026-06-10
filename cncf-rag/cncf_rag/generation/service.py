"""Answer generation via claude-sonnet-4-5 (DECISIONS.md 6.1)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import structlog
from anthropic import AsyncAnthropic

from cncf_rag.generation.prompt_builder import PromptBuilder
from cncf_rag.vectorstore.qdrant_store import ScoredChunk

logger = structlog.get_logger(__name__)

# PRICING WARNING: hardcoded from anthropic.com pricing, June 2026 — prices
# change without notice; these constants exist for cost VISIBILITY, not billing.
_INPUT_PRICE_PER_MTOK = 3.00   # claude-sonnet-4-5 input USD per million tokens
_OUTPUT_PRICE_PER_MTOK = 15.00  # claude-sonnet-4-5 output

# Below this top similarity score, retrieval found nothing meaningful —
# skip the LLM call entirely (saves money AND prevents the model from
# being tempted to answer from weak context).
_MIN_SIMILARITY = 0.70


class GenerationParseError(Exception):
    """The model returned non-JSON or schema-violating output."""


@dataclass
class GenerationResult:
    answer: str
    citations: list[dict] = field(default_factory=list)
    confidence: float = 0.0
    cannot_answer: bool = False
    version_warning: str | None = None
    tokens_input: int = 0
    tokens_output: int = 0
    cost_usd: float = 0.0
    model_used: str = ""


class GenerationService:
    def __init__(
        self,
        anthropic_client: AsyncAnthropic | None = None,
        model: str = "claude-sonnet-4-5",
    ) -> None:
        self._client = anthropic_client or AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self._prompt_builder = PromptBuilder()
        self.model = model

    async def generate(self, query: str, chunks: list[ScoredChunk]) -> GenerationResult:
        # Short-circuit: no chunks, or best score too weak for grounded answering.
        if not chunks or max(c.score for c in chunks) < _MIN_SIMILARITY:
            logger.info(
                "generation_short_circuit",
                query=query,
                top_score=max((c.score for c in chunks), default=0.0),
            )
            return GenerationResult(
                answer="",
                cannot_answer=True,
                confidence=0.0,
                model_used="none (similarity short-circuit)",
            )

        system_prompt, user_message = self._prompt_builder.build(query, chunks)
        response = await self._client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        raw = response.content[0].text
        parsed = self._parse_response(raw)

        tokens_in = response.usage.input_tokens
        tokens_out = response.usage.output_tokens
        cost = (
            tokens_in * _INPUT_PRICE_PER_MTOK / 1_000_000
            + tokens_out * _OUTPUT_PRICE_PER_MTOK / 1_000_000
        )
        return GenerationResult(
            answer=parsed.get("answer", ""),
            citations=parsed.get("citations", []),
            confidence=float(parsed.get("confidence", 0.0)),
            cannot_answer=bool(parsed.get("cannot_answer", False)),
            version_warning=parsed.get("version_warning"),
            tokens_input=tokens_in,
            tokens_output=tokens_out,
            cost_usd=round(cost, 6),
            model_used=self.model,
        )

    @staticmethod
    def _parse_response(raw: str) -> dict:
        """Parse and schema-validate the model's JSON response.

        Strips Markdown fences first — models occasionally wrap JSON despite
        instructions, and that is recoverable; missing required fields is not.
        """
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text
            text = text.rsplit("```", 1)[0]
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            logger.error("generation_parse_failed", raw_response=raw)
            raise GenerationParseError(f"Model returned non-JSON output: {exc}") from exc
        if not isinstance(parsed, dict) or "answer" not in parsed:
            logger.error("generation_schema_invalid", raw_response=raw)
            raise GenerationParseError("Model JSON missing required 'answer' field")
        return parsed
