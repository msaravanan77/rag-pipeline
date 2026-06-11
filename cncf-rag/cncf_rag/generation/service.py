"""Answer generation — Anthropic (default) or OpenAI (GENERATION_PROVIDER=openai).

Switch providers via environment:
  GENERATION_PROVIDER=openai          → uses OPENAI_API_KEY
  OPENAI_GENERATION_MODEL=gpt-4o      → override model (default gpt-4o-mini)
  GENERATION_PROVIDER=anthropic       → uses ANTHROPIC_API_KEY (default)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import structlog

from cncf_rag.generation.prompt_builder import PromptBuilder
from cncf_rag.vectorstore.qdrant_store import ScoredChunk

logger = structlog.get_logger(__name__)

# PRICING WARNING: hardcoded from provider pricing pages, June 2026.
_ANTHROPIC_INPUT_PER_MTOK = 3.00    # claude-sonnet-4-5
_ANTHROPIC_OUTPUT_PER_MTOK = 15.00
_OPENAI_INPUT_PER_MTOK = 0.15       # gpt-4o-mini
_OPENAI_OUTPUT_PER_MTOK = 0.60

_MIN_SIMILARITY = 0.55


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
    def __init__(self) -> None:
        provider = os.environ.get("GENERATION_PROVIDER", "anthropic").lower()
        self._prompt_builder = PromptBuilder()

        if provider == "openai":
            from openai import AsyncOpenAI
            self._provider = "openai"
            self._openai = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
            self.model = os.environ.get("OPENAI_GENERATION_MODEL", "gpt-4o-mini")
            self._anthropic = None
        else:
            from anthropic import AsyncAnthropic
            self._provider = "anthropic"
            self._anthropic = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
            self.model = "claude-sonnet-4-5"
            self._openai = None

    async def generate(self, query: str, chunks: list[ScoredChunk]) -> GenerationResult:
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

        if self._provider == "openai":
            return await self._generate_openai(query, chunks)
        return await self._generate_anthropic(query, chunks)

    async def _generate_anthropic(self, query: str, chunks: list[ScoredChunk]) -> GenerationResult:
        system_prompt, user_message = self._prompt_builder.build(query, chunks)
        response = await self._anthropic.messages.create(
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
            tokens_in * _ANTHROPIC_INPUT_PER_MTOK / 1_000_000
            + tokens_out * _ANTHROPIC_OUTPUT_PER_MTOK / 1_000_000
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

    async def _generate_openai(self, query: str, chunks: list[ScoredChunk]) -> GenerationResult:
        system_prompt, user_message = self._prompt_builder.build(query, chunks)
        response = await self._openai.chat.completions.create(
            model=self.model,
            response_format={"type": "json_object"},
            max_tokens=4096,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
        raw = response.choices[0].message.content or ""
        parsed = self._parse_response(raw)
        tokens_in = response.usage.prompt_tokens
        tokens_out = response.usage.completion_tokens
        # Use gpt-4o pricing if model is gpt-4o, else gpt-4o-mini
        if "gpt-4o" in self.model and "mini" not in self.model:
            in_price, out_price = 2.50, 10.00
        else:
            in_price, out_price = _OPENAI_INPUT_PER_MTOK, _OPENAI_OUTPUT_PER_MTOK
        cost = (tokens_in * in_price + tokens_out * out_price) / 1_000_000
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
