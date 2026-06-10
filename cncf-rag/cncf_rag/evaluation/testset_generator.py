"""Test set construction (DECISIONS.md 7.1).

Tier 1: real StackOverflow questions citing kubernetes.io (StackExchange API).
Tier 2: hand-crafted version-edge cases (loaded from a curated JSON file).
Tier 3: LLM-generated synthetic questions, marked synthetic=True and reported
        separately — synthetic-only evaluation hides real-user failure modes.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import httpx
import structlog
from anthropic import AsyncAnthropic

from cncf_rag.evaluation.ragas_runner import EvalTestCase

logger = structlog.get_logger(__name__)

_STACKEXCHANGE_API = "https://api.stackexchange.com/2.3/search/advanced"

_SYNTH_PROMPT = """Based on this documentation excerpt, write one realistic question a platform \
engineer might ask, and the correct answer derived ONLY from the excerpt.
Return JSON: {{"question": "...", "answer": "..."}}. Return ONLY the JSON.

Excerpt:
{excerpt}"""


class TestsetGenerator:
    def __init__(self, anthropic_client: AsyncAnthropic | None = None) -> None:
        self._client = anthropic_client

    async def fetch_stackoverflow_tier1(self, max_questions: int = 100) -> list[EvalTestCase]:
        """Tier 1: accepted-answer k8s questions whose answers cite kubernetes.io."""
        cases: list[EvalTestCase] = []
        async with httpx.AsyncClient(timeout=30) as client:
            page = 1
            while len(cases) < max_questions and page <= 10:
                response = await client.get(
                    _STACKEXCHANGE_API,
                    params={
                        "order": "desc",
                        "sort": "votes",
                        "tagged": "kubernetes",
                        "accepted": "True",
                        "site": "stackoverflow",
                        "filter": "withbody",
                        "pagesize": 50,
                        "page": page,
                    },
                )
                response.raise_for_status()
                data = response.json()
                for item in data.get("items", []):
                    # Only questions whose accepted answer cites the official docs
                    # qualify as ground truth grounded in our corpus.
                    body = item.get("body", "")
                    if "kubernetes.io" not in body:
                        continue
                    cases.append(
                        EvalTestCase(
                            question=item["title"],
                            answer="",  # filled by running the pipeline
                            contexts=[],
                            ground_truth=body[:2000],
                            synthetic=False,
                        )
                    )
                if not data.get("has_more"):
                    break
                page += 1
        logger.info("tier1_fetched", count=len(cases))
        return cases[:max_questions]

    def load_tier2_version_cases(self, path: Path) -> list[EvalTestCase]:
        """Tier 2: hand-curated version-specific edge cases from JSON."""
        raw = json.loads(path.read_text())
        return [
            EvalTestCase(
                question=item["question"],
                answer="",
                contexts=[],
                ground_truth=item["ground_truth"],
                expected_version=item["expected_version"],
                synthetic=False,
            )
            for item in raw
        ]

    async def generate_tier3_synthetic(
        self, excerpts: list[str], max_cases: int = 50
    ) -> list[EvalTestCase]:
        """Tier 3: LLM-generated questions, explicitly flagged synthetic."""
        client = self._client or AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        cases: list[EvalTestCase] = []
        for excerpt in excerpts[:max_cases]:
            response = await client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=500,
                messages=[{"role": "user", "content": _SYNTH_PROMPT.format(excerpt=excerpt[:3000])}],
            )
            try:
                parsed = json.loads(response.content[0].text.strip())
                cases.append(
                    EvalTestCase(
                        question=parsed["question"],
                        answer="",
                        contexts=[excerpt],
                        ground_truth=parsed["answer"],
                        synthetic=True,
                    )
                )
            except (json.JSONDecodeError, KeyError):
                logger.warning("tier3_generation_skipped_bad_json")
        return cases
