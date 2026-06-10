"""Version staleness — the corpus-specific metric no standard benchmark covers.

A system can score 0.85 faithfulness while citing v1.21 docs for a v1.29
question — faithful to the WRONG version. For Kubernetes users that means
implementing removed APIs. This evaluator counts exactly that failure.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import structlog

logger = structlog.get_logger(__name__)

_VERSION_RE = re.compile(r"v?(\d+)\.(\d+)")


def _parse_version(tag: str | None) -> tuple[int, int] | None:
    if not tag:
        return None
    match = _VERSION_RE.search(tag)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


@dataclass
class VersionedTestCase:
    query: str
    expected_version: str
    answer_citations: list[dict]  # each: {"version": "v1.29", ...}


class VersionStalenessEvaluator:
    def evaluate(self, test_cases: list[VersionedTestCase]) -> float:
        """Return the stale fraction: cases where ANY citation predates expected_version.

        ANY (not all): a single stale citation can inject a removed API into the
        answer, so one is enough to mark the whole case stale.
        """
        if not test_cases:
            return 0.0
        stale = 0
        for case in test_cases:
            expected = _parse_version(case.expected_version)
            if expected is None:
                continue
            cited = [_parse_version(c.get("version")) for c in case.answer_citations]
            if any(v is not None and v < expected for v in cited):
                stale += 1
                logger.info("stale_case", query=case.query, expected=case.expected_version)
        return stale / len(test_cases)
