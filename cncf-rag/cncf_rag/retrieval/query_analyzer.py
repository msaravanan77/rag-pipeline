"""Query classification for retrieval routing (DECISIONS.md 5.1).

Rules first — instant and free for ~80% of queries; the rule outcomes are
deterministic and unit-testable, which LLM routing is not.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum


class QueryType(str, Enum):
    FACTUAL = "factual"
    PROCEDURAL = "procedural"
    CONCEPTUAL = "conceptual"
    VERSION_SPECIFIC = "version_specific"
    CROSS_PROJECT = "cross_project"
    EXPLORATORY = "exploratory"


@dataclass
class QueryAnalysis:
    query: str
    query_type: QueryType
    detected_projects: list[str] = field(default_factory=list)
    detected_version: str | None = None
    confidence: float = 1.0
    used_llm_fallback: bool = False


_VERSION_RE = re.compile(r"\bv?\d+\.\d+(\.\d+)?\b")

# Project aliases as users actually type them, not just canonical names.
_PROJECT_ALIASES: dict[str, list[str]] = {
    "kubernetes": ["kubernetes", "k8s", "kubectl", "kubelet", "kube-"],
    "helm": ["helm", "chart museum", "helm chart"],
    "argocd": ["argo cd", "argocd", "argo-cd"],
    "prometheus": ["prometheus", "promql", "alertmanager"],
}

_PROCEDURAL_RE = re.compile(
    r"^\s*(how (do|can|to|should) |steps to |configure |install |set up |setup |deploy |create |enable |upgrade )",
    re.IGNORECASE,
)
_CONCEPTUAL_RE = re.compile(
    r"^\s*(what (is|are) |why (does|is|do) |explain |when should |difference between |compare )",
    re.IGNORECASE,
)


class QueryAnalyzer:
    def analyze(self, query: str) -> QueryAnalysis:
        lowered = query.lower()
        projects = [
            name
            for name, aliases in _PROJECT_ALIASES.items()
            if any(alias in lowered for alias in aliases)
        ]
        version_match = _VERSION_RE.search(query)
        version = version_match.group(0) if version_match else None
        if version and not version.startswith("v"):
            version = "v" + version

        # Rule precedence (DECISIONS.md 5.1): version anchors beat everything —
        # answering from the wrong version is the corpus's worst failure mode.
        if version is not None:
            qtype, confidence = QueryType.VERSION_SPECIFIC, 0.95
        elif len(projects) >= 2:
            qtype, confidence = QueryType.CROSS_PROJECT, 0.9
        elif _PROCEDURAL_RE.match(query):
            qtype, confidence = QueryType.PROCEDURAL, 0.85
        elif _CONCEPTUAL_RE.match(query):
            qtype, confidence = QueryType.CONCEPTUAL, 0.85
        elif len(query.split()) <= 3 and "?" not in query:
            # "service mesh", "operators" — broad single terms with no question
            # structure: the user is exploring, diversity beats precision.
            qtype, confidence = QueryType.EXPLORATORY, 0.7
        else:
            qtype, confidence = QueryType.FACTUAL, 0.6

        return QueryAnalysis(
            query=query,
            query_type=qtype,
            detected_projects=projects,
            detected_version=version,
            confidence=confidence,
        )
