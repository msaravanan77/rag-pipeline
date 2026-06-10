"""Builds Qdrant payload filters from query analysis (DECISIONS.md 5.2)."""

from __future__ import annotations

from cncf_rag.retrieval.query_analyzer import QueryAnalysis, QueryType


class FilterBuilder:
    def build_from_analysis(self, analysis: QueryAnalysis) -> dict:
        filters: dict = {}
        # Single detected project narrows the search space; with 2+ the
        # multi-query strategy handles per-project filtering itself.
        if len(analysis.detected_projects) == 1:
            filters["project"] = analysis.detected_projects[0]

        match analysis.query_type:
            case QueryType.PROCEDURAL:
                # "how do I X" answered from a concept doc returns an explanation,
                # not steps — force task docs (highest-ROI filter, see 5.2).
                filters["doc_type"] = "task"
            case QueryType.CONCEPTUAL:
                filters["doc_type"] = "concept"
            case QueryType.VERSION_SPECIFIC if analysis.detected_version:
                filters["version_tag"] = analysis.detected_version
            case _:
                pass
        return filters
