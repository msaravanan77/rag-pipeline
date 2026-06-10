"""RAGAS evaluation runner (DECISIONS.md 7.1): four standard metrics + version staleness."""

from __future__ import annotations

from dataclasses import dataclass

import structlog
from rich.console import Console
from rich.table import Table

from cncf_rag.evaluation.version_staleness import VersionedTestCase, VersionStalenessEvaluator

logger = structlog.get_logger(__name__)

# Targets from configs/evaluation.yaml — duplicated as code constants so the
# report object is self-describing without a config dependency.
TARGETS: dict[str, float] = {
    "faithfulness": 0.85,
    "answer_relevancy": 0.80,
    "context_precision": 0.75,
    "context_recall": 0.70,
}
VERSION_STALENESS_MAX = 0.10


@dataclass
class EvalTestCase:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    expected_version: str | None = None
    answer_citations: list[dict] | None = None
    synthetic: bool = False


@dataclass
class EvaluationReport:
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    version_staleness: float
    passed: dict[str, bool]

    @property
    def all_passed(self) -> bool:
        return all(self.passed.values())


class RAGASRunner:
    async def evaluate(self, test_cases: list[EvalTestCase]) -> EvaluationReport:
        # Import inside the method: ragas pulls in heavy deps (datasets, langchain
        # internals) that should not load for users who never run evaluation.
        from datasets import Dataset
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

        dataset = Dataset.from_dict(
            {
                "question": [c.question for c in test_cases],
                "answer": [c.answer for c in test_cases],
                "contexts": [c.contexts for c in test_cases],
                "ground_truth": [c.ground_truth for c in test_cases],
            }
        )
        ragas_result = ragas_evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        )
        scores = {
            "faithfulness": float(ragas_result["faithfulness"]),
            "answer_relevancy": float(ragas_result["answer_relevancy"]),
            "context_precision": float(ragas_result["context_precision"]),
            "context_recall": float(ragas_result["context_recall"]),
        }

        versioned = [
            VersionedTestCase(
                query=c.question,
                expected_version=c.expected_version,
                answer_citations=c.answer_citations or [],
            )
            for c in test_cases
            if c.expected_version
        ]
        staleness = VersionStalenessEvaluator().evaluate(versioned)

        passed = {name: scores[name] >= target for name, target in TARGETS.items()}
        passed["version_staleness"] = staleness <= VERSION_STALENESS_MAX
        return EvaluationReport(
            faithfulness=scores["faithfulness"],
            answer_relevancy=scores["answer_relevancy"],
            context_precision=scores["context_precision"],
            context_recall=scores["context_recall"],
            version_staleness=staleness,
            passed=passed,
        )

    def print_scorecard(self, report: EvaluationReport) -> None:
        console = Console()
        table = Table(title="RAG Evaluation Scorecard")
        table.add_column("Metric")
        table.add_column("Score", justify="right")
        table.add_column("Target", justify="right")
        table.add_column("Status", justify="center")

        rows = [
            ("Faithfulness", report.faithfulness, f"≥{TARGETS['faithfulness']}", "faithfulness"),
            ("Answer Relevancy", report.answer_relevancy, f"≥{TARGETS['answer_relevancy']}", "answer_relevancy"),
            ("Context Precision", report.context_precision, f"≥{TARGETS['context_precision']}", "context_precision"),
            ("Context Recall", report.context_recall, f"≥{TARGETS['context_recall']}", "context_recall"),
            ("Version Staleness", report.version_staleness, f"≤{VERSION_STALENESS_MAX}", "version_staleness"),
        ]
        for label, score, target, key in rows:
            ok = report.passed[key]
            status = "[green]PASS[/green]" if ok else "[red]FAIL[/red]"
            color = "green" if ok else "red"
            table.add_row(label, f"[{color}]{score:.3f}[/{color}]", target, status)
        console.print(table)
