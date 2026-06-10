"""Evaluation CLI: run RAGAS + version staleness against a test set.

Exits 1 if any metric misses its target — wire into CI to block regressions.

Usage:
    uv run python scripts/run_eval.py
    uv run python scripts/run_eval.py --test-set tests/fixtures/test_set.json --output-json report.json
"""

from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import asdict
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console

app = typer.Typer()
console = Console()


@app.command()
def main(
    test_set: Path = typer.Option(Path("tests/fixtures/test_set.json"), "--test-set"),
    output_json: Path | None = typer.Option(None, "--output-json"),
) -> None:
    load_dotenv()
    exit_code = asyncio.run(_run(test_set, output_json))
    raise typer.Exit(code=exit_code)


async def _run(test_set: Path, output_json: Path | None) -> int:
    from cncf_rag.evaluation.ragas_runner import EvalTestCase, RAGASRunner

    if not test_set.exists():
        console.print(f"[red]Test set not found: {test_set}[/red]")
        console.print("Generate one via cncf_rag.evaluation.testset_generator, or start with")
        console.print("the Tier 2 cases in tests/fixtures/version_cases.json.")
        return 1

    raw = json.loads(test_set.read_text())
    cases = [EvalTestCase(**item) for item in raw]
    # Synthetic cases are reported separately (DECISIONS.md 7.1) — synthetic-only
    # scores systematically overestimate quality on real user phrasing.
    real = [c for c in cases if not c.synthetic]
    synthetic = [c for c in cases if c.synthetic]

    runner = RAGASRunner()
    console.rule("Real test cases")
    report = await runner.evaluate(real if real else cases)
    runner.print_scorecard(report)

    if synthetic:
        console.rule("Synthetic test cases (reported separately)")
        synth_report = await runner.evaluate(synthetic)
        runner.print_scorecard(synth_report)

    if output_json:
        output_json.write_text(json.dumps(asdict(report), indent=2))
        console.print(f"Report saved to {output_json}")

    return 0 if report.all_passed else 1


if __name__ == "__main__":
    app()
