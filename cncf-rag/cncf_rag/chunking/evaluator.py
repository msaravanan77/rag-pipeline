"""Chunking quality metrics — the gate before spending embedding quota.

Run via `scripts/ingest.py --dry-run`: if split_code_block_count > 0 or
p95 tokens balloon past 1000, fix chunking before embedding anything.
"""

from __future__ import annotations

import statistics
from collections import Counter

from rich.console import Console
from rich.table import Table

from cncf_rag.chunking.models import Chunk, ChunkingReport


class ChunkingEvaluator:
    def evaluate(self, chunks: list[Chunk]) -> ChunkingReport:
        if not chunks:
            return ChunkingReport(0, 0, 0.0, 0, 0, 0, 0, 0, {})
        token_counts = sorted(c.token_count for c in chunks)
        n = len(token_counts)
        return ChunkingReport(
            total_chunks=n,
            total_tokens=sum(token_counts),
            mean_token_count=statistics.mean(token_counts),
            p50_token_count=token_counts[n // 2],
            p95_token_count=token_counts[min(int(n * 0.95), n - 1)],
            max_token_count=token_counts[-1],
            chunks_under_40_tokens=sum(1 for t in token_counts if t < 40),
            split_code_block_count=sum(1 for c in chunks if c.content.count("```") % 2 != 0),
            chunks_by_strategy=dict(Counter(c.chunking_strategy for c in chunks)),
        )

    def print_report(self, report: ChunkingReport) -> None:
        console = Console()
        table = Table(title="Chunking Report")
        table.add_column("Metric")
        table.add_column("Value", justify="right")
        table.add_row("Total chunks", str(report.total_chunks))
        table.add_row("Total tokens", f"{report.total_tokens:,}")
        table.add_row("Mean tokens/chunk", f"{report.mean_token_count:.1f}")
        table.add_row("p50 tokens", str(report.p50_token_count))
        table.add_row("p95 tokens", str(report.p95_token_count))
        table.add_row("Max tokens", str(report.max_token_count))
        table.add_row("Chunks < 40 tokens", str(report.chunks_under_40_tokens))
        split_style = "red" if report.split_code_block_count else "green"
        table.add_row(
            "Split code blocks", f"[{split_style}]{report.split_code_block_count}[/{split_style}]"
        )
        for strategy, count in sorted(report.chunks_by_strategy.items()):
            table.add_row(f"  via {strategy}", str(count))
        console.print(table)
