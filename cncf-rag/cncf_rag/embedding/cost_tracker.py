"""Embedding cost accounting.

Tracks tokens per batch and computes spend from a hardcoded pricing table.
On a Cohere trial key the dollar cost is $0, but the tracker still matters:
it shows what the same run WOULD cost on a paid key — the number you need
before deciding to upgrade.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rich.console import Console
from rich.table import Table

# PRICING WARNING: hardcoded from public pricing pages, June 2026. Providers
# change prices without notice — verify before trusting estimates for budgeting.
_PRICE_PER_MILLION_TOKENS: dict[str, float] = {
    "embed-english-v3.0": 0.10,
    "embed-multilingual-v3.0": 0.10,
    "text-embedding-3-small": 0.02,
    "text-embedding-3-large": 0.13,
}


@dataclass
class EmbeddingCostTracker:
    total_tokens: int = 0
    total_batches: int = 0
    tokens_by_model: dict[str, int] = field(default_factory=dict)

    def record_batch(self, model: str, tokens: int) -> None:
        self.total_tokens += tokens
        self.total_batches += 1
        self.tokens_by_model[model] = self.tokens_by_model.get(model, 0) + tokens

    def total_cost_usd(self) -> float:
        return sum(
            tokens * _PRICE_PER_MILLION_TOKENS.get(model, 0.0) / 1_000_000
            for model, tokens in self.tokens_by_model.items()
        )

    def print_summary(self) -> None:
        console = Console()
        table = Table(title="Embedding Cost Summary")
        table.add_column("Model")
        table.add_column("Tokens", justify="right")
        table.add_column("Cost (paid-tier USD)", justify="right")
        for model, tokens in self.tokens_by_model.items():
            cost = tokens * _PRICE_PER_MILLION_TOKENS.get(model, 0.0) / 1_000_000
            table.add_row(model, f"{tokens:,}", f"${cost:.4f}")
        table.add_row("TOTAL", f"{self.total_tokens:,}", f"${self.total_cost_usd():.4f}")
        console.print(table)
        console.print("[dim]OpenAI paid: billed at rates above. Cohere trial reranking: $0.00.[/dim]")
