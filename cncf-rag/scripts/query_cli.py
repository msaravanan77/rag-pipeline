"""Query CLI: ask the pipeline a question from the terminal.

Usage:
    uv run python scripts/query_cli.py "What is a Kubernetes Pod?"
    uv run python scripts/query_cli.py "operators" --show-chunks --strategy mmr
"""

from __future__ import annotations

import asyncio

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

app = typer.Typer()
console = Console()


@app.command()
def main(
    query: str = typer.Argument(..., help="Natural language question"),
    show_chunks: bool = typer.Option(False, "--show-chunks", help="Print retrieved chunks with scores"),
    strategy: str | None = typer.Option(None, "--strategy", help="Override: filtered_topk|mmr|multi_query"),
) -> None:
    load_dotenv()
    asyncio.run(_run(query, show_chunks, strategy))


async def _run(query: str, show_chunks: bool, strategy: str | None) -> None:
    from cncf_rag.embedding.cohere_provider import CohereEmbeddingService
    from cncf_rag.generation.service import GenerationService
    from cncf_rag.retrieval.pipeline import RetrievalPipeline
    from cncf_rag.retrieval.reranker import CohereReranker
    from cncf_rag.vectorstore.qdrant_store import QdrantVectorStore

    embedder = CohereEmbeddingService()
    store = QdrantVectorStore()
    pipeline = RetrievalPipeline(embedder, store, reranker=CohereReranker())
    generator = GenerationService()

    result = await pipeline.retrieve(query, strategy_override=strategy)
    console.print(
        f"[dim]query_type={result.analysis.query_type.value} "
        f"strategy={result.strategy_used} reranked={result.reranked} "
        f"retrieval={result.latency_ms:.0f}ms[/dim]\n"
    )

    if show_chunks:
        for i, chunk in enumerate(result.chunks, 1):
            console.print(
                Panel(
                    chunk.content[:600] + ("…" if len(chunk.content) > 600 else ""),
                    title=f"#{i} score={chunk.score:.3f} {chunk.payload.get('source_path', '?')}",
                )
            )

    generation = await generator.generate(query, result.chunks)
    if generation.cannot_answer:
        console.print("[red bold]Cannot answer from the indexed documentation.[/red bold]")
        if generation.answer:
            console.print(generation.answer)
        return

    console.print(Panel(generation.answer, title="Answer"))
    if generation.version_warning:
        console.print(f"[yellow]⚠ Version warning: {generation.version_warning}[/yellow]")
    if generation.citations:
        console.print("[bold]Citations:[/bold]")
        for citation in generation.citations:
            console.print(f"  • {citation}")
    console.print(
        f"[dim]confidence={generation.confidence:.2f} "
        f"tokens={generation.tokens_input}+{generation.tokens_output} "
        f"cost=${generation.cost_usd:.4f} model={generation.model_used}[/dim]"
    )


if __name__ == "__main__":
    app()
