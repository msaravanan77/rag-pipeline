"""Ingestion CLI: corpus → parse → classify → chunk → embed → Qdrant.

Usage:
    uv run python scripts/ingest.py --project kubernetes --dry-run   # chunk report only
    uv run python scripts/ingest.py --project kubernetes             # full ingest
    uv run python scripts/ingest.py                                  # all projects
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from cncf_rag.chunking.evaluator import ChunkingEvaluator
from cncf_rag.chunking.factory import ChunkerFactory
from cncf_rag.embedding.cost_tracker import EmbeddingCostTracker
from cncf_rag.ingestion.corpus_loader import CorpusLoader
from cncf_rag.ingestion.models import Project

app = typer.Typer()
console = Console()


@app.command()
def main(
    project: str = typer.Option("all", help="kubernetes|helm|argocd|prometheus|all"),
    corpus_dir: Path = typer.Option(Path("corpus"), help="Directory holding project doc clones"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Parse + chunk only; no embedding/upserting"),
) -> None:
    load_dotenv()
    projects = list(Project) if project == "all" else [Project(project)]
    asyncio.run(_ingest(projects, corpus_dir, dry_run))


async def _ingest(projects: list[Project], corpus_dir: Path, dry_run: bool) -> None:
    start = time.perf_counter()
    tracker = EmbeddingCostTracker()

    embedder = None
    store = None
    classifier_client = None
    if not dry_run:
        from anthropic import AsyncAnthropic

        from cncf_rag.embedding.cohere_provider import CohereEmbeddingService
        from cncf_rag.vectorstore.qdrant_store import QdrantVectorStore

        embedder = CohereEmbeddingService(cost_tracker=tracker)
        store = QdrantVectorStore()
        await store.ensure_collection()
        classifier_client = AsyncAnthropic()

    from cncf_rag.ingestion.classifier import DocTypeClassifier

    loader = CorpusLoader(classifier=DocTypeClassifier(classifier_client))
    factory = ChunkerFactory(embedding_service=embedder)

    all_chunks = []
    doc_count = 0
    # Buffer chunks across documents and embed in FULL 96-text batches.
    # Per-document embedding (~6 chunks/call) would spend ~15x more API
    # calls — fatal on a Cohere trial key with a monthly call cap.
    buffer: list = []
    flush_threshold = 96 * 4

    async def flush() -> None:
        if not buffer or embedder is None or store is None:
            return
        vectors = await embedder.embed_documents([c.content for c in buffer])
        for chunk, vector in zip(buffer, vectors):
            chunk.embedding = vector
        await store.upsert_chunks(buffer)
        buffer.clear()

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as progress:
        task = progress.add_task("Ingesting...", total=None)
        async for document in loader.load_all(corpus_dir, projects):
            doc_count += 1
            chunks = await factory.chunk_document(document)
            all_chunks.extend(chunks)
            progress.update(task, description=f"{doc_count} docs → {len(all_chunks)} chunks ({document.metadata.source_path})")

            if not dry_run:
                buffer.extend(chunks)
                if len(buffer) >= flush_threshold:
                    await flush()
        if not dry_run:
            await flush()

    report = ChunkingEvaluator().evaluate(all_chunks)
    ChunkingEvaluator().print_report(report)

    elapsed = time.perf_counter() - start
    console.print(f"\n[bold]Docs ingested:[/bold] {doc_count}")
    console.print(f"[bold]Chunks created:[/bold] {len(all_chunks)}")
    if not dry_run:
        tracker.print_summary()
    console.print(f"[bold]Elapsed:[/bold] {elapsed:.1f}s")
    if dry_run:
        console.print("[yellow]DRY RUN — nothing embedded or indexed[/yellow]")


if __name__ == "__main__":
    app()
