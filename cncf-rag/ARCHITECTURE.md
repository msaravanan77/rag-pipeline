# ARCHITECTURE.md — CNCF RAG Pipeline

## What This System Is

A RAG-powered question-answering system over the official documentation of
**Kubernetes, Helm, Argo CD, and Prometheus**. Users ask natural-language questions
and receive accurate, cited answers grounded in source documentation.

## Why This Corpus

The CNCF documentation was chosen over Wikipedia, arXiv, or Django docs for five reasons:

1. **Six structurally distinct document types in one corpus** (concept guides, how-tos,
   API reference, DSL reference, blog/changelog, operational runbooks) — forces real
   per-type chunking decisions instead of one-size-fits-all.
2. **Versioned content across Kubernetes v1.24–v1.34+** — surfaces the version staleness
   problem absent from generic RAG tutorials.
3. **CC BY 4.0 / Apache 2.0 licensed** — freely redistributable corpus and test set.
4. **Real community Q&A ground truth** from StackOverflow and GitHub Issues.
5. **Domain recognition** — every cloud engineer already uses these tools.

## Data Flow

```
                      INGESTION (batch, scripts/ingest.py)
corpus/ (git clones) ──► CorpusLoader ──► MarkdownParser ──► DocTypeClassifier
    │ language filter:                    (SHA-256 skip)       │
    │ - kubernetes: skip /content/<lang>/ except /en/          ├─ rule-based (free)
    │ - argocd: skip everything except docs/ subdir            └─ LLM fallback (Haiku)
    │                                                             disabled by default
    ▼
 Preprocessor ──► ChunkerFactory (strategy per DocType)
                         │
                         ▼
          OpenAI text-embedding-3-small (1536 dims, 1M TPM)
                         │
                         ▼
               Qdrant v1.18.2 (HNSW, payload-indexed)

                      QUERY TIME (online, api/app.py)
user query ──► QueryAnalyzer (rules) ──► FilterBuilder
                                             │
                               user's explicit project_filter/version_filter
                               merged here (override analyzer inferences)
                                             │
                                      Strategy Router
                              ┌──────────────┼─────────────┐
                              ▼              ▼             ▼
                        filtered top-k      MMR      multi-query + RRF
                              └─────────────┴─────────────┘
                                             │
                          Cohere rerank (EXPLORATORY / CROSS_PROJECT only)
                                             │
                                             ▼
                  PromptBuilder ──► GPT-4o-mini (or claude-sonnet-4-5) ──► cited JSON
```

## Technology Choices (full rationale in DECISIONS.md)

| Layer | Current choice | Original design | Decision |
|-------|---------------|-----------------|---------|
| Markdown parsing | marko + python-frontmatter | Same | 1.1 |
| Doc classification | Rules + Haiku fallback (disabled by default) | Same | 1.2 |
| Incremental ingest | SHA-256 + SQLite | Same | 1.3 |
| Chunking | Per-DocType strategies | Same | 2.1 |
| **Embeddings** | **OpenAI text-embedding-3-small (1536 dims)** | Cohere embed-english-v3.0 | **3.1** |
| Vector store | Qdrant v1.18.2 (self-hosted) | Qdrant | 4.1 |
| HNSW params | m=16, ef_construct=100, ef=50 | Same | 4.2 |
| Query routing | Rule-based analyzer | Same | 5.1 |
| Retrieval | Strategy per query type; user filters server-side | Same (bug fix: was post-filter) | 5.2 |
| Reranking | Cohere rerank-english-v3.0, selective | Same | 5.3 |
| **Generation** | **GPT-4o-mini (default) or claude-sonnet-4-5** | claude-sonnet-4-5 only | **6.1** |
| Context assembly | Score-desc, 50k token cap | Same | 6.2 |
| Evaluation | RAGAS + version staleness | Same | 7.1 |

> **Why embeddings changed from Cohere to OpenAI:** Cohere trial key is capped at
> 100k tokens/minute. OpenAI paid tier is 1M tokens/minute — 10× faster for ingestion
> at effectively the same cost (~$0.02/1M tokens). Decision 3.1 in DECISIONS.md
> still explains the original reasoning; the actual tradeoff table is updated there.

> **Why generation defaults to OpenAI:** The Anthropic API has a monthly spending cap
> on trial accounts. GPT-4o-mini is the safer default until the cap is lifted.
> Switch via `GENERATION_PROVIDER=anthropic` env var — zero code change required.

## Infrastructure

Single AWS EC2 instance (t3.xlarge) running both Qdrant and the FastAPI application.
S3 stores corpus tarballs + Qdrant index snapshot. Terraform provisions everything.

```
Internet ──► EC2 t3.xlarge (your IP only, ports 22 + 8000)
              ├── Qdrant v1.18.2  localhost:6333 (never exposed externally)
              ├── FastAPI         0.0.0.0:8000
              └── systemd manages both services
              └── IAM instance profile → S3 corpus bucket only (least privilege)

S3 bucket (cncf-rag-corpus-msaravanan77):
  bootstrap/code.tar.gz           application code
  bootstrap/cncf-rag.env          API keys (bucket is private + SSE-S3)
  backups/qdrant-storage.tar.gz   full vector index snapshot (169 MB)
  raw/{kubernetes,helm,argocd,prometheus}.tar.gz  raw corpora
  ingest-index/.ingest_index.db   checksum index for incremental re-ingest
```

**Cold start time**: ~3 minutes from `terraform apply` to first answered query.
Qdrant index is restored from S3 — no re-embedding, no API calls.
