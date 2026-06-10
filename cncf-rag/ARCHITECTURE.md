# ARCHITECTURE.md — CNCF RAG Pipeline

## What This System Is

A RAG-powered question-answering system over the official documentation of
**Kubernetes, Helm, Argo CD, and Prometheus**. Users — platform engineers, SREs,
DevOps practitioners — ask natural-language questions and receive accurate, cited
answers grounded in source documentation.

## Why This Corpus

The CNCF documentation was chosen over Wikipedia, arXiv papers, Django docs, and
API documentation for five specific reasons:

1. **Six structurally distinct document types in one corpus** (concept guides,
   how-tos, API reference, DSL reference, blog/changelog, operational runbooks) —
   forces real per-type chunking decisions instead of one-size-fits-all.
2. **Versioned content across Kubernetes v1.24–v1.34+** — surfaces the version
   staleness problem absent from generic RAG tutorials.
3. **CC BY 4.0 / Apache 2.0 licensed** — freely redistributable corpus and test set.
4. **Real community Q&A ground truth** from StackOverflow and GitHub Issues —
   no synthetic-only evaluation.
5. **Domain recognition** — every AI/ML engineer already uses these tools, giving
   the project immediate relevance.

## Data Flow

```
                          INGESTION (batch, scripts/ingest.py)
corpus/ (git clones) ──► CorpusLoader ──► MarkdownParser ──► DocTypeClassifier
                              │ (SHA-256 skip via SQLite)         │
                              ▼                                   ▼
                         Preprocessor ──► ChunkerFactory (strategy per DocType)
                                                │
                                                ▼
                              Cohere embed-english-v3.0 (input_type=search_document)
                                                │
                                                ▼
                                      Qdrant (HNSW, payload-indexed)

                          QUERY TIME (online, api/app.py)
user query ──► QueryAnalyzer (rules) ──► FilterBuilder ──► Strategy Router
                                                                │
                              ┌─────────────────────────────────┤
                              ▼                ▼                ▼
                        filtered top-k        MMR          multi-query + RRF
                              └────────────────┴────────────────┘
                                                │
                                  Cohere Rerank (EXPLORATORY / CROSS_PROJECT only)
                                                │
                                                ▼
                          PromptBuilder ──► claude-sonnet-4-5 ──► cited JSON answer
```

## Technology Choices (full rationale in DECISIONS.md)

| Layer | Choice | Why (one line) | Decision |
|-------|--------|----------------|----------|
| Markdown parsing | marko + python-frontmatter | Typed AST heading nodes required for heading-aware chunking | 1.1 |
| Doc classification | Rules + LLM fallback | Rules free and instant for 80%; LLM only for ambiguous files | 1.2 |
| Incremental ingest | SHA-256 + SQLite | Source-agnostic change detection, zero ops overhead | 1.3 |
| Chunking | Per-DocType strategies | Chunking determines retrieval unit; one size fails 3 of 6 types | 2.1 |
| Embeddings | Cohere embed-english-v3.0 | Free trial tier, higher MTEB than OpenAI small, one fewer provider | 3.1 |
| Vector store | Qdrant (self-hosted) | Server-side payload filters; trivial single-binary deployment | 4.1 |
| ANN params | HNSW m=16/ef_construct=100/ef=50 | Defaults already >0.95 recall at 200k vectors | 4.2 |
| Query routing | Rule-based analyzer | LLM routing adds 200–400ms for marginal gain on 80% of queries | 5.1 |
| Retrieval | Strategy per query type | doc_type filters are the highest-ROI improvement for this corpus | 5.2 |
| Reranking | Cohere rerank, selective | Only where initial ranking is noisy; saves latency and quota | 5.3 |
| Generation | claude-sonnet-4-5 | Faithfulness-first; strongest context-only instruction following | 6.1 |
| Context assembly | Score-desc, 50k cap | "Lost in the middle" mitigation | 6.2 |
| Evaluation | RAGAS + version staleness | Standard metrics + the corpus-specific failure mode | 7.1 |

## Infrastructure

Single AWS EC2 instance (t3.xlarge) running both Qdrant and the FastAPI application.
S3 for corpus storage. Terraform for provisioning. No container orchestration —
minimal infrastructure, maximum RAG learning. See [infra/README.md](infra/README.md).

```
Internet ──► EC2 t3.xlarge (your IP only, ports 22 + 8000)
              ├── Qdrant      localhost:6333 (never exposed externally)
              ├── FastAPI     0.0.0.0:8000
              └── systemd manages both
              └── IAM instance profile → S3 corpus bucket only (least privilege)
```
