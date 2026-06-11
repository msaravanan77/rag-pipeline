# CNCF RAG — RAG Pipeline over CNCF Documentation

A RAG-powered question-answering system over the official docs of
**Kubernetes, Helm, Argo CD, and Prometheus**, with cited answers.

- **Why every technology was chosen:** [DECISIONS.md](DECISIONS.md)
- **System design and data flow:** [ARCHITECTURE.md](ARCHITECTURE.md)
- **AWS deployment, stop/start, costs:** [infra/README.md](infra/README.md)
- **Day-to-day usage after deployment:** [USAGE.md](USAGE.md)

## Stack at a glance

| Layer | Technology | Notes |
|---|---|---|
| Embeddings | OpenAI `text-embedding-3-small` (1536 dims) | 1M TPM paid tier |
| Reranking | Cohere `rerank-english-v3.0` | Per-query; trial key sufficient |
| Generation | OpenAI `gpt-4o-mini` (default) | Swap to `claude-sonnet-4-5` via env var |
| Vector store | Qdrant v1.18.2 (self-hosted, single binary) | |
| API | FastAPI + uvicorn | |
| Infra | Terraform → 1× EC2 t3.xlarge + S3 | |

> **Generation provider swap:** set `GENERATION_PROVIDER=anthropic` +
> `ANTHROPIC_API_KEY=...` to use Claude instead of GPT-4o-mini.
> The default is OpenAI because the Anthropic trial key has a monthly spending cap.

## Current corpus (as of June 2026)

| Project | Vectors | Chunking note |
|---------|---------|---------------|
| Kubernetes | 15,646 | Rule-classified (frontmatter); heading-aware chunks |
| Helm | 7,025 | UNKNOWN type (no frontmatter); fixed-size chunks |
| Argo CD | 2,266 | Only `docs/` subdir of argoproj/argo-cd |
| Prometheus | 775 | Fixed-size chunks |
| **Total** | **25,712** | |

Re-enabling LLM classification (`DISABLE_LLM_CLASSIFICATION=false`) and re-ingesting
will improve Helm/Argo CD/Prometheus chunking. Costs ~$0.03 with Haiku after the
monthly cap resets.

## Local quickstart

```bash
# 1. Dependencies (Python 3.12 via uv)
uv sync --extra dev

# 2. Secrets
cp .env.example .env   # fill in OPENAI_API_KEY and COHERE_API_KEY at minimum

# 3. Local Qdrant (Docker)
docker compose up -d

# 4. Clone a corpus (start with Kubernetes only)
git clone --depth 1 https://github.com/kubernetes/website corpus/kubernetes

# 5. Dry-run to verify chunking before spending embedding quota
uv run python scripts/ingest.py --project kubernetes --dry-run

# 6. Ingest for real (embeds + indexes; ~5 min for kubernetes)
uv run python scripts/ingest.py --project kubernetes

# 7. Ask a question
uv run python scripts/query_cli.py "What is a Kubernetes Pod?"

# 8. Run the API
uv run uvicorn api.app:app --port 8000
curl localhost:8000/health
```

## AWS quickstart (restore from S3 backup — no re-ingestion needed)

```bash
cd infra
terraform init
terraform apply -var="your_ip_cidr=$(curl -4 -s ifconfig.me)/32"
# Wait ~3 min for user-data to complete, then:
curl http://<output_ip>:8000/health
```

The user-data script automatically restores the 169 MB Qdrant index from S3.
A new EC2 is query-ready in ~3 minutes with zero API calls.

## Tests

```bash
uv run pytest tests/ -q     # 65 tests, no network required
```

## Repository layout

```
cncf_rag/ingestion/    parse → classify → load (marko, hybrid classifier, SQLite checksums)
cncf_rag/chunking/     per-DocType strategies (heading-aware, fixed, semantic) + evaluator
cncf_rag/embedding/    OpenAI provider with batching and 429 retry
cncf_rag/vectorstore/  Qdrant store: HNSW config, payload indexes, filtered search
cncf_rag/retrieval/    query analyzer → filters → top-k / MMR / multi-query+RRF → rerank
cncf_rag/generation/   faithfulness-first prompting + dual-provider (OpenAI/Anthropic) JSON answers
cncf_rag/evaluation/   RAGAS runner + version staleness + test set generator
api/                   FastAPI: POST /query, GET /health, GET /metrics
scripts/               ingest.py, query_cli.py, backup_to_s3.sh
infra/                 Terraform: EC2 + S3, user-data bootstrap
```
