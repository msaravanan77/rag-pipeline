# CNCF RAG — Enterprise-Grade RAG Pipeline over CNCF Documentation

A RAG-powered question-answering system over the official docs of
**Kubernetes, Helm, Argo CD, and Prometheus**, with cited, version-aware answers.

- **Why every technology was chosen over its alternatives:** [DECISIONS.md](DECISIONS.md) (13 entries)
- **System design and data flow:** [ARCHITECTURE.md](ARCHITECTURE.md)
- **AWS deployment, costs, stop/start:** [infra/README.md](infra/README.md)
- **Day-to-day usage after deployment:** [USAGE.md](USAGE.md) ← start here once deployed

## Stack at a glance

| Layer | Technology |
|---|---|
| Embeddings + Reranking | Cohere `embed-english-v3.0` / `rerank-english-v3.0` (free trial tier) |
| Generation | Anthropic `claude-sonnet-4-5` |
| Vector store | Qdrant (self-hosted, single binary) |
| API | FastAPI + uvicorn |
| Evaluation | RAGAS + custom version-staleness metric |
| Infra | Terraform → 1× EC2 t3.xlarge + S3 + Elastic IP |

## Local quickstart

```bash
# 1. Dependencies (Python 3.12 via uv)
uv sync --extra dev

# 2. Secrets
cp .env.example .env        # then fill in COHERE_API_KEY and ANTHROPIC_API_KEY

# 3. Local Qdrant
docker compose up -d

# 4. Get a corpus (start with Kubernetes only)
git clone --depth 1 https://github.com/kubernetes/website corpus/kubernetes

# 5. Verify chunking before spending API quota
uv run python scripts/ingest.py --project kubernetes --dry-run

# 6. Ingest for real (embeds + indexes; paced for the Cohere trial key)
uv run python scripts/ingest.py --project kubernetes

# 7. Ask a question
uv run python scripts/query_cli.py "What is a Kubernetes Pod?"

# 8. Run the API
uv run uvicorn api.app:app --port 8000
curl localhost:8000/health
```

## AWS quickstart

```bash
cd infra
terraform init
terraform apply -var="your_ip_cidr=$(curl -s ifconfig.me)/32"
# then follow the `next_steps` output
```

## Tests

```bash
uv run pytest tests/ -q     # 65 tests, no network required
```

## Repository layout

```
cncf_rag/ingestion/    parse → classify → load (marko, hybrid classifier, SQLite checksums)
cncf_rag/chunking/     per-DocType strategies (heading-aware, fixed, semantic) + evaluator
cncf_rag/embedding/    Cohere provider with batching, pacing, cost tracking
cncf_rag/vectorstore/  Qdrant store: HNSW config, payload indexes, filtered search
cncf_rag/retrieval/    query analyzer → filters → top-k / MMR / multi-query+RRF → rerank
cncf_rag/generation/   faithfulness-first prompting + claude-sonnet-4-5 JSON answers
cncf_rag/evaluation/   RAGAS runner + version staleness + test set generator
api/                   FastAPI: POST /query, GET /health, GET /metrics
scripts/               ingest.py, query_cli.py, run_eval.py
infra/                 Terraform: EC2 + S3 + EIP, user-data bootstrap
```
