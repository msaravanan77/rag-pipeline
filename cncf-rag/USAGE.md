# USAGE.md — How to Operate Your CNCF RAG System

This is the day-to-day manual for the deployed system. It assumes
`terraform apply` already ran (see [infra/README.md](infra/README.md)).

---

## 1. The pieces you own

| Thing | Where | What it does |
|---|---|---|
| EC2 instance `cncf-rag` | us-east-1 | Runs Qdrant + the FastAPI app |
| Elastic IP | attached to EC2 | Stable address — survives stop/start |
| S3 bucket `cncf-rag-corpus-msaravanan77` | us-east-1 | Corpus backup storage |
| SSH key | `/Users/saravanan/aws/key/us-east-1-mac-key.pem` | Your door into the instance |

Get the current IP anytime:
```bash
cd infra && terraform output -raw ec2_public_ip
```

## 2. SSH in

```bash
ssh -i /Users/saravanan/aws/key/us-east-1-mac-key.pem ec2-user@<ELASTIC_IP>
```

## 3. One-time setup after first boot

The instance bootstraps itself (Qdrant installed and running, repo cloned,
Python env built). You only need to add your API keys:

```bash
sudo nano /etc/cncf-rag.env
# Set:  OPENAI_API_KEY=sk-proj-...     (from platform.openai.com — used for embeddings)
#       ANTHROPIC_API_KEY=sk-ant-...   (from console.anthropic.com — used for generation)
#       COHERE_API_KEY=...             (your trial key — used for reranking only)
sudo systemctl start cncf-rag
```

> **If S3 was seeded** with `backup_to_s3.sh`, all three keys are restored automatically
> and no manual editing is needed.

Verify both services:
```bash
sudo systemctl status qdrant cncf-rag
curl localhost:8000/health        # {"status":"ok","qdrant":"connected",...}
```

## 4. Load the corpus and ingest

```bash
cd /opt/cncf-rag

# Pull the docs (Kubernetes alone is the biggest; start there)
sudo git clone --depth 1 https://github.com/kubernetes/website corpus/kubernetes
sudo git clone --depth 1 https://github.com/helm/helm-www corpus/helm
sudo git clone --depth 1 https://github.com/argoproj/argo-cd corpus/argocd
sudo git clone --depth 1 https://github.com/prometheus/docs corpus/prometheus

# ALWAYS dry-run first — prints the chunking report, costs nothing
sudo /root/.local/bin/uv run python scripts/ingest.py --project kubernetes --dry-run

# Real ingestion (paced to the Cohere trial limit; kubernetes takes ~25-40 min)
sudo /root/.local/bin/uv run python scripts/ingest.py --project kubernetes
```

Re-running ingestion is cheap: the SQLite checksum index skips unchanged files
automatically (see DECISIONS.md 1.3).

## 5. Ask questions

**From the instance (CLI):**
```bash
sudo /root/.local/bin/uv run python scripts/query_cli.py "What is a Kubernetes Pod?"
sudo /root/.local/bin/uv run python scripts/query_cli.py "operators" --show-chunks --strategy mmr
```

**From your Mac (HTTP API):**
```bash
curl -X POST http://<ELASTIC_IP>:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How do I configure liveness probes?"}'

# Optional filters:
#   "project_filter": "kubernetes" | "helm" | "argocd" | "prometheus"
#   "version_filter": "v1.29"
```

Interactive API docs: `http://<ELASTIC_IP>:8000/docs`
Prometheus metrics:   `http://<ELASTIC_IP>:8000/metrics`

### How your question is routed (automatic)

| You ask… | Detected as | What happens |
|---|---|---|
| "What is a Service?" | CONCEPTUAL | top-5, concept docs only |
| "How do I install Helm?" | PROCEDURAL | top-5, task docs only |
| "Ingress in v1.22?" | VERSION_SPECIFIC | top-5, anchored to v1.22 docs |
| "Helm vs Argo CD rollbacks?" | CROSS_PROJECT | per-project sub-queries, RRF-merged, reranked |
| "service mesh" | EXPLORATORY | MMR diversity retrieval, reranked |
| anything else | FACTUAL | plain filtered top-5 |

If the system can't ground an answer in the docs it says so
(`cannot_answer: true`) instead of guessing.

## 6. Evaluate quality

```bash
cd /opt/cncf-rag
sudo /root/.local/bin/uv run python scripts/run_eval.py --test-set tests/fixtures/test_set.json
```
Exit code 1 means a metric missed target (faithfulness ≥0.85, relevancy ≥0.80,
precision ≥0.75, recall ≥0.70, staleness ≤0.10). The 20 hand-curated
version-edge cases live in `tests/fixtures/version_cases.json`.

## 7. Cost control — read this

The instance costs **~$0.166/hour (~$120/month) while running**. Stop it whenever
you're not using it; the Qdrant index survives on the EBS volume.

```bash
# From your Mac:
cd infra
aws ec2 stop-instances  --instance-ids $(terraform output -raw instance_id)
aws ec2 start-instances --instance-ids $(terraform output -raw instance_id)
```
After start, the same Elastic IP works and both services auto-start via systemd
(give it ~60 seconds).

## 8. Update the application code

```bash
ssh -i /Users/saravanan/aws/key/us-east-1-mac-key.pem ec2-user@<ELASTIC_IP>
cd /opt/rag-pipeline && sudo git pull
cd /opt/cncf-rag && sudo /root/.local/bin/uv sync
sudo systemctl restart cncf-rag
```

## 9. Troubleshooting

| Symptom | Check |
|---|---|
| `/health` shows `"qdrant": "error"` | `sudo systemctl status qdrant`; `sudo journalctl -u qdrant -n 50` |
| App won't start | `sudo journalctl -u cncf-rag -n 50` — usually a missing key in `/etc/cncf-rag.env` |
| 429 / rate-limit errors during ingestion | Normal on the trial key; the embedder retries with backoff. Just let it run. |
| `cannot_answer` for everything | Did ingestion finish? `curl localhost:6333/collections/cncf_docs` should show points_count > 0 |
| Query from Mac times out | Your public IP changed — re-run `terraform apply -var="your_ip_cidr=$(curl -s ifconfig.me)/32"` to update the security group |
| Bootstrap seems incomplete | `sudo tail -100 /var/log/cloud-init-output.log` |

## 10. Destroy everything except S3, then resurrect with one command

The S3 bucket is the system's source of truth. **Before destroying**, snapshot
the running instance into S3 (takes ~2 minutes):

```bash
ssh -i /Users/saravanan/aws/key/us-east-1-mac-key.pem ec2-user@<ELASTIC_IP> \
  'sudo bash /opt/cncf-rag/scripts/backup_to_s3.sh'
```

This uploads to the bucket:

| S3 key | Contents |
|---|---|
| `bootstrap/code.tar.gz` | the application code as deployed |
| `bootstrap/cncf-rag.env` | env file **including your API keys** |
| `backups/qdrant-storage.tar.gz` | the vector index — restoring it means **no re-embedding** |
| `raw/kubernetes.tar.gz` (etc.) | the corpora |
| `ingest-index/.ingest_index.db` | checksum index so re-ingestion skips unchanged files |

Then destroy compute (S3 survives because `force_destroy = false` plus
terraform refuses to delete a non-empty bucket):

```bash
cd infra && terraform destroy -var="your_ip_cidr=$(curl -4 -s ifconfig.me)/32"
```

**To resurrect later — this is the whole procedure:**

```bash
cd infra
terraform apply -var="your_ip_cidr=$(curl -4 -s ifconfig.me)/32"
```

The new instance's user-data pulls code, env (keys included), Qdrant index, and
corpus from S3 via its IAM role and starts both services. ~5 minutes after
apply, `curl http://<NEW_ELASTIC_IP>:8000/health` works and queries answer
immediately — no SSH, no key pasting, no re-ingestion. You get a NEW Elastic IP
(shown in the apply output); your security group is built fresh from the
`your_ip_cidr` you pass, so it also heals the "my ISP changed my IP" problem.

Security note: the env file in S3 contains real API keys. That's acceptable for
a single-user private bucket (public access blocked, SSE-S3 encrypted, IAM-scoped),
but for anything shared, move keys to SSM Parameter Store instead.

## 11. Tear down completely

```bash
cd infra
terraform destroy -var="your_ip_cidr=$(curl -s ifconfig.me)/32"
```
Destroys EC2 + EBS (index gone). The S3 bucket refuses to die while it has
objects (`force_destroy = false`) — that's intentional corpus protection.
