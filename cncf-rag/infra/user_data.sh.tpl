#!/bin/bash
# CNCF RAG bootstrap — fully unattended when the S3 bucket is seeded.
# Log: /var/log/cloud-init-output.log
#
# S3 layout this script consumes (all optional — graceful fallbacks):
#   s3://<bucket>/bootstrap/code.tar.gz          application code bundle
#   s3://<bucket>/bootstrap/cncf-rag.env         populated environment file (API keys)
#   s3://<bucket>/backups/qdrant-storage.tar.gz  vector index snapshot (skips re-embedding!)
#   s3://<bucket>/raw/kubernetes.tar.gz          corpus (only needed for re-ingestion)
#
# Seed/refresh these with:  scripts/backup_to_s3.sh  (run on the instance)
set -euxo pipefail

BUCKET="${s3_bucket_name}"

# ---------- System packages ----------
dnf update -y
dnf install -y python3.12 git sqlite

# ---------- uv (Python package manager) ----------
export HOME=/root
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="/root/.local/bin:$PATH"

# ---------- Qdrant (single static binary) ----------
QDRANT_VERSION="v1.18.2"
cd /tmp
wget -q "https://github.com/qdrant/qdrant/releases/download/$${QDRANT_VERSION}/qdrant-x86_64-unknown-linux-musl.tar.gz"
tar -xzf qdrant-x86_64-unknown-linux-musl.tar.gz
mv qdrant /usr/local/bin/qdrant
mkdir -p /var/lib/qdrant

cat > /etc/systemd/system/qdrant.service << 'EOF'
[Unit]
Description=Qdrant Vector Store
After=network.target

[Service]
ExecStart=/usr/local/bin/qdrant
WorkingDirectory=/var/lib/qdrant
Restart=always
RestartSec=5
# Bind to localhost ONLY — Qdrant must never be reachable from outside.
Environment=QDRANT__SERVICE__HTTP_PORT=6333
Environment=QDRANT__SERVICE__HOST=127.0.0.1

[Install]
WantedBy=multi-user.target
EOF

# ---------- Restore Qdrant index from S3 (BEFORE first start) ----------
# This is what makes a destroyed-and-recreated instance answer queries
# immediately — no re-embedding, no trial-quota spend.
if aws s3 cp "s3://$${BUCKET}/backups/qdrant-storage.tar.gz" /tmp/qdrant-storage.tar.gz; then
  tar xzf /tmp/qdrant-storage.tar.gz -C /var/lib/qdrant
  echo "Qdrant index restored from S3"
else
  echo "No Qdrant backup in S3 — starting with an empty index (run ingestion)"
fi

# ---------- Application code: S3 bundle first, git clone fallback ----------
# S3 bundle is primary (faster, pre-built). Git clone works as fallback because
# the GitHub repo is public — no authentication required.
mkdir -p /opt/rag-pipeline
if aws s3 cp "s3://$${BUCKET}/bootstrap/code.tar.gz" /tmp/code.tar.gz; then
  tar xzf /tmp/code.tar.gz -C /opt/rag-pipeline
  echo "Code restored from S3 bundle"
else
  git clone ${app_repo_url} /opt/rag-pipeline
fi
ln -sfn /opt/rag-pipeline/${app_subdir} /opt/cncf-rag
cd /opt/cncf-rag
uv python install 3.12
uv sync

# ---------- Environment file: S3 first, placeholder fallback ----------
if aws s3 cp "s3://$${BUCKET}/bootstrap/cncf-rag.env" /etc/cncf-rag.env; then
  echo "Environment (with API keys) restored from S3"
else
  cat > /etc/cncf-rag.env << EOF
# ACTION REQUIRED: populate all three keys, then: sudo systemctl restart cncf-rag
OPENAI_API_KEY=REPLACE_ME
ANTHROPIC_API_KEY=REPLACE_ME
COHERE_API_KEY=REPLACE_ME
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=cncf_docs
AWS_REGION=${aws_region}
S3_CORPUS_BUCKET=$${BUCKET}
LOG_LEVEL=INFO
API_PORT=8000
SEMANTIC_CHUNKING=off
EOF
fi
chmod 600 /etc/cncf-rag.env

# ---------- Corpus from S3 (only needed if you plan to re-ingest) ----------
mkdir -p /opt/cncf-rag/corpus
for project in kubernetes helm argocd prometheus; do
  if aws s3 cp "s3://$${BUCKET}/raw/$${project}.tar.gz" "/tmp/$${project}.tar.gz"; then
    tar xzf "/tmp/$${project}.tar.gz" -C /opt/cncf-rag/corpus
    rm -f "/tmp/$${project}.tar.gz"
  fi
done
# Restore the ingestion checksum index so unchanged files are skipped.
aws s3 cp "s3://$${BUCKET}/ingest-index/.ingest_index.db" /opt/cncf-rag/corpus/.ingest_index.db || true

# ---------- FastAPI service ----------
cat > /etc/systemd/system/cncf-rag.service << 'EOF'
[Unit]
Description=CNCF RAG FastAPI Application
After=network.target qdrant.service

[Service]
ExecStart=/root/.local/bin/uv run uvicorn api.app:app --host 0.0.0.0 --port 8000
WorkingDirectory=/opt/cncf-rag
EnvironmentFile=/etc/cncf-rag.env
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable qdrant cncf-rag
systemctl start qdrant
sleep 3
systemctl start cncf-rag

echo "Bootstrap complete. If S3 was seeded, the system is fully up: curl localhost:8000/health"
