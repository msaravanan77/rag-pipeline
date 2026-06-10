#!/bin/bash
# CNCF RAG bootstrap — runs once at first boot via EC2 user data.
# Log: /var/log/cloud-init-output.log
set -euxo pipefail

# ---------- System packages ----------
dnf update -y
dnf install -y python3.12 git sqlite

# ---------- uv (Python package manager) ----------
export HOME=/root
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="/root/.local/bin:$PATH"

# ---------- Qdrant (single static binary, no Docker needed) ----------
QDRANT_VERSION="v1.9.0"
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

# ---------- Application ----------
git clone ${app_repo_url} /opt/rag-pipeline
ln -s /opt/rag-pipeline/${app_subdir} /opt/cncf-rag
cd /opt/cncf-rag
/root/.local/bin/uv python install 3.12
/root/.local/bin/uv sync

# ---------- Environment file ----------
# COHERE key is injected post-boot by the deployer (never baked into user data,
# which is visible in the EC2 console). ANTHROPIC key: paste manually.
cat > /etc/cncf-rag.env << 'EOF'
# ACTION REQUIRED: populate ANTHROPIC_API_KEY after SSH-ing in
ANTHROPIC_API_KEY=REPLACE_ME
COHERE_API_KEY=REPLACE_ME
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=cncf_docs
AWS_REGION=${aws_region}
S3_CORPUS_BUCKET=${s3_bucket_name}
LOG_LEVEL=INFO
API_PORT=8000
EOF
chmod 600 /etc/cncf-rag.env

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

# Enable both; start Qdrant now (needs no keys). The app starts only after
# the operator populates /etc/cncf-rag.env.
systemctl daemon-reload
systemctl enable qdrant cncf-rag
systemctl start qdrant

echo "Bootstrap complete. SSH in, populate /etc/cncf-rag.env, then: sudo systemctl start cncf-rag"
