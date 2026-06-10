#!/bin/bash
# Snapshot the running CNCF RAG instance into S3 so a future
# `terraform apply` recreates a fully working system with zero manual steps.
#
# Run ON THE EC2 INSTANCE as root:  sudo bash scripts/backup_to_s3.sh
#
# What it uploads:
#   bootstrap/code.tar.gz          application code (as deployed)
#   bootstrap/cncf-rag.env         env file INCLUDING API keys (bucket is private + SSE)
#   backups/qdrant-storage.tar.gz  the vector index — restoring it skips re-embedding
#   raw/<project>.tar.gz           corpora present under corpus/
#   ingest-index/.ingest_index.db  checksum index for incremental ingestion
set -euo pipefail

BUCKET="$(grep '^S3_CORPUS_BUCKET=' /etc/cncf-rag.env | cut -d= -f2)"
echo "Backing up to s3://$BUCKET"

# Code (excluding venv — uv sync rebuilds it, and corpus — uploaded separately)
tar czf /tmp/code.tar.gz -C /opt/rag-pipeline --exclude='.venv' --exclude='cncf-rag/corpus' .
aws s3 cp /tmp/code.tar.gz "s3://$BUCKET/bootstrap/code.tar.gz"

# Env file with keys. Acceptable here because the bucket blocks all public
# access, is SSE-encrypted, and the IAM policy scopes access to this instance
# + your own user. For team/production use, move keys to SSM Parameter Store.
aws s3 cp /etc/cncf-rag.env "s3://$BUCKET/bootstrap/cncf-rag.env"

# Qdrant index — stop writes for a consistent snapshot, then resume.
systemctl stop cncf-rag
systemctl stop qdrant
tar czf /tmp/qdrant-storage.tar.gz -C /var/lib/qdrant .
systemctl start qdrant
sleep 3
systemctl start cncf-rag
aws s3 cp /tmp/qdrant-storage.tar.gz "s3://$BUCKET/backups/qdrant-storage.tar.gz"

# Corpora
for dir in /opt/cncf-rag/corpus/*/; do
  [ -d "$dir" ] || continue
  project="$(basename "$dir")"
  tar czf "/tmp/$project.tar.gz" -C /opt/cncf-rag/corpus "$project"
  aws s3 cp "/tmp/$project.tar.gz" "s3://$BUCKET/raw/$project.tar.gz"
  rm -f "/tmp/$project.tar.gz"
done

# Incremental-ingestion checksum index
[ -f /opt/cncf-rag/corpus/.ingest_index.db ] && \
  aws s3 cp /opt/cncf-rag/corpus/.ingest_index.db "s3://$BUCKET/ingest-index/.ingest_index.db"

rm -f /tmp/code.tar.gz /tmp/qdrant-storage.tar.gz
echo "Backup complete:"
aws s3 ls "s3://$BUCKET" --recursive --human-readable | tail -15
