# Infrastructure — CNCF RAG on AWS

Single EC2 instance + S3 bucket + Elastic IP. Nothing else. See ARCHITECTURE.md
for why there is no ECS, no RDS, and no load balancer (short version: they cost
money and teach nothing about RAG).

## Prerequisites

1. **AWS CLI** configured with credentials (`aws sts get-caller-identity` works)
2. **Terraform ≥ 1.7** installed
3. **An EC2 key pair** already created in us-east-1
   (this project uses `us-east-1-mac-key`, with the .pem at
   `/Users/saravanan/aws/key/us-east-1-mac-key.pem`)

## Deploy

```bash
cd infra
terraform init
terraform apply -var="your_ip_cidr=$(curl -s ifconfig.me)/32"
```

`key_pair_name`, `s3_bucket_name`, and `app_repo_url` have defaults set for this
project (see variables.tf). Override with `-var=` if anything differs.

After apply, follow the `next_steps` output verbatim (SSH in, populate
`/etc/cncf-rag.env`, start the service, ingest, query).

## Cost estimate

```
Resource              Monthly (24/7)   Notes
─────────────────────────────────────────────────────────────────
EC2 t3.xlarge         ~$120.00         Stop when not in use: $0.166/hr
Elastic IP            ~$0.00           Free while attached to running instance
                      ~$3.60           If instance is stopped a full month
S3 Storage            ~$0.50           400MB corpus at $0.023/GB
S3 Requests           ~$0.01           Negligible at this scale
Cohere embeddings     $0.00            Trial key (rate-limited, not billed)
Anthropic generation  ~$0.02–0.05/query
─────────────────────────────────────────────────────────────────
Total (instance on)   ~$121/month + per-query LLM cost
Total (instance off)  ~$4.11/month
```

## Stop / start without data loss

Qdrant data lives on the EBS **root volume**, which persists across
`stop-instances` / `start-instances`. Only **terminating** the instance (or
`terraform destroy`) deletes the volume.

```bash
# Stop (billing for compute stops; EBS ~$4/month continues)
aws ec2 stop-instances --instance-ids $(terraform output -raw instance_id)

# Start again — Elastic IP stays the same, Qdrant auto-starts via systemd
aws ec2 start-instances --instance-ids $(terraform output -raw instance_id)
```

## Destroy everything

```bash
terraform destroy -var="your_ip_cidr=$(curl -s ifconfig.me)/32"
```

⚠ Warnings:
- Destroys the EBS root volume → **the Qdrant index is gone**; re-ingestion required.
- The S3 bucket is protected by `force_destroy = false` — destroy fails if it
  contains objects. That is deliberate corpus protection; empty the bucket
  manually (or flip the flag) if you truly want it gone.

## Updating the application

```bash
ssh -i /Users/saravanan/aws/key/us-east-1-mac-key.pem ec2-user@$(terraform output -raw ec2_public_ip)
cd /opt/rag-pipeline && sudo git pull
cd /opt/cncf-rag && sudo /root/.local/bin/uv sync
sudo systemctl restart cncf-rag
```
