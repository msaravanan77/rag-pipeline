output "instance_id" {
  description = "EC2 instance ID (for stop/start commands)."
  value       = aws_instance.cncf_rag.id
}

output "ec2_public_ip" {
  description = "Elastic IP of the EC2 instance."
  value       = aws_eip.cncf_rag.public_ip
}

output "api_base_url" {
  description = "Base URL for the CNCF RAG API."
  value       = "http://${aws_eip.cncf_rag.public_ip}:8000"
}

output "api_health_url" {
  description = "Health check endpoint."
  value       = "http://${aws_eip.cncf_rag.public_ip}:8000/health"
}

output "s3_bucket" {
  description = "S3 bucket for corpus storage."
  value       = aws_s3_bucket.cncf_rag_corpus.id
}

output "ssh_command" {
  description = "SSH command to access the instance."
  value       = "ssh -i /Users/saravanan/aws/key/${var.key_pair_name}.pem ec2-user@${aws_eip.cncf_rag.public_ip}"
}

output "monthly_cost_estimate" {
  description = "Estimated monthly cost when running 24/7."
  value       = "t3.xlarge: ~$120/month | EIP: ~$0/month (attached) | S3: <$1/month | Total: ~$121/month. STOP INSTANCE WHEN NOT IN USE: saves ~$0.166/hour."
}

output "next_steps" {
  description = "Required manual steps after terraform apply completes."
  value       = <<-EOT

  ============================================================
  NEXT STEPS (complete in this order):
  ============================================================
  1. SSH in:
     ssh -i /Users/saravanan/aws/key/${var.key_pair_name}.pem ec2-user@${aws_eip.cncf_rag.public_ip}

  2. Populate API keys:
     sudo nano /etc/cncf-rag.env
     (Replace REPLACE_ME values for ANTHROPIC_API_KEY and COHERE_API_KEY)

  3. Start the app (Qdrant is already running):
     sudo systemctl start cncf-rag

  4. Verify health:
     curl http://${aws_eip.cncf_rag.public_ip}:8000/health

  5. Download corpus and run initial ingestion (start with one project):
     cd /opt/cncf-rag
     sudo git clone --depth 1 https://github.com/kubernetes/website corpus/kubernetes
     sudo /root/.local/bin/uv run python scripts/ingest.py --project kubernetes --dry-run
     sudo /root/.local/bin/uv run python scripts/ingest.py --project kubernetes

  6. Test a query:
     sudo /root/.local/bin/uv run python scripts/query_cli.py "What is a Kubernetes Pod?"

  ============================================================
  COST CONTROL: Stop instance when not actively using:
  aws ec2 stop-instances --instance-ids ${aws_instance.cncf_rag.id}
  ============================================================
  EOT
}
