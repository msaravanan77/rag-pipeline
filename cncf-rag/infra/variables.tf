variable "aws_region" {
  description = "AWS region. us-east-1 is cheapest for most workloads."
  type        = string
  default     = "us-east-1"
}

variable "instance_type" {
  description = "t3.xlarge = 4vCPU/16GB = ~$120/month. Do not downsize below t3.large (8GB) or Qdrant will OOM during ingestion."
  type        = string
  default     = "t3.xlarge"
}

variable "your_ip_cidr" {
  description = "Your public IP in CIDR notation. Find it: curl ifconfig.me. Append /32. Example: 1.2.3.4/32"
  type        = string
  # No default — must be supplied. Security requirement: never open to 0.0.0.0/0.
}

variable "key_pair_name" {
  description = "Name of an existing EC2 key pair in this region. Must exist before terraform apply."
  type        = string
  default     = "us-east-1-mac-key"
}

variable "s3_bucket_name" {
  description = "S3 bucket for corpus. Must be globally unique."
  type        = string
  default     = "cncf-rag-corpus-msaravanan77"
}

variable "app_repo_url" {
  description = "GitHub URL to clone the application repo onto the EC2 instance."
  type        = string
  default     = "https://github.com/msaravanan77/rag-pipeline.git"
}

variable "app_subdir" {
  description = "Subdirectory inside the repo that holds the cncf-rag project."
  type        = string
  default     = "cncf-rag"
}
