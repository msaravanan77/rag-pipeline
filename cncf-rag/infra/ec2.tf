# Always use the latest Amazon Linux 2023 AMI for security patches.
# Pin a specific AMI ID only if byte-for-byte reproducibility matters more
# than patching (it doesn't, for this project).
data "aws_ami" "al2023" {
  most_recent = true
  owners      = ["amazon"]
  filter {
    name   = "name"
    values = ["al2023-ami-2023*-x86_64"]
  }
  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

resource "aws_security_group" "cncf_rag_sg" {
  name        = "cncf-rag-sg"
  description = "CNCF RAG: SSH and API access from one IP only"

  # Never open SSH to 0.0.0.0/0 — this allows only your IP.
  ingress {
    description = "SSH from your IP only"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.your_ip_cidr]
  }

  # FastAPI port — only your IP; widen only if sharing with teammates.
  ingress {
    description = "FastAPI from your IP only"
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = [var.your_ip_cidr]
  }

  # Qdrant port 6333 is INTENTIONALLY NOT in this security group —
  # Qdrant listens on localhost only; no external access ever.

  # EC2 needs outbound to reach Anthropic and Cohere APIs, S3, and GitHub.
  egress {
    description = "All outbound"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Project = "cncf-rag" }
}

# Least-privilege IAM — this EC2 instance can touch ONLY the corpus bucket,
# no other S3 bucket and no other AWS service.
resource "aws_iam_role" "cncf_rag" {
  name = "cncf-rag-ec2-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy" "corpus_bucket_only" {
  name = "cncf-rag-s3-corpus-only"
  role = aws_iam_role.cncf_rag.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["s3:GetObject", "s3:PutObject"]
        Resource = "arn:aws:s3:::${var.s3_bucket_name}/*"
      },
      {
        Effect   = "Allow"
        Action   = ["s3:ListBucket"]
        Resource = "arn:aws:s3:::${var.s3_bucket_name}"
      }
    ]
  })
}

resource "aws_iam_instance_profile" "cncf_rag" {
  name = "cncf-rag-instance-profile"
  role = aws_iam_role.cncf_rag.name
}

# t3.xlarge: Qdrant ~600MB (200k vectors x 1024 dims) + FastAPI ~500MB +
# ingestion peaks ~4GB => 16GB RAM comfortable. ~$0.166/hr = ~$120/month
# on-demand. STOP THE INSTANCE WHEN NOT IN USE.
resource "aws_instance" "cncf_rag" {
  ami                    = data.aws_ami.al2023.id
  instance_type          = var.instance_type
  key_name               = var.key_pair_name
  vpc_security_group_ids = [aws_security_group.cncf_rag_sg.id]
  iam_instance_profile   = aws_iam_instance_profile.cncf_rag.name

  # gp3: same performance as gp2, ~20% cheaper.
  # 50GB = Qdrant storage + Python env + corpus cache.
  root_block_device {
    volume_size = 50
    volume_type = "gp3"
  }

  user_data = templatefile("${path.module}/user_data.sh.tpl", {
    app_repo_url   = var.app_repo_url
    app_subdir     = var.app_subdir
    aws_region     = var.aws_region
    s3_bucket_name = var.s3_bucket_name
  })

  tags = {
    Name    = "cncf-rag"
    Project = "cncf-rag"
  }
}

# Elastic IP = stable public address that doesn't change on instance stop/start.
# Free while attached to a RUNNING instance; ~$0.005/hr while the instance is stopped.
resource "aws_eip" "cncf_rag" {
  instance = aws_instance.cncf_rag.id
  domain   = "vpc"
  tags     = { Project = "cncf-rag" }
}
