# Corpus bucket — ~$0.50/month for a ~400MB corpus.
resource "aws_s3_bucket" "cncf_rag_corpus" {
  bucket = var.s3_bucket_name
  # Protect corpus — accidental destroy would require re-cloning and
  # re-ingesting all docs. Set true only when intentionally tearing down.
  force_destroy = false
  tags          = { Project = "cncf-rag" }
}

# Recover from accidental overwrites; versioning cost is negligible at this size.
resource "aws_s3_bucket_versioning" "cncf_rag_corpus" {
  bucket = aws_s3_bucket.cncf_rag_corpus.id
  versioning_configuration {
    status = "Enabled"
  }
}

# Encrypt at rest; SSE-S3 is free; no reason not to enable.
resource "aws_s3_bucket_server_side_encryption_configuration" "cncf_rag_corpus" {
  bucket = aws_s3_bucket.cncf_rag_corpus.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# Corpus is never public; block all four public access paths.
resource "aws_s3_bucket_public_access_block" "cncf_rag_corpus" {
  bucket                  = aws_s3_bucket.cncf_rag_corpus.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Corpus is written once; STANDARD_IA saves ~45% storage cost after the first
# month; retrieval cost if ever needed is negligible at this scale.
resource "aws_s3_bucket_lifecycle_configuration" "cncf_rag_corpus" {
  bucket = aws_s3_bucket.cncf_rag_corpus.id
  rule {
    id     = "transition-to-ia"
    status = "Enabled"
    filter {}
    transition {
      days          = 30
      storage_class = "STANDARD_IA"
    }
  }
}
