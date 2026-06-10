"""Extracts and validates YAML manifests from fenced code blocks.

Kubernetes docs are full of YAML manifests; knowing a chunk contains a valid
manifest (vs prose) lets the chunk payload advertise has_code_blocks accurately
and lets future features (e.g. manifest-only search) filter on it.
"""

from __future__ import annotations

import yaml

from cncf_rag.ingestion.models import CodeBlock


class YamlBlockParser:
    """Identifies which fenced code blocks are parseable Kubernetes-style YAML."""

    def extract_manifests(self, code_blocks: list[CodeBlock]) -> list[dict]:
        """Return parsed YAML documents that look like k8s manifests (have kind/apiVersion).

        Silently skips invalid YAML — docs often contain intentionally partial
        snippets (e.g. "...add this under spec:") that must not fail ingestion.
        """
        manifests: list[dict] = []
        for block in code_blocks:
            if block.language not in ("yaml", "yml", ""):
                continue
            try:
                for doc in yaml.safe_load_all(block.content):
                    if isinstance(doc, dict) and ("kind" in doc or "apiVersion" in doc):
                        manifests.append(doc)
            except yaml.YAMLError:
                continue  # partial snippet, not an error
        return manifests
