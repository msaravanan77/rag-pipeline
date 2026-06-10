"""Shared fixtures: a realistic Kubernetes doc snippet and prebuilt Documents."""

from __future__ import annotations

import hashlib

import pytest

from cncf_rag.ingestion.models import Document, DocumentMetadata, DocType, Project
from cncf_rag.ingestion.parsers.markdown_parser import MarkdownParser
from cncf_rag.ingestion.preprocessor import Preprocessor

# A condensed but structurally faithful kubernetes/website page: frontmatter,
# Hugo shortcodes, headings, fenced YAML, prose.
K8S_DOC_SNIPPET = """---
title: "Pod Lifecycle"
content_type: concept
weight: 30
---

<!-- overview -->

This page describes the lifecycle of a Pod.

{{< note >}}
Pods are ephemeral.
{{< /note >}}

## Pod phase

A Pod's `status` field is a PodStatus object, which has a `phase` field.
The phase is a simple, high-level summary of where the Pod is in its lifecycle.
The phase value transitions through Pending, Running, and then Succeeded or Failed
depending on the outcome of containers in the Pod.

### Container states

Kubernetes tracks the state of each container inside a Pod. The kubelet reports
container states through the Pod status as the containers start, run, and stop.

## Pod conditions

A Pod has a PodStatus, which has an array of PodConditions. Conditions describe
whether the pod has been scheduled, whether its containers are ready, and whether
the pod overall is considered ready to serve requests.

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: nginx
spec:
  containers:
  - name: nginx
    image: nginx:1.14.2
```

Restart policy applies to all containers in the Pod.
"""


@pytest.fixture
def k8s_snippet() -> str:
    return K8S_DOC_SNIPPET


@pytest.fixture
def parsed_k8s_doc():
    return MarkdownParser().parse(
        K8S_DOC_SNIPPET, "kubernetes/content/en/docs/concepts/workloads/pods/pod-lifecycle.md"
    )


def make_document(
    content: str,
    doc_type: DocType = DocType.CONCEPT,
    headings: list | None = None,
    title: str = "Test Doc",
) -> Document:
    parser = MarkdownParser()
    parsed = parser.parse(content, "test/doc.md")
    # Use parser output directly (not Preprocessor) so heading char_offsets
    # stay valid — Preprocessor's blank-line collapsing shifts offsets, which
    # production code handles by re-locating headings, but tests want determinism.
    cleaned = parsed.content
    return Document(
        doc_id=hashlib.sha256(b"test/doc.md").hexdigest(),
        project=Project.KUBERNETES,
        doc_type=doc_type,
        doc_type_confidence=1.0,
        content=cleaned,
        headings=headings if headings is not None else parsed.headings,
        code_blocks=parsed.code_blocks,
        metadata=DocumentMetadata(title=title, source_path="test/doc.md"),
        checksum="abc",
    )


@pytest.fixture
def concept_document(k8s_snippet) -> Document:
    return make_document(k8s_snippet, title="Pod Lifecycle")
