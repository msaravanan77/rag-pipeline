"""Prompt assembly: the faithfulness-first system prompt + numbered context blocks."""

from __future__ import annotations

import structlog
import tiktoken

from cncf_rag.vectorstore.qdrant_store import ScoredChunk

logger = structlog.get_logger(__name__)

_ENCODING = tiktoken.get_encoding("cl100k_base")

# 50k cap (DECISIONS.md 6.2): leaves 150k of the 200k window for system
# prompt, query, and the answer itself.
_CONTEXT_TOKEN_BUDGET = 50_000
_WARN_TOKEN_THRESHOLD = 100_000

SYSTEM_PROMPT = """You are a technical documentation assistant for the CNCF ecosystem (Kubernetes,
Helm, Argo CD, and Prometheus). You answer questions for platform engineers,
SREs, and DevOps practitioners.

STRICT RULES:
1. Answer ONLY from the CONTEXT sections provided below. Do not use knowledge
   from your training data that is not reflected in the provided context.
2. If the context does not contain sufficient information, respond with
   cannot_answer: true and explain what specific information is missing.
3. If the context contains answers for multiple Kubernetes versions, present ALL
   versions explicitly and state which version each answer applies to.
4. If any context mentions that an API or feature is deprecated, you MUST include
   the deprecation notice in your answer even if the user did not ask about it.
5. Every factual claim must include an inline citation:
   [Source: <project> | <doc_type> | <version> | <url>]
6. Reproduce code blocks exactly as they appear in context. Never paraphrase code.
7. Do not include "see also" links that appear in context — only cite content
   directly present in the provided context sections.

RESPONSE FORMAT: Return valid JSON only. No text outside the JSON object.
The JSON object must have these fields:
{
  "answer": "the answer text with inline citations",
  "citations": [{"project": "...", "doc_type": "...", "version": "...", "source_path": "..."}],
  "confidence": 0.0-1.0,
  "cannot_answer": false,
  "version_warning": "string or null - set when the answer differs across versions"
}"""


class PromptBuilder:
    def build(self, query: str, chunks: list[ScoredChunk]) -> tuple[str, str]:
        """Return (system_prompt, user_message) with context within budget.

        Chunks are ordered score-descending and dropped from the tail when over
        budget — "lost in the middle" means the front of the context gets the
        best recall, so the best chunk goes first (DECISIONS.md 6.2).
        """
        ordered = sorted(chunks, key=lambda c: c.score, reverse=True)

        blocks: list[str] = []
        total_tokens = 0
        for i, chunk in enumerate(ordered, start=1):
            heading = chunk.payload.get("heading_path") or ""
            header = (
                f"CONTEXT {i} "
                f"[project: {chunk.payload.get('project', '?')} | "
                f"doc_type: {chunk.payload.get('doc_type', '?')} | "
                f"version: {chunk.payload.get('version_tag') or 'unversioned'} | "
                f"source: {chunk.payload.get('source_path', '?')}]"
            )
            block = f"{header}\n{heading}\n{chunk.content}" if heading else f"{header}\n{chunk.content}"
            block_tokens = len(_ENCODING.encode(block))
            if total_tokens + block_tokens > _CONTEXT_TOKEN_BUDGET:
                logger.info("context_budget_hit", included=i - 1, dropped=len(ordered) - i + 1)
                break
            blocks.append(block)
            total_tokens += block_tokens

        user_message = f"QUESTION: {query}\n\n" + "\n\n---\n\n".join(blocks)
        prompt_tokens = len(_ENCODING.encode(SYSTEM_PROMPT)) + len(_ENCODING.encode(user_message))
        if prompt_tokens > _WARN_TOKEN_THRESHOLD:
            logger.warning("prompt_unusually_large", tokens=prompt_tokens)
        return SYSTEM_PROMPT, user_message
