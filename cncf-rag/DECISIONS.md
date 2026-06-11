# DECISIONS.md — Architectural Decision Log
# Project: CNCF RAG Pipeline
# Format: One entry per decision. Written BEFORE implementation. Measured impact filled AFTER.
# Total planned entries: 13 (1.1 through 7.1)

---

## Decision 1.1 — Markdown Parser: marko + python-frontmatter

```
Context: The CNCF corpus is 2,650+ Markdown files. Each file has YAML frontmatter
(title, content_type, weight, description) and a Markdown body with headings (H1–H3),
fenced code blocks (YAML, bash, Go), and tables. The parser must preserve the heading
hierarchy as structured data — not as rendered HTML — because heading-aware chunking
requires typed Heading nodes with level and text as separate fields.

Options Considered:
| Option          | Pros                                | Cons                                          |
|-----------------|-------------------------------------|-----------------------------------------------|
| python-markdown | Standard library familiar           | HTML output only; heading level lost in HTML  |
| mistune         | Fastest Markdown parser available   | HTML output; no AST; no heading tree          |
| marko           | Full AST; typed nodes; spec-correct | ~3x slower than mistune                       |
| Custom regex    | Full control                        | Brittle on nested code blocks                 |

Decision: python-frontmatter + marko
Reason: marko produces Heading(level=2, children=[RawText("Configuring TLS")]) — the
heading level and text are first-class typed objects. This is a structural requirement
for heading_aware.py. The 3x speed cost of marko vs mistune adds ~2 minutes to one-time
ingestion of 2,650 files — a non-issue for a batch process.
Tradeoffs accepted: Slower parsing. marko is less widely used than mistune.
Revisit when: Ingestion time exceeds 30 minutes (currently estimated at 8–10 minutes).
```

---

## Decision 1.2 — Document Type Classifier: Hybrid rule-based + LLM fallback

```
Context: Each document must be assigned one of six types (CONCEPT, TASK, API_REF,
DSL_REF, BLOG, RUNBOOK). Kubernetes has explicit content_type in frontmatter; Helm
and Argo CD do not. A single strategy cannot handle all four projects.

Options Considered:
| Option          | Pros                            | Cons                                      |
|-----------------|---------------------------------|-------------------------------------------|
| Pure rule-based | Zero API cost; instant          | Fails on ~30% of Helm/Argo files          |
| Pure LLM        | High accuracy                   | $2–5 cost; 45 min latency for 2,650 files |
| Hybrid          | Accurate + bounded cost         | Two code paths to maintain                |

Decision: Hybrid — rules first, LLM (claude-haiku via Anthropic API) for confidence < 0.80
Reason: Rules handle Kubernetes perfectly (explicit content_type field). LLM fallback
handles the ~20% of Helm/Argo CD files without content_type. Total LLM cost bounded
to ~$0.50 (500 files × ~1k tokens). Note: original design used OpenAI for fallback;
this project has no OpenAI key, so the fallback uses the Anthropic key already required
for generation — one fewer provider dependency.
Tradeoffs accepted: Two classification code paths. LLM adds latency for ambiguous files.
Revisit when: Classification accuracy (measured on hand-labeled sample) drops below 90%.

Rule mapping:
- frontmatter.content_type == "concept"              → CONCEPT  (confidence 1.0)
- frontmatter.content_type == "task"                 → TASK     (confidence 1.0)
- frontmatter.content_type == "reference"            → API_REF  (confidence 1.0)
- path contains "/blog/"                             → BLOG     (confidence 0.95)
- code_block_ratio > 0.6 AND avg_sentence_len < 8    → DSL_REF  (confidence 0.85)
- project == ARGOCD AND "/operator-manual/" in path  → RUNBOOK  (confidence 0.90)
- else                                               → LLM classify (confidence unknown)
```

---

## Decision 1.3 — Incremental Ingestion: SHA-256 checksum with SQLite index

```
Context: The corpus is 2,650 files. Re-embedding everything on every corpus update
wastes API quota (and, on a Cohere trial key, rate-limited time). Need a way to skip
unchanged files.

Options Considered:
| Option          | Pros                              | Cons                                       |
|-----------------|-----------------------------------|--------------------------------------------|
| Full re-ingest  | Simple; always correct            | Hours per run; burns trial-key quota       |
| Git diff-based  | Only truly changed files          | Couples ingestion to git; fails with zips  |
| Checksum-based  | Self-contained; works any source  | Whitespace changes trigger false positives |

Decision: SHA-256 checksum stored in SQLite (corpus/.ingest_index.db)
Reason: Self-contained — works whether corpus comes from git clone or S3 download.
SQLite adds zero operational overhead (no server). False positives (whitespace changes)
cause unnecessary re-embedding but no correctness errors.
Tradeoffs accepted: Whitespace-only changes trigger re-ingestion unnecessarily.
Revisit when: False positive rate causes perceptible cost increase.
```

---

## Decision 2.1 — Chunking Strategy per Document Type: Heading-aware primary, semantic for blogs

```
Context: The CNCF corpus has six document types with fundamentally different structures.
A single strategy applied uniformly produces wrong chunks for at least three of the six types.

The central insight: chunking determines the retrieval unit. Wrong chunks cannot be fixed
downstream — no embedding model or retrieval strategy can reconstruct what was destroyed
at chunk time.

Options Considered per type:

CONCEPT docs:
- Fixed-size: Splits mid-paragraph; destroys semantic units. Rejected.
- Sentence: No structure awareness; loses heading context. Rejected.
- Heading-aware: Section per H2/H3; preserves complete explanations. CHOSEN.

TASK docs:
- Fixed-size: Splits numbered steps across chunks; destroys procedural coherence. Rejected.
- Heading-aware with overlap: Steps live under headings; overlap preserves step references. CHOSEN.

API_REF docs (e.g. PodSpec with 50+ fields):
- Fixed-size alone: Field descriptions split mid-entry. Rejected as sole strategy.
- Heading-aware + fixed fallback: Heading per field group; fixed-size for overlong sections. CHOSEN.

DSL_REF docs (Helm template reference):
- Sentence: Zero prose sentences; fails completely. Rejected.
- Heading-aware + small fixed fallback: Parameter groups under headings; 256-token max preserves entries. CHOSEN.

BLOG docs:
- Heading-aware: Blog posts often have no headings or misleading ones. Unreliable. Rejected.
- Semantic: Embedding-based boundary detection finds topic shifts in narrative text. CHOSEN.
  Cost tradeoff: embedding every sentence at index time — free on Cohere trial, but consumes
  rate-limited quota; applied to the blog corpus only.

RUNBOOK docs:
- Heading-aware with overlap: Operational steps under headings; overlap preserves command context. CHOSEN.

Strategy map and parameters:
| DocType  | Strategy                        | max_tokens | overlap_tokens |
|----------|---------------------------------|------------|----------------|
| CONCEPT  | heading_aware                   | 512        | 0              |
| TASK     | heading_aware                   | 400        | 64             |
| API_REF  | heading_aware + fixed fallback  | 384        | 48             |
| DSL_REF  | heading_aware + fixed fallback  | 256        | 32             |
| BLOG     | semantic                        | 300 target | N/A            |
| RUNBOOK  | heading_aware                   | 512        | 64             |
| UNKNOWN  | fixed_size (with warning log)   | 512        | 64             |

Tradeoffs accepted: Semantic chunking for blogs consumes extra embedding quota at index time.
Different code paths per type add maintenance surface.
Revisit when: ChunkingEvaluator shows split_code_block_count > 0 or p95_token_count > 1000.
```

---

## Decision 3.1 — Embedding Model: OpenAI text-embedding-3-small (1536 dimensions)

```
Context: Need to embed ~25,000–200,000 chunks. Choice affects retrieval quality (MTEB
score), ingestion speed, cost per run, and operational complexity.

Original design used Cohere embed-english-v3.0 (1024 dims, trial key, 100k TPM).
Switched to OpenAI text-embedding-3-small during initial ingestion because the Cohere
trial key's 100k tokens/minute cap made a full-corpus ingest take 40+ minutes with
unpredictable rate-limit errors. The OpenAI paid tier provides 1M tokens/minute — 10×
faster — at essentially the same cost.

Options Considered:
| Model                           | MTEB Retrieval | TPM (trial)  | TPM (paid)  | Dims | Decision    |
|---------------------------------|----------------|--------------|-------------|------|-------------|
| embed-english-v3.0 (Cohere)     | 64.5           | 100k (trial) | 10M (paid)  | 1024 | Original    |
| text-embedding-3-small (OpenAI) | 62.3           | —            | 1M          | 1536 | **CURRENT** |
| text-embedding-3-large (OpenAI) | 64.6           | —            | 1M          | 3072 | Rejected    |
| BAAI/bge-large-en-v1.5 (local)  | 63.7           | —            | GPU server  | 1024 | Rejected    |

Decision: OpenAI text-embedding-3-small at 1536 dimensions
Reason: 1M TPM paid tier makes a full ingestion of all four projects take ~9 minutes
total. MTEB score difference vs Cohere (62.3 vs 64.5) is negligible for this use case.
The Cohere key is still used for reranking (separate API, trial quota sufficient for
per-query usage).

Note on Qdrant dimension change: switching from Cohere (1024 dims) to OpenAI (1536 dims)
requires deleting and recreating the Qdrant collection. The ensure_collection() method
in qdrant_store.py detects this mismatch automatically and recreates the collection.
Qdrant v1.18.2+ stores vectors in a new storage format incompatible with v1.9.0 data
files — a clean start (delete /var/lib/qdrant/storage/) is needed when upgrading Qdrant.

The input_type asymmetry: OpenAI text-embedding-3-small does NOT require separate
document/query input types (unlike Cohere v3). However, EmbeddingService still exposes
embed_documents() and embed_query() methods — this preserves the interface contract and
allows switching back to Cohere on the paid tier without changing call sites.

Tradeoffs accepted: Slightly lower MTEB score than Cohere (62.3 vs 64.5). Requires
OpenAI account. The 1536-dim vectors take ~50% more storage than 1024-dim.
Revisit when: RAGAS answer relevance falls below 0.75 and embeddings are the bottleneck.
```

---

## Decision 4.1 — Vector Store: Qdrant

```
Context: Need persistent vector storage with metadata filtering for version_tag, project,
doc_type. Filters must execute server-side (pre-retrieval) not in application code
(post-retrieval). Must self-host on EC2.

Options Considered:
| Store    | Server-side filters  | Self-host | Production-ready | Decision |
|----------|----------------------|-----------|------------------|----------|
| Qdrant   | Rich payload filters | Yes       | Yes              | CHOSEN   |
| Weaviate | GraphQL-based        | Yes       | Yes              | Rejected |
| Pinecone | Basic + namespaces   | No        | Yes              | Rejected |
| pgvector | Full SQL WHERE       | Yes       | Yes              | Rejected |
| ChromaDB | Basic                | Yes       | Dev only         | Rejected |

Decision: Qdrant
Reason: Server-side payload filtering means irrelevant vectors are never scored —
O(filter_result) not O(total_vectors). For 200k+ chunks filtered to version_tag="v1.29",
only ~15k vectors are scored. Qdrant's Python client is idiomatic; startup is instant;
no configuration needed for development.
Weaviate rejected: GraphQL query syntax adds learning surface with no benefit.
Pinecone rejected: no self-hosting; per-vector pricing; data residency concerns.
pgvector rejected: requires Postgres operational overhead for this project.
ChromaDB rejected: not production-ready; inadequate filter expressiveness.
Tradeoffs accepted: Qdrant is less widely known than Pinecone in the industry.
Revisit when: corpus grows beyond 5M vectors (Qdrant clustering becomes relevant).
```

---

## Decision 4.2 — HNSW Parameters: m=16, ef_construct=100, ef=50 at query time

```
Context: HNSW parameters control the tradeoff between index quality, build time,
memory usage, and query-time recall.

m=16: bidirectional connections per HNSW node. Default. Appropriate for 200k vectors.
Increasing to m=32 doubles memory usage; justified only if recall < 0.90.

ef_construct=100: search width during index build. Higher = better graph = better recall.
100 builds a high-quality graph for this corpus size. 200 would add ~2x build time.

ef=50 at query time: controls recall vs latency tradeoff per query. Set in the search
call, not at collection creation — can be tuned without re-indexing.

Decision: m=16, ef_construct=100, ef=50 (defaults for this corpus size)
Reason: At 200k vectors, default parameters give >0.95 recall for technical documentation
queries. Aggressive tuning is premature before measuring actual recall degradation.
Tradeoffs accepted: Potentially leaving some recall on the table vs higher parameters.
Revisit when: RAGAS context_recall falls below 0.70 and retrieval is identified as the
gap (not chunking).
```

---

## Decision 5.1 — Query Analyzer: Rule-based with LLM fallback

```
Context: Incoming queries must be routed to the correct retrieval strategy. Misrouting
a VERSION_SPECIFIC query to plain top-k retrieves wrong version docs. Misrouting a
CROSS_PROJECT query retrieves only one project's docs.

Decision: Rule-based classification; LLM fallback for ambiguous queries (< 20% expected)
Reason: LLM query classification adds 200–400ms before retrieval. Rules cover 80% of
cases accurately and instantaneously. The latency cost of LLM for every query is not
justified when simple patterns reliably detect version strings and project names.
Tradeoffs accepted: Rule edge cases (e.g. query mentioning a version incidentally).
Revisit when: Query misrouting rate (measured from user feedback) exceeds 10%.

Rules:
- Contains r"v\d+\.\d+" → VERSION_SPECIFIC
- ≥2 project names detected → CROSS_PROJECT
- Starts with how-to patterns → PROCEDURAL
- Starts with explanatory patterns → CONCEPTUAL
- Single broad term, no specifics → EXPLORATORY
- Default → FACTUAL
```

---

## Decision 5.2 — Retrieval Strategy per Query Type

```
Decision:
FACTUAL       → filtered top-k (k=5)                          no filter override
PROCEDURAL    → filtered top-k (k=5) + doc_type=task filter   forces task documents
CONCEPTUAL    → filtered top-k (k=5) + doc_type=concept       forces concept documents
VERSION_SPEC  → filtered top-k (k=5) + version_tag filter     anchors to requested version
CROSS_PROJECT → multi-query, k=3 per project, merged by RRF   guarantees multi-project coverage
EXPLORATORY   → MMR (k=8, lambda=0.6)                         diversifies retrieved set

Reason for doc_type filters on PROCEDURAL/CONCEPTUAL: Retrieving a concept doc to answer
"how do I configure X" returns an explanation, not steps. The filter forces the retrieval
to the document type that actually answers the query type. This is the highest-ROI retrieval
improvement for this corpus.
Tradeoffs accepted: Doc_type filter may miss relevant content in the wrong doc type category.
Revisit when: Type-filtered retrieval produces "cannot_answer" for queries that should be
answerable (measure this in evaluation).
```

---

## Decision 5.3 — Reranking: Cohere Rerank on EXPLORATORY and CROSS_PROJECT only

```
Decision: Apply Cohere rerank-english-v3.0 for EXPLORATORY and CROSS_PROJECT query types.
Skip reranking for FACTUAL, PROCEDURAL, CONCEPTUAL, VERSION_SPECIFIC.

Reason: Reranking adds ~300ms latency. For factual and procedural queries, top-1 cosine
similarity result is correct in >90% of cases — reranking rarely changes it. For exploratory
and cross-project queries, initial retrieval ranking is noisier (diverse topics, multiple
projects) — a cross-encoder that sees the full (query, chunk) pair scores these significantly
more accurately. The trial key's rerank quota (1000 calls/month) also argues for selective
application: reserving rerank for the two query types where it actually moves results.

Latency budget: target p95 < 3s total. Embedding: ~200ms, retrieval: ~100ms, reranking:
~300ms (when applied), generation: ~1.5s. Total with reranking: ~2.1s. Budget satisfied.

Tradeoffs accepted: Exploratory queries consume rerank quota; factual queries do not.
Revisit when: Reranking latency exceeds 500ms (switch to self-hosted cross-encoder).
```

---

## Decision 6.1 — Generation Model: Dual-provider (GPT-4o-mini default, claude-sonnet-4-5 optional)

```
Original design: anthropic claude-sonnet-4-5 only.

Change: GenerationService now supports both providers via GENERATION_PROVIDER env var.
Default is OpenAI GPT-4o-mini. Reason for change: the Anthropic trial account has a
monthly spending cap (~$5 by default). During initial corpus ingestion, the LLM
classifier made ~15,000 API calls (Haiku) for every document — this exhausted the
monthly cap before ingestion completed. GPT-4o-mini became the necessary fallback.

GENERATION_PROVIDER=openai  (default)
  Model: gpt-4o-mini ($0.15/1M input, $0.60/1M output)
  Response format: JSON mode (response_format={"type": "json_object"})
  Cost per query: ~$0.0004–0.0006

GENERATION_PROVIDER=anthropic
  Model: claude-sonnet-4-5 ($3.00/1M input, $15.00/1M output)
  Response format: Instructed JSON in system prompt (no native JSON mode)
  Cost per query: ~$0.02–0.05
  Faithfulness: Higher — Claude follows "answer only from context" more reliably

Which to use:
- For learning/exploration: GPT-4o-mini (default). Adequate faithfulness, very cheap.
- For production or accuracy evaluation: switch to claude-sonnet-4-5 (set GENERATION_PROVIDER=anthropic).

LLM classification (separate from generation):
  Uses claude-haiku-4-5-20251001. Default: DISABLED (DISABLE_LLM_CLASSIFICATION=true).
  Cost if enabled: ~$0.03 for a full re-ingest of all 4 projects (English docs only).
  Burned $2.40 in practice because haiku was called for 11 non-English helm translations
  before the DISABLE flag existed. The try/except and env guard now prevent this.

Tradeoffs accepted:
- GPT-4o-mini may hallucinate slightly more than Claude (lower faithfulness ceiling).
- Two code paths add maintenance surface.
Revisit when: RAGAS faithfulness drops below 0.75 on GPT-4o-mini (switch to Claude).
```

---

## Decision 6.2 — Context Assembly: Score-descending order; 50k token budget cap

```
Decision: Order chunks by similarity score descending. If total tokens exceed 50,000,
drop lowest-scoring chunks until within budget.
Reason: "Lost in the middle" phenomenon means LLMs recall beginning and end of context
better than middle. Score-descending order puts the most relevant chunk first — where
recall is highest. 50k token cap leaves 150k tokens of the 200k context window for
system prompt, query, and generated answer.
Tradeoffs accepted: Lowest-scoring retrieved chunks may be dropped. If these contain
necessary context, faithfulness degrades — this is detectable in RAGAS context_recall.
Revisit when: context_recall falls below 0.70 (increase token budget or reduce chunk size).
```

---

## Decision 7.1 — Evaluation Framework: RAGAS + custom VersionStalenessEvaluator

```
Decision: RAGAS for four standard metrics + custom version staleness metric.

Why RAGAS: Covers the four ways a RAG pipeline fails:
- Faithfulness (≥0.85): Claims grounded in retrieved context? Catches hallucination.
- Answer Relevance (≥0.80): Answer addresses the actual question? Catches wrong-context retrieval.
- Context Precision (≥0.75): Retrieved chunks contributed to the answer? Catches noisy retrieval.
- Context Recall (≥0.70): All needed information was present in context? Catches missing chunks.

Why a custom Version Staleness metric: No standard benchmark measures this. The CNCF
corpus uniquely presents the problem of multiple-version answers. A system scoring 0.85
faithfulness but 15% version staleness is production-unsafe for Kubernetes users — they
could implement a removed API based on old documentation. This is the metric that makes
this project's evaluation distinct from generic RAG benchmarks.

Test set construction:
- Tier 1 (primary, ≥100): Real Kubernetes StackOverflow questions where accepted answer
  cites kubernetes.io. Scraped via StackExchange API.
- Tier 2 (edge cases, 20 hand-crafted): Version-specific queries where correct answer
  changed between k8s versions (e.g., extensions/v1beta1 Ingress removal in v1.22).
- Tier 3 (supplement, 50 LLM-generated): Marked synthetic=True; reported separately.

Tradeoffs accepted: Tier 1 scraping requires StackExchange API access. Tier 2 requires
manual knowledge of version-specific changes — document 20 cases in a reference table.
Revisit when: Test set size exceeds 500 (switch to automated synthetic with human review).
```
