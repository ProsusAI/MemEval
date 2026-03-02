# PropMem

PropMem is a proposition-based memory system built to improve factual precision in
multi-person conversations while controlling token cost.

The motivation is simple: chunk-only retrieval often mixes facts across people or
returns broad context when the question needs a small, person-specific detail.
PropMem addresses this by extracting atomic facts at ingestion time and making
entity-aware retrieval the default path at answer time.

## Foundation

PropMem extends the same core chunk-and-search backbone used by
[OpenClaw](https://docs.openclaw.ai/concepts/memory):

- conversation chunking
- vector embeddings
- BM25 lexical retrieval
- hybrid rank merging

Chunks remain available as fallback context, but PropMem prioritizes proposition
memory first.

## What PropMem Does

### 1) Extract proposition memory at ingestion

For each session, an LLM converts dialogue into atomic `{entity, fact}` entries
with session date context. The extraction prompt is tuned to keep facts
self-contained, preserve concrete values, and resolve relative dates.

Implementation details:

- extraction runs in parallel across sessions for throughput
- malformed/truncated JSON is partially recovered instead of dropped
- extracted propositions are embedded once and reused at query time

### 2) Retrieve with entity discipline

At question time, PropMem first identifies the target entity from the question.
If an entity is found, proposition retrieval is strictly scoped to that entity to
reduce cross-person contamination.

Retrieval combines vector similarity with BM25 scoring over propositions, then
deduplicates near-duplicate facts before final selection. BM25 search uses an AND
query first and falls back to OR when strict matching is empty.

Chunks are retrieved as supporting context. If entity-filtered chunk matches are
available, they are preferred; otherwise retrieval falls back to broader chunk
results.

For large proposition sets where entity matching is unclear, PropMem also uses
embedding-space clustering as a topic pre-filter before ranking. This path is
used only when clustering is enabled and the proposition pool is large, to avoid
excluding relevant facts on smaller conversations.

### 3) Generate constrained answers from evidence

PropMem uses structured JSON output (`reasoning`, `answer`) and explicit prompt
constraints:

This is a constrained chain-of-thought style step: the model first reasons over
retrieved evidence and then returns a concise final answer.

- stay grounded in retrieved evidence
- enforce entity consistency
- answer in the same language as the question
- use inferential reasoning prompts for "would/could/likely" style questions

For single-user conversation formats, PropMem switches prompt style to a
user-centric variant while keeping the same retrieval pipeline.

## Why this design

PropMem is designed as a practical balance between quality and cost:

- proposition memory improves precision on person-specific questions
- fallback chunks preserve broad context when propositions are incomplete
- hybrid retrieval avoids over-reliance on either lexical or semantic matching

Benchmark outcomes and token tradeoffs are reported in the main
[README](README.md) tables.
