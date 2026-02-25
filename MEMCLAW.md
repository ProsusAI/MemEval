# MemClaw

MemClaw is the only custom-built memory system in this benchmark. It extends [OpenClaw](https://docs.openclaw.ai/concepts/memory)'s chunk-and-search infrastructure with three layers: LLM-extracted propositions, entity-filtered retrieval, and chain-of-thought answer generation.

## Base: OpenClaw

OpenClaw chunks conversations into ~400-token markdown segments with 80-token overlap, indexed into SQLite with both BM25 full-text search and vector embeddings. At query time, retrieval merges the two signals (weighted 0.7 vector / 0.3 BM25 for chunks, 0.8 / 0.2 for propositions). No LLM calls at ingestion — only at query time when top-K chunks are fed to the model.

## What MemClaw Adds

**1. Proposition extraction (at ingestion)**
For each conversation session, an LLM extracts atomic facts as date-stamped `{entity, fact}` pairs — e.g. "[7 May 2023] Caroline went to the LGBTQ support group". Each proposition is ~25 words vs ~100 words per raw chunk. The extraction prompt covers both social facts (hobbies, relationships, feelings) and technical content (parameters, configurations, specifications), with instructions to preserve exact values verbatim. A `_recover_partial_json()` fallback handles truncated LLM output.

**2. Entity-filtered retrieval (at query time)**
The target entity is identified from the question via word-boundary matching, then both proposition and chunk search are scoped to that entity only. This prevents cross-person contamination — the dominant error mode in base OpenClaw for multi-person conversations. Propositions are deduplicated (case-insensitive, punctuation-stripped) so the top-30 slots contain diverse facts rather than near-identical repeats. BM25 falls back from AND to OR queries when the strict match returns zero results.

**3. Chain-of-thought answer generation**
Structured JSON output with `{reasoning, answer}` fields forces the model to cite specific evidence before committing to an answer. Prompt rules enforce entity discipline ("ONLY use facts about {entity_name}", "If not clearly stated, say None"). An `_is_inferential()` detector routes questions like "would X prefer..." to a specialized prompt that reasons from known facts and preferences rather than requiring literal evidence. All prompts include "Answer in the SAME LANGUAGE as the question" for multilingual support.

## Cost

Each question prompt uses ~30 propositions (~1,000 tokens) + 3 fallback chunks (~1,200 tokens) + template (~300 tokens) ≈ 2,500 tokens. Total across the full LoCoMo benchmark (1,986 QA pairs, 10 conversations): 5.3M tokens — 68% fewer than base OpenClaw (16.5M).