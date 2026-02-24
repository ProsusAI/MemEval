# OpenClaw+ Design & Evolution

OpenClaw+ is the only custom-built system in this benchmark. It wraps [OpenClaw](https://docs.openclaw.ai/concepts/memory)'s hybrid search with LLM-extracted propositions per entity and entity-filtered retrieval. This document covers the design decisions and incremental improvements that got it from F1=0.277 (OpenClaw base) to F1=0.556 on LoCoMo.

## Architecture

Three layers on top of OpenClaw's chunk-and-search infrastructure:

1. **Proposition extraction** (LLM at ingestion) — For each conversation session, the LLM extracts atomic facts as `{entity, fact}` pairs. Each proposition is a self-contained sentence like "Caroline went to the LGBTQ support group on 7 May 2023". ~25 words per proposition vs ~100 words per raw chunk.

2. **Entity-filtered retrieval** — At query time, identify the target entity from the question (string matching), then search only that entity's propositions. Prevents cross-person contamination — the #1 source of errors in OpenClaw.

3. **CoT answer generation** — Structured JSON output with `{reasoning, answer}` fields. The reasoning step forces the model to cite specific evidence before committing to an answer.

## OpenClaw Base (starting point)

Chunk-and-search: markdown files chunked at ~400 tokens with 80-token overlap, indexed into SQLite with BM25 full-text and vector embeddings. Retrieval merges the two signals (0.7 vector / 0.3 BM25 for chunks, 0.8 / 0.2 for propositions since they're shorter). No LLM calls during ingestion — only at query time when top-K chunks are fed to the LLM.

**Strengths**: Simple, no ingestion LLM cost, good adversarial resistance (0.790). With gpt-4.1, reaches F1=0.557.

**Weaknesses**: Highly model-dependent — F1 drops from 0.557 (gpt-4.1) to 0.277 (gpt-4.1-mini). Raw chunks mix speakers causing cross-person confusion. No entity awareness. Temporal reasoning very poor with mini (0.069) because relative dates aren't resolved. Uses 16.5M tokens across the benchmark (~20 raw chunks per answer prompt).

## v0 — Propositions + Entity Reasoning (F1=0.410)

Added the three layers described above on top of OpenClaw's infrastructure.

**Result (single conversation, gpt-4.1)**: F1=0.410. Worse than expected — proposition extraction was noisy and prompts too permissive.

## v1 — Five Targeted Fixes (F1=0.625)

Each fix addressed a specific failure mode observed in v0 results:

### 1. Date-stamped propositions

**Problem**: Temporal questions returned "None" because propositions had no dates. "When did X happen?" was unanswerable.

**Fix**: Added session dates to extraction prompt. Each proposition now carries `[date]` prefix in evidence context.

**Impact**: Temporal F1 jumped significantly — the model can now resolve "when did X happen?".

### 2. Proposition deduplication

**Problem**: Same fact extracted across multiple sessions wasted retrieval slots. If a recurring topic appeared in 5 sessions, the top-30 propositions would contain 5 near-identical entries.

**Fix**: Normalized-text dedup during retrieval (case-insensitive, strip whitespace).

**Impact**: More diverse facts in the top-30 slots, better multi-hop coverage.

### 3. Entity-scoped chunk search

**Problem**: Even with entity-filtered propositions, the chunk retrieval (used as fallback context) still mixed entities. Chunks containing both speakers leaked cross-person information.

**Fix**: Filtered chunks to prefer those mentioning the target entity. Falls back to unfiltered only if zero entity-specific matches.

**Impact**: Adversarial F1 improved — fewer cross-person hallucinations from chunk context.

### 4. Prompt tightening

**Problem**: The model over-answered — fabricated details not in evidence, combined facts from different people. Particularly bad on adversarial questions where the answer should be "None".

**Fix**: Added explicit rules to answer prompt: "ONLY use facts about {entity_name}", "If not clearly stated, say None", "Do NOT attribute another person's experiences to {entity_name}".

**Impact**: Adversarial F1 reached 0.791+ — the model learned to refuse confidently.

### 5. Inferential question routing

**Problem**: "Would X prefer..." questions answered "None" because the answer isn't literally stated in any proposition. The standard prompt only does literal lookup.

**Fix**: Added `_is_inferential()` detector (keyword matching on "would", "could", "likely", etc.) that routes to a specialized prompt encouraging reasoning from known facts, personality, and preferences.

**Impact**: Inferential F1 improved — model reasons from preferences instead of literal lookup.

**Combined result (10 conversations, gpt-4.1-mini)**: F1=0.625, 10.9M tokens. Best system overall.

## Token Reduction (F1=0.611, tokens -53%)

**Problem**: Answer prompt dominated by raw chunks: 10 chunks x ~400 tokens = ~4,000 tokens per question (75% of prompt). Propositions (30 x ~33 tokens = ~1,000 tokens) carried most of the signal.

**Fix**: Reduced `top_k_chunks` from 10 to 3. Kept `top_k_props` at 30.

| Component | Before | After |
|-----------|--------|-------|
| Propositions (30) | ~1,000 tokens | ~1,000 tokens |
| Chunks | 10 x 400 = ~4,000 | 3 x 400 = ~1,200 |
| Template + question | ~300 | ~300 |
| **Total per question** | **~5,300** | **~2,500** |

**Result**: F1=0.611 (within noise of 0.625), tokens dropped from 10.9M to 5.2M (-53%). The 3 remaining chunks act as a safety net for proposition extraction failures (~2% of sessions).

## Generalization — Technical Content (LoCoMo 0.611 → 0.554)

Testing on technical conversations (long sessions with parameter codes, configurations, multi-language content) exposed three failures that didn't surface on LoCoMo's social conversations:

### Problem 1: JSON truncation

`max_tokens=2000` on extraction calls truncated JSON output for long technical sessions (up to ~14K input tokens producing 100+ facts). Truncated JSON → parse error → 0 propositions for that session.

**Fix**: Raised `max_tokens` to 8192. Only pay for actual output tokens, not the ceiling. Added `_recover_partial_json()` — regex-based extraction of complete `{"entity": ..., "fact": ...}` objects from truncated JSON as a fallback.

### Problem 2: Domain-blind extraction prompt

The original prompt listed only social fact categories (hobbies, relationships, feelings). Technical content — parameter codes, configurations, troubleshooting steps, specifications — was silently dropped.

**Fix**: Added technical fact categories to the extraction prompt:
- Technical values, parameters, configurations, specifications
- Recommendations, procedures, troubleshooting steps
- Added rule: "Preserve exact technical values (numbers, codes, formulas, URLs) verbatim"

Changed from JSON array format to JSON object `{"facts": [...]}` for more reliable parsing.

### Problem 3: Language mismatch

English-only prompts caused the model to answer in English even for non-English questions, destroying token-F1 against non-English ground truths.

**Fix**: Added "Answer in the SAME LANGUAGE as the question" to answer prompts. Changed answer format from "1-5 words max" to "direct listing of facts" for better recall on detailed ground truths.

All three fixes are zero-cost — no extra LLM calls, no extra tokens. LoCoMo regressed slightly (0.611 → 0.554) because longer answers reduce precision against its terse ground truths (mean 4-5 words). Factual dropped most (-0.084) while Adversarial improved (+0.013).

## v4 — Retrieval Tuning (F1=0.556)

Six zero-cost fixes to retrieval logic — no extra LLM calls, tokens, or latency:

1. **BM25 AND→OR fallback** — When AND query returns zero results (e.g. a token doesn't appear in any proposition), falls back to OR. Catches partial matches that AND misses entirely.
2. **Safer entity matching** — Replaced 3-char prefix matching ("Car" matching "career") with word-boundary matching on name parts ≥4 characters. Prevents false entity identification.
3. **Stricter chunk entity filtering** — Only falls back to unfiltered chunks when entity-filtered returns zero results (was <3). Reduces cross-person leakage.
4. **Punctuation-stripped dedup** — Dedup normalization now strips punctuation, catching near-duplicates like "likes hiking." vs "likes hiking".
5. **Expanded inferential detection** — Added patterns: "could/should" prefixes, "do you think", "how would", "is it likely", "prefer" in questions.
6. **Vector weight 0.8 for propositions** — Propositions are ~25 words, too short for BM25 to be discriminative. Changed from 0.7/0.3 to 0.8/0.2 vector/BM25 split.

### Results

| Benchmark | Before (v3) | After (v4) | Delta |
|-----------|-------------|------------|-------|
| LoCoMo (10 conv, 1986 QA) | 0.554 | 0.556 | +0.002 |

Marginal gains — the retrieval logic was already reasonable. The remaining gap is likely in proposition extraction quality and LLM answer generation, not retrieval ranking.

## Summary — Full Evolution

| Version | F1 (LoCoMo) | Tokens | Key change |
|---------|-------------|--------|------------|
| OpenClaw base | 0.277 | 16.5M | Chunk-and-search baseline |
| v0 (propositions + entity) | 0.410 | ~11M | Added 3 layers on top of OpenClaw |
| v1 (5 targeted fixes) | 0.625 | 10.9M | Dates, dedup, entity scope, prompt tightening, inferential routing |
| v2 (token reduction) | 0.611 | 5.2M | top_k_chunks 10→3 |
| v3 (generalization) | 0.554 | 5.3M | Technical extraction, language matching, answer format |
| v4 (retrieval tuning) | 0.556 | 5.3M | BM25 fallback, entity matching, dedup, inferential detection, vector weight |

OpenClaw+ is #1 on LoCoMo (0.556 vs Full Context 0.545) using 68% fewer tokens than OpenClaw base (5.3M vs 16.5M).

## Ideas Explored but Not Pursued

After v4, we investigated several architecturally different approaches to push beyond the current ceiling. None cleared the bar for novelty + clear improvement over existing work.

### Hierarchical Memory with Auto-Clustering

Extract propositions → cluster by embedding (k-means or HDBSCAN) → summarize each cluster → search summaries first, drill into the best cluster for full propositions. Inspired by how human memory organizes by theme, not chronology.

**Why not**: [TraceMem](https://arxiv.org/abs/2502.06835) (Feb 2025) already does this with HDBSCAN clustering and hierarchical retrieval. [RAPTOR](https://arxiv.org/abs/2401.18059) applies the same tree-of-summaries mechanism to documents. Novelty is too narrow.

### Pre-computed QA Pairs for Conversational Memory

At ingestion, generate anticipated Q&A pairs with full session context (the LLM sees the entire conversation, not just a chunk). At query time, match the incoming question against pre-computed questions via embedding similarity. Direct answer lookup for hits, fall back to standard retrieval for misses.

**Prior art**: [PAQ/RePAQ](https://arxiv.org/abs/2106.02223) (Facebook 2021) proved this works at scale for Wikipedia (65M QA pairs). [HyPE](https://arxiv.org/abs/2410.13880) generates hypothetical questions at index time for document retrieval. Nobody has applied it to persistent conversational agent memory specifically — but the gap is narrow and the ingestion cost is high (many LLM calls per session to generate QA pairs).

### Narrative Document Memory

Instead of proposition lists, maintain a living prose document per entity — updated incrementally after each session. LLMs are trained on narrative text (next-token prediction), so a well-written paragraph about Caroline's life is closer to training distribution than a bullet list of facts.

**Why not**: We prototyped this. F1=0.237 — roughly half of OpenClaw+, more tokens per question. Two failure modes: (1) incremental rewrites lose specific details (model names, parameter values, exact codes), (2) the 3K-word narrative is too long for gpt-4.1-mini to pinpoint the right fact, causing verbose answers that kill precision. Propositions win because they give the LLM pre-digested, focused evidence rather than a haystack. [PersonaMem-v2](https://arxiv.org/abs/2412.10675) (Dec 2025) is also 80% of this idea.

### Memory as In-Context Demonstrations (MemICD)

Pre-compute entity-tagged QA pairs at ingestion. At query time, retrieve the most relevant pre-computed QA pairs and present them as few-shot ICL demonstrations before the actual question. The LLM sees "here's how questions about Caroline are answered" before answering a new one — enabling generalization to unanticipated questions.

**Prior art**: [DoubleDipper](https://aclanthology.org/2025.ijcnlp-main.8/) (IJCNLP-AACL 2025) generates QA pairs and uses them as ICL demos, but only for single documents — not persistent multi-session memory. The specific combination (persistent user QA pairs + entity tagging + ICL framing) is novel but incremental.

### Why We Stopped Here

The remaining F1 gap is mostly in factual (0.379) and inferential (0.234) categories, where the bottleneck is proposition extraction quality and LLM reasoning — not the retrieval architecture. All five ideas above change how memory is stored or retrieved, but the errors we actually see are: (1) the extraction LLM missing a fact, (2) the answer LLM failing to reason from available evidence. Those are model capability limits, not architecture limits. A better model (gpt-4.1 vs mini) improves scores more than any retrieval change we tested.
