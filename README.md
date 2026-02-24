# Agents Memory

Semantic memory for agent platforms — what the agent knows *about the user*. Preferences, configurations, history, and personal details extracted from conversations, stored as atomic propositions and retrieved via hybrid search.

- Scoped per user per agent
- Stored as atomic propositions: `"user is allergic to peanuts"`, `"slope setting is 0.5"`
- Retrieved via hybrid search (vector + BM25) and fed to the LLM as context

Our implementation (OpenClaw+) extracts propositions via LLM, embeds them, and retrieves with entity-filtered hybrid search. See [OPENCLAW_PLUS.md](OPENCLAW_PLUS.md) for the full design.

## Benchmark

Benchmarking the semantic memory layer on **factual recall** — can agents recall facts, dates, and relationships from long conversations? Compares 7 systems on the [LoCoMo](https://arxiv.org/abs/2402.17753) dataset.

## Setup

```bash
uv sync --all-extras
```

## Systems

| System | Architecture | Retrieval |
|--------|-------------|-----------|
| **OpenClaw+** | OpenClaw + entity reasoning (custom-built) | Entity-filtered proposition search + CoT reasoning |
| **OpenClaw** | Chunk-and-search ([docs](https://docs.openclaw.ai/concepts/memory)) | Hybrid BM25 + vector search, top-K chunks to LLM |
| **Graphiti** | Temporal knowledge graph ([Zep AI](https://github.com/getzep/graphiti)) | Graph search over entity nodes and relationship edges |
| **Full Context** | Brute force | Entire conversation in the prompt |
| **SimpleMem** | Raw text + planning | Multi-round reflection (5+ LLM calls/question) |
| **Mem0** | Fact extraction + search | Vector search over extracted facts |
| **MemU** | Summary extraction | Vector search with intention routing |

**OpenClaw+** wraps OpenClaw's hybrid search with LLM-extracted propositions per entity and entity-filtered retrieval. See [OPENCLAW_PLUS.md](OPENCLAW_PLUS.md) for design decisions and evolution. The rest are open-source baselines used as-is.

## LoCoMo Benchmark

[LoCoMo](https://arxiv.org/abs/2402.17753) evaluates long-term memory using realistic multi-session dialogues (1,986 QA pairs across 10 conversations). Categories: factual (36%), adversarial (25%), temporal (21%), multi-hop (15%), inferential (4%).

All results with gpt-4.1-mini, text-embedding-3-small, token-level F1 scoring. Token counts include both ingestion (memory construction) and query (answering) — tracked via monkey-patching the OpenAI SDK so all LLM calls are captured regardless of which library makes them.

### F1 Scores (all 5 categories)

| System | Overall | Factual | Temporal | Multi-hop | Adversarial | Tokens |
|--------|---------|---------|----------|-----------|-------------|--------|
| **OpenClaw+** | **0.556 ± 0.037** | 0.379 | 0.506 | 0.540 | **0.812** | 5.3M |
| Full Context | 0.545 ± 0.036 | **0.517** | 0.380 | **0.675** | 0.504 | 37.5M |
| SimpleMem | 0.470 ± 0.043 | 0.393 | **0.583** | 0.555 | 0.299 | 22.5M |
| Graphiti | 0.415 ± 0.031 | 0.279 | 0.135 | 0.367 | **0.875** | 0.7M |
| Mem0 | 0.345 ± 0.037 | 0.279 | 0.121 | 0.344 | 0.595 | 3.0M |
| MemU | 0.310 ± 0.028 | 0.192 | 0.064 | 0.235 | 0.760 | 6.9M |
| OpenClaw | 0.277 ± 0.028 | 0.230 | 0.069 | 0.120 | 0.790 | 16.5M |

**Note on adversarial:** Adversarial questions (25% of LoCoMo) test whether a system correctly refuses to answer trick questions about the wrong person — e.g. "What did Caroline realize after her charity race?" when Caroline never ran one. Published papers ([SimpleMem](https://arxiv.org/abs/2601.02553), [Mem0](https://arxiv.org/abs/2504.19413)) typically exclude adversarial from their reported averages. Without adversarial, the ranking changes:

| System | F1 (excl. adversarial) | Tokens |
|--------|------------------------|--------|
| Full Context | 0.556 | 37.5M |
| SimpleMem | 0.520 | 22.5M |
| **OpenClaw+** | **0.482** | **5.3M** |
| Graphiti | 0.281 | 0.7M |
| Mem0 | 0.273 | 3.0M |
| MemU | 0.180 | 6.9M |
| OpenClaw | 0.129 | 16.5M |

OpenClaw+ drops to #3 on pure retrieval but remains the most token-efficient of the top 3 (5.3M vs 37.5M and 22.5M). Its advantage is specifically adversarial resistance — entity-filtered retrieval prevents cross-person hallucination, which matters in multi-person conversations but isn't tested by the other 4 categories.

### Key Findings

- **Full Context and SimpleMem lead on pure retrieval** (excluding adversarial). Full Context has the advantage of seeing everything; SimpleMem's multi-round reflection is genuinely effective.
- **OpenClaw+ is the best cost/quality tradeoff** — #3 on retrieval at 7x fewer tokens than Full Context, #1 when adversarial resistance matters.
- **Adversarial resistance correlates with entity awareness**, not retrieval quality. Graphiti (0.875) and OpenClaw+ (0.812) both have entity-centric architectures. SimpleMem (0.299) has none.
- **OpenClaw collapses with gpt-4.1-mini (0.277)** — raw chunks need a strong model. With gpt-4.1 it jumps to 0.557.
- **Structured retrieval compensates for weaker models** — OpenClaw+ pre-digests information so even cheap models answer correctly.

### Model Sensitivity (gpt-4.1 vs gpt-4.1-mini)

Tested on 2 conversations (304 QA pairs) with gpt-4.1:

| System | gpt-4.1-mini | gpt-4.1 | Delta | Tokens (gpt-4.1) |
|--------|-------------|---------|-------|-------------------|
| OpenClaw+ | 0.556 | 0.601* | +0.045 | 780K |
| OpenClaw | 0.277 | **0.557** | **+0.280** | 2,481K |
| Graphiti | 0.415 | 0.337 | -0.078 | 96K |

OpenClaw doubles with gpt-4.1 (+101%). OpenClaw+ barely changes — propositions do the hard work upfront. OpenClaw costs 3.2x more tokens even when it catches up on F1.

*gpt-4.1 OpenClaw+ number is pre-generalization; current code would be slightly lower.

## Reproduce Results

Run all systems on the full dataset (10 conversations, 1,986 QA pairs):


```bash
uv run python scripts/run_full_benchmark.py \
--systems all \
--num-samples 10 \
--skip-judge
```
Results are saved to `data/` as JSON files.

## Evaluate Your Own System

The benchmark harness is designed for head-to-head comparison. You can plug in any memory system and get comparable F1 scores against the same dataset.

### Quick start: use the evaluation functions directly

```python
from agents_memory.evaluation import compute_f1
from agents_memory.locomo import download_locomo

data = download_locomo()  # downloads LoCoMo once, caches locally

for conv in data[:1]:  # start with 1 conversation
    # --- Your system: ingest the conversation ---
    your_system.ingest(conv)

    # --- Evaluate against ground truth ---
    for qa in conv["qa"]:
        predicted = your_system.answer(qa["question"])
        f1 = compute_f1(predicted, qa["answer"])
        print(f"F1={f1:.3f}  Q: {qa['question'][:60]}")
```

`compute_f1` is token-level F1 (same metric used in the LoCoMo paper and all results above). It handles adversarial questions (empty ground truth) automatically.

### Add your system to the benchmark runner

Write an adapter function and register it:

```python
# In scripts/run_full_benchmark.py

def run_mysystem(conv: dict, llm_model: str, run_judge: bool) -> list[dict]:
    """Your system: describe what it does."""
    # 1. Ingest conversation (conv has LoCoMo format — see below)
    your_system = MyMemorySystem(model=llm_model)
    your_system.ingest(conv)

    # 2. Evaluate all QA pairs (handles scoring + optional LLM judge)
    return _qa_results(
        conv,
        lambda q: your_system.answer(q),  # str -> str
        run_judge,
    )

# Register it
SYSTEMS["mysystem"] = {
    "fn": run_mysystem,
    "architecture": "your architecture description",
    "infrastructure": "your dependencies",
}
```

Then run:

```bash
# Your system vs OpenClaw+ on 1 conversation
uv run python scripts/run_full_benchmark.py --systems mysystem,cortex --num-samples 1 --skip-judge

# Full benchmark (10 conversations, 1986 QA pairs)
uv run python scripts/run_full_benchmark.py --systems mysystem --num-samples 10 --skip-judge

# With LLM-as-judge evaluation (uses gpt-5.2, costs more)
uv run python scripts/run_full_benchmark.py --systems mysystem --num-samples 10

# Custom dataset instead of LoCoMo
uv run python scripts/run_full_benchmark.py --systems mysystem --data-file data/your_data.json
```

### Data format

LoCoMo format (used by both the standard dataset and custom benchmarks):

```json
{
  "sample_id": "conv-1",
  "conversation": {
    "speaker_a": "Alice",
    "speaker_b": "Bob",
    "session_1": [
      {"speaker": "Alice", "text": "I just moved to Berlin", "dia_id": "1"},
      {"speaker": "Bob", "text": "How's the weather?", "dia_id": "2"}
    ],
    "session_1_date_time": "2024-01-15 14:30:00",
    "session_2": [...],
    "session_2_date_time": "2024-02-20 10:00:00"
  },
  "qa": [
    {"question": "Where does Alice live?", "answer": "Berlin", "category": 1},
    {"question": "When did Alice move?", "answer": "January 2024", "category": 2}
  ]
}
```

QA categories: 1=Factual, 2=Temporal, 3=Inferential, 4=Multi-hop, 5=Adversarial (empty answer = correct refusal).

### Metrics

| Metric | What it measures | How to get it |
|--------|-----------------|---------------|
| **Token F1** | Word-overlap between predicted and ground truth | `compute_f1(predicted, ground_truth)` — default, free |
| **LLM Judge** | Relevance + completeness + accuracy (3 binary dimensions) | `evaluate_with_judge(question, expected, predicted)` — uses gpt-5.2 |

Token F1 is the primary metric for comparison. LLM judge is supplementary — useful when token F1 is misleading (e.g., correct answer phrased differently from ground truth).

## Fairness Notes

- **Graphiti** uses the open-source `graphiti-core` library with Kuzu (embedded graph DB), not the commercial Zep platform which uses Neo4j + BGE-m3 reranking. Zep's published numbers (75-80% accuracy) use a different metric (LLM-judge accuracy, not token F1) and their commercial infrastructure. The Mem0 paper independently measured Zep's platform at token F1 ~0.35-0.50 per category — our 0.415 with the open-source library is in the same range.
- **Mem0** has a known timestamp bug ([mem0ai/mem0#3944](https://github.com/mem0ai/mem0/issues/3944)) where the platform uses current system date instead of conversation timestamps, degrading temporal reasoning. Our Mem0 temporal F1 (0.121) is far below the paper's (0.489). This likely depresses our Mem0 overall F1.
- **MemU** claims "92% accuracy" on LoCoMo but uses LLM-judge binary accuracy — a fundamentally different metric from token F1. Not directly comparable.
- **SimpleMem** results are close to the paper's: our 4-category average is 45.8 vs paper's 43.2 (temporal matches exactly at 58.3 vs 58.6).

## Future Considerations
- **Tool-calling integration** — expose `recall()` and `forget()` as agent tools for mid-conversation memory management
- **Database backend** — replace JSON file persistence with a vector database (pgvector, Qdrant, LanceDB) for scale; the `Memory` API stays the same
- **Async support** — add async wrappers for high-throughput platforms

## References

| Paper | Link |
|-------|------|
| LoCoMo benchmark | [arXiv:2402.17753](https://arxiv.org/abs/2402.17753) |
| Memory in the Age of AI Agents (survey) | [arXiv:2512.13564](https://arxiv.org/abs/2512.13564) |
| Mem0 | [arXiv:2504.19413](https://arxiv.org/abs/2504.19413) |
| SimpleMem | [arXiv:2601.02553](https://arxiv.org/abs/2601.02553) |
| Graphiti (Zep AI) | [github.com/getzep/graphiti](https://github.com/getzep/graphiti) |
