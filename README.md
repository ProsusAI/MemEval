<p align="center">
  <img src="assets/logo.png" alt="MemEval" width="140"/>
  <br/>
  <em>Fair evaluation framework for agent memory</em>
  <br/><br/>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue"/></a>
  <img src="https://img.shields.io/badge/python-3.13%2B-blue"/>
  <a href="https://github.com/ProsusAI/MemEval"><img src="https://img.shields.io/badge/github-ProsusAI%2FMemEval-black?logo=github"/></a>
  <a href="https://github.com/ProsusAI/MemEval/pulls"><img src="https://img.shields.io/badge/PRs-welcome-brightgreen"/></a>
</p>

<p align="center">
  <img src="assets/benchmark.png" width="85%"/>
</p>

MemEval is a reproducible benchmark for agent memory systems, built on [LoCoMo](https://arxiv.org/abs/2402.17753) (1,986 QA pairs across 10 multi-session conversations). It evaluates factual recall, temporal reasoning, multi-hop inference, and adversarial resistance — with identical models, prompts, and scoring across all 8 systems.

We also built **PropMem**, a proposition-based memory that achieves the best cost/quality tradeoff. Rather than building a knowledge graph, PropMem extracts atomic facts, tags each with an entity, and filters by entity at query time. See [PROPMEM.md](PROPMEM.md) for the design.

---

## Results

All 8 systems evaluated on the same 10 LoCoMo conversations (1,986 QA pairs) with **gpt-4.1-mini** and **text-embedding-3-small**. Judge scores from **gpt-5.2** (3 binary dimensions: relevance, completeness, accuracy). Token counts track LLM calls only (chat completions + responses API); embedding calls are excluded.\*

### Token F1

| Rank | System | Overall F1 | Factual | Temporal | Multi-hop | Inferential | Adversarial | Tokens |
|:----:|--------|:----------:|:-------:|:--------:|:---------:|:-----------:|:-----------:|-------:|
| 1 | **PropMem** | **0.561** | 0.362 | **0.585** | 0.528 | **0.234** | **0.803** | **5.9M** |
| 2 | OpenClaw | 0.557 | **0.464** | 0.482 | **0.670** | 0.213 | 0.528 | 16.8M |
| 3 | Full Context | 0.542 | 0.517 | 0.369 | 0.674 | 0.197 | 0.509 | 37.8M |
| 4 | Hindsight | 0.489 | 0.431 | 0.306 | 0.526 | 0.206 | 0.647 | 24.5M |
| 5 | Graphiti | 0.416 | 0.296 | 0.151 | 0.349 | 0.120 | 0.873 | 5.1M |
| 6 | SimpleMem | 0.358 | 0.245 | 0.320 | 0.237 | 0.136 | 0.734 | 11.7M |
| 7 | Mem0 | 0.344 | 0.267 | 0.104 | 0.330 | 0.174 | 0.629 | 3.3M |
| 8 | MemU | 0.299 | 0.190 | 0.068 | 0.233 | 0.076 | 0.704 | 7.1M |

### LLM Judge (gpt-5.2)

| Rank | System | Relevant | Complete | Accurate | Judge Avg | Tokens |
|:----:|--------|:--------:|:--------:|:--------:|:---------:|-------:|
| 1 | **PropMem** | **0.921** | **0.752** | **0.792** | **0.822** | **5.9M** |
| 2 | OpenClaw | 0.904 | 0.595 | 0.676 | 0.725 | 16.8M |
| 3 | Full Context | 0.919 | 0.536 | 0.672 | 0.709 | 37.8M |
| 4 | Hindsight | 0.855 | 0.547 | 0.625 | 0.676 | 24.5M |
| 5 | Graphiti | 0.712 | 0.459 | 0.546 | 0.573 | 5.1M |
| 6 | Mem0 | 0.663 | 0.360 | 0.469 | 0.497 | 3.3M |
| 7 | SimpleMem | 0.568 | 0.426 | 0.441 | 0.478 | 11.7M |
| 8 | MemU | 0.527 | 0.297 | 0.374 | 0.399 | 7.1M |

### Judge Accuracy by Category

| System | Factual | Temporal | Multi-hop | Inferential | Adversarial |
|--------|:-------:|:--------:|:---------:|:-----------:|:-----------:|
| **PropMem** | 0.606 | **0.773** | 0.834 | **0.604** | 0.883 |
| OpenClaw | **0.631** | 0.449 | 0.834 | 0.458 | 0.619 |
| Full Context | 0.691 | 0.315 | **0.850** | 0.479 | 0.623 |
| Hindsight | 0.535 | 0.442 | 0.692 | 0.438 | 0.729 |
| Graphiti | 0.500 | 0.181 | 0.546 | 0.271 | **0.899** |
| Mem0 | 0.472 | 0.093 | 0.492 | 0.354 | 0.717 |
| SimpleMem | 0.305 | 0.355 | 0.347 | 0.323 | 0.789 |
| MemU | 0.319 | 0.040 | 0.333 | 0.229 | 0.758 |

*\*Embedding API calls (text-embedding-3-small) are not included — these are cheap (~$0.02/M tokens) and used by all retrieval-based systems.*

---

## Key Findings

- **PropMem is #1 on both F1 and judge** — and at 5.9M tokens, it's 6.4x cheaper than Full Context (37.8M) and 2.8x cheaper than OpenClaw (16.8M).
- **F1 understates PropMem's lead.** F1 shows a near-tie with OpenClaw (0.561 vs 0.557), but judge completeness (0.752 vs 0.595) and accuracy (0.792 vs 0.676) reveal a clear gap — PropMem gives more complete, correct answers.
- **PropMem dominates temporal reasoning.** Judge accuracy 0.773, nearly double the next best (OpenClaw 0.449). Absolute date extraction during ingestion means the model doesn't need to resolve "last week" at query time.
- **Adversarial resistance correlates with entity awareness.** Graphiti (0.899) and PropMem (0.883) both have entity-centric architectures. Systems without entity filtering (SimpleMem 0.789, MemU 0.758) are worse, and chunk-based systems (OpenClaw 0.619, Full Context 0.623) struggle most.
- **Propositions pre-digest information for cheap models.** Full Context sends raw conversations to gpt-4.1-mini and still scores well on F1 — but at 37.8M tokens. PropMem extracts atomic facts once, so even cheap models answer correctly from pre-processed evidence.

---

## Examples

Real outputs from the LoCoMo benchmark (gpt-4.1-mini, conv-26):

**Factual** — can the system recall specific facts about a person?

```diff
  Question: "What are Melanie's pets' names?"
- OpenClaw: "None"                     [missed entirely]
- Mem0:     "Oliver and Luna"          [missing Bailey]
+ PropMem:  "Bailey, Luna, Oliver"     [all three]
```

**Temporal** — can the system answer questions involving dates and time?

```diff
  Question: "When did Melanie sign up for a pottery class?"
- OpenClaw: "None"                     [missed entirely]
- Mem0:     "None"                     [missed entirely]
+ PropMem:  "2 July, 2023"             [exact date]
```

**Multi-hop** — can the system connect facts across multiple conversation sessions?

```diff
  Question: "Where did Oliver hide his bone once?"
- OpenClaw: "None"                     [missed entirely]
- Mem0:     "None"                     [missed entirely]
+ PropMem:  "in Melanie's slipper"     [correct]
```

---

## Systems

| System | Architecture | Retrieval |
|--------|-------------|-----------|
| **PropMem** | Entity-filtered propositions ([design](PROPMEM.md)) | Entity-scoped proposition search + CoT reasoning |
| **OpenClaw** | Chunk-and-search | Hybrid BM25 + vector search, top-K chunks to LLM |
| **Full Context** | Brute force | Entire conversation in the prompt |
| **Hindsight** | Chunk + hierarchical summary | Summaries for routing, chunks for answering |
| **Graphiti** | Temporal knowledge graph | Graph search over entity nodes and relationship edges |
| **SimpleMem** | Raw text + planning | Multi-round reflection (5+ LLM calls/question) |
| **Mem0** | Fact extraction + search | Vector search over extracted facts |
| **MemU** | Summary extraction | Vector search with intention routing |

---

## Quick Start

```bash
uv sync --all-extras
```

Run the full benchmark:

```bash
uv run python scripts/run_full_benchmark.py --systems all --num-samples 10 --skip-judge
```

Results are saved to `data/` as JSON files.

---

## Add Your System

Write an adapter function and register it in the `SYSTEMS` dict in `scripts/run_full_benchmark.py`:

```python
def run_mysystem(conv: dict, llm_model: str, run_judge: bool) -> list[dict]:
    """Your system: describe what it does."""
    your_system = MyMemorySystem(model=llm_model)
    your_system.ingest(conv)
    return _qa_results(conv, lambda q: your_system.answer(q), run_judge)

SYSTEMS: dict[str, dict] = {
    # ... existing systems ...
    "mysystem": {
        "fn": run_mysystem,
        "architecture": "your architecture description",
        "infrastructure": "your dependencies",
    },
}
```

Then run:

```bash
# Your system vs PropMem on 1 conversation
uv run python scripts/run_full_benchmark.py --systems mysystem,propmem --num-samples 1 --skip-judge

# Full benchmark (10 conversations, 1986 QA pairs)
uv run python scripts/run_full_benchmark.py --systems mysystem --num-samples 10 --skip-judge

# With LLM-as-judge evaluation
uv run python scripts/run_full_benchmark.py --systems mysystem --num-samples 10

# Custom dataset
uv run python scripts/run_full_benchmark.py --systems mysystem --data-file data/your_data.json
```

**Evaluate your own system directly:**

```python
from agents_memory.evaluation import compute_f1
from agents_memory.locomo import download_locomo

data = download_locomo()  # downloads LoCoMo once, caches locally

for conv in data[:1]:
    your_system.ingest(conv)
    for qa in conv["qa"]:
        predicted = your_system.answer(qa["question"])
        f1 = compute_f1(predicted, qa["answer"])
        print(f"F1={f1:.3f}  Q: {qa['question'][:60]}")
```

`compute_f1` is token-level F1 (same metric used in the LoCoMo paper). It handles adversarial questions (empty ground truth) automatically.

### Metrics

| Metric | What it measures | How to get it |
|--------|-----------------|---------------|
| **Token F1** | Word-overlap between predicted and ground truth | `compute_f1(predicted, ground_truth)` — default, free |
| **LLM Judge** | Relevance + completeness + accuracy (3 binary dimensions) | `evaluate_with_judge(question, expected, predicted)` — uses gpt-5.2 |

Token F1 is the primary metric. LLM judge is supplementary — useful when token F1 is misleading (e.g., correct answer phrased differently from ground truth).

### Data format

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
    "session_2": ["..."],
    "session_2_date_time": "2024-02-20 10:00:00"
  },
  "qa": [
    {"question": "Where does Alice live?", "answer": "Berlin", "category": 1},
    {"question": "When did Alice move?", "answer": "January 2024", "category": 2}
  ]
}
```

QA categories: 1=Factual, 2=Temporal, 3=Inferential, 4=Multi-hop, 5=Adversarial (empty answer = correct refusal).

---

## Fairness Notes

- **Graphiti** uses the open-source `graphiti-core` library with Kuzu (embedded graph DB), not the commercial Zep platform which uses Neo4j + BGE-m3 reranking. Zep's published numbers (75-80% accuracy) use a different metric (LLM-judge accuracy, not token F1) and their commercial infrastructure. The Mem0 paper independently measured Zep's platform at token F1 ~0.35-0.50 per category — our 0.416 with the open-source library is in the same range.
- **Mem0** has a known timestamp bug ([mem0ai/mem0#3944](https://github.com/mem0ai/mem0/issues/3944)) where the platform uses current system date instead of conversation timestamps, degrading temporal reasoning. Our Mem0 temporal F1 (0.104) is far below the paper's (0.489). This likely depresses our Mem0 overall F1.
- **MemU** claims "92% accuracy" on LoCoMo but uses LLM-judge binary accuracy — a fundamentally different metric from token F1. Not directly comparable.
- **Hindsight** uses hierarchical summarization (conversation summaries + chunk retrieval). It's the only system that builds both summaries and chunks, explaining the high token count (24.5M).

---

## License

Apache License 2.0 — see [LICENSE](LICENSE).

---

## References

- **Benchmark**: [LoCoMo](https://arxiv.org/abs/2402.17753) — Long-context memory evaluation framework
- **Survey**: [Memory in the Age of AI Agents](https://arxiv.org/abs/2512.13564) — Overview of memory systems for agents
- **Mem0**: [Mem0](https://arxiv.org/abs/2504.19413) — Fact extraction + vector search memory
- **SimpleMem**: [SimpleMem](https://arxiv.org/abs/2601.02553) — Multi-round reflection memory system
- **Graphiti**: [Graphiti](https://github.com/getzep/graphiti) — Temporal knowledge graph by Zep AI
- **Hindsight**: [Hindsight](https://arxiv.org/abs/2503.02972) — Hierarchical summarization memory
- **OpenClaw**: [OpenClaw](https://docs.openclaw.ai/concepts/memory) — OpenClaw memory system
- **MemU**: [MemU](https://github.com/NevaMind-AI/memU) — Summary-based memory with intention routing
