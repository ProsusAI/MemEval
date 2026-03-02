<p align="center">
  <img src="assets/logo.png" alt="MemEval" width="140"/>
  <br/>
  <em>Fair evaluation framework for agent memory</em>
  <br/><br/>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue"/></a>
  <a href="https://github.com/ProsusAI/MemEval"><img src="https://img.shields.io/badge/github-ProsusAI%2FMemEval-black?logo=github"/></a>
  <a href="https://github.com/ProsusAI/MemEval/pulls"><img src="https://img.shields.io/badge/PRs-welcome-brightgreen"/></a>
</p>

<p align="center">
  <img src="assets/benchmark.png" width="85%"/>
</p>

Agent memory systems are hard to compare fairly. They are typically evaluated with different LLMs, embedding models, and metrics. MemEval standardizes the setup: same LLM, same embeddings, same scoring pipeline, and end-to-end token cost tracking across ingestion, retrieval, and answer generation. Cost reporting matters because LLM calls often differ by an order of magnitude across architectures.

Evaluation combines token F1 with LLM-as-judge scores, with per-category breakdowns to show where each system actually wins. The framework ships with 9 memory systems and 2 benchmarks ([LoCoMo](https://arxiv.org/abs/2402.17753) and [LongMemEval](https://arxiv.org/abs/2410.10813)), and adding new systems or datasets is straightforward.

We also introduce **PropMem**, which provides the strongest measured quality-to-cost tradeoff in our runs. It extracts atomic facts, tags them by entity, and filters retrieval by entity at query time. See [PROPMEM.md](PROPMEM.md) for the design.

## Results

### LoCoMo

Overall metrics:

<div align="center">

| Rank | System | F1 | Judge | Tokens |
|:----:|:------:|:--:|:-----:|:------:|
| 1 | **PropMem** | **0.605** | **0.823** | 5.9M |
| 2 | OpenClaw | 0.557 | 0.725 | 16.4M |
| 3 | Full Context | 0.542 | 0.709 | 37.5M |
| 4 | Hindsight | 0.489 | 0.676 | 24.2M |
| 5 | Graphiti | 0.416 | 0.573 | 5.1M |
| 6 | Memory-R1 | 0.389 | 0.569 | 3.4M |
| 7 | SimpleMem | 0.358 | 0.478 | 11.4M |
| 8 | Mem0 | 0.344 | 0.497 | **3.0M** |
| 9 | MemU | 0.299 | 0.399 | 6.7M |

</div>

<p align="center"><em>`Tokens` = total system LLM prompt + completion tokens across ingestion, retrieval, and answering; excludes embedding and judge calls.</em></p>

Per-category F1:

<div align="center">

| System | Factual | Temporal | Multi-hop | Inferential | Adversarial |
|:------:|:-------:|:--------:|:---------:|:-----------:|:-----------:|
| PropMem | 0.431 | **0.615** | 0.599 | **0.289** | 0.794 |
| OpenClaw | 0.464 | 0.482 | 0.670 | 0.213 | 0.528 |
| Full Context | **0.517** | 0.369 | **0.674** | 0.197 | 0.509 |
| Hindsight | 0.431 | 0.306 | 0.526 | 0.206 | 0.647 |
| Graphiti | 0.296 | 0.151 | 0.349 | 0.120 | **0.873** |
| Memory-R1 | 0.370 | 0.116 | 0.460 | 0.193 | 0.504 |
| SimpleMem | 0.245 | 0.320 | 0.237 | 0.136 | 0.734 |
| Mem0 | 0.267 | 0.104 | 0.330 | 0.174 | 0.629 |
| MemU | 0.190 | 0.068 | 0.233 | 0.076 | 0.704 |

</div>

<p align="center"><em>10 conversations, 1,986 QA pairs. LLM: gpt-4.1-mini. Embeddings: text-embedding-3-small. Judge: gpt-5.2 (avg of relevance, completeness, accuracy).</em></p>

### LongMemEval

Overall metrics:

<div align="center">

| Rank | System | F1 | Judge | Tokens |
|:----:|:------:|:--:|:-----:|:------:|
| 1 | **PropMem** | **0.550** | **0.716** | 23.1M |
| 2 | SimpleMem | 0.480 | 0.667 | 20.8M |
| 3 | OpenClaw | 0.244 | 0.598 | **0.7M** |
| 4 | Full Context | 0.222 | 0.520 | 10.6M |

</div>

Per-category scores:

<div align="center">

| System | SS-U | SS-A | SS-P | MS | Temp | K-Update |
|:------:|:----:|:----:|:----:|:--:|:----:|:--------:|
| PropMem | **0.851** | **0.767** | 0.147 | **0.582** | 0.424 | **0.528** |
| SimpleMem | 0.752 | 0.566 | 0.126 | 0.382 | **0.578** | 0.475 |
| OpenClaw | 0.401 | 0.432 | 0.127 | 0.082 | 0.185 | 0.234 |
| Full Context | 0.265 | 0.415 | **0.177** | 0.062 | 0.212 | 0.202 |

</div>

Legend: `SS-U` = Single-Session User, `SS-A` = Single-Session Assistant, `SS-P` = Single-Session Preference, `MS` = Multi-Session, `Temp` = Temporal, `K-Update` = Knowledge Update.

<p align="center"><em>Stratified sample of 102 questions (17 per category), conversations up to 500 turns. LLM: gpt-4.1. Embeddings: text-embedding-3-small. Judge: gpt-4o (LongMemEval native binary accuracy, matches the paper's evaluation protocol).</em></p>

**Note on token consumption:** Token costs depend on architecture. Systems with LLM-heavy ingestion (PropMem, SimpleMem) are more expensive on longer conversations. Systems with mostly query-time usage (OpenClaw, Full Context) get cheaper when fewer questions are evaluated. Call count and token count can also move in opposite directions if individual calls are shorter.

## Systems

<div align="center">

| System | Architecture | Retrieval |
|:------:|:------------:|:---------:|
| **PropMem** | Entity-filtered propositions ([design](PROPMEM.md)) | Entity-scoped proposition search + CoT reasoning |
| **OpenClaw** | Chunk-and-search | Hybrid BM25 + vector search, top-K chunks to LLM |
| **Full Context** | Brute force | Entire conversation in the prompt |
| **Hindsight** | Chunk + hierarchical summary | Summaries for routing, chunks for answering |
| **Graphiti** | Temporal knowledge graph | Graph search over entity nodes and relationship edges |
| **SimpleMem** | Raw text + planning | Multi-round reflection (5+ LLM calls/question) |
| **Mem0** | Fact extraction + search | Vector search over extracted facts |
| **Memory-R1** | Two-agent RL ([arXiv:2508.19828](https://arxiv.org/abs/2508.19828)) | SFT+GRPO fine-tuned Qwen-2.5-7B (Memory Manager + Answer Agent) |
| **MemU** | Summary extraction | Vector search with intention routing |

</div>

## Quick Start

Requirements: Python `>=3.11` and `OPENAI_API_KEY` set in your environment (or `.env`).
For full parity with all 9 systems (including MemU and Memory-R1), use Python `>=3.13`.

```bash
uv sync --all-extras
```

Reproduce LoCoMo results:

```bash
uv run python scripts/run_full_benchmark.py --systems all --num-samples 10 --llm-model gpt-4.1-mini
```

Run a single system with a specific LLM (no judge):

```bash
uv run python scripts/run_full_benchmark.py --systems propmem --num-samples 1 --llm-model gpt-4.1-mini --skip-judge
```

Run a single system with judge enabled:

```bash
uv run python scripts/run_full_benchmark.py --systems propmem --num-samples 1 --llm-model gpt-4.1-mini
```

Reproduce LongMemEval results:

```bash
uv run python scripts/run_full_benchmark.py --benchmark longmemeval --data-file data/longmemeval_s_stratified_102.json --systems propmem,simplemem,openclaw,fullcontext --num-samples 102 --llm-model gpt-4.1
```

Generate the LongMemEval stratified sample used in this README:

```bash
uv run python scripts/stratified_sample.py --split s --total 102 --output data/longmemeval_s_stratified_102.json
```

Results are saved to `data/` as JSON files.

## Use PropMem in Your Agent

You can use PropMem directly as an app memory layer (no benchmark runner required):

```python
from agents_memory import PropMemMemory

memory = PropMemMemory(
    user_name="John",
    assistant_name="Assistant",
    llm_model="gpt-4.1-mini",
)

memory.add_session(
    [
        {"speaker": "John", "text": "I prefer quiet coffee shops for work."},
        {"speaker": "Assistant", "text": "Noted. You prefer quiet coffee shops."},
    ],
    session_date="2026-03-01 10:30:00",
)

answer = memory.ask("Where does John prefer to work?")
print(answer)
```

For multiple conversations, call `add_session(...)` for each new session, then query
with `ask(...)` or `ask_with_details(...)`.

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
# Quick test: your system vs PropMem on 1 conversation
uv run python scripts/run_full_benchmark.py --systems mysystem,propmem --num-samples 1 --skip-judge

# Full benchmark with judge (10 conversations, 1986 QA pairs)
uv run python scripts/run_full_benchmark.py --systems mysystem --num-samples 10
```

## Add Your Benchmark

Any QA dataset works. Register a loader in `scripts/run_full_benchmark.py` and run with `--benchmark mybench`. See [CONTRIBUTING.md](CONTRIBUTING.md) for the data format and details.

## Fairness Notes

- **Graphiti** uses the open-source `graphiti-core` library with Kuzu (embedded graph DB), not the commercial Zep platform which uses Neo4j + BGE-m3 reranking. Zep's published numbers (75-80% accuracy) use a different metric (LLM-judge accuracy, not token F1) and their commercial infrastructure. The Mem0 paper independently measured Zep's platform at token F1 ~0.35-0.50 per category. Our 0.416 with the open-source library is in the same range.
- **Mem0**: At evaluation time, there was a reported timestamp-handling issue on the Mem0 platform ([mem0ai/mem0#3944](https://github.com/mem0ai/mem0/issues/3944)) that may affect temporal reasoning. Our Mem0 temporal F1 (0.104) is materially lower than the paper's reported value (0.489), which may depress overall Mem0 performance in this benchmark.
- **MemU** claims "92% accuracy" on LoCoMo but uses LLM-judge binary accuracy, a fundamentally different metric from token F1. Not directly comparable.
- **Hindsight** builds both summaries and chunks, explaining the high token count (24.2M).
- **Memory-R1** is the only system using a fine-tuned local model (Qwen-2.5-7B) rather than API-based LLMs. Results here use a model trained for 100 GRPO steps (undertrained vs. the paper’s schedule). Token usage is 3.4M total (1,986 questions; ~1,705 prompt / ~5.3 completion per question), between Mem0 (3.0M) and Graphiti (5.1M) in efficiency.

## License

Apache License 2.0. See [LICENSE](LICENSE).
Third-party attribution and notices: [NOTICE](NOTICE).

## Disclaimer

MemEval is provided "as is," without warranty of any kind, express or implied,
including but not limited to the warranties of merchantability, fitness for a
particular purpose, and noninfringement. In no event shall the authors or
copyright holders be liable for any claim, damages, or other liability, whether
in an action of contract, tort, or otherwise, arising from, out of, or in
connection with the software or the use or other dealings in the software.

## References

- [LoCoMo](https://arxiv.org/abs/2402.17753) (benchmark)
- [LongMemEval](https://arxiv.org/abs/2410.10813) (benchmark)
- [Memory in the Age of AI Agents](https://arxiv.org/abs/2512.13564) (survey)
- [PropMem Design](PROPMEM.md) (this repo)
- [Mem0](https://arxiv.org/abs/2504.19413)
- [SimpleMem](https://arxiv.org/abs/2601.02553)
- [Graphiti](https://github.com/getzep/graphiti)
- [Hindsight](https://arxiv.org/abs/2503.02972)
- [OpenClaw Memory](https://docs.openclaw.ai/concepts/memory)
- [MemU](https://github.com/NevaMind-AI/memU)
- [Memory-R1](https://arxiv.org/abs/2508.19828) 