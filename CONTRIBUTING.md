# Contributing

## Add Your Benchmark

MemEval isn't tied to LoCoMo or LongMemEval. Any QA dataset works as long as each conversation has a list of question/answer pairs. Register a loader in `scripts/run_full_benchmark.py`:

```python
BENCHMARKS["mybench"] = {
    "loader": lambda path: load_my_data(path),  # returns list of conversation dicts
    "categories": {1: "Cat-A", 2: "Cat-B"},      # optional category mapping
}
```

Then run with `--benchmark mybench --data-file data/mybench.json`.

The only requirement is that your data follows the same conversation + QA format (see Data Format below). All systems, scoring, and judge evaluation work automatically.

## Evaluate Directly

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

`compute_f1` is token-level F1 (same metric as the LoCoMo paper). It handles adversarial questions (empty ground truth) automatically.

## Metrics

| Metric | What it measures | How to get it |
|--------|-----------------|---------------|
| **Token F1** | Word overlap between predicted and ground truth | `compute_f1(predicted, ground_truth)` |
| **LLM Judge** | Relevance + completeness + accuracy (3 binary dimensions) | `evaluate_with_judge(question, expected, predicted)` uses gpt-5.2 |

Token F1 is the primary metric. The LLM judge is supplementary, useful when F1 is misleading (e.g. correct answer phrased differently from ground truth).

## Data Format

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
