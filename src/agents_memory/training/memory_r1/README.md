# Memory-R1: Two-Agent RL for Memory Management

Implementation of the Memory-R1 system ([arXiv:2508.19828](https://arxiv.org/abs/2508.19828)) — a two-agent reinforcement learning approach to long-term memory management.

## Architecture

```
Dialogue turns ──► Memory Manager (MM) ──► Memory Bank ──► Answer Agent (AA) ──► Answer
                   ADD / UPDATE / DELETE      structured       distill + answer
                                              facts
```

Both agents are fine-tuned from **Qwen-2.5-7B-Instruct** using:
1. **SFT** — Supervised fine-tuning with QLoRA on LoCoMo conversation data
2. **GRPO** — Group Relative Policy Optimization (reinforcement learning)
   - Phase 1: Train AA with direct Exact Match reward
   - Phase 2: Train MM with indirect EM reward via frozen AA

## Directory Structure

```
training/memory_r1/
├── prompts.py             # System prompts for fact extraction, MM, and AA
├── prepare_data.py        # Generate SFT training data from LoCoMo
├── sft.py                 # QLoRA SFT training (both agents)
├── grpo.py                # GRPO RL training (two-phase)
├── rewards.py             # EM/F1 metrics + MM reward computer
├── eval.py                # Validation evaluation during training
├── callbacks.py           # Structured JSON-lines metrics logging
├── local_token_tracker.py # Token counting for local models
└── README.md
```

The benchmark adapter lives at `systems/memory_r1.py` and follows the standard registry interface.

## Quick Start

### 1. Install training dependencies

```bash
uv sync --extra training
```

### 2. Prepare training data

```bash
# Full run (requires OPENAI_API_KEY for embeddings + optional teacher model)
./scripts/prepare_memory_r1_data.sh

# Without teacher model (uses LoCoMo observations directly)
./scripts/prepare_memory_r1_data.sh --skip-teacher

# Dry run — validate data counts, no API calls
./scripts/prepare_memory_r1_data.sh --dry-run
```

Output: `data/r1_training/` with JSONL files + memory banks.

### 3. SFT training

```bash
# Smoke test (CPU, tiny model, 3 steps)
./scripts/train_memory_r1_sft.sh --smoke-test

# Train Memory Manager
./scripts/train_memory_r1_sft.sh --agent mm

# Train Answer Agent
./scripts/train_memory_r1_sft.sh --agent aa

# Both sequentially
./scripts/train_memory_r1_sft.sh --agent both
```

Output: adapters in `models/adapter_memory_manager/` and `models/adapter_answer_agent/`.

### 4. GRPO RL training

```bash
# Phase 1: Answer Agent (direct EM reward)
./scripts/train_memory_r1_grpo.sh --phase aa --max-steps 500

# Phase 2: Memory Manager (indirect EM via frozen AA)
./scripts/train_memory_r1_grpo.sh --phase mm --max-steps 500

# Both phases sequentially
./scripts/train_memory_r1_grpo.sh --phase both
```

Output: RL adapters in `models/memory-r1-rl/`.

### 5. Run benchmark evaluation

```bash
# Requires trained adapters in models/
uv run python scripts/run_full_benchmark.py --systems memory_r1 --num-samples 1
```

## Training Data Split

| Split | Conversation | Non-adversarial QAs | Turns | Observations |
|-------|-------------|---------------------|-------|--------------|
| Train | conv-26     | 152                 | 419   | 184          |
| Val   | conv-30     | 81                  | 369   | 169          |
| Test  | conv-41–50  | 1,307               | —     | —            |

## Hyperparameters

**SFT**: QLoRA rank=64, alpha=64, dropout=0.05, lr=2e-4, 5 epochs, effective batch=16

**GRPO**: group_size=8, KL=0.01, lr=5e-5, temperature=0.7, max_completion=512 (AA) / 1024 (MM)

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | Required for data prep (embeddings) |
| `TEACHER_MODEL` | `gpt-4o-mini` | Teacher model for MM label generation |
| `MEMORY_R1_BASE_MODEL` | `Qwen/Qwen2.5-7B-Instruct` | Base model override |
| `MEMORY_R1_MM_ADAPTER` | `models/memory-r1-adapters/adapter_memory_manager` | MM adapter path |
| `MEMORY_R1_AA_ADAPTER` | `models/memory-r1-adapters/adapter_answer_agent` | AA adapter path |
