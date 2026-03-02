"""Memory-R1: Two-agent RL system for memory management (arXiv:2508.19828).

Uses fine-tuned Qwen-2.5-7B with SFT+GRPO for memory management and QA.
Requires trained adapters in models/ — see training/memory_r1/ for training pipeline.
"""

import json
import os
from collections import defaultdict
from pathlib import Path

from agents_memory.locomo import extract_dialogues
from agents_memory.systems._helpers import _qa_results
from agents_memory.training.memory_r1.prompts import ANSWER_AGENT_PROMPT, MEMORY_MANAGER_PROMPT
from agents_memory.training.memory_r1.rewards import (
    apply_memory_operations,
    parse_mm_output,
)

SYSTEM_INFO = {
    "architecture": "Two-agent RL (Memory Manager + Answer Agent), SFT+GRPO on Qwen-2.5-7B",
    "infrastructure": "Fine-tuned local model (requires adapters in models/)",
}

# Default adapter paths (relative to repo root)
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_DEFAULT_MM_ADAPTER = _REPO_ROOT / "models" / "memory-r1-adapters" / "adapter_memory_manager"
_DEFAULT_AA_ADAPTER = _REPO_ROOT / "models" / "memory-r1-adapters" / "adapter_answer_agent"

# Base model
_DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def _load_model(base_model: str, adapter_path: Path):
    """Load base model with LoRA adapter."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        base_model, dtype=dtype, trust_remote_code=True,
        device_map=device if device != "cpu" else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if adapter_path.exists():
        model = PeftModel.from_pretrained(model, str(adapter_path))
        model = model.merge_and_unload()

    model.eval()
    return model, tokenizer, device


def run(
    conv: dict, llm_model: str, run_judge: bool,
    category_names: dict | None = None, judge_fn: str | None = None,
) -> list[dict]:
    import torch
    from agents_memory.training.memory_r1 import local_token_tracker

    base_model = os.environ.get("MEMORY_R1_BASE_MODEL", _DEFAULT_BASE_MODEL)
    mm_adapter = Path(os.environ.get("MEMORY_R1_MM_ADAPTER", str(_DEFAULT_MM_ADAPTER)))
    aa_adapter = Path(os.environ.get("MEMORY_R1_AA_ADAPTER", str(_DEFAULT_AA_ADAPTER)))

    # Load models
    print("    Loading Memory Manager model...")
    mm_model, tokenizer, device = _load_model(base_model, mm_adapter)
    print("    Loading Answer Agent model...")
    aa_model, _, _ = _load_model(base_model, aa_adapter)

    dialogues = extract_dialogues(conv)

    # --- 1. INGEST: Run Memory Manager over all dialogue turns ---
    memory_bank: list[dict] = []
    next_id = 0

    for d in dialogues:
        turn_text = f"[{d['speaker']}] ({d['timestamp']}) {d['text']}"

        # Format related memories for prompt
        related_for_prompt = [
            {"id": m["id"], "text": m["text"], "speaker": m.get("speaker", "")}
            for m in memory_bank[-10:]  # Use last 10 as context window
        ]

        user_content = MEMORY_MANAGER_PROMPT.format(
            related_memories=json.dumps(related_for_prompt, indent=2) if related_for_prompt
            else "No existing memories yet.",
            new_facts=f"- {turn_text}",
        )

        messages = [{"role": "user", "content": user_content}]
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(input_text, return_tensors="pt").to(device)

        local_token_tracker.count_prompt(input_text, tokenizer)

        with torch.no_grad():
            outputs = mm_model.generate(
                **inputs, max_new_tokens=1024, do_sample=False, temperature=1.0,
            )
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        completion = tokenizer.decode(generated, skip_special_tokens=True)

        local_token_tracker.count_completion(completion, tokenizer)

        # Parse and apply operations
        operations = parse_mm_output(completion)
        for op in operations:
            event = op.get("event", "NONE").upper()
            if event == "ADD":
                memory_bank.append({
                    "id": str(next_id),
                    "text": op.get("text", ""),
                    "speaker": d["speaker"],
                })
                next_id += 1
            elif event == "UPDATE":
                for mem in memory_bank:
                    if mem["id"] == op.get("id"):
                        mem["text"] = op.get("text", mem["text"])
                        break
            elif event == "DELETE":
                memory_bank = [m for m in memory_bank if m["id"] != op.get("id")]

    print(f"    Memory bank: {len(memory_bank)} entries after ingestion")

    # --- 2. ANSWER: Use Answer Agent with memory bank ---
    def answer_fn(question: str) -> str:
        # Group memories by speaker
        by_speaker = defaultdict(list)
        for mem in memory_bank:
            by_speaker[mem.get("speaker", "Unknown")].append(mem)

        memory_lines = []
        for speaker in sorted(by_speaker):
            memory_lines.append(f"\n### {speaker}")
            for i, mem in enumerate(by_speaker[speaker], 1):
                memory_lines.append(f"{i}. {mem['text']}")

        user_content = ANSWER_AGENT_PROMPT.format(
            memories="\n".join(memory_lines),
            question=question,
        )

        messages = [{"role": "user", "content": user_content}]
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(input_text, return_tensors="pt").to(device)

        local_token_tracker.count_prompt(input_text, tokenizer)

        with torch.no_grad():
            outputs = aa_model.generate(
                **inputs, max_new_tokens=512, do_sample=False, temperature=1.0,
            )
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        completion = tokenizer.decode(generated, skip_special_tokens=True)

        local_token_tracker.count_completion(completion, tokenizer)

        # Extract answer after **Answer:** marker
        if "**Answer:**" in completion:
            return completion.split("**Answer:**")[-1].strip()
        if "Answer:" in completion:
            return completion.split("Answer:")[-1].strip()
        lines = [l.strip() for l in completion.splitlines() if l.strip()]
        return lines[-1] if lines else ""

    # --- 3. EVALUATE ---
    results = _qa_results(
        conv, answer_fn, run_judge,
        category_names=category_names, judge_fn=judge_fn,
    )

    local_token_tracker.print_stats("    Memory-R1 token usage:")
    return results
