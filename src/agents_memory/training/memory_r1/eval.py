"""Validation evaluation functions for Memory-R1 RL training.

Provides periodic evaluation during GRPO training:
- evaluate_aa: Greedy decode on val QA pairs, returns EM/F1
- evaluate_mm: Run MM on val turns, apply ops, evaluate via frozen AA
"""

from __future__ import annotations

import json
import random
from typing import TYPE_CHECKING

import torch
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from transformers import AutoModelForCausalLM, AutoTokenizer

from agents_memory.training.memory_r1.rewards import (
    apply_memory_operations,
    compute_em,
    compute_f1,
    extract_answer_from_completion,
    parse_mm_output,
)


def evaluate_aa(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    val_dataset: Dataset | list[dict],
    device: str = "cuda",
    max_new_tokens: int = 512,
    **kwargs,
) -> dict:
    """Evaluate Answer Agent on validation set with greedy decoding.

    Args:
        model: The AA model (potentially with LoRA adapter).
        tokenizer: Tokenizer for the model.
        val_dataset: List of dicts with "prompt" and "gold_answer" keys.
        device: Device for inference.
        max_new_tokens: Maximum generation length.

    Returns:
        Dict with val_em, val_f1, val_em_by_category, n.
    """
    model.eval()
    em_scores = []
    f1_scores = []
    category_ems: dict[str, list[float]] = {}

    for example in val_dataset:
        prompt = example["prompt"]
        gold = example["gold_answer"]
        category = example.get("category", "unknown")

        messages = [{"role": "user", "content": prompt}]
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(input_text, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
            )
        generated = outputs[0][inputs["input_ids"].shape[1] :]
        completion = tokenizer.decode(generated, skip_special_tokens=True)
        predicted = extract_answer_from_completion(completion)

        em = compute_em(predicted, gold)
        f1 = compute_f1(predicted, gold)
        em_scores.append(em)
        f1_scores.append(f1)

        if category not in category_ems:
            category_ems[category] = []
        category_ems[category].append(em)

    n = len(em_scores)
    val_em = sum(em_scores) / n if n > 0 else 0.0
    val_f1 = sum(f1_scores) / n if n > 0 else 0.0
    val_em_by_category = {
        cat: sum(scores) / len(scores) for cat, scores in category_ems.items()
    }

    model.train()
    return {
        "val_em": val_em,
        "val_f1": val_f1,
        "val_em_by_category": val_em_by_category,
        "n": n,
    }


def evaluate_mm(
    model=None,
    frozen_aa=None,
    tokenizer=None,
    val_dataset=None,
    device: str = "cuda",
    max_new_tokens_mm: int = 1024,
    max_new_tokens_aa: int = 512,
    max_eval_samples: int = 20,
    **kwargs,
) -> dict:
    """Evaluate Memory Manager on validation set.

    Runs MM on val turns, applies ops to memory bank, evaluates via frozen AA.
    Subsamples for efficiency since MM reward is expensive.

    Args:
        mm_model: The MM model (potentially with LoRA adapter).
        frozen_aa: Frozen Answer Agent for indirect evaluation.
        tokenizer: Tokenizer shared by both models.
        val_dataset: List of dicts with "prompt", "memory_bank_state", "qa_pairs".
        device: Device for inference.
        max_new_tokens_mm: Maximum generation length for MM.
        max_new_tokens_aa: Maximum generation length for AA.
        max_eval_samples: Max examples to evaluate (subsampling for speed).

    Returns:
        Dict with val_em, val_f1, n.
    """
    mm_model = model
    mm_model.eval()
    frozen_aa.eval()

    # Subsample for efficiency
    examples = list(val_dataset)
    if len(examples) > max_eval_samples:
        examples = random.sample(examples, max_eval_samples)

    all_em_scores = []
    all_f1_scores = []

    for example in examples:
        prompt = example["prompt"]
        bank_json = example.get("memory_bank_state", "[]")
        qa_json = example.get("qa_pairs", "[]")

        # Generate MM output
        messages = [{"role": "user", "content": prompt}]
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(input_text, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = mm_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens_mm,
                do_sample=False,
                temperature=1.0,
            )
        generated = outputs[0][inputs["input_ids"].shape[1] :]
        mm_completion = tokenizer.decode(generated, skip_special_tokens=True)

        # Parse and apply operations
        operations = parse_mm_output(mm_completion)
        try:
            bank = json.loads(bank_json) if isinstance(bank_json, str) else bank_json
        except json.JSONDecodeError:
            bank = []
        updated_bank = apply_memory_operations(bank, operations)

        # Evaluate via frozen AA
        try:
            qa_list = json.loads(qa_json) if isinstance(qa_json, str) else qa_json
        except json.JSONDecodeError:
            qa_list = []

        for qa in qa_list:
            question = qa.get("question", "")
            gold = qa.get("answer", "")

            memory_text = "\n".join(
                f"- [{m.get('key', '?')}]: {m.get('content', '')}"
                for m in updated_bank
            )
            aa_prompt = (
                "You are a helpful assistant. Use the following memories to answer the question.\n\n"
                f"**Memories:**\n{memory_text}\n\n"
                f"**Question:** {question}\n\n"
                "**Answer:**"
            )
            aa_messages = [{"role": "user", "content": aa_prompt}]
            aa_input_text = tokenizer.apply_chat_template(
                aa_messages, tokenize=False, add_generation_prompt=True
            )
            aa_inputs = tokenizer(aa_input_text, return_tensors="pt").to(device)

            with torch.no_grad():
                aa_outputs = frozen_aa.generate(
                    **aa_inputs,
                    max_new_tokens=max_new_tokens_aa,
                    do_sample=False,
                    temperature=1.0,
                )
            aa_generated = aa_outputs[0][aa_inputs["input_ids"].shape[1] :]
            aa_completion = tokenizer.decode(aa_generated, skip_special_tokens=True)
            predicted = extract_answer_from_completion(aa_completion)

            all_em_scores.append(compute_em(predicted, gold))
            all_f1_scores.append(compute_f1(predicted, gold))

    mm_model.train()

    n = len(all_em_scores)
    return {
        "val_em": sum(all_em_scores) / n if n > 0 else 0.0,
        "val_f1": sum(all_f1_scores) / n if n > 0 else 0.0,
        "n": n,
    }
