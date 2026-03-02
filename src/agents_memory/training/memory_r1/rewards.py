"""Reward functions for Memory-R1 GRPO training.

Implements reward functions matching TRL GRPOTrainer's `reward_funcs` interface:
    fn(completions, **kwargs) -> list[float]

Two reward modes:
1. Answer Agent (AA): Direct EM/F1 against gold answer (Paper Eq. 4)
2. Memory Manager (MM): Indirect EM via frozen AA after applying memory operations
"""

from __future__ import annotations

import json
import re
import string
import unicodedata
from collections import Counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Text normalization helpers
# ---------------------------------------------------------------------------

def normalize_answer(text: str) -> str:
    """Normalize answer text for EM/F1 comparison (SQuAD-style)."""
    text = text.lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = " ".join(text.split())
    return text.strip()


def extract_answer_from_completion(completion: str) -> str:
    """Extract answer text after **Answer:** marker from model completion."""
    # Try bold markdown format: **Answer:** <text>
    match = re.search(r'\*\*Answer:\*\*\s*(.*)', completion, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback: plain "Answer:" without bold markers
    match = re.search(r'Answer:\s*(.*)', completion, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    # Last resort: return the last non-empty line
    lines = [l.strip() for l in completion.splitlines() if l.strip()]
    return lines[-1] if lines else ""


# ---------------------------------------------------------------------------
# Core EM / F1 metrics
# ---------------------------------------------------------------------------

def compute_em(predicted: str, gold: str) -> float:
    """Binary exact match after normalization. Returns 1.0 or 0.0."""
    return 1.0 if normalize_answer(predicted) == normalize_answer(gold) else 0.0


def compute_f1(predicted: str, gold: str) -> float:
    """Token-level F1 score after normalization."""
    pred_tokens = normalize_answer(predicted).split()
    gold_tokens = normalize_answer(gold).split()
    if not gold_tokens:
        return 1.0 if not pred_tokens else 0.0
    if not pred_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Memory Manager reward helpers
# ---------------------------------------------------------------------------

def parse_mm_output(completion: str) -> list[dict]:
    """Parse Memory Manager JSON output into a list of operations.

    Expected: {"memory": [...]} or just [...]. Handles markdown fences and partial JSON.
    """
    text = completion.strip()
    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    # Try parsing as {"memory": [...]}
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and "memory" in parsed:
            return parsed["memory"]
        if isinstance(parsed, list):
            return parsed
        return [parsed] if isinstance(parsed, dict) else []
    except json.JSONDecodeError:
        pass

    # Try extracting JSON object or array
    for pattern in [r'\{.*\}', r'\[.*\]']:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group())
                if isinstance(parsed, dict) and "memory" in parsed:
                    return parsed["memory"]
                if isinstance(parsed, list):
                    return parsed
                return [parsed] if isinstance(parsed, dict) else []
            except json.JSONDecodeError:
                continue
    return []


def apply_memory_operations(memory_bank: list[dict], operations: list[dict]) -> list[dict]:
    """Apply parsed MM operations to a memory bank state."""
    bank = [m.copy() for m in memory_bank]
    next_id = max((int(m.get("id", 0)) for m in bank), default=-1) + 1

    for op in operations:
        event = op.get("event", "NONE").upper()

        if event == "ADD":
            bank.append({
                "id": op.get("id", str(next_id)),
                "text": op.get("text", ""),
            })
            next_id += 1
        elif event == "UPDATE":
            op_id = op.get("id", "")
            for mem in bank:
                if mem.get("id") == op_id:
                    if "text" in op:
                        mem["text"] = op["text"]
                    break
        elif event == "DELETE":
            op_id = op.get("id", "")
            bank = [m for m in bank if m.get("id") != op_id]

    return bank


# ---------------------------------------------------------------------------
# Memory Manager reward computer (callable class)
# ---------------------------------------------------------------------------

class MMRewardComputer:
    """Callable reward function for Memory Manager training.

    Holds a frozen AA. For each MM completion:
    1. Parse JSON ops -> apply to memory bank -> run frozen AA -> average EM = reward
    """

    def __init__(
        self,
        frozen_aa_model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        max_new_tokens: int = 512,
        device: str | None = None,
    ):
        import torch

        self.frozen_aa = frozen_aa_model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.frozen_aa.eval()
        self.__name__ = "mm_reward"

    def _run_frozen_aa(self, memories: list[dict], question: str) -> str:
        """Run frozen AA on a single QA pair with given memories."""
        import torch

        memory_text = "\n".join(
            f"- [{m.get('id', '?')}]: {m.get('text', '')}" for m in memories
        )
        prompt = (
            "You are answering a question about a conversation between two people, "
            "using retrieved memories.\n\n"
            f"## Retrieved Memories\n{memory_text}\n\n"
            f"## Question\n{question}\n\n"
            "**Answer:**"
        )
        messages = [{"role": "user", "content": prompt}]
        input_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.frozen_aa.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=1.0,
            )
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def __call__(
        self,
        completions: list[str],
        memory_bank_state: list[str],
        qa_pairs: list[str],
        **kwargs,
    ) -> list[float]:
        """Compute MM reward for a batch of completions.

        Args:
            completions: Plain string completions from GRPOTrainer.
            memory_bank_state: JSON-serialized memory banks (one per prompt).
            qa_pairs: JSON-serialized QA pair lists (one per prompt).
        """
        rewards = []
        for completion, bank_json, qa_json in zip(completions, memory_bank_state, qa_pairs):
            operations = parse_mm_output(completion)
            try:
                bank = json.loads(bank_json) if isinstance(bank_json, str) else bank_json
            except json.JSONDecodeError:
                bank = []
            updated_bank = apply_memory_operations(bank, operations)

            try:
                qa_list = json.loads(qa_json) if isinstance(qa_json, str) else qa_json
            except json.JSONDecodeError:
                qa_list = []

            if not qa_list:
                rewards.append(0.0)
                continue

            em_scores = []
            for qa in qa_list:
                question = qa.get("question", "")
                gold = qa.get("answer", qa.get("gold_answer", ""))
                predicted = self._run_frozen_aa(updated_bank, question)
                answer = extract_answer_from_completion(predicted)
                em_scores.append(compute_em(answer, gold))

            rewards.append(sum(em_scores) / len(em_scores))

        return rewards
