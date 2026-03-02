"""Track token consumption during local model training/inference.

Unlike the main token_tracker (which monkey-patches the OpenAI client for API-based
systems), this tracker is for local/open-source models where tokens are counted
directly from the tokenizer.
"""

import threading
from typing import Optional

_lock = threading.Lock()
_stats = {
    "prompts": 0,
    "completions": 0,
    "prompt_tokens": 0,
    "completion_tokens": 0,
}


def count_prompt(prompt: str, tokenizer) -> int:
    """Count tokens in a prompt and update stats."""
    tokens = tokenizer.encode(prompt, add_special_tokens=True)
    num_tokens = len(tokens)

    with _lock:
        _stats["prompts"] += 1
        _stats["prompt_tokens"] += num_tokens

    return num_tokens


def count_completion(completion: str, tokenizer) -> int:
    """Count tokens in a completion and update stats."""
    tokens = tokenizer.encode(completion, add_special_tokens=False)
    num_tokens = len(tokens)

    with _lock:
        _stats["completions"] += 1
        _stats["completion_tokens"] += num_tokens

    return num_tokens


def count_prompt_completion(prompt: str, completion: str, tokenizer) -> tuple[int, int]:
    """Count tokens in both prompt and completion."""
    prompt_tokens = count_prompt(prompt, tokenizer)
    completion_tokens = count_completion(completion, tokenizer)
    return prompt_tokens, completion_tokens


def get_stats() -> dict:
    """Return current token statistics."""
    with _lock:
        return {
            "prompts": _stats["prompts"],
            "completions": _stats["completions"],
            "prompt_tokens": _stats["prompt_tokens"],
            "completion_tokens": _stats["completion_tokens"],
            "total_tokens": _stats["prompt_tokens"] + _stats["completion_tokens"],
        }


def reset() -> None:
    """Reset all counters to zero."""
    with _lock:
        _stats["prompts"] = 0
        _stats["completions"] = 0
        _stats["prompt_tokens"] = 0
        _stats["completion_tokens"] = 0


def print_stats(prefix: str = "") -> None:
    """Print current statistics in a readable format."""
    stats = get_stats()
    if prefix:
        print(f"\n{prefix}")
    print(f"  Prompts: {stats['prompts']:,}")
    print(f"  Completions: {stats['completions']:,}")
    print(f"  Prompt tokens: {stats['prompt_tokens']:,}")
    print(f"  Completion tokens: {stats['completion_tokens']:,}")
    print(f"  Total tokens: {stats['total_tokens']:,}")

    if stats['prompts'] > 0:
        print(f"  Avg tokens/prompt: {stats['prompt_tokens'] / stats['prompts']:.1f}")
    if stats['completions'] > 0:
        print(f"  Avg tokens/completion: {stats['completion_tokens'] / stats['completions']:.1f}")
