"""Shared utilities for memory system adapters."""

from __future__ import annotations

import asyncio

from agents_memory.evaluation import (
    compute_f1,
    evaluate_longmemeval,
    evaluate_with_judge,
)
from agents_memory.locomo import CATEGORY_NAMES as _DEFAULT_CATEGORIES

ANSWER_PROMPT_SHORT = (
    "Answer the question concisely (1-5 words) using ONLY the "
    "provided memories. If not found, say 'None'."
)
ANSWER_PROMPT_NATURAL = (
    "Answer the question concisely but completely using ONLY the "
    "provided memories. If not found, say 'None'."
)


def answer_prompt(judge_fn: str | None) -> str:
    """Return the appropriate answer prompt based on judge type."""
    return ANSWER_PROMPT_NATURAL if judge_fn == "longmemeval" else ANSWER_PROMPT_SHORT


def print_ingest_stats(ingest: dict) -> None:
    """Print ingestion stats from a PropMem ingest result."""
    parts = [
        f"chunks={ingest['num_chunks']}",
        f"propositions={ingest['num_propositions']}",
    ]
    clusters = ingest.get("num_clusters", 0)
    if clusters:
        parts.append(f"clusters={clusters}")
    print(f"    Ingested: {', '.join(parts)}")


async def _qa_results_async(
    conv: dict,
    answer_fn,
    run_judge: bool,
    category_names: dict | None = None,
    judge_fn: str | None = None,
) -> list[dict]:
    """Evaluate all QA pairs for a conversation.

    Accepts both sync and async answer_fn -- dispatches accordingly.

    Parameters
    ----------
    judge_fn : str | None
        Which judge to use.  ``"longmemeval"`` uses the native LongMemEval
        binary-accuracy judge (matching the ICLR 2025 paper prompts).
        ``None`` (default) uses the generic 3-dimension judge.
    """
    cats = category_names or _DEFAULT_CATEGORIES
    is_async = asyncio.iscoroutinefunction(answer_fn)
    qa_pairs = conv.get("qa", [])
    sample_id = conv.get("sample_id", "unknown")
    results = []

    for i, qa in enumerate(qa_pairs, 1):
        question = qa.get("question", "")
        ground_truth = qa.get("answer", "")
        category = qa.get("category", 0)
        question_id = qa.get("question_id", "")

        try:
            predicted = (await answer_fn(question)) if is_async else answer_fn(question)
        except Exception as err:
            print(f"    Error on Q{i}: {err}")
            predicted = ""

        f1 = compute_f1(predicted, ground_truth)

        row = {
            "sample_id": sample_id,
            "question": question,
            "ground_truth": ground_truth,
            "predicted": predicted,
            "category": category,
            "category_name": cats.get(category, str(category)),
            "f1": f1,
        }

        if run_judge:
            if judge_fn == "longmemeval":
                row.update(evaluate_longmemeval(
                    question, ground_truth, predicted,
                    category=str(category),
                    question_id=question_id,
                ))
            else:
                row.update(evaluate_with_judge(
                    question, ground_truth, predicted,
                ))

        results.append(row)

        if i % 20 == 0:
            print(f"    QA {i}/{len(qa_pairs)} - F1={f1:.3f}")

    return results


def _qa_results(
    conv: dict,
    answer_fn,
    run_judge: bool,
    category_names: dict | None = None,
    judge_fn: str | None = None,
) -> list[dict]:
    """Sync wrapper around _qa_results_async."""
    return asyncio.run(
        _qa_results_async(
            conv, answer_fn, run_judge,
            category_names=category_names,
            judge_fn=judge_fn,
        )
    )


def run_async(async_fn):
    """Wrap an async adapter into the sync signature the runner expects."""
    def wrapper(
        conv, llm_model, run_judge,
        category_names=None, judge_fn=None,
    ):
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                async_fn(
                    conv, llm_model, run_judge,
                    category_names=category_names,
                    judge_fn=judge_fn,
                )
            )
        finally:
            loop.close()
    wrapper.__doc__ = async_fn.__doc__
    return wrapper
