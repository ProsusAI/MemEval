"""<System Name>: <one-line description>.

Copy this file, rename it, fill in the three sections below.
Run: uv run python scripts/run_full_benchmark.py --systems <filename> --num-samples 1 --skip-judge
"""

import os

from openai import OpenAI

from agents_memory.locomo import extract_dialogues
from agents_memory.systems._helpers import _qa_results

SYSTEM_INFO = {
    "architecture": "<describe the retrieval/memory approach>",
    "infrastructure": "<key dependencies>",
}


def run(conv: dict, llm_model: str, run_judge: bool) -> list[dict]:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    dialogues = extract_dialogues(conv)

    # --- 1. INGEST: store dialogues in your memory system ---
    # my_system = MySystem(model=llm_model)
    # my_system.ingest(dialogues)
    # print(f"    Ingested: {len(dialogues)} turns")

    # --- 2. ANSWER: define a function that takes a question and returns a short answer ---
    def answer_fn(question: str) -> str:
        # context = my_system.retrieve(question)
        # return my_system.answer(question, context)
        raise NotImplementedError("Fill in answer logic")

    # --- 3. EVALUATE: this handles scoring, don't change ---
    return _qa_results(conv, answer_fn, run_judge)
