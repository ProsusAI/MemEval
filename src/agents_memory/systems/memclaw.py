"""MemClaw: proposition-based entity-centric retrieval."""

import os

from openai import OpenAI

from agents_memory.memclaw import MemClawSystem
from agents_memory.systems._helpers import _qa_results

SYSTEM_INFO = {
    "architecture": "proposition-based entity-centric retrieval",
    "infrastructure": "vector store + BM25",
}


def run(conv: dict, llm_model: str, run_judge: bool) -> list[dict]:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    system = MemClawSystem()
    ingest = system.ingest_conversation(conv, client, llm_model)
    print(
        f"    Ingested: chunks={ingest['num_chunks']}, "
        f"propositions={ingest['num_propositions']}"
    )

    return _qa_results(
        conv,
        lambda q: system.answer_question(q, client, llm_model)["answer"],
        run_judge,
    )
