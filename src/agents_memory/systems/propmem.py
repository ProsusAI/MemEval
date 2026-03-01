"""PropMem: proposition-based entity-centric retrieval."""

import os

from openai import OpenAI

from agents_memory.propmem import PropMemSystem
from agents_memory.systems._helpers import _qa_results

SYSTEM_INFO = {
    "architecture": "proposition-based entity-centric retrieval",
    "infrastructure": "vector store + BM25",
}


def run(
    conv: dict, llm_model: str, run_judge: bool,
    category_names: dict | None = None, judge_fn: str | None = None,
) -> list[dict]:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    system = PropMemSystem()
    ingest = system.ingest_conversation(conv, client, llm_model)
    print(
        f"    Ingested: chunks={ingest['num_chunks']}, "
        f"propositions={ingest['num_propositions']}, "
        f"clusters={ingest.get('num_clusters', 0)}"
    )

    return _qa_results(
        conv,
        lambda q: system.answer_question(q, client, llm_model)["answer"],
        run_judge,
        category_names=category_names,
        judge_fn=judge_fn,
    )
