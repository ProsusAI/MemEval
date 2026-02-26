"""Graphiti: temporal knowledge graph (Kuzu embedded)."""

from agents_memory.systems._helpers import _qa_results_async, run_async

SYSTEM_INFO = {
    "architecture": "temporal knowledge graph (Kuzu embedded)",
    "infrastructure": "Graphiti + Kuzu",
}


@run_async
async def run(
    conv: dict, llm_model: str, run_judge: bool, category_names: dict | None = None
) -> list[dict]:
    from agents_memory.answer_openai import AsyncOpenAIAnswerAgent
    from agents_memory.graphiti_system import GraphitiSystem

    system = GraphitiSystem(llm_model=llm_model)
    answer_agent = AsyncOpenAIAnswerAgent(model=llm_model, prompt_style="json_f1")

    try:
        ingest = await system.ingest_conversation(conv)
        print(
            f"    Ingested: episodes={ingest['num_episodes']}, turns={ingest['num_turns']}"
        )

        async def answer_fn(question: str) -> str:
            result = await system.answer_question(question, answer_agent)
            return result["answer"]

        return await _qa_results_async(conv, answer_fn, run_judge, category_names=category_names)
    finally:
        await system.close()
