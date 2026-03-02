"""Mem0: memory extraction + vector search."""

import os

from openai import OpenAI

from agents_memory.locomo import extract_dialogues
from agents_memory.systems._helpers import _qa_results, answer_prompt

SYSTEM_INFO = {
    "architecture": "memory extraction + vector search",
    "infrastructure": "mem0 library",
}


def run(
    conv: dict, llm_model: str, run_judge: bool,
    category_names: dict | None = None, judge_fn: str | None = None,
) -> list[dict]:
    from mem0 import Memory

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    dialogues = extract_dialogues(conv)
    sample_id = conv.get("sample_id", "unknown")
    user_id = f"locomo-{sample_id}"

    config = {
        "llm": {
            "provider": "openai",
            "config": {
                "model": llm_model,
                "temperature": 0.1,
                "api_key": os.environ["OPENAI_API_KEY"],
            },
        },
        "embedder": {
            "provider": "openai",
            "config": {
                "model": "text-embedding-3-small",
                "api_key": os.environ["OPENAI_API_KEY"],
            },
        },
        "version": "v1.1",
    }
    memory = Memory.from_config(config)

    # Add dialogues in batches
    batch_size = 10
    for i in range(0, len(dialogues), batch_size):
        batch = dialogues[i : i + batch_size]
        messages = [
            {
                "role": "user",
                "content": f"[{d['speaker']}] ({d['timestamp']}): {d['text']}",
            }
            for d in batch
        ]
        try:
            memory.add(messages, user_id=user_id)
        except Exception as e:
            print(f"    Error adding batch {i // batch_size}: {e}")

    print(f"    Dialogues ingested: {len(dialogues)}")

    sys_prompt = answer_prompt(judge_fn)

    def answer_fn(question: str) -> str:
        search_results = memory.search(query=question, user_id=user_id, limit=20)
        memories_list = (
            search_results.get("results", [])
            if isinstance(search_results, dict)
            else search_results
        )
        memory_text = "\n".join(
            m.get("memory", "") if isinstance(m, dict) else str(m)
            for m in memories_list[:20]
        )
        if not memory_text.strip():
            return "None"
        response = client.chat.completions.create(
            model=llm_model,
            messages=[
                {"role": "system", "content": sys_prompt},
                {
                    "role": "user",
                    "content": f"Memories:\n{memory_text}\n\nQuestion: {question}",
                },
            ],
            max_tokens=50,
            temperature=0.1,
        )
        return response.choices[0].message.content.strip()

    results = _qa_results(
        conv, answer_fn, run_judge,
        category_names=category_names, judge_fn=judge_fn,
    )

    # Cleanup
    try:
        memory.delete_all(user_id=user_id)
    except Exception:
        pass

    return results
