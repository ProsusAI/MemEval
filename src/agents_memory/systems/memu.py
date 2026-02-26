"""MemU: memory service with file-based memorize."""

import json
import os
import tempfile
from pathlib import Path

from openai import OpenAI

from agents_memory.locomo import extract_dialogues
from agents_memory.systems._helpers import _qa_results_async, run_async

SYSTEM_INFO = {
    "architecture": "memory service with file-based memorize",
    "infrastructure": "MemU library",
}


@run_async
async def run(conv: dict, llm_model: str, run_judge: bool) -> list[dict]:
    from memu.app.service import MemoryService

    dialogues = extract_dialogues(conv)
    sample_id = conv.get("sample_id", "unknown")
    user_id = f"locomo-{sample_id}"

    service = MemoryService(
        llm_profiles={
            "default": {
                "api_key": os.environ["OPENAI_API_KEY"],
                "chat_model": llm_model,
                "embed_model": "text-embedding-3-small",
            }
        },
        database_config={"metadata_store": {"provider": "inmemory"}},
        retrieve_config={"route_intention": False},
    )

    # Format for MemU
    content = [
        {
            "role": "user" if d["speaker"] == dialogues[0]["speaker"] else "assistant",
            "content": f"[{d['speaker']}]: {d['text']}",
        }
        for d in dialogues
    ]
    conv_data = {"metadata": {"user_id": user_id}, "content": content}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(conv_data, f)
        temp_path = f.name

    try:
        await service.memorize(
            resource_url=temp_path,
            modality="conversation",
            user={"user_id": user_id},
        )
        print(f"    Dialogues ingested: {len(dialogues)}")

        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

        async def answer_fn(question: str) -> str:
            retrieval = await service.retrieve(
                queries=[{"role": "user", "content": {"text": question}}],
                where={"user_id": user_id},
            )
            memories = retrieval.get("items", [])
            memory_text = "\n".join(m.get("summary", "") for m in memories[:10])
            if not memory_text.strip():
                return "None"
            response = client.chat.completions.create(
                model=llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Answer the question concisely (1-5 words) using ONLY the "
                            "provided memories. If not found, say 'None'."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Memories:\n{memory_text}\n\nQuestion: {question}",
                    },
                ],
                max_tokens=50,
                temperature=0.1,
            )
            return response.choices[0].message.content.strip()

        return await _qa_results_async(conv, answer_fn, run_judge)
    finally:
        Path(temp_path).unlink(missing_ok=True)
