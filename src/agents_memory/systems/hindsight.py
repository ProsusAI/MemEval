"""Hindsight: biomimetic agent memory with multi-strategy retrieval.

Requires a running Hindsight server configured with OpenAI. Start with:

    export OPENAI_API_KEY=sk-xxx
    docker run --rm -p 8888:8888 \
        -e HINDSIGHT_API_LLM_PROVIDER=openai \
        -e HINDSIGHT_API_LLM_MODEL=gpt-4.1 \
        -e HINDSIGHT_API_LLM_API_KEY=$OPENAI_API_KEY \
        -e HINDSIGHT_API_EMBEDDINGS_PROVIDER=openai \
        -e HINDSIGHT_API_EMBEDDINGS_OPENAI_API_KEY=$OPENAI_API_KEY \
        -e HINDSIGHT_API_EMBEDDINGS_OPENAI_MODEL=text-embedding-3-small \
        ghcr.io/vectorize-io/hindsight:latest

Set HINDSIGHT_URL env var if not running on localhost:8888.
"""

import os
from datetime import datetime

from openai import OpenAI

from agents_memory.locomo import extract_dialogues
from agents_memory.systems._helpers import _qa_results_async, answer_prompt, run_async
from agents_memory.token_tracker import _record_call, _record_usage


def _parse_timestamp(ts: str) -> str | None:
    """Parse LoCoMo timestamps like '1:56 pm on 8 May, 2023' to ISO 8601."""
    formats = [
        "%I:%M %p on %d %B, %Y",  # 1:56 pm on 8 May, 2023
        "%I:%M %p on %B %d, %Y",  # 1:56 pm on May 8, 2023
        "%Y-%m-%dT%H:%M:%S",      # already ISO
    ]
    for fmt in formats:
        try:
            return datetime.strptime(ts.strip(), fmt).isoformat()
        except ValueError:
            continue
    return None


def _track_hindsight_usage(response, model: str = "hindsight-server") -> None:
    """Record token usage reported by Hindsight's API into our tracker."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return
    prompt = getattr(usage, "input_tokens", 0) or 0
    completion = getattr(usage, "output_tokens", 0) or 0
    if prompt or completion:
        _record_call(model)
        _record_usage(prompt, completion, model)


SYSTEM_INFO = {
    "architecture": "biomimetic memory with multi-strategy retrieval (TEMPR)",
    "infrastructure": "Hindsight server + PostgreSQL + pgvector",
}


@run_async
async def run(
    conv: dict, llm_model: str, run_judge: bool,
    category_names: dict | None = None, judge_fn: str | None = None,
) -> list[dict]:
    from hindsight_client import Hindsight

    api_key = os.environ["OPENAI_API_KEY"]
    base_url = os.environ.get("HINDSIGHT_URL", "http://localhost:8888")
    sample_id = conv.get("sample_id", "unknown")
    bank_id = f"locomo-{sample_id}"

    hs = Hindsight(base_url=base_url)

    # Create a fresh bank for this conversation
    try:
        await hs.acreate_bank(bank_id=bank_id, name=f"LoCoMo {sample_id}")
    except Exception:
        # Bank may already exist from a previous run
        pass

    # Ingest dialogues
    dialogues = extract_dialogues(conv)
    batch_size = 5
    for i in range(0, len(dialogues), batch_size):
        batch = dialogues[i : i + batch_size]
        items = []
        for d in batch:
            text = f"[{d['speaker']}]: {d['text']}"
            item = {"content": text, "context": "conversation"}
            if d.get("timestamp"):
                iso_ts = _parse_timestamp(d["timestamp"])
                if iso_ts:
                    item["timestamp"] = iso_ts
            items.append(item)
        try:
            resp = await hs.aretain_batch(bank_id=bank_id, items=items)
            _track_hindsight_usage(resp, llm_model)
        except Exception:
            # Fall back to individual retain
            for item in items:
                try:
                    resp = await hs.aretain(
                        bank_id=bank_id,
                        content=item["content"],
                        context=item.get("context"),
                        timestamp=item.get("timestamp"),
                    )
                    _track_hindsight_usage(resp, llm_model)
                except Exception as e:
                    print(f"    Retain error: {e}")
    print(f"    Ingested: {len(dialogues)} dialogue turns")

    # Answer questions using recall + OpenAI
    client = OpenAI(api_key=api_key)
    sys_prompt = answer_prompt(judge_fn)

    async def answer_fn(question: str) -> str:
        recall_resp = await hs.arecall(
            bank_id=bank_id,
            query=question,
            budget="high",
            max_tokens=4096,
        )
        # Extract text from recall results
        results = recall_resp.results if hasattr(recall_resp, "results") else recall_resp
        if isinstance(results, list):
            memory_text = "\n".join(
                r.text if hasattr(r, "text") else str(r) for r in results
            )
        else:
            memory_text = str(results)

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

    return await _qa_results_async(
        conv, answer_fn, run_judge,
        category_names=category_names, judge_fn=judge_fn,
    )
