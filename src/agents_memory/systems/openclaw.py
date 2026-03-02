"""OpenClaw: hybrid BM25 + vector chunk retrieval."""

import os

from openai import OpenAI

from agents_memory.locomo import extract_dialogues, format_as_markdown
from agents_memory.openclaw import chunk_markdown, embed_texts, hybrid_search
from agents_memory.systems._helpers import _qa_results

SYSTEM_INFO = {
    "architecture": "hybrid BM25 + vector chunk retrieval",
    "infrastructure": "vector store + BM25",
}

_ANSWER_PROMPT_SHORT = (
    "Answer the question concisely (1-5 words) using ONLY the "
    "provided memories. If not found, say 'None'."
)
_ANSWER_PROMPT_NATURAL = (
    "Answer the question concisely but completely using ONLY the "
    "provided memories. If not found, say 'None'."
)


def run(
    conv: dict, llm_model: str, run_judge: bool,
    category_names: dict | None = None, judge_fn: str | None = None,
) -> list[dict]:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    dialogues = extract_dialogues(conv)
    markdown_text = format_as_markdown(dialogues)

    chunks = chunk_markdown(markdown_text, tokens=400, overlap=80)
    chunk_embeddings = embed_texts(
        [c.text for c in chunks], model="text-embedding-3-small"
    )
    print(f"    Chunks: {len(chunks)}")

    sys_prompt = (
        _ANSWER_PROMPT_NATURAL if judge_fn == "longmemeval"
        else _ANSWER_PROMPT_SHORT
    )

    def answer_fn(question: str) -> str:
        query_emb = embed_texts([question], model="text-embedding-3-small")[0]
        search_results = hybrid_search(
            query=question,
            chunks=chunks,
            chunk_embeddings=chunk_embeddings,
            query_embedding=query_emb,
            top_k=20,
            vector_weight=0.7,
            text_weight=0.3,
        )
        memory_text = "\n---\n".join(c.text for c, _s in search_results)
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

    return _qa_results(
        conv, answer_fn, run_judge,
        category_names=category_names, judge_fn=judge_fn,
    )
