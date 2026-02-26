"""Full context baseline: entire conversation in prompt."""

import os

from openai import OpenAI

from agents_memory.locomo import extract_dialogues
from agents_memory.systems._helpers import _qa_results

SYSTEM_INFO = {
    "architecture": "full conversation in prompt (upper bound)",
    "infrastructure": "none",
}


def run(
    conv: dict, llm_model: str, run_judge: bool, category_names: dict | None = None
) -> list[dict]:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    dialogues = extract_dialogues(conv)

    # Format conversation
    lines = []
    current_timestamp = None
    for d in dialogues:
        if d["timestamp"] != current_timestamp:
            current_timestamp = d["timestamp"]
            lines.append(f"\n--- {current_timestamp} ---\n")
        lines.append(f"[{d['speaker']}]: {d['text']}")
    conversation_text = "\n".join(lines)
    print(f"    Context: ~{len(conversation_text) // 4} tokens")

    prompt_template = (
        "You are answering questions about a conversation between two people.\n"
        "The conversation history is provided below. Answer based ONLY on information "
        "in the conversation.\n\n"
        "Rules:\n"
        "1. Give the SHORTEST answer possible - just the key fact (1-5 words max)\n"
        "2. Use EXACT words from the conversation when possible\n"
        "3. NO full sentences, NO explanations\n"
        "4. For dates, use the format from the conversation\n"
        "5. If the answer is truly not in the conversation, say 'None'\n\n"
        "CONVERSATION:\n{conversation}\n\nNow answer this question: {question}"
    )

    def answer_fn(question: str) -> str:
        response = client.chat.completions.create(
            model=llm_model,
            messages=[
                {
                    "role": "user",
                    "content": prompt_template.format(
                        conversation=conversation_text,
                        question=question,
                    ),
                }
            ],
            max_tokens=50,
            temperature=0.1,
        )
        return response.choices[0].message.content.strip()

    return _qa_results(conv, answer_fn, run_judge, category_names=category_names)
