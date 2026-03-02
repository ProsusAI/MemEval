"""Prompts for Memory-R1 system (based on arXiv:2508.19828).

Three prompts for the two-agent architecture:
1. FACT_EXTRACTION_PROMPT - Extract facts from dialogue turns (both speakers)
2. MEMORY_MANAGER_PROMPT - Decide ADD/UPDATE/DELETE/NOOP on memory bank
3. ANSWER_AGENT_PROMPT - Distill relevant memories and answer questions
"""

FACT_EXTRACTION_PROMPT = """\
Extract key facts from this dialogue turn. Include facts from BOTH speakers.

Rules:
1. Each fact should be a single, self-contained statement.
2. Include the speaker's name with each fact (e.g., "Caroline attended..." not "She attended...").
3. Preserve exact dates, times, and temporal references as stated.
4. Include personal facts, events, opinions, feelings, and plans.
5. If the turn is just a greeting or has no meaningful new information, return an empty list.

Turn timestamp: {timestamp}
Speaker: {speaker}
Text: {text}

Return JSON only: {{"facts": ["fact1", "fact2", ...]}}"""

MEMORY_MANAGER_PROMPT = """\
You are a memory manager that maintains a structured memory bank about people in a conversation.

You can perform four operations:
1. **ADD**: New information not present in existing memories. Generate a new unique ID.
2. **UPDATE**: Information that refines, extends, or corrects an existing memory. Use the \
existing memory's ID. Keep the version with MORE detail and precision. Store the previous \
text in old_memory.
3. **DELETE**: Information that directly contradicts an existing memory. Use the existing \
memory's ID.
4. **NONE**: The fact is already captured in existing memories. No change needed.

Guidelines:
- Preserve temporal precision: dates, times, durations must be kept exactly as stated.
- Include speaker attribution (e.g., "Caroline attended..." not "User attended...").
- When two facts about the same topic have updated information, UPDATE the old one.
- Only DELETE when there is a clear contradiction, not just a different aspect.

## Existing Related Memories
{related_memories}

## New Facts to Process
{new_facts}

Return JSON only:
{{"memory": [\
{{"id": "<existing ID for UPDATE/DELETE/NONE, new for ADD>", \
"text": "<memory content>", \
"event": "ADD|UPDATE|DELETE|NONE", \
"old_memory": "<previous text, only for UPDATE>"}}\
]}}"""

ANSWER_AGENT_PROMPT = """\
You are answering a question about a conversation between two people, using retrieved memories.

Instructions:
1. Carefully analyze ALL provided memories from both speakers.
2. Pay special attention to timestamps to determine temporal answers.
3. Always convert relative time references to specific dates, months, or years when possible.
4. IMPORTANT: First SELECT the memories that are useful for answering the question. \
Output them before your answer.
5. IMPORTANT: Output your final answer after "**Answer:**"
6. Keep your answer to 5-6 words or fewer. Use exact words from memories when possible.
7. If the information is truly NOT in the memories, answer "None".

## Retrieved Memories
{memories}

## Question
{question}

First list the relevant memories you found, then give your answer."""
