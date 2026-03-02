#!/usr/bin/env python3
"""Prepare training data for Memory-R1 SFT from LoCoMo dataset.

Generates Memory Manager and Answer Agent SFT datasets following the
Memory-R1 paper (arXiv:2508.19828).

Data split (conversation-level, matching paper):
  Train: conv-26 (152 non-adversarial QAs, 419 turns, 184 observations)
  Val:   conv-30 (81 non-adversarial QAs, 369 turns, 169 observations)
  Test:  conv-41–conv-50 (1,307 non-adversarial QAs)

Output (data/r1_training/):
  memory_manager_train.jsonl  - MM SFT examples from conv-26 turns
  memory_manager_val.jsonl    - MM SFT examples from conv-30 turns
  answer_agent_train.jsonl    - AA SFT examples from conv-26 QAs
  answer_agent_val.jsonl      - AA SFT examples from conv-30 QAs
  memory_banks/               - Pre-built memory banks per conversation
  stats.json                  - Data preparation statistics

Usage:
    uv run python -m agents_memory.training.memory_r1.prepare_data
    uv run python -m agents_memory.training.memory_r1.prepare_data --skip-teacher
    uv run python -m agents_memory.training.memory_r1.prepare_data --dry-run
"""

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

load_dotenv()

# ============================================================
# Configuration
# ============================================================
DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data"
LOCOMO_PATH = DATA_DIR / "locomo10.json"
OUTPUT_DIR = DATA_DIR / "r1_training"

TEACHER_MODEL = os.environ.get("TEACHER_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = "text-embedding-3-small"
ADVERSARIAL_CATEGORY = 5

# Conversation-level split (matching paper exactly)
TRAIN_CONV = "conv-26"  # 152 non-adversarial QAs
VAL_CONV = "conv-30"  # 81 non-adversarial QAs

# Retrieval settings (from paper)
RETRIEVAL_K_UPDATE = 10  # Top-K memories for Memory Manager context
RETRIEVAL_K_ANSWER = 60  # Top-K memories for Answer Agent (30 per speaker)

# ============================================================
# Data Loading & Parsing
# ============================================================


def load_locomo() -> list[dict]:
    """Load LoCoMo dataset."""
    print(f"Loading LoCoMo from {LOCOMO_PATH}")
    with open(LOCOMO_PATH) as f:
        return json.load(f)


def get_conversation(data: list[dict], sample_id: str) -> dict:
    """Get a specific conversation by sample_id."""
    for conv in data:
        if conv.get("sample_id") == sample_id:
            return conv
    raise ValueError(f"Conversation {sample_id} not found")


def extract_dialogue_turns(conv: dict) -> list[dict]:
    """Extract all dialogue turns with metadata from a conversation."""
    turns = []
    conv_data = conv.get("conversation", {})

    session_nums = sorted(
        int(k.split("_")[1])
        for k in conv_data
        if k.startswith("session_") and not k.endswith(("_date_time", "_observation"))
    )

    for num in session_nums:
        session_key = f"session_{num}"
        datetime_key = f"session_{num}_date_time"
        session_time = conv_data.get(datetime_key, "")
        session_turns = conv_data.get(session_key, [])

        if not isinstance(session_turns, list):
            continue

        for turn in session_turns:
            if isinstance(turn, dict):
                turns.append(
                    {
                        "speaker": turn.get("speaker", "Unknown"),
                        "text": turn.get("text", ""),
                        "dia_id": turn.get("dia_id", ""),
                        "session_num": num,
                        "timestamp": session_time,
                    }
                )

    return turns


def extract_observations(conv: dict) -> list[dict]:
    """Extract all observations with metadata from a conversation.

    Returns list of dicts with: speaker, text, evidence_ref, session_num
    """
    observations = []
    obs_data = conv.get("observation", {})

    for key, session_obs in obs_data.items():
        if not key.endswith("_observation"):
            continue
        session_num = int(key.replace("session_", "").replace("_observation", ""))

        if not isinstance(session_obs, dict):
            continue

        for speaker, entries in session_obs.items():
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if isinstance(entry, list) and len(entry) >= 2:
                    ref = entry[1]
                    # Handle evidence_ref that might be a list
                    if isinstance(ref, list):
                        ref = ref[0] if ref else ""
                    observations.append(
                        {
                            "speaker": speaker,
                            "text": entry[0],
                            "evidence_ref": str(ref),
                            "session_num": session_num,
                        }
                    )

    return sorted(observations, key=lambda x: (x["session_num"], str(x["evidence_ref"])))


def extract_qa_pairs(conv: dict, exclude_adversarial: bool = True) -> list[dict]:
    """Extract QA pairs, optionally excluding adversarial."""
    qa_pairs = conv.get("qa", [])
    if exclude_adversarial:
        qa_pairs = [qa for qa in qa_pairs if qa.get("category") != ADVERSARIAL_CATEGORY]
    return qa_pairs


# ============================================================
# Embeddings
# ============================================================


class EmbeddingCache:
    """Cache embeddings to avoid redundant API calls."""

    def __init__(self, model: str = EMBEDDING_MODEL, cache_path: Path | None = None):
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.model = model
        self.cache: dict[str, list[float]] = {}
        self.cache_path = cache_path
        if cache_path and cache_path.exists():
            with open(cache_path) as f:
                self.cache = json.load(f)
            print(f"  Loaded {len(self.cache)} cached embeddings")

    def get(self, text: str) -> list[float]:
        """Get embedding for text, using cache."""
        key = hashlib.md5(text.encode()).hexdigest()
        if key not in self.cache:
            response = self.client.embeddings.create(model=self.model, input=text)
            self.cache[key] = response.data[0].embedding
        return self.cache[key]

    def get_batch(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings for multiple texts, batching uncached ones."""
        results = []
        uncached_texts = []
        uncached_indices = []

        for i, text in enumerate(texts):
            key = hashlib.md5(text.encode()).hexdigest()
            if key in self.cache:
                results.append(self.cache[key])
            else:
                results.append(None)
                uncached_texts.append(text)
                uncached_indices.append(i)

        if uncached_texts:
            # Batch embed (OpenAI supports up to 2048 inputs per call)
            for batch_start in range(0, len(uncached_texts), 2048):
                batch = uncached_texts[batch_start : batch_start + 2048]
                response = self.client.embeddings.create(model=self.model, input=batch)
                for j, emb_data in enumerate(response.data):
                    idx = uncached_indices[batch_start + j]
                    key = hashlib.md5(texts[idx].encode()).hexdigest()
                    self.cache[key] = emb_data.embedding
                    results[idx] = emb_data.embedding

        return results

    def save(self):
        """Save cache to disk."""
        if self.cache_path:
            with open(self.cache_path, "w") as f:
                json.dump(self.cache, f)
            print(f"  Saved {len(self.cache)} embeddings to cache")


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a_np, b_np = np.array(a), np.array(b)
    norm = np.linalg.norm(a_np) * np.linalg.norm(b_np)
    if norm == 0:
        return 0.0
    return float(np.dot(a_np, b_np) / norm)


def retrieve_top_k(
    query_emb: list[float],
    memory_bank: list[dict],
    k: int,
    speaker_filter: str | None = None,
) -> list[dict]:
    """Retrieve top-K memories by cosine similarity."""
    candidates = memory_bank
    if speaker_filter:
        candidates = [m for m in candidates if m.get("speaker") == speaker_filter]

    if not candidates:
        return []

    similarities = []
    for mem in candidates:
        sim = cosine_similarity(query_emb, mem["embedding"])
        similarities.append((sim, mem))

    similarities.sort(key=lambda x: x[0], reverse=True)
    return [mem for _, mem in similarities[:k]]


# ============================================================
# Memory Manager Data Generation
# ============================================================


def build_observation_memory_bank(
    observations: list[dict],
    up_to_session: int,
    embeddings: EmbeddingCache,
) -> list[dict]:
    """Build a memory bank from observations up to (inclusive) a session.

    Each observation becomes a memory entry with an embedding.
    """
    filtered = [obs for obs in observations if obs["session_num"] <= up_to_session]
    if not filtered:
        return []

    texts = [obs["text"] for obs in filtered]
    embs = embeddings.get_batch(texts)

    memory_bank = []
    for i, obs in enumerate(filtered):
        memory_bank.append(
            {
                "id": str(i),
                "text": obs["text"],
                "speaker": obs["speaker"],
                "evidence_ref": obs["evidence_ref"],
                "session_num": obs["session_num"],
                "embedding": embs[i],
            }
        )

    return memory_bank


def generate_mm_labels_from_observations(
    turns: list[dict],
    observations: list[dict],
    embeddings: EmbeddingCache,
) -> list[dict]:
    """Generate Memory Manager SFT labels from observations (no teacher model).

    For each turn:
    - If observations reference this turn → ADD those observations
    - Otherwise → NOOP

    This only produces ADD and NOOP labels (no UPDATE/DELETE).
    """
    # Map dia_id → observations
    obs_by_turn = defaultdict(list)
    for obs in observations:
        obs_by_turn[obs["evidence_ref"]].append(obs)

    examples = []
    memory_bank = []
    next_id = 0

    for turn in tqdm(turns, desc="  MM labels (observations)"):
        dia_id = turn["dia_id"]
        turn_obs = obs_by_turn.get(dia_id, [])

        # Get related memories for context
        if memory_bank and turn["text"].strip():
            turn_emb = embeddings.get(turn["text"])
            related = retrieve_top_k(turn_emb, memory_bank, k=RETRIEVAL_K_UPDATE)
        else:
            related = []

        # Build the operations
        operations = []
        if turn_obs:
            for obs in turn_obs:
                operations.append(
                    {
                        "id": str(next_id),
                        "text": obs["text"],
                        "event": "ADD",
                    }
                )
                # Add to memory bank for future retrieval
                emb = embeddings.get(obs["text"])
                memory_bank.append(
                    {
                        "id": str(next_id),
                        "text": obs["text"],
                        "speaker": obs["speaker"],
                        "embedding": emb,
                    }
                )
                next_id += 1
        else:
            # NOOP - pick the most related memory to show as context
            if related:
                operations.append(
                    {
                        "id": related[0]["id"],
                        "text": related[0]["text"],
                        "event": "NONE",
                    }
                )

        # Format related memories for prompt (without embeddings)
        related_for_prompt = [
            {"id": m["id"], "text": m["text"], "speaker": m.get("speaker", "")}
            for m in related[:RETRIEVAL_K_UPDATE]
        ]

        examples.append(
            {
                "turn": {
                    "speaker": turn["speaker"],
                    "text": turn["text"],
                    "timestamp": turn["timestamp"],
                    "dia_id": turn["dia_id"],
                    "session_num": turn["session_num"],
                },
                "related_memories": related_for_prompt,
                "operations": operations,
                "memory_bank_size": len(memory_bank),
            }
        )

    return examples


def generate_mm_labels_with_teacher(
    turns: list[dict],
    observations: list[dict],
    embeddings: EmbeddingCache,
    teacher_model: str = TEACHER_MODEL,
) -> list[dict]:
    """Generate Memory Manager SFT labels using a teacher model.

    The teacher model processes each turn and decides ADD/UPDATE/DELETE/NOOP
    given the current memory bank and new turn content.
    """
    from agents_memory.training.memory_r1.prompts import MEMORY_MANAGER_PROMPT

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    # Build initial memory bank from observations for reference
    obs_by_turn = defaultdict(list)
    for obs in observations:
        obs_by_turn[obs["evidence_ref"]].append(obs)

    examples = []
    memory_bank = []
    next_id = 0

    for turn in tqdm(turns, desc="  MM labels (teacher)"):
        dia_id = turn["dia_id"]
        text = turn["text"].strip()

        if not text or len(text) < 10:
            # Skip very short turns (greetings, etc.)
            examples.append(
                {
                    "turn": {
                        "speaker": turn["speaker"],
                        "text": turn["text"],
                        "timestamp": turn["timestamp"],
                        "dia_id": turn["dia_id"],
                        "session_num": turn["session_num"],
                    },
                    "related_memories": [],
                    "operations": [],
                    "memory_bank_size": len(memory_bank),
                }
            )
            continue

        # Retrieve related memories
        turn_emb = embeddings.get(text)
        related = retrieve_top_k(turn_emb, memory_bank, k=RETRIEVAL_K_UPDATE)

        related_for_prompt = [
            {"id": m["id"], "text": m["text"], "speaker": m.get("speaker", "")}
            for m in related
        ]

        # New facts = the turn content + any observations for this turn
        turn_obs = obs_by_turn.get(dia_id, [])
        new_facts = [f"[{turn['speaker']}] {text}"]
        for obs in turn_obs:
            new_facts.append(f"[Observation] {obs['text']}")

        prompt = MEMORY_MANAGER_PROMPT.format(
            related_memories=json.dumps(related_for_prompt, indent=2) if related_for_prompt
            else "No existing memories yet.",
            new_facts="\n".join(f"- {f}" for f in new_facts),
        )

        try:
            response = client.chat.completions.create(
                model=teacher_model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0,
                max_tokens=1024,
            )
            result = json.loads(response.choices[0].message.content)
            operations = result.get("memory", [])
        except Exception as e:
            print(f"    Teacher error on {dia_id}: {e}")
            # Fallback: use observations if available
            operations = []
            for obs in turn_obs:
                operations.append(
                    {"id": str(next_id), "text": obs["text"], "event": "ADD"}
                )

        # Apply operations to memory bank
        for op in operations:
            event = op.get("event", "NONE").upper()
            if event == "ADD":
                op["id"] = str(next_id)
                emb = embeddings.get(op["text"])
                memory_bank.append(
                    {
                        "id": str(next_id),
                        "text": op["text"],
                        "speaker": turn["speaker"],
                        "embedding": emb,
                    }
                )
                next_id += 1
            elif event == "UPDATE":
                target_id = op.get("id")
                for mem in memory_bank:
                    if mem["id"] == target_id:
                        op["old_memory"] = mem["text"]
                        mem["text"] = op["text"]
                        mem["embedding"] = embeddings.get(op["text"])
                        break
            elif event == "DELETE":
                target_id = op.get("id")
                memory_bank = [m for m in memory_bank if m["id"] != target_id]

        examples.append(
            {
                "turn": {
                    "speaker": turn["speaker"],
                    "text": turn["text"],
                    "timestamp": turn["timestamp"],
                    "dia_id": turn["dia_id"],
                    "session_num": turn["session_num"],
                },
                "related_memories": related_for_prompt,
                "operations": operations,
                "memory_bank_size": len(memory_bank),
            }
        )

    return examples


# ============================================================
# Answer Agent Data Generation
# ============================================================


def build_full_memory_bank(
    observations: list[dict],
    embeddings: EmbeddingCache,
) -> list[dict]:
    """Build complete memory bank from all observations for a conversation."""
    texts = [obs["text"] for obs in observations]
    embs = embeddings.get_batch(texts)

    memory_bank = []
    for i, obs in enumerate(observations):
        memory_bank.append(
            {
                "id": str(i),
                "text": obs["text"],
                "speaker": obs["speaker"],
                "evidence_ref": obs["evidence_ref"],
                "session_num": obs["session_num"],
                "embedding": embs[i],
            }
        )

    return memory_bank


def generate_aa_examples(
    qa_pairs: list[dict],
    memory_bank: list[dict],
    observations: list[dict],
    embeddings: EmbeddingCache,
) -> list[dict]:
    """Generate Answer Agent SFT examples.

    For each QA pair:
    1. Retrieve top-60 memories (30 per speaker)
    2. Map evidence refs to identify relevant memories
    3. Format as SFT example with distillation target
    """
    # Get unique speakers
    speakers = sorted(set(m["speaker"] for m in memory_bank))

    # Map evidence_ref → observation text for distillation target
    obs_by_ref = {}
    for obs in observations:
        obs_by_ref[obs["evidence_ref"]] = obs["text"]

    examples = []
    per_speaker_k = RETRIEVAL_K_ANSWER // max(len(speakers), 1)

    for qa in tqdm(qa_pairs, desc="  AA examples"):
        question = qa.get("question", "")
        answer = qa.get("answer", "")
        evidence_refs = qa.get("evidence", [])
        category = qa.get("category", 0)

        # Ensure answer is string
        answer = str(answer) if not isinstance(answer, str) else answer

        # Retrieve top-K per speaker
        q_emb = embeddings.get(question)
        retrieved = []
        for speaker in speakers:
            speaker_memories = retrieve_top_k(
                q_emb, memory_bank, k=per_speaker_k, speaker_filter=speaker
            )
            for mem in speaker_memories:
                retrieved.append(
                    {
                        "id": mem["id"],
                        "text": mem["text"],
                        "speaker": mem["speaker"],
                        "session_num": mem["session_num"],
                    }
                )

        # Identify which retrieved memories match evidence
        relevant_memories = []
        for ref in evidence_refs:
            obs_text = obs_by_ref.get(ref)
            if obs_text:
                for mem in retrieved:
                    if mem["text"] == obs_text:
                        relevant_memories.append(mem)
                        break

        # Build target output (distillation + answer)
        target_parts = []
        if relevant_memories:
            target_parts.append("## Selected Relevant Memories")
            for mem in relevant_memories:
                target_parts.append(f"- [{mem['speaker']}] {mem['text']}")
            target_parts.append("")

        target_parts.append(f"**Answer:** {answer}")
        target_output = "\n".join(target_parts)

        examples.append(
            {
                "question": question,
                "answer": answer,
                "category": category,
                "evidence_refs": evidence_refs,
                "retrieved_memories": retrieved,
                "relevant_memories": relevant_memories,
                "target_output": target_output,
                "num_retrieved": len(retrieved),
                "num_relevant_found": len(relevant_memories),
                "num_evidence_refs": len(evidence_refs),
            }
        )

    return examples


# ============================================================
# ChatML Formatting (Qwen-2.5 template)
# ============================================================


def format_mm_chatml(example: dict) -> dict:
    """Format a Memory Manager example as ChatML for SFT."""
    from agents_memory.training.memory_r1.prompts import MEMORY_MANAGER_PROMPT

    turn = example["turn"]
    related = example["related_memories"]
    operations = example["operations"]

    # Format the user prompt
    new_facts = f"[{turn['speaker']}] ({turn['timestamp']}) {turn['text']}"

    user_content = MEMORY_MANAGER_PROMPT.format(
        related_memories=json.dumps(related, indent=2) if related
        else "No existing memories yet.",
        new_facts=f"- {new_facts}",
    )

    # Format the assistant response (target)
    assistant_content = json.dumps({"memory": operations}, indent=2)

    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
    }


def format_aa_chatml(example: dict) -> dict:
    """Format an Answer Agent example as ChatML for SFT."""
    from agents_memory.training.memory_r1.prompts import ANSWER_AGENT_PROMPT

    # Group retrieved memories by speaker
    by_speaker = defaultdict(list)
    for mem in example["retrieved_memories"]:
        by_speaker[mem["speaker"]].append(mem)

    memory_lines = []
    for speaker in sorted(by_speaker):
        memory_lines.append(f"\n### {speaker}")
        for i, mem in enumerate(by_speaker[speaker], 1):
            memory_lines.append(f"{i}. [Session {mem['session_num']}] {mem['text']}")

    user_content = ANSWER_AGENT_PROMPT.format(
        memories="\n".join(memory_lines),
        question=example["question"],
    )

    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": example["target_output"]},
        ]
    }


# ============================================================
# Main
# ============================================================


def main():
    parser = argparse.ArgumentParser(description="Prepare Memory-R1 training data")
    parser.add_argument(
        "--skip-teacher",
        action="store_true",
        help="Use observations only for MM labels (no API calls for teacher)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and validate data without API calls or file output",
    )
    args = parser.parse_args()

    if not args.dry_run and not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set. Use --dry-run to skip API calls.")
        sys.exit(1)

    # Load data
    data = load_locomo()
    print(f"Loaded {len(data)} conversations")

    # Get train/val conversations
    train_conv = get_conversation(data, TRAIN_CONV)
    val_conv = get_conversation(data, VAL_CONV)

    # Extract components
    print(f"\n--- Train: {TRAIN_CONV} ---")
    train_turns = extract_dialogue_turns(train_conv)
    train_obs = extract_observations(train_conv)
    train_qa = extract_qa_pairs(train_conv, exclude_adversarial=True)
    print(f"  Turns: {len(train_turns)}, Observations: {len(train_obs)}, QAs: {len(train_qa)}")

    print(f"\n--- Val: {VAL_CONV} ---")
    val_turns = extract_dialogue_turns(val_conv)
    val_obs = extract_observations(val_conv)
    val_qa = extract_qa_pairs(val_conv, exclude_adversarial=True)
    print(f"  Turns: {len(val_turns)}, Observations: {len(val_obs)}, QAs: {len(val_qa)}")

    # Validate expected counts
    assert len(train_qa) == 152, f"Expected 152 train QAs, got {len(train_qa)}"
    assert len(val_qa) == 81, f"Expected 81 val QAs, got {len(val_qa)}"

    if args.dry_run:
        print("\n--- Dry Run Summary ---")
        print(f"Train turns: {len(train_turns)} → MM train examples")
        print(f"Train QAs: {len(train_qa)} → AA train examples")
        print(f"Val turns: {len(val_turns)} → MM val examples")
        print(f"Val QAs: {len(val_qa)} → AA val examples")

        # Show sample observation mapping
        obs_by_turn = defaultdict(list)
        for obs in train_obs:
            obs_by_turn[obs["evidence_ref"]].append(obs)
        turns_with_obs = sum(1 for t in train_turns if t["dia_id"] in obs_by_turn)
        print(f"\nTurns with observations: {turns_with_obs}/{len(train_turns)}")
        print(f"Turns without (NOOP): {len(train_turns) - turns_with_obs}/{len(train_turns)}")

        # Show speaker distribution
        train_speakers = sorted(set(t["speaker"] for t in train_turns))
        val_speakers = sorted(set(t["speaker"] for t in val_turns))
        print(f"\nTrain speakers: {train_speakers}")
        print(f"Val speakers: {val_speakers}")
        return

    # Setup
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "memory_banks").mkdir(exist_ok=True)
    cache_path = OUTPUT_DIR / "embeddings_cache.json"
    embeddings = EmbeddingCache(cache_path=cache_path)

    # ---- Memory Manager Data ----
    print("\n=== Memory Manager Data ===")

    if args.skip_teacher:
        print("Generating MM labels from observations only (no teacher)...")
        mm_train = generate_mm_labels_from_observations(train_turns, train_obs, embeddings)
        mm_val = generate_mm_labels_from_observations(val_turns, val_obs, embeddings)
    else:
        print(f"Generating MM labels with teacher model ({TEACHER_MODEL})...")
        mm_train = generate_mm_labels_with_teacher(train_turns, train_obs, embeddings)
        mm_val = generate_mm_labels_with_teacher(val_turns, val_obs, embeddings)

    # Count operation types
    for name, examples in [("train", mm_train), ("val", mm_val)]:
        ops_count = defaultdict(int)
        for ex in examples:
            for op in ex["operations"]:
                ops_count[op.get("event", "NONE").upper()] += 1
        print(f"  MM {name}: {len(examples)} examples, ops: {dict(ops_count)}")

    # Format as ChatML and save
    mm_train_chatml = [format_mm_chatml(ex) for ex in mm_train if ex["operations"]]
    mm_val_chatml = [format_mm_chatml(ex) for ex in mm_val if ex["operations"]]

    mm_train_path = OUTPUT_DIR / "memory_manager_train.jsonl"
    mm_val_path = OUTPUT_DIR / "memory_manager_val.jsonl"

    with open(mm_train_path, "w") as f:
        for ex in mm_train_chatml:
            f.write(json.dumps(ex) + "\n")
    with open(mm_val_path, "w") as f:
        for ex in mm_val_chatml:
            f.write(json.dumps(ex) + "\n")

    print(f"  Saved: {mm_train_path} ({len(mm_train_chatml)} examples)")
    print(f"  Saved: {mm_val_path} ({len(mm_val_chatml)} examples)")

    # Save raw MM data (with dia_id + operations) for RL overlap matching
    for name, examples in [("train", mm_train), ("val", mm_val)]:
        raw_path = OUTPUT_DIR / f"memory_manager_{name}_raw.jsonl"
        with open(raw_path, "w") as f:
            for ex in examples:
                if not ex["operations"]:
                    continue
                raw = {
                    "turn": ex["turn"],
                    "operations": ex["operations"],
                    "related_memories": ex["related_memories"],
                }
                f.write(json.dumps(raw) + "\n")
        print(f"  Saved raw: {raw_path}")

    # ---- Answer Agent Data ----
    print("\n=== Answer Agent Data ===")

    # Build full memory banks from observations
    print("Building memory banks...")
    train_memory_bank = build_full_memory_bank(train_obs, embeddings)
    val_memory_bank = build_full_memory_bank(val_obs, embeddings)

    # Save memory banks (without embeddings for readability)
    for name, bank in [("train", train_memory_bank), ("val", val_memory_bank)]:
        bank_path = OUTPUT_DIR / "memory_banks" / f"{name}_memory_bank.json"
        bank_no_emb = [{k: v for k, v in m.items() if k != "embedding"} for m in bank]
        with open(bank_path, "w") as f:
            json.dump(bank_no_emb, f, indent=2)
        print(f"  Memory bank ({name}): {len(bank)} entries → {bank_path}")

    # Generate AA examples
    print("Generating Answer Agent examples...")
    aa_train = generate_aa_examples(train_qa, train_memory_bank, train_obs, embeddings)
    aa_val = generate_aa_examples(val_qa, val_memory_bank, val_obs, embeddings)

    # Stats on evidence coverage
    for name, examples in [("train", aa_train), ("val", aa_val)]:
        total_refs = sum(ex["num_evidence_refs"] for ex in examples)
        found_refs = sum(ex["num_relevant_found"] for ex in examples)
        coverage = found_refs / total_refs * 100 if total_refs > 0 else 0
        print(f"  AA {name}: {len(examples)} examples, "
              f"evidence coverage: {found_refs}/{total_refs} ({coverage:.1f}%)")

    # Format as ChatML and save
    aa_train_chatml = [format_aa_chatml(ex) for ex in aa_train]
    aa_val_chatml = [format_aa_chatml(ex) for ex in aa_val]

    aa_train_path = OUTPUT_DIR / "answer_agent_train.jsonl"
    aa_val_path = OUTPUT_DIR / "answer_agent_val.jsonl"

    with open(aa_train_path, "w") as f:
        for ex in aa_train_chatml:
            f.write(json.dumps(ex) + "\n")
    with open(aa_val_path, "w") as f:
        for ex in aa_val_chatml:
            f.write(json.dumps(ex) + "\n")

    print(f"  Saved: {aa_train_path} ({len(aa_train_chatml)} examples)")
    print(f"  Saved: {aa_val_path} ({len(aa_val_chatml)} examples)")

    # Save raw AA data (with evidence_refs) for RL overlap matching
    for name, examples in [("train", aa_train), ("val", aa_val)]:
        raw_path = OUTPUT_DIR / f"answer_agent_{name}_raw.jsonl"
        with open(raw_path, "w") as f:
            for ex in examples:
                raw = {
                    "question": ex["question"],
                    "answer": ex["answer"],
                    "category": ex["category"],
                    "evidence_refs": ex["evidence_refs"],
                }
                f.write(json.dumps(raw) + "\n")
        print(f"  Saved raw: {raw_path}")

    # ---- Save embeddings cache ----
    embeddings.save()

    # ---- Save stats ----
    stats = {
        "timestamp": datetime.now().isoformat(),
        "teacher_model": TEACHER_MODEL if not args.skip_teacher else "observations_only",
        "embedding_model": EMBEDDING_MODEL,
        "train_conv": TRAIN_CONV,
        "val_conv": VAL_CONV,
        "memory_manager": {
            "train_examples": len(mm_train_chatml),
            "val_examples": len(mm_val_chatml),
            "train_turns_total": len(train_turns),
            "val_turns_total": len(val_turns),
        },
        "answer_agent": {
            "train_examples": len(aa_train_chatml),
            "val_examples": len(aa_val_chatml),
        },
        "observations": {
            "train": len(train_obs),
            "val": len(val_obs),
        },
    }
    stats_path = OUTPUT_DIR / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\n  Stats saved to {stats_path}")

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("DATA PREPARATION COMPLETE")
    print("=" * 60)
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  MM train: {len(mm_train_chatml)} examples")
    print(f"  MM val:   {len(mm_val_chatml)} examples")
    print(f"  AA train: {len(aa_train_chatml)} examples")
    print(f"  AA val:   {len(aa_val_chatml)} examples")
    print(f"  Embeddings cached: {len(embeddings.cache)}")


if __name__ == "__main__":
    main()
