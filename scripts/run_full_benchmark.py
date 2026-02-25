#!/usr/bin/env python3
"""Unified benchmark runner for all memory systems on LoCoMo.

Runs any combination of systems across all 10 LoCoMo conversations with
configurable LLM model and optional judge evaluation.

Usage:
    # Single system, 1 conversation
    uv run python scripts/run_full_benchmark.py --systems memclaw --num-samples 1 --skip-judge

    # All systems, all conversations, gpt-4.1-mini
    uv run python scripts/run_full_benchmark.py \
        --systems all --num-samples 10 --llm-model gpt-4.1-mini --skip-judge

    # Specific systems
    uv run python scripts/run_full_benchmark.py \
        --systems memclaw,graphiti --num-samples 10 --skip-judge
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from agents_memory.evaluation import compute_f1, evaluate_with_judge
from agents_memory.locomo import CATEGORY_NAMES, download_locomo, extract_dialogues
from agents_memory.token_tracker import get_stats, get_stats_by_model
from agents_memory.token_tracker import reset as reset_tracker
from agents_memory.token_tracker import start as start_tracking

DATA_DIR = Path(__file__).parent.parent / "data"

# Ensure unbuffered output
sys.stdout.reconfigure(line_buffering=True)


# ---------------------------------------------------------------------------
# Common helpers
# ---------------------------------------------------------------------------


def _qa_results(
    conv: dict,
    answer_fn,
    run_judge: bool,
) -> list[dict]:
    """Evaluate all QA pairs for a conversation using the given answer function.

    answer_fn(question: str) -> str  (returns predicted answer)
    """
    qa_pairs = conv.get("qa", [])
    sample_id = conv.get("sample_id", "unknown")
    results = []

    for i, qa in enumerate(qa_pairs, 1):
        question = qa.get("question", "")
        ground_truth = qa.get("answer", "")
        category = qa.get("category", 0)

        try:
            predicted = answer_fn(question)
        except Exception as err:
            print(f"    Error on Q{i}: {err}")
            predicted = ""

        f1 = compute_f1(predicted, ground_truth)

        row = {
            "sample_id": sample_id,
            "question": question,
            "ground_truth": ground_truth,
            "predicted": predicted,
            "category": category,
            "category_name": CATEGORY_NAMES.get(category, "Unknown"),
            "f1": f1,
        }

        if run_judge:
            row.update(evaluate_with_judge(question, ground_truth, predicted))

        results.append(row)

        if i % 20 == 0:
            print(f"    QA {i}/{len(qa_pairs)} - F1={f1:.3f}")

    return results


async def _qa_results_async(
    conv: dict,
    answer_fn,
    run_judge: bool,
) -> list[dict]:
    """Async version of _qa_results.

    answer_fn(question: str) -> Awaitable[str]
    """
    qa_pairs = conv.get("qa", [])
    sample_id = conv.get("sample_id", "unknown")
    results = []

    for i, qa in enumerate(qa_pairs, 1):
        question = qa.get("question", "")
        ground_truth = qa.get("answer", "")
        category = qa.get("category", 0)

        try:
            predicted = await answer_fn(question)
        except Exception as err:
            print(f"    Error on Q{i}: {err}")
            predicted = ""

        f1 = compute_f1(predicted, ground_truth)

        row = {
            "sample_id": sample_id,
            "question": question,
            "ground_truth": ground_truth,
            "predicted": predicted,
            "category": category,
            "category_name": CATEGORY_NAMES.get(category, "Unknown"),
            "f1": f1,
        }

        if run_judge:
            row.update(evaluate_with_judge(question, ground_truth, predicted))

        results.append(row)

        if i % 20 == 0:
            print(f"    QA {i}/{len(qa_pairs)} - F1={f1:.3f}")

    return results


# ---------------------------------------------------------------------------
# System adapters
# ---------------------------------------------------------------------------


def run_memclaw(conv: dict, llm_model: str, run_judge: bool) -> list[dict]:
    """MemClaw: proposition-based entity-centric retrieval."""
    from agents_memory.memclaw import MemClawSystem

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


def run_openclaw(conv: dict, llm_model: str, run_judge: bool) -> list[dict]:
    """OpenClaw: hybrid BM25 + vector chunk retrieval."""
    from agents_memory.openclaw import chunk_markdown, embed_texts, hybrid_search

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    dialogues = extract_dialogues(conv)

    # Format as markdown
    lines = []
    current_date = None
    for d in dialogues:
        timestamp = d.get("timestamp", "")
        date_part = timestamp.split(" ")[0] if timestamp else ""
        if date_part and date_part != current_date:
            current_date = date_part
            lines.append(f"\n## {current_date}\n")
        speaker = d["speaker"]
        text = d["text"]
        if timestamp:
            lines.append(f"**{speaker}** ({timestamp}): {text}")
        else:
            lines.append(f"**{speaker}**: {text}")
    markdown_text = "\n".join(lines)

    chunks = chunk_markdown(markdown_text, tokens=400, overlap=80)
    chunk_embeddings = embed_texts(
        [c.text for c in chunks], model="text-embedding-3-small"
    )
    print(f"    Chunks: {len(chunks)}")

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

    return _qa_results(conv, answer_fn, run_judge)


def run_simplemem(conv: dict, llm_model: str, run_judge: bool) -> list[dict]:
    """SimpleMem: multi-round retrieval with parallel processing."""
    from simplemem import SimpleMemConfig, SimpleMemSystem, set_config
    from simplemem.models.memory_entry import Dialogue

    config = SimpleMemConfig(
        openai_api_key=os.environ["OPENAI_API_KEY"],
        llm_model=llm_model,
    )
    set_config(config)

    dialogues = extract_dialogues(conv)
    memory = SimpleMemSystem(clear_db=True)

    dialogue_objects = [
        Dialogue(
            dialogue_id=i + 1,
            speaker=d["speaker"],
            content=d["text"],
            timestamp=d["timestamp"] or datetime.now().isoformat(),
        )
        for i, d in enumerate(dialogues)
    ]

    memory.add_dialogues(dialogue_objects)
    memory.finalize()
    print(f"    Dialogues ingested: {len(dialogue_objects)}")

    return _qa_results(
        conv,
        lambda q: memory.ask(q),
        run_judge,
    )


def run_mem0(conv: dict, llm_model: str, run_judge: bool) -> list[dict]:
    """Mem0: memory extraction + vector search."""
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

    results = _qa_results(conv, answer_fn, run_judge)

    # Cleanup
    try:
        memory.delete_all(user_id=user_id)
    except Exception:
        pass

    return results


def run_memu(conv: dict, llm_model: str, run_judge: bool) -> list[dict]:
    """MemU: memory service with file-based memorize."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(_run_memu_async(conv, llm_model, run_judge))
    finally:
        loop.close()


async def _run_memu_async(conv: dict, llm_model: str, run_judge: bool) -> list[dict]:
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


def run_fullcontext(conv: dict, llm_model: str, run_judge: bool) -> list[dict]:
    """Full context baseline: entire conversation in prompt."""
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

    return _qa_results(conv, answer_fn, run_judge)


def run_graphiti(conv: dict, llm_model: str, run_judge: bool) -> list[dict]:
    """Graphiti: temporal knowledge graph (Kuzu embedded)."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(_run_graphiti_async(conv, llm_model, run_judge))
    finally:
        loop.close()


async def _run_graphiti_async(
    conv: dict, llm_model: str, run_judge: bool
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

        return await _qa_results_async(conv, answer_fn, run_judge)
    finally:
        await system.close()


# ---------------------------------------------------------------------------
# System registry
# ---------------------------------------------------------------------------

SYSTEMS: dict[str, dict] = {
    "memclaw": {
        "fn": run_memclaw,
        "architecture": "proposition-based entity-centric retrieval",
        "infrastructure": "vector store + BM25",
    },
    "openclaw": {
        "fn": run_openclaw,
        "architecture": "hybrid BM25 + vector chunk retrieval",
        "infrastructure": "vector store + BM25",
    },
    "simplemem": {
        "fn": run_simplemem,
        "architecture": "multi-round retrieval with parallel processing",
        "infrastructure": "SimpleMem library",
    },
    "mem0": {
        "fn": run_mem0,
        "architecture": "memory extraction + vector search",
        "infrastructure": "mem0 library",
    },
    "memu": {
        "fn": run_memu,
        "architecture": "memory service with file-based memorize",
        "infrastructure": "MemU library",
    },
    "fullcontext": {
        "fn": run_fullcontext,
        "architecture": "full conversation in prompt (upper bound)",
        "infrastructure": "none",
    },
    "graphiti": {
        "fn": run_graphiti,
        "architecture": "temporal knowledge graph (Kuzu embedded)",
        "infrastructure": "Graphiti + Kuzu",
    },
}


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified benchmark runner for all memory systems on LoCoMo"
    )
    parser.add_argument(
        "--systems",
        type=str,
        default="all",
        help="Comma-separated system names or 'all' (default: all)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of conversations to evaluate (default: 10)",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default=None,
        help="LLM model to use (default: from LLM_MODEL env or gpt-4.1)",
    )
    parser.add_argument(
        "--skip-judge",
        action="store_true",
        help="Skip LLM-as-judge evaluation (F1 only)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: data/)",
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default=None,
        help="Custom data file in LoCoMo format (default: download LoCoMo)",
    )
    return parser.parse_args()


def _compute_summary(all_results: list[dict]) -> dict:
    """Compute mean +/- std F1 overall and by category."""
    if not all_results:
        return {}

    # Group by conversation for per-conversation F1
    by_conv: dict[str, list[float]] = {}
    by_category: dict[str, list[float]] = {}

    for r in all_results:
        sid = r["sample_id"]
        by_conv.setdefault(sid, []).append(r["f1"])
        cat_name = r.get("category_name", "Unknown")
        by_category.setdefault(cat_name, []).append(r["f1"])

    # Per-conversation F1
    conv_f1s = {sid: np.mean(scores) for sid, scores in by_conv.items()}
    overall_f1s = list(conv_f1s.values())

    summary = {
        "overall_f1_mean": float(np.mean(overall_f1s)),
        "overall_f1_std": float(np.std(overall_f1s)),
        "n_conversations": len(conv_f1s),
        "n_questions": len(all_results),
        "per_conversation": {
            sid: {"f1": float(f1), "n_questions": len(by_conv[sid])}
            for sid, f1 in conv_f1s.items()
        },
        "by_category": {},
    }

    for cat_name, scores in sorted(by_category.items()):
        summary["by_category"][cat_name] = {
            "f1_mean": float(np.mean(scores)),
            "f1_std": float(np.std(scores)),
            "n": len(scores),
        }

    return summary


def main() -> None:
    args = parse_args()
    load_dotenv()
    start_tracking()

    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set. Create a .env file or export it.")
        return

    llm_model = args.llm_model or os.environ.get("LLM_MODEL", "gpt-4.1")
    judge_model = os.environ.get("JUDGE_MODEL", "gpt-5.2")
    run_judge = not args.skip_judge
    output_dir = Path(args.output_dir) if args.output_dir else DATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve system list
    if args.systems.lower() == "all":
        system_names = list(SYSTEMS.keys())
    else:
        system_names = [s.strip() for s in args.systems.split(",")]
        unknown = [s for s in system_names if s not in SYSTEMS]
        if unknown:
            print(f"Error: Unknown systems: {unknown}")
            print(f"Available: {list(SYSTEMS.keys())}")
            return

    # Load data
    if args.data_file:
        print(f"Loading custom data from {args.data_file}")
        with open(args.data_file) as f:
            data = json.load(f)
    else:
        data = download_locomo()
    conversations = data if isinstance(data, list) else [data]
    conversations = conversations[: args.num_samples]

    print("=" * 70)
    print("Full Benchmark Runner")
    print("=" * 70)
    print(f"  Systems: {', '.join(system_names)}")
    print(f"  LLM Model: {llm_model}")
    print(f"  Judge: {'enabled (' + judge_model + ')' if run_judge else 'disabled'}")
    print(f"  Conversations: {len(conversations)}")
    print(f"  Output: {output_dir}")
    print("=" * 70)

    # Model name for file naming (sanitize dots)
    model_tag = llm_model.replace(".", "")
    if args.data_file:
        dataset_name = Path(args.data_file).stem.replace(".", "")
        model_tag = f"{dataset_name}_{model_tag}"

    all_summaries = {}

    for sys_name in system_names:
        sys_info = SYSTEMS[sys_name]
        run_fn = sys_info["fn"]

        print(f"\n{'='*60}")
        print(f"  SYSTEM: {sys_name} ({sys_info['architecture']})")
        print(f"{'='*60}")

        # Reset tracker for this system
        reset_tracker()
        all_results = []

        for conv in tqdm(conversations, desc=sys_name):
            sample_id = conv.get("sample_id", "unknown")
            qa_count = len(conv.get("qa", []))
            print(f"\n  [{sys_name}] Conv {sample_id}: {qa_count} QA pairs")

            try:
                results = run_fn(conv, llm_model, run_judge)
                all_results.extend(results)
            except Exception as err:
                print(f"  ERROR on conv {sample_id}: {err}")
                import traceback

                traceback.print_exc()
                continue

            # Progress
            if all_results:
                running_f1 = np.mean([r["f1"] for r in all_results])
                print(f"  Running F1: {running_f1:.3f} ({len(all_results)} questions)")

        if not all_results:
            print(f"  No results for {sys_name}, skipping")
            continue

        # Compute summary
        summary = _compute_summary(all_results)
        stats = get_stats()
        model_stats = get_stats_by_model()

        # Print results
        print(f"\n  {sys_name} Results:")
        mean, std = summary["overall_f1_mean"], summary["overall_f1_std"]
        print(f"    Overall F1: {mean:.3f} +/- {std:.3f}")
        print(f"    Questions: {summary['n_questions']}")
        for cat, cs in summary.get("by_category", {}).items():
            print(
                f"    {cat:12}: F1={cs['f1_mean']:.3f} +/- {cs['f1_std']:.3f} (N={cs['n']})"
            )
        print(f"    Tokens: {stats['total_tokens']} " f"({stats['calls']} calls)")

        # Save per-system results
        payload = {
            "system": sys_name,
            "llm_model": llm_model,
            "timestamp": datetime.now().isoformat(),
            "config": {
                "architecture": sys_info["architecture"],
                "infrastructure": sys_info["infrastructure"],
                "llm_model": llm_model,
                "judge_model": judge_model if run_judge else None,
            },
            "summary": summary,
            "token_usage": stats,
            "token_breakdown": model_stats,
            "results": all_results,
        }

        out_path = output_dir / f"{sys_name}_{model_tag}_results.json"
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"    Saved: {out_path}")

        all_summaries[sys_name] = {
            "overall_f1_mean": summary["overall_f1_mean"],
            "overall_f1_std": summary["overall_f1_std"],
            "n_questions": summary["n_questions"],
            "n_conversations": summary["n_conversations"],
            "by_category": summary.get("by_category", {}),
            "token_usage": stats,
            "token_breakdown": model_stats,
        }

    # Save aggregate summary
    if all_summaries:
        summary_path = output_dir / f"benchmark_summary_{model_tag}.json"
        summary_payload = {
            "timestamp": datetime.now().isoformat(),
            "llm_model": llm_model,
            "judge_model": judge_model if run_judge else None,
            "num_conversations": len(conversations),
            "systems": all_summaries,
        }
        with open(summary_path, "w") as f:
            json.dump(summary_payload, f, indent=2)
        print(f"\nSummary saved: {summary_path}")

        # Print comparison table
        print(f"\n{'='*70}")
        print(f"BENCHMARK SUMMARY (model={llm_model})")
        print(f"{'='*70}")
        print(
            f"  {'System':14} {'F1 Mean':>8} {'F1 Std':>8} {'N QA':>6} {'Tokens':>10}"
        )
        print(f"  {'-'*14} {'-'*8} {'-'*8} {'-'*6} {'-'*10}")
        for sys_name, s in sorted(
            all_summaries.items(),
            key=lambda x: x[1]["overall_f1_mean"],
            reverse=True,
        ):
            print(
                f"  {sys_name:14} {s['overall_f1_mean']:>8.3f} "
                f"{s['overall_f1_std']:>8.3f} "
                f"{s['n_questions']:>6} "
                f"{s['token_usage']['total_tokens']:>10}"
            )

        # Category comparison
        all_cats = set()
        for s in all_summaries.values():
            all_cats.update(s.get("by_category", {}).keys())

        if all_cats:
            for cat in sorted(all_cats):
                print(f"\n  {cat}:")
                print(f"    {'System':14} {'F1 Mean':>8} {'F1 Std':>8} {'N':>6}")
                print(f"    {'-'*14} {'-'*8} {'-'*8} {'-'*6}")
                for sys_name, s in sorted(
                    all_summaries.items(),
                    key=lambda x: x[1]
                    .get("by_category", {})
                    .get(cat, {})
                    .get("f1_mean", 0),
                    reverse=True,
                ):
                    cs = s.get("by_category", {}).get(cat, {})
                    if cs:
                        print(
                            f"    {sys_name:14} {cs['f1_mean']:>8.3f} "
                            f"{cs['f1_std']:>8.3f} {cs['n']:>6}"
                        )


if __name__ == "__main__":
    main()
