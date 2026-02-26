#!/usr/bin/env python3
"""Unified benchmark runner for memory systems.

Supports multiple benchmarks (LoCoMo, LongMemEval) and any combination of
memory systems with configurable LLM model and optional judge evaluation.

Usage:
    # LoCoMo (default benchmark), single system, 1 conversation
    uv run python scripts/run_full_benchmark.py --systems propmem --num-samples 1 --skip-judge

    # All systems on LoCoMo, gpt-4.1-mini
    uv run python scripts/run_full_benchmark.py \
        --systems all --num-samples 10 --llm-model gpt-4.1-mini --skip-judge

    # LongMemEval oracle split (cheapest), 1 question
    uv run python scripts/run_full_benchmark.py \
        --benchmark longmemeval --split oracle --systems fullcontext --num-samples 1 --skip-judge

    # LongMemEval S split (115k tokens/question)
    uv run python scripts/run_full_benchmark.py \
        --benchmark longmemeval --split s --systems fullcontext --num-samples 1 --skip-judge
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm

from agents_memory.benchmarks import BENCHMARKS
from agents_memory.locomo import CATEGORY_NAMES as _DEFAULT_CATEGORIES
from agents_memory.systems import SYSTEMS
from agents_memory.token_tracker import get_stats, get_stats_by_model
from agents_memory.token_tracker import reset as reset_tracker
from agents_memory.token_tracker import start as start_tracking

DATA_DIR = Path(__file__).parent.parent / "data"

# Ensure unbuffered output
sys.stdout.reconfigure(line_buffering=True)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified benchmark runner for memory systems"
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="locomo",
        choices=list(BENCHMARKS.keys()),
        help=f"Benchmark to run (default: locomo). Available: {', '.join(BENCHMARKS)}",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Benchmark-specific split (e.g., oracle/s/m for longmemeval)",
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
        help="Custom data file in LoCoMo format (default: use benchmark registry)",
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

    # Resolve benchmark
    benchmark_key = args.benchmark
    bench = BENCHMARKS[benchmark_key]
    bench_name = bench["name"]
    category_names = bench.get("category_names", _DEFAULT_CATEGORIES)

    # Load data
    if args.data_file:
        print(f"Loading custom data from {args.data_file}")
        with open(args.data_file) as f:
            data = json.load(f)
        conversations = data if isinstance(data, list) else [data]
        conversations = conversations[: args.num_samples]
    else:
        conversations = bench["download"](
            split=args.split, num_samples=args.num_samples
        )

    print("=" * 70)
    print("Full Benchmark Runner")
    print("=" * 70)
    print(
        f"  Benchmark: {bench_name}"
        + (f" (split={args.split})" if args.split else "")
    )
    print(f"  Systems: {', '.join(system_names)}")
    print(f"  LLM Model: {llm_model}")
    print(
        f"  Judge: {'enabled (' + judge_model + ')' if run_judge else 'disabled'}"
    )
    print(f"  Conversations: {len(conversations)}")
    print(f"  Output: {output_dir}")
    print("=" * 70)

    # Build unique run tag: benchmark_model_timestamp (never overwrites)
    model_tag = llm_model.replace(".", "")
    bench_tag = benchmark_key
    if args.split:
        bench_tag = f"{benchmark_key}_{args.split}"
    if args.data_file:
        bench_tag = Path(args.data_file).stem.replace(".", "")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_tag = f"{bench_tag}_{model_tag}_{ts}"

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
                results = run_fn(
                    conv, llm_model, run_judge, category_names=category_names
                )
                all_results.extend(results)
            except Exception as err:
                print(f"  ERROR on conv {sample_id}: {err}")
                import traceback

                traceback.print_exc()
                continue

            # Progress
            if all_results:
                running_f1 = np.mean([r["f1"] for r in all_results])
                print(
                    f"  Running F1: {running_f1:.3f} ({len(all_results)} questions)"
                )

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
                f"    {cat:12}: F1={cs['f1_mean']:.3f} "
                f"+/- {cs['f1_std']:.3f} (N={cs['n']})"
            )
        print(
            f"    Tokens: {stats['total_tokens']} ({stats['calls']} calls)"
        )

        # Save per-system results
        payload = {
            "system": sys_name,
            "benchmark": bench_name,
            "split": args.split,
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

        out_path = output_dir / f"{sys_name}_{run_tag}_results.json"
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
        summary_path = output_dir / f"benchmark_summary_{run_tag}.json"
        summary_payload = {
            "timestamp": datetime.now().isoformat(),
            "benchmark": bench_name,
            "split": args.split,
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
        print(f"BENCHMARK SUMMARY ({bench_name}, model={llm_model})")
        print(f"{'='*70}")
        print(
            f"  {'System':14} {'F1 Mean':>8} {'F1 Std':>8} "
            f"{'N QA':>6} {'Tokens':>10}"
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
                print(
                    f"    {'System':14} {'F1 Mean':>8} "
                    f"{'F1 Std':>8} {'N':>6}"
                )
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
