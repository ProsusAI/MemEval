#!/usr/bin/env python3
"""Create a stratified sample from LongMemEval for balanced category evaluation.

Samples equally from each category (or proportionally if a category has
fewer items than the per-category quota).

Usage:
    uv run python scripts/stratified_sample.py --split s --total 100 --output data/longmemeval_s_stratified_100.json
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="s", choices=["oracle", "s", "m"])
    parser.add_argument("--total", type=int, default=100)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(__file__).resolve().parents[1] / "data" / "longmemeval"
    split_files = {"oracle": "longmemeval_oracle.json", "s": "longmemeval_s_cleaned.json", "m": "longmemeval_m_cleaned.json"}
    with open(data_dir / split_files[args.split]) as f:
        data = json.load(f)

    # Group by category
    by_cat = defaultdict(list)
    for item in data:
        by_cat[item.get("question_type", "unknown")].append(item)

    cats = sorted(by_cat.keys())
    print(f"Categories ({len(cats)}): {cats}")
    print(f"Distribution: {[(c, len(by_cat[c])) for c in cats]}")

    # Equal allocation per category
    per_cat = args.total // len(cats)
    remainder = args.total - per_cat * len(cats)

    rng = random.Random(args.seed)
    sampled = []
    for i, cat in enumerate(cats):
        n = per_cat + (1 if i < remainder else 0)
        pool = by_cat[cat]
        n = min(n, len(pool))
        sampled.extend(rng.sample(pool, n))

    rng.shuffle(sampled)

    # Normalize to LoCoMo-compatible format for --data-file
    from agents_memory.benchmarks.longmemeval import _normalize, CATEGORY_NAMES, BENCHMARK_INFO
    normalized = [_normalize(item) for item in sampled]

    with open(args.output, "w") as f:
        json.dump(normalized, f)

    print(f"\nSampled {len(sampled)} questions → {args.output}")
    from collections import Counter
    dist = Counter(item["question_type"] for item in sampled)
    for cat in cats:
        print(f"  {cat:30s}: {dist[cat]}")


if __name__ == "__main__":
    main()
