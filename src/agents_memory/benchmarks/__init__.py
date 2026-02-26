"""Benchmark registry — auto-discovers benchmark modules in this package.

Each benchmark module must export:
    BENCHMARK_INFO : dict with "name", "description", "category_names"
    download(split, num_samples) -> list[dict]   (LoCoMo-compatible format)
"""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path

BENCHMARKS: dict[str, dict] = {}


def _discover() -> None:
    """Import all sibling modules that expose BENCHMARK_INFO.

    Modules whose dependencies are not installed are silently skipped.
    """
    pkg_dir = Path(__file__).resolve().parent
    for info in pkgutil.iter_modules([str(pkg_dir)]):
        if info.name.startswith("_"):
            continue
        try:
            mod = importlib.import_module(f"{__package__}.{info.name}")
        except ImportError:
            continue
        bench_info = getattr(mod, "BENCHMARK_INFO", None)
        download_fn = getattr(mod, "download", None)
        if bench_info and download_fn:
            key = info.name
            BENCHMARKS[key] = {
                **bench_info,
                "download": download_fn,
            }


_discover()
