"""LoCoMo benchmark wrapper.

Delegates to the canonical :func:`agents_memory.locomo.download_locomo` and
re-exports the category map so the benchmark registry has a uniform interface.
"""

from __future__ import annotations

from agents_memory.locomo import CATEGORY_NAMES
from agents_memory.locomo import download_locomo as _download_locomo

BENCHMARK_INFO: dict = {
    "name": "LoCoMo",
    "description": (
        "Long conversation memory benchmark — 10 conversations, "
        "1986 QA pairs across 5 categories"
    ),
    "category_names": CATEGORY_NAMES,
}


def download(split: str | None = None, num_samples: int = 10) -> list[dict]:
    """Return LoCoMo conversations (split is ignored — single dataset)."""
    data = _download_locomo()
    conversations = data if isinstance(data, list) else [data]
    return conversations[:num_samples]
