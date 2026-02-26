"""LongMemEval benchmark loader.

Downloads from HuggingFace (``xiaowu0162/longmemeval-cleaned``) and normalizes
each question into the LoCoMo-compatible format used by all system adapters.

Splits
------
- ``oracle``  — evidence sessions only (~15 MB, ~3 sessions/question)
- ``s``       — small haystack (~277 MB, ~53 sessions/question, ~115k tokens)
- ``m``       — medium haystack (~2.7 GB, ~1.5M tokens/question)
"""

from __future__ import annotations

import json
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "xiaowu0162/longmemeval-cleaned"

_SPLIT_FILES: dict[str, str] = {
    "oracle": "longmemeval_oracle.json",
    "s": "longmemeval_s_cleaned.json",
    "m": "longmemeval_m_cleaned.json",
}

CATEGORY_NAMES: dict[str, str] = {
    "single-session-user": "Single-Session User",
    "single-session-assistant": "Single-Session Assistant",
    "single-session-preference": "Single-Session Preference",
    "multi-session": "Multi-Session",
    "temporal-reasoning": "Temporal Reasoning",
    "knowledge-update": "Knowledge Update",
}

BENCHMARK_INFO: dict = {
    "name": "LongMemEval",
    "description": (
        "Long-term memory evaluation — 500 questions across 6 categories "
        "with variable-size haystacks"
    ),
    "category_names": CATEGORY_NAMES,
}

DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "longmemeval"


def _download_split(split: str) -> list[dict]:
    """Download a LongMemEval split from HuggingFace and cache locally."""
    filename = _SPLIT_FILES[split]
    local_path = DATA_DIR / filename

    if local_path.exists():
        print(f"Loading LongMemEval ({split}) from {local_path}")
        with open(local_path) as f:
            return json.load(f)

    print(f"Downloading LongMemEval ({split}) from HuggingFace...")
    downloaded = hf_hub_download(
        repo_id=REPO_ID,
        filename=filename,
        repo_type="dataset",
    )

    # Copy to our data dir so subsequent loads are fast
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(downloaded) as f:
        data = json.load(f)
    with open(local_path, "w") as f:
        json.dump(data, f)
    print(f"Saved to {local_path}")
    return data


def _normalize(item: dict) -> dict:
    """Convert a single LongMemEval question into LoCoMo-compatible format.

    LoCoMo format expects:
        sample_id, conversation (with session_N / session_N_date_time keys,
        speaker_a, speaker_b), and qa list.
    """
    conversation: dict = {
        "speaker_a": "user",
        "speaker_b": "assistant",
    }

    haystack_sessions = item.get("haystack_sessions", [])
    haystack_dates = item.get("haystack_dates", [])
    haystack_ids = item.get("haystack_session_ids", [])

    for idx, session in enumerate(haystack_sessions):
        session_num = idx + 1
        date_str = haystack_dates[idx] if idx < len(haystack_dates) else ""
        session_id = haystack_ids[idx] if idx < len(haystack_ids) else f"session_{session_num}"

        turns: list[dict] = []
        for turn_idx, msg in enumerate(session):
            role = msg.get("role", "user")
            turns.append(
                {
                    "speaker": role,
                    "text": msg.get("content", ""),
                    "dia_id": f"{session_id}_{turn_idx}",
                }
            )

        conversation[f"session_{session_num}"] = turns
        conversation[f"session_{session_num}_date_time"] = date_str

    answer = item.get("answer", "")
    if not isinstance(answer, str):
        answer = str(answer)

    return {
        "sample_id": item.get("question_id", "unknown"),
        "conversation": conversation,
        "qa": [
            {
                "question": item.get("question", ""),
                "answer": answer,
                "category": item.get("question_type", "unknown"),
            }
        ],
    }


def download(split: str | None = "oracle", num_samples: int = 500) -> list[dict]:
    """Download LongMemEval and return normalized LoCoMo-compatible dicts.

    Parameters
    ----------
    split : str
        One of ``"oracle"``, ``"s"``, ``"m"``.  Defaults to ``"oracle"``.
    num_samples : int
        Maximum number of questions to return.
    """
    split = split or "oracle"
    if split not in _SPLIT_FILES:
        raise ValueError(f"Unknown LongMemEval split '{split}'. Choose from: {list(_SPLIT_FILES)}")

    raw = _download_split(split)
    normalized = [_normalize(item) for item in raw[:num_samples]]
    print(f"LongMemEval: {len(normalized)} questions loaded (split={split})")
    return normalized
