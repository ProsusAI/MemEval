"""LoCoMo dataset loading and constants."""

import json
from pathlib import Path

import httpx

LOCOMO_URL = (
    "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
)
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
LOCOMO_PATH = DATA_DIR / "locomo10.json"

CATEGORY_NAMES = {
    1: "Factual",
    2: "Temporal",
    3: "Inferential",
    4: "Multi-hop",
    5: "Adversarial",
}


def download_locomo() -> list | dict:
    """Download LoCoMo dataset if not present, otherwise load from disk."""
    if LOCOMO_PATH.exists():
        print(f"Loading LoCoMo from {LOCOMO_PATH}")
        with open(LOCOMO_PATH) as f:
            return json.load(f)

    print(f"Downloading LoCoMo dataset from {LOCOMO_URL}...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    response = httpx.get(LOCOMO_URL, timeout=60)
    response.raise_for_status()
    data = response.json()

    with open(LOCOMO_PATH, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved to {LOCOMO_PATH}")
    return data


def extract_dialogues(conversation: dict) -> list[dict]:
    """Extract all dialogue turns from a LoCoMo conversation.

    LoCoMo format has sessions stored as 'session_N' keys with
    'session_N_date_time' for timestamps.
    """
    dialogues = []
    conv_data = conversation.get("conversation", {})

    # Find all session keys (session_1, session_2, etc.)
    session_nums = []
    for key in conv_data.keys():
        if key.startswith("session_") and not key.endswith("_date_time"):
            try:
                num = int(key.split("_")[1])
                session_nums.append(num)
            except (ValueError, IndexError):
                pass

    # Process sessions in order
    for num in sorted(session_nums):
        session_key = f"session_{num}"
        datetime_key = f"session_{num}_date_time"

        session_time = conv_data.get(datetime_key, "")
        session_turns = conv_data.get(session_key, [])

        if not isinstance(session_turns, list):
            continue

        for turn in session_turns:
            if isinstance(turn, dict):
                dialogues.append(
                    {
                        "speaker": turn.get("speaker", "Unknown"),
                        "text": turn.get("text", ""),
                        "dia_id": turn.get("dia_id", ""),
                        "timestamp": session_time,
                    }
                )

    return dialogues


def format_as_markdown(dialogues: list[dict]) -> str:
    """Format LoCoMo dialogue turns as a markdown document."""
    lines = []
    current_date = None
    for d in dialogues:
        timestamp = d.get("timestamp", "")
        date_part = timestamp.split(" ")[0] if timestamp else ""
        if date_part and date_part != current_date:
            current_date = date_part
            lines.append(f"\n## {current_date}\n")
        speaker = d.get("speaker", "Unknown")
        text = d.get("text", "")
        if timestamp:
            lines.append(f"**{speaker}** ({timestamp}): {text}")
        else:
            lines.append(f"**{speaker}**: {text}")
    return "\n".join(lines)
