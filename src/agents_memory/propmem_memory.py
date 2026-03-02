"""Standalone PropMem wrapper for agent-memory use cases.

This adapter exposes a small API so users can use PropMem directly in apps
without running the full benchmark pipeline.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

from openai import OpenAI

from agents_memory.propmem import PropMemSystem


class PropMemMemory:
    """Simple PropMem memory interface for agents.

    Typical flow:
    1) add_session(turns, session_date=...)
    2) ask("question")
    """

    def __init__(
        self,
        *,
        user_name: str = "User",
        assistant_name: str = "Assistant",
        llm_model: str = "gpt-4.1",
        embedding_model: str = "text-embedding-3-small",
        use_llm_classifier: bool = False,
        use_temporal_boost: bool = True,
        use_knowledge_updates: bool = True,
        openai_api_key: str | None = None,
    ) -> None:
        api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. Pass openai_api_key or set env var."
            )

        self._client = OpenAI(api_key=api_key)
        self._llm_model = llm_model
        self._system_config: dict[str, Any] = {
            "embedding_model": embedding_model,
            "use_llm_classifier": use_llm_classifier,
            "use_temporal_boost": use_temporal_boost,
            "use_knowledge_updates": use_knowledge_updates,
        }
        self._conversation: dict[str, Any] = {
            "conversation": {
                "speaker_a": user_name,
                "speaker_b": assistant_name,
            }
        }
        self._next_session_num = 1
        self._system: PropMemSystem | None = None

    def add_session(
        self,
        turns: list[dict[str, str]],
        *,
        session_date: str | None = None,
    ) -> dict[str, Any]:
        """Add one session and rebuild PropMem indices.

        `turns` format:
        [{"speaker": "User", "text": "..."}, {"speaker": "Assistant", "text": "..."}]
        """
        if not turns:
            raise ValueError("turns must not be empty")

        normalized_turns: list[dict[str, str]] = []
        for turn in turns:
            normalized_turns.append(
                {
                    "speaker": str(turn.get("speaker", "Unknown")),
                    "text": str(turn.get("text", "")),
                }
            )

        session_key = f"session_{self._next_session_num}"
        date_key = f"{session_key}_date_time"
        if not session_date:
            session_date = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

        self._conversation["conversation"][session_key] = normalized_turns
        self._conversation["conversation"][date_key] = session_date
        self._next_session_num += 1

        # PropMem ingestion is batch-oriented, so rebuild once from all sessions.
        self._system = PropMemSystem(**self._system_config)
        return self._system.ingest_conversation(
            self._conversation, self._client, self._llm_model
        )

    def ask(self, question: str) -> str:
        """Ask a question against memory and return only the answer string."""
        result = self.ask_with_details(question)
        return str(result.get("answer", "")).strip()

    def ask_with_details(self, question: str) -> dict[str, Any]:
        """Ask a question and return PropMem retrieval metadata."""
        if not self._system:
            raise RuntimeError("No memory indexed yet. Call add_session() first.")
        return self._system.answer_question(question, self._client, self._llm_model)
