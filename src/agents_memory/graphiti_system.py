"""Graphiti: temporal knowledge graph memory.

Wraps graphiti-core with Kuzu embedded graph DB for benchmarking.
Each conversation gets its own temp DB directory for isolation.

Architecture:
  1. Ingestion — add_episode() per session with group_id isolation
  2. Retrieval — search() returns edge facts ranked by relevance
  3. Answer — OpenAIAnswerAgent over retrieved facts
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from agents_memory.answer_openai import AsyncOpenAIAnswerAgent

# LoCoMo dates look like "1:56 pm on 8 May, 2023"
_MONTH_MAP = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}


def parse_locomo_datetime(raw: str) -> datetime:
    """Parse LoCoMo datetime string to a naive datetime.

    LoCoMo format: "1:56 pm on 8 May, 2023"
    Returns naive datetime (no timezone) to avoid Kuzu timezone issues.
    """
    if not raw:
        return datetime(2023, 1, 1)

    # Try to extract time, day, month, year
    m = re.match(
        r"(\d{1,2}):(\d{2})\s*(am|pm)\s+on\s+(\d{1,2})\s+(\w+),?\s+(\d{4})",
        raw.strip(),
        re.IGNORECASE,
    )
    if m:
        hour = int(m.group(1))
        minute = int(m.group(2))
        ampm = m.group(3).lower()
        day = int(m.group(4))
        month_str = m.group(5).lower()
        year = int(m.group(6))

        if ampm == "pm" and hour != 12:
            hour += 12
        elif ampm == "am" and hour == 12:
            hour = 0

        month = _MONTH_MAP.get(month_str, 1)
        return datetime(year, month, day, hour, minute)

    # Fallback: try to extract just day month year
    m2 = re.search(r"(\d{1,2})\s+(\w+),?\s+(\d{4})", raw)
    if m2:
        day = int(m2.group(1))
        month_str = m2.group(2).lower()
        year = int(m2.group(3))
        month = _MONTH_MAP.get(month_str, 1)
        return datetime(year, month, day)

    return datetime(2023, 1, 1)


def _format_session_text(turns: list[dict]) -> str:
    """Format session turns into a single text block for Graphiti."""
    lines = []
    for t in turns:
        speaker = t.get("speaker", "?")
        text = t.get("text", "")
        lines.append(f"{speaker}: {text}")
    return "\n".join(lines)


@dataclass
class GraphitiSystem:
    """Graphiti temporal knowledge graph wrapper for benchmarking.

    Uses Kuzu embedded DB (no external Neo4j needed).
    Each instance creates a temp directory for DB isolation.
    """

    llm_model: str = "gpt-4.1"
    num_search_results: int = 10
    entity_names: list[str] = field(default_factory=list)

    # Internal state
    _graphiti: Any = field(default=None, init=False)
    _kuzu_driver: Any = field(default=None, init=False)
    _db_dir: str = field(default="", init=False)

    async def _init_graphiti(self) -> None:
        """Initialize Graphiti with a fresh Kuzu DB."""
        from graphiti_core import Graphiti
        from graphiti_core.driver.kuzu_driver import KuzuDriver
        from graphiti_core.embedder import OpenAIEmbedder, OpenAIEmbedderConfig
        from graphiti_core.llm_client import LLMConfig, OpenAIClient

        self._db_dir = tempfile.mkdtemp(prefix="graphiti_bench_")
        db_path = os.path.join(self._db_dir, "kuzu_db")

        llm_config = LLMConfig(
            api_key=os.environ["OPENAI_API_KEY"],
            model=self.llm_model,
            small_model=self.llm_model,
        )
        llm_client = OpenAIClient(config=llm_config)

        emb_config = OpenAIEmbedderConfig(
            api_key=os.environ["OPENAI_API_KEY"],
            embedding_model="text-embedding-3-small",
        )
        embedder = OpenAIEmbedder(config=emb_config)

        self._kuzu_driver = KuzuDriver(db=db_path)

        self._graphiti = Graphiti(
            llm_client=llm_client,
            embedder=embedder,
            graph_driver=self._kuzu_driver,
            store_raw_episode_content=False,
        )
        await self._graphiti.build_indices_and_constraints()
        # Create empty FTS indexes so add_episode() searches don't crash
        await self._build_fts_indexes()

    async def _build_fts_indexes(self) -> None:
        """Create Kuzu FTS indexes after ingestion.

        Workaround for graphiti-core bug (#1112): build_indices_and_constraints()
        is Neo4j-only and doesn't create FTS indexes for Kuzu.
        Must be called after ingestion since Kuzu FTS only indexes
        entries that exist at creation time.
        """
        if self._kuzu_driver is None:
            return
        fts_indexes = [
            ("Entity", "node_name_and_summary", "['name', 'summary']"),
            ("RelatesToNode_", "edge_name_and_fact", "['name', 'fact']"),
            ("Community", "community_name", "['name']"),
            ("Episodic", "episode_content", "['content']"),
        ]
        for table, index_name, columns in fts_indexes:
            try:
                await self._kuzu_driver.execute_query(
                    f"CALL CREATE_FTS_INDEX('{table}', '{index_name}', {columns})"
                )
            except Exception as err:
                print(f"  FTS index {table}/{index_name}: {err}")

    async def _rebuild_fts_indexes(self) -> None:
        """Drop and recreate FTS indexes to include newly ingested data."""
        if self._kuzu_driver is None:
            return
        fts_indexes = [
            ("Entity", "node_name_and_summary", "['name', 'summary']"),
            ("RelatesToNode_", "edge_name_and_fact", "['name', 'fact']"),
            ("Community", "community_name", "['name']"),
            ("Episodic", "episode_content", "['content']"),
        ]
        for table, index_name, columns in fts_indexes:
            try:
                await self._kuzu_driver.execute_query(
                    f"CALL DROP_FTS_INDEX('{table}', '{index_name}')"
                )
            except Exception:
                pass
            try:
                await self._kuzu_driver.execute_query(
                    f"CALL CREATE_FTS_INDEX('{table}', '{index_name}', {columns})"
                )
            except Exception as err:
                print(f"  FTS rebuild {table}/{index_name}: {err}")

    async def ingest_conversation(self, conv: dict) -> dict:
        """Ingest a LoCoMo conversation into Graphiti."""
        sample_id = conv.get("sample_id", "unknown")
        conversation = conv["conversation"]
        speaker_a = conversation.get("speaker_a", "User A")
        speaker_b = conversation.get("speaker_b", "User B")
        self.entity_names = [speaker_a, speaker_b]

        group_id = f"locomo-{sample_id}"
        await self._init_graphiti()

        # Find session keys
        session_keys = sorted(
            [
                k
                for k in conversation.keys()
                if k.startswith("session_") and not k.endswith("_date_time")
            ],
            key=lambda x: int(x.split("_")[1]),
        )

        num_episodes = 0
        num_turns = 0

        for sk in session_keys:
            session_num = sk.split("_")[1]
            date_key = f"session_{session_num}_date_time"
            date_str = conversation.get(date_key, "")
            turns = conversation[sk]

            if not isinstance(turns, list) or not turns:
                continue

            num_turns += len(turns)
            session_text = _format_session_text(turns)
            ref_time = parse_locomo_datetime(date_str)

            await self._graphiti.add_episode(
                name=f"Session {session_num}",
                episode_body=session_text,
                source_description=(
                    f"Conversation between {speaker_a} and {speaker_b}"
                ),
                reference_time=ref_time,
            )
            num_episodes += 1

        # Rebuild FTS indexes so Kuzu indexes all ingested entities/edges
        # (Kuzu FTS only indexes data present at creation time)
        await self._rebuild_fts_indexes()

        return {
            "num_turns": num_turns,
            "num_episodes": num_episodes,
            "entities": self.entity_names,
            "group_id": group_id,
        }

    async def answer_question(
        self,
        question: str,
        answer_agent: AsyncOpenAIAnswerAgent,
    ) -> dict[str, Any]:
        """Search Graphiti and answer a question."""
        if not question.strip():
            return {"answer": "None", "num_facts": 0}

        results = await self._graphiti.search(
            query=question,
            num_results=self.num_search_results,
        )

        # Build memory text from edge facts
        fact_lines = []
        for edge in results:
            fact = getattr(edge, "fact", None) or str(edge)
            fact_lines.append(f"- {fact}")

        memory_text = "\n".join(fact_lines) if fact_lines else ""
        answer = await answer_agent.answer(question, memory_text)

        return {
            "answer": answer,
            "num_facts": len(fact_lines),
        }

    async def close(self) -> None:
        """Cleanup Graphiti and delete temp DB directory."""
        if self._graphiti is not None:
            try:
                await self._graphiti.close()
            except Exception:
                pass
            self._graphiti = None

        if self._db_dir and os.path.exists(self._db_dir):
            shutil.rmtree(self._db_dir, ignore_errors=True)
            self._db_dir = ""
