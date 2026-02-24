"""Semantic memory for agents — stores and retrieves user facts from conversations.

Designed for agent platforms where one user talks to one agent across sessions.
Extracts atomic propositions from conversations, stores with embeddings,
retrieves by hybrid vector + BM25 search. The agent generates answers — this
is a retrieval system, not an answer system.

Architecture borrowed from OpenClaw and OpenClaw+:
- Hybrid search (vector + BM25 via SQLite FTS5) from OpenClaw
- Proposition extraction via LLM from OpenClaw+
- 0.8/0.2 vector/BM25 weight split for short propositions from OpenClaw+
- BM25 AND→OR fallback from OpenClaw+
- Partial JSON recovery for truncated LLM output from OpenClaw+

Usage:
    from agents_memory import Memory

    memory = Memory(api_key="sk-...")

    # After a conversation session
    memory.add_session([
        {"role": "user", "content": "I love Thai food but I'm allergic to peanuts"},
        {"role": "assistant", "content": "Noted!"},
    ], session_date="2024-01-15")

    # Before generating a response — retrieve relevant memories
    memories = memory.recall("What food restrictions does the user have?")
    # → [{"text": "user is allergic to peanuts", "date": "2024-01-15", "score": 0.87}]

    # User asks to forget something
    memory.forget("peanut allergy")

    memory.save("user_123.json")
    memory = Memory.load("user_123.json", api_key="sk-...")
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import sqlite3
import struct
import tempfile
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class Proposition:
    """An atomic fact about the user, extracted from a conversation session."""

    text: str
    date: str
    session_id: str


# ---------------------------------------------------------------------------
# Extraction prompt
# ---------------------------------------------------------------------------

EXTRACT_PROMPT = """\
Extract facts about the user from this conversation that would be useful in FUTURE conversations.

CORE FILTER: For each candidate fact, ask: "Would knowing this improve a future conversation \
with this user?" If not, skip it.

Session date: {date}

Conversation:
{turns_text}

Extract facts about the user including:
- Personal details, preferences, opinions, decisions
- Events, activities, plans
- Relationships, feelings, reactions
- Names, places, people, tools, systems mentioned
- Career, education, family, professional background
- Technical values, parameters, configurations, specifications
- Recurring patterns and durable preferences ("always use X", "every time Y happens", \
"I prefer Z for tasks like this") — these are high-value memories

Return a JSON object:
{{"facts": [
  {{"fact": "self-contained fact about the user [{date}]"}}
]}}

Rules:
- Each fact must be a COMPLETE, self-contained sentence
- End every fact with the session date in brackets: "user is allergic to peanuts [{date}]"
- ALWAYS resolve relative dates to absolute dates using the session date. \
"next week Tuesday" on session date 2025-03-01 becomes "2025-03-04". \
"last month" becomes the actual month/year. "yesterday" becomes the actual date. \
Never store "next week", "tomorrow", "last year" — these become meaningless later
- For facts with their own temporal context, include both: \
"user is planning a trip to Japan on 2025-03-15 [{date}]"
- Preserve exact technical values (numbers, codes, formulas, URLs) verbatim
- Be specific and factual, not interpretive
- One atomic fact per entry — do NOT combine multiple facts

Avoid redundancy:
- Do NOT repeat the same contextual detail (like trip dates, locations) in multiple facts. \
Mention it once in the most relevant fact, then refer to it briefly in others. \
BAD: "user is going to NYC 2025-03-04 to 2025-03-06", \
"user's team of 5 is going to NYC 2025-03-04 to 2025-03-06", \
"user is presenting at NYC quarterly review 2025-03-04 to 2025-03-06" \
GOOD: "user is going to NYC 2025-03-04 to 2025-03-06 for quarterly review with team of 5", \
"user is presenting Q2 roadmap at the NYC quarterly review"
- If one fact is a subset of another, keep only the more complete one. \
BAD: "user is vegetarian" AND "two of user's team including user are vegetarian" \
GOOD: "user is vegetarian" (standalone personal fact), \
"user's teammate Jin is gluten-free" (separate person)

Skip these — they are NOT useful memories:
- Greetings, thank-yous, conversational filler
- Session actions: "user asked for recommendations", "user requested help with X"
- General knowledge not specific to the user: "flight from SF to NYC takes 5 hours"
- One-time content edits or specific data requests without reusable context: \
"user asked to change chart title", "user wanted weather in Amsterdam"
- Restaurant/hotel/booking preferences that are one-time logistics, \
UNLESS they reveal a durable preference (e.g. "user prefers aisle seats" is durable)
- Vague preferences without enough context to be actionable in the future"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _strip_thinking(text: str) -> str:
    """Remove reasoning/thinking blocks from assistant messages."""
    text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"```thinking\n.*?\n```", "", text, flags=re.DOTALL)
    return text.strip()


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    length = min(len(a), len(b))
    dot = sum(a[i] * b[i] for i in range(length))
    norm_a = math.sqrt(sum(x * x for x in a[:length]))
    norm_b = math.sqrt(sum(x * x for x in b[:length]))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _embed_texts(
    texts: list[str],
    model: str,
    client: OpenAI,
) -> list[list[float]]:
    """Embed texts via OpenAI API. Batches at 512."""
    if not texts:
        return []
    embeddings: list[list[float]] = []
    for i in range(0, len(texts), 512):
        batch = texts[i : i + 512]
        response = client.embeddings.create(model=model, input=batch)
        for item in response.data:
            embeddings.append(item.embedding)
    return embeddings


def _recover_partial_json(content: str) -> list[dict]:
    """Extract complete {"fact": ...} objects from truncated JSON."""
    pattern = r'\{\s*"fact"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"\s*\}'
    items = []
    for m in re.finditer(pattern, content):
        fact = m.group(1).replace('\\"', '"').replace("\\n", " ")
        if fact:
            items.append({"fact": fact})
    return items


def _pack_embedding(embedding: list[float]) -> bytes:
    """Pack a float list into a compact bytes blob (float32)."""
    return struct.pack(f"<{len(embedding)}f", *embedding)


def _unpack_embedding(blob: bytes) -> list[float]:
    """Unpack a bytes blob back into a float list."""
    n = len(blob) // 4  # 4 bytes per float32
    return list(struct.unpack(f"<{n}f", blob))


def _bm25_search(
    query: str,
    texts: list[str],
    indices: list[int] | None = None,
    db: sqlite3.Connection | None = None,
    db_id_map: list[int] | None = None,
) -> dict[int, float]:
    """BM25 keyword search over texts using SQLite FTS5.

    Ported from OpenClaw's hybrid.ts — AND-joined query first, OR fallback.
    Score conversion: 1 / (1 + max(0, rank)).

    When ``db`` is a persistent connection with a populated ``propositions_fts``
    table, queries it directly (much faster — no temp table creation).  The
    ``db_id_map`` list maps in-memory indices to SQLite rowids so the caller
    gets back the same index space as the in-memory lists.
    """
    tokens = re.findall(r"[A-Za-z0-9_]+", query)
    if not tokens:
        return {}

    if indices is None:
        indices = list(range(len(texts)))

    fts_query_and = " AND ".join(f'"{t}"' for t in tokens)
    fts_query_or = " OR ".join(f'"{t}"' for t in tokens)

    # --- Persistent FTS5 path ---
    if db is not None and db_id_map is not None:
        rowid_to_idx = {rowid: idx for idx, rowid in enumerate(db_id_map)}
        try:
            rows = db.execute(
                "SELECT rowid, bm25(propositions_fts) AS rank "
                "FROM propositions_fts WHERE propositions_fts MATCH ? "
                "ORDER BY rank ASC LIMIT ?",
                (fts_query_and, len(indices)),
            ).fetchall()
            if not rows:
                rows = db.execute(
                    "SELECT rowid, bm25(propositions_fts) AS rank "
                    "FROM propositions_fts WHERE propositions_fts MATCH ? "
                    "ORDER BY rank ASC LIMIT ?",
                    (fts_query_or, len(indices)),
                ).fetchall()
        except sqlite3.OperationalError:
            return {}

        scores: dict[int, float] = {}
        for row in rows:
            rowid = int(row[0])
            idx = rowid_to_idx.get(rowid)
            if idx is not None and idx in set(indices):
                rank = float(row[1])
                scores[idx] = 1.0 / (1.0 + max(0.0, rank))
        return scores

    # --- Temp-table path (original, for db_path=None) ---
    tmp_db = sqlite3.connect(":memory:")
    tmp_db.execute("CREATE VIRTUAL TABLE prop_fts USING fts5(prop_id, text)")
    for idx in indices:
        tmp_db.execute(
            "INSERT INTO prop_fts(prop_id, text) VALUES (?, ?)",
            (str(idx), texts[idx]),
        )

    try:
        rows = tmp_db.execute(
            "SELECT prop_id, bm25(prop_fts) AS rank "
            "FROM prop_fts WHERE prop_fts MATCH ? "
            "ORDER BY rank ASC LIMIT ?",
            (fts_query_and, len(indices)),
        ).fetchall()
        if not rows:
            rows = tmp_db.execute(
                "SELECT prop_id, bm25(prop_fts) AS rank "
                "FROM prop_fts WHERE prop_fts MATCH ? "
                "ORDER BY rank ASC LIMIT ?",
                (fts_query_or, len(indices)),
            ).fetchall()
    except sqlite3.OperationalError:
        tmp_db.close()
        return {}

    scores = {}
    for row in rows:
        idx = int(row[0])
        rank = float(row[1])
        scores[idx] = 1.0 / (1.0 + max(0.0, rank))

    tmp_db.close()
    return scores


def _hybrid_scores(
    query_emb: list[float],
    embeddings: list[list[float]],
    query: str,
    texts: list[str],
    vector_weight: float = 0.8,
    text_weight: float = 0.2,
    db: sqlite3.Connection | None = None,
    db_id_map: list[int] | None = None,
) -> dict[int, float]:
    """Compute hybrid vector + BM25 scores for all propositions.

    Weight split: 0.8 vector / 0.2 BM25 — propositions are ~25 words,
    too short for BM25 to be highly discriminative on its own, but it
    catches exact keyword matches that embeddings miss (codes, numbers).
    """
    # Vector scores
    vec_scores: dict[int, float] = {}
    for i, emb in enumerate(embeddings):
        vec_scores[i] = _cosine_similarity(query_emb, emb)

    # BM25 scores
    bm25_scores = _bm25_search(query, texts, db=db, db_id_map=db_id_map)

    # Merge
    merged: dict[int, float] = {}
    for i in range(len(embeddings)):
        vs = vec_scores.get(i, 0.0)
        bs = bm25_scores.get(i, 0.0)
        merged[i] = vector_weight * vs + text_weight * bs

    return merged


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------


class Memory:
    """Semantic memory for agents.

    Extracts atomic propositions from conversations, stores them with
    embeddings, and retrieves by vector similarity. No answer generation —
    the agent uses retrieved facts as context for its own responses.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4.1-mini",
        embedding_model: str = "text-embedding-3-small",
        entity_name: str = "user",
        assistant_name: str = "assistant",
        max_propositions: int = 1000,
        dedup_threshold: float = 0.95,
        db_path: str | None = None,
    ):
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._model = model
        self._embedding_model = embedding_model
        self._entity_name = entity_name
        self._assistant_name = assistant_name
        self._max_propositions = max_propositions
        self._dedup_threshold = dedup_threshold
        self._client = OpenAI(api_key=self._api_key)
        self._propositions: list[Proposition] = []
        self._embeddings: list[list[float]] = []
        self._session_counter = 0

        # SQLite persistence (optional)
        self._db: sqlite3.Connection | None = None
        self._db_path: str | None = db_path
        self._db_ids: list[int] = []  # parallel to _propositions — SQLite rowids
        if db_path is not None:
            self._init_db(db_path)
            self._load_from_db()

    def __len__(self) -> int:
        return len(self._propositions)

    def __repr__(self) -> str:
        return f"Memory({len(self._propositions)} propositions, model={self._model!r})"

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def add_session(
        self,
        messages: list[dict[str, str]] | str,
        session_id: str = "",
        session_date: str = "",
    ) -> dict[str, Any]:
        """Extract and store facts from a conversation session.

        Args:
            messages: One of:
                - List of dicts with "role" and "text" or "content" keys
                  (OpenAI chat format works directly)
                - Raw text string (ingested as a single user message)
            session_id: Optional session identifier. Auto-generated if empty.
            session_date: Optional date string (e.g. "2024-01-15").

        Returns:
            Stats dict with num_new, num_dupes, total.
        """
        if isinstance(messages, str):
            messages = [{"role": "user", "text": messages}]

        if not messages:
            return {
                "session_id": "",
                "num_new": 0,
                "num_dupes": 0,
                "total": len(self._propositions),
            }

        if not session_id:
            self._session_counter += 1
            session_id = f"session_{self._session_counter}"

        # Filter to user + assistant visible messages only
        turns = self._filter_messages(messages)

        if not turns:
            return {
                "session_id": session_id,
                "num_new": 0,
                "num_dupes": 0,
                "total": len(self._propositions),
            }

        # Extract propositions (LLM call)
        new_props = self._extract_facts(turns, session_date, session_id)

        if not new_props:
            return {
                "session_id": session_id,
                "num_new": 0,
                "num_dupes": 0,
                "total": len(self._propositions),
            }

        # Embed new propositions
        new_embs = _embed_texts(
            [p.text for p in new_props],
            self._embedding_model,
            self._client,
        )

        # Dedup against existing propositions
        props_to_add = []
        embs_to_add = []
        num_dupes = 0

        for prop, emb in zip(new_props, new_embs):
            if self._is_duplicate(emb):
                num_dupes += 1
            else:
                props_to_add.append(prop)
                embs_to_add.append(emb)

        self._propositions.extend(props_to_add)
        self._embeddings.extend(embs_to_add)

        # Write-through to SQLite
        if self._db is not None and props_to_add:
            for prop, emb in zip(props_to_add, embs_to_add):
                cur = self._db.execute(
                    "INSERT INTO propositions (text, date, session_id, embedding) "
                    "VALUES (?, ?, ?, ?)",
                    (prop.text, prop.date, prop.session_id, _pack_embedding(emb)),
                )
                self._db_ids.append(cur.lastrowid)
            self._db.execute(
                "INSERT OR REPLACE INTO config (key, value) VALUES (?, ?)",
                ("session_counter", str(self._session_counter)),
            )
            self._db.commit()

        # Enforce max size — drop oldest
        self._enforce_limit()

        return {
            "session_id": session_id,
            "num_new": len(props_to_add),
            "num_dupes": num_dupes,
            "total": len(self._propositions),
        }

    # Alias
    add_messages = add_session

    # ------------------------------------------------------------------
    # Recall
    # ------------------------------------------------------------------

    def recall(
        self,
        query: str,
        top_k: int = 10,
        min_score: float = 0.3,
    ) -> list[dict[str, Any]]:
        """Retrieve relevant memories via hybrid vector + BM25 search.

        Returns list of {"text", "date", "session_id", "score"} sorted by
        relevance. Only returns results above min_score. On equal scores,
        more recently added propositions rank first (recency tiebreaker).
        """
        if not self._propositions or not self._embeddings or not query.strip():
            return []

        query_emb = _embed_texts([query], self._embedding_model, self._client)[0]
        texts = [p.text for p in self._propositions]
        scores = _hybrid_scores(
            query_emb,
            self._embeddings,
            query,
            texts,
            db=self._db,
            db_id_map=self._db_ids or None,
        )

        scored = [(i, s) for i, s in scores.items() if s >= min_score]
        # Sort by score descending, then by index descending (newer = higher index)
        scored.sort(key=lambda x: (x[1], x[0]), reverse=True)

        results = []
        for idx, score in scored[:top_k]:
            p = self._propositions[idx]
            results.append(
                {
                    "text": p.text,
                    "date": p.date,
                    "session_id": p.session_id,
                    "score": round(score, 4),
                }
            )
        return results

    # ------------------------------------------------------------------
    # Forget
    # ------------------------------------------------------------------

    def forget(
        self,
        query: str,
        threshold: float = 0.7,
    ) -> dict[str, Any]:
        """Delete memories matching the query.

        If one match scores > 0.9, deletes it immediately.
        Otherwise returns candidates above threshold for confirmation.

        Returns:
            {"action": "deleted", "deleted": [...]} or
            {"action": "candidates", "candidates": [...]} or
            {"action": "none"}
        """
        if not self._propositions or not self._embeddings or not query.strip():
            return {"action": "none", "message": "No memories to search."}

        query_emb = _embed_texts([query], self._embedding_model, self._client)[0]
        texts = [p.text for p in self._propositions]
        scores = _hybrid_scores(
            query_emb,
            self._embeddings,
            query,
            texts,
            db=self._db,
            db_id_map=self._db_ids or None,
        )

        candidates = [(i, s) for i, s in scores.items() if s >= threshold]
        candidates.sort(key=lambda x: x[1], reverse=True)

        if not candidates:
            return {"action": "none", "message": "No matching memories found."}

        # Single high-confidence match — auto-delete.
        # Hybrid scores max at ~1.0 only when both vector and BM25 agree.
        # A score of 0.75+ means very high vector similarity (cosine ≥ 0.9)
        # even when BM25 contributes nothing.
        if len(candidates) == 1 and candidates[0][1] > 0.75:
            idx = candidates[0][0]
            deleted_text = self._propositions[idx].text
            self._delete_at(idx)
            return {"action": "deleted", "deleted": [deleted_text]}

        # Multiple or ambiguous — return candidates
        candidate_list = []
        for idx, score in candidates[:5]:
            candidate_list.append(
                {
                    "index": idx,
                    "text": self._propositions[idx].text,
                    "date": self._propositions[idx].date,
                    "score": round(score, 4),
                }
            )
        return {"action": "candidates", "candidates": candidate_list}

    def forget_index(self, index: int) -> dict[str, Any]:
        """Delete a specific proposition by index (from forget() candidates)."""
        if index < 0 or index >= len(self._propositions):
            return {"action": "error", "message": f"Invalid index: {index}"}
        deleted_text = self._propositions[index].text
        self._delete_at(index)
        return {"action": "deleted", "deleted": [deleted_text]}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save memory to a JSON file.

        Writes atomically — data goes to a temp file first, then is renamed
        to the target path so a crash mid-write never corrupts existing data.
        """
        data = {
            "version": 2,
            "config": {
                "model": self._model,
                "embedding_model": self._embedding_model,
                "entity_name": self._entity_name,
                "assistant_name": self._assistant_name,
                "max_propositions": self._max_propositions,
                "dedup_threshold": self._dedup_threshold,
            },
            "propositions": [
                {"text": p.text, "date": p.date, "session_id": p.session_id}
                for p in self._propositions
            ],
            "embeddings": self._embeddings,
            "session_counter": self._session_counter,
        }
        # Atomic write: temp file in same directory (same filesystem) → rename
        dir_path = os.path.dirname(os.path.abspath(path))
        fd, tmp_path = tempfile.mkstemp(dir=dir_path, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f)
            os.replace(tmp_path, path)
        except BaseException:
            # Clean up temp file on any failure
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    @classmethod
    def load(
        cls,
        path: str,
        api_key: str | None = None,
        model: str | None = None,
        db_path: str | None = None,
    ) -> Memory:
        """Load memory from a JSON file.

        Validates that proposition and embedding counts match, and that the
        saved embedding model is consistent with the loaded instance.

        If ``db_path`` is provided, writes all loaded data into a new SQLite
        database (JSON → SQLite migration).
        """
        with open(path) as f:
            data = json.load(f)

        config = data["config"]
        propositions = [Proposition(**p) for p in data["propositions"]]
        embeddings = data.get("embeddings", [])

        # Validate proposition/embedding count consistency
        if len(propositions) != len(embeddings):
            raise ValueError(
                f"Corrupt save file: {len(propositions)} propositions "
                f"but {len(embeddings)} embeddings"
            )

        # Don't pass db_path to __init__ — we'll populate it manually to
        # avoid loading an empty DB then overwriting with JSON data.
        mem = cls(
            api_key=api_key,
            model=model or config["model"],
            embedding_model=config["embedding_model"],
            entity_name=config.get("entity_name", "user"),
            assistant_name=config.get("assistant_name", "assistant"),
            max_propositions=config.get("max_propositions", 1000),
            dedup_threshold=config.get("dedup_threshold", 0.95),
        )
        mem._session_counter = data.get("session_counter", 0)
        mem._propositions = propositions
        mem._embeddings = embeddings

        # Migrate to SQLite if db_path given
        if db_path is not None:
            mem._db_path = db_path
            mem._init_db(db_path)
            for prop, emb in zip(propositions, embeddings):
                cur = mem._db.execute(
                    "INSERT INTO propositions (text, date, session_id, embedding) "
                    "VALUES (?, ?, ?, ?)",
                    (prop.text, prop.date, prop.session_id, _pack_embedding(emb)),
                )
                mem._db_ids.append(cur.lastrowid)
            mem._db.execute(
                "INSERT OR REPLACE INTO config (key, value) VALUES (?, ?)",
                ("session_counter", str(mem._session_counter)),
            )
            mem._db.commit()

        return mem

    # ------------------------------------------------------------------
    # SQLite persistence
    # ------------------------------------------------------------------

    def _init_db(self, db_path: str) -> None:
        """Open (or create) the SQLite database and ensure schema exists."""
        self._db = sqlite3.connect(db_path)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS propositions ("
            "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "  text TEXT NOT NULL,"
            "  date TEXT DEFAULT '',"
            "  session_id TEXT DEFAULT '',"
            "  embedding BLOB NOT NULL"
            ")"
        )
        # FTS5 content-sync table
        self._db.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS propositions_fts USING fts5("
            "  text, content='propositions', content_rowid='id'"
            ")"
        )
        # Auto-sync triggers
        self._db.execute(
            "CREATE TRIGGER IF NOT EXISTS propositions_ai AFTER INSERT ON propositions BEGIN"
            "  INSERT INTO propositions_fts(rowid, text) VALUES (new.id, new.text);"
            "END"
        )
        self._db.execute(
            "CREATE TRIGGER IF NOT EXISTS propositions_ad AFTER DELETE ON propositions BEGIN"
            "  INSERT INTO propositions_fts(propositions_fts, rowid, text)"
            "    VALUES ('delete', old.id, old.text);"
            "END"
        )
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS config ("
            "  key TEXT PRIMARY KEY, value TEXT NOT NULL"
            ")"
        )
        # Store embedding model for consistency checks
        self._db.execute(
            "INSERT OR REPLACE INTO config (key, value) VALUES (?, ?)",
            ("embedding_model", self._embedding_model),
        )
        self._db.commit()

    def _load_from_db(self) -> None:
        """Populate in-memory lists from SQLite on startup."""
        if self._db is None:
            return
        rows = self._db.execute(
            "SELECT id, text, date, session_id, embedding "
            "FROM propositions ORDER BY id"
        ).fetchall()
        for row_id, text, date, session_id, emb_blob in rows:
            self._propositions.append(
                Proposition(text=text, date=date, session_id=session_id)
            )
            self._embeddings.append(_unpack_embedding(emb_blob))
            self._db_ids.append(row_id)
        # Restore session counter from config if available
        row = self._db.execute(
            "SELECT value FROM config WHERE key = 'session_counter'"
        ).fetchone()
        if row:
            self._session_counter = int(row[0])

    def close(self) -> None:
        """Commit and close the SQLite connection (if open)."""
        if self._db is not None:
            try:
                self._db.commit()
                self._db.close()
            except sqlite3.ProgrammingError:
                pass  # already closed
            self._db = None

    def __del__(self) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _filter_messages(self, messages: list[dict]) -> list[dict]:
        """Filter to user + assistant visible text only."""
        turns = []
        for msg in messages:
            role = msg.get("role", "user")

            if role in ("tool", "system", "function"):
                continue

            if (
                role == "assistant"
                and msg.get("tool_calls")
                and not msg.get("content")
                and not msg.get("text")
            ):
                continue

            text = msg.get("text") or msg.get("content", "")
            if not text or not isinstance(text, str):
                continue

            if role == "assistant":
                text = _strip_thinking(text)
                if not text.strip():
                    continue

            speaker = self._entity_name if role == "user" else self._assistant_name
            turns.append({"speaker": speaker, "text": text})
        return turns

    def _extract_facts(
        self,
        turns: list[dict],
        date: str,
        session_id: str,
    ) -> list[Proposition]:
        """Extract atomic facts from a session using LLM."""
        turns_text = "\n".join(f"{t['speaker']}: {t['text']}" for t in turns)
        prompt = EXTRACT_PROMPT.format(date=date, turns_text=turns_text)

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                max_tokens=8192,
                temperature=0.1,
            )
            content = response.choices[0].message.content or ""

            items: list[dict] = []
            try:
                parsed = json.loads(content)
                if isinstance(parsed, dict):
                    items = parsed.get("facts", [])
                    if not items:
                        for v in parsed.values():
                            if isinstance(v, list):
                                items = v
                                break
                elif isinstance(parsed, list):
                    items = parsed
            except json.JSONDecodeError:
                items = _recover_partial_json(content)

            propositions = []
            for item in items:
                if isinstance(item, dict):
                    fact = item.get("fact", "")
                    if fact:
                        propositions.append(
                            Proposition(
                                text=fact,
                                date=date,
                                session_id=session_id,
                            )
                        )
            return propositions

        except Exception as err:
            logger.warning("Extraction error (%s): %s", session_id, err)
            return []

    def _is_duplicate(self, new_emb: list[float]) -> bool:
        """Check if a new embedding is a near-duplicate of any existing one."""
        for existing_emb in self._embeddings:
            if _cosine_similarity(new_emb, existing_emb) >= self._dedup_threshold:
                return True
        return False

    def _enforce_limit(self) -> None:
        """Drop oldest propositions if over max_propositions."""
        if len(self._propositions) > self._max_propositions:
            excess = len(self._propositions) - self._max_propositions
            if self._db is not None and self._db_ids:
                rowids_to_delete = self._db_ids[:excess]
                self._db.executemany(
                    "DELETE FROM propositions WHERE id = ?",
                    [(rid,) for rid in rowids_to_delete],
                )
                self._db.commit()
                self._db_ids = self._db_ids[excess:]
            self._propositions = self._propositions[excess:]
            self._embeddings = self._embeddings[excess:]

    def _delete_at(self, index: int) -> None:
        """Delete a proposition and its embedding by index."""
        if self._db is not None and self._db_ids:
            rowid = self._db_ids.pop(index)
            self._db.execute("DELETE FROM propositions WHERE id = ?", (rowid,))
            self._db.commit()
        self._propositions.pop(index)
        self._embeddings.pop(index)
