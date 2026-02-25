"""OpenClaw memory system — Python port for benchmarking.

Ported from the OpenClaw TypeScript source:
- Chunking: src/memory/internal.ts (chunkMarkdown)
- Hybrid search: src/memory/hybrid.ts (mergeHybridResults)
- Vector search: src/memory/manager-search.ts (searchVector)
- BM25 scoring: src/memory/hybrid.ts (bm25RankToScore)

Reference: https://github.com/openclaw/openclaw
Docs: https://docs.openclaw.ai/concepts/memory
"""
# OpenClaw is MIT licensed — https://github.com/openclaw/openclaw

from __future__ import annotations

import hashlib
import math
import os
import re
from dataclasses import dataclass, field

from openai import OpenAI

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class MemoryChunk:
    """A chunk of Markdown text with line range metadata."""

    text: str
    start_line: int
    end_line: int
    hash: str = field(default="")

    def __post_init__(self) -> None:
        if not self.hash:
            self.hash = hashlib.sha256(self.text.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Chunking — port of OpenClaw chunkMarkdown (internal.ts)
# ---------------------------------------------------------------------------


def chunk_markdown(
    content: str,
    tokens: int = 400,
    overlap: int = 80,
) -> list[MemoryChunk]:
    """Split Markdown content into overlapping chunks.

    OpenClaw estimates characters as tokens * 4. Chunks are split on newline
    boundaries and carry ``overlap`` tokens of trailing context into the next
    chunk to preserve continuity.
    """
    lines = content.split("\n")
    if not lines:
        return []

    max_chars = max(32, tokens * 4)
    overlap_chars = max(0, overlap * 4)

    chunks: list[MemoryChunk] = []
    current: list[tuple[str, int]] = []  # (line_text, 1-based line_no)
    current_chars = 0

    def flush() -> None:
        if not current:
            return
        text = "\n".join(line for line, _ in current)
        start_line = current[0][1]
        end_line = current[-1][1]
        chunks.append(MemoryChunk(text=text, start_line=start_line, end_line=end_line))

    def carry_overlap() -> tuple[list[tuple[str, int]], int]:
        """Keep the last ~overlap_chars of text for the next chunk."""
        if overlap_chars <= 0 or not current:
            return [], 0
        kept: list[tuple[str, int]] = []
        acc = 0
        for line, line_no in reversed(current):
            acc += len(line) + 1
            kept.insert(0, (line, line_no))
            if acc >= overlap_chars:
                break
        chars = sum(len(line) + 1 for line, _ in kept)
        return kept, chars

    for i, line in enumerate(lines):
        line_no = i + 1
        if line:
            segments = [
                line[start : start + max_chars]
                for start in range(0, len(line), max_chars)
            ]
        else:
            segments = [""]

        for segment in segments:
            line_size = len(segment) + 1
            if current_chars + line_size > max_chars and current:
                flush()
                current, current_chars = carry_overlap()

            current.append((segment, line_no))
            current_chars += line_size

    flush()
    return chunks


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------


def embed_texts(
    texts: list[str],
    model: str = "text-embedding-3-small",
    client: OpenAI | None = None,
) -> list[list[float]]:
    """Embed a list of texts using OpenAI embeddings API."""
    if not texts:
        return []

    if client is None:
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    embeddings: list[list[float]] = []
    batch_size = 512
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.embeddings.create(model=model, input=batch)
        for item in response.data:
            embeddings.append(item.embedding)

    return embeddings


# ---------------------------------------------------------------------------
# Cosine similarity — port of OpenClaw cosineSimilarity (internal.ts)
# ---------------------------------------------------------------------------


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if not a or not b:
        return 0.0
    length = min(len(a), len(b))
    dot = sum(a[i] * b[i] for i in range(length))
    norm_a = math.sqrt(sum(x * x for x in a[:length]))
    norm_b = math.sqrt(sum(x * x for x in b[:length]))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def bm25_search(
    query: str,
    chunks: list[MemoryChunk],
    limit: int,
) -> list[tuple[int, float]]:
    """BM25 keyword search over chunks using SQLite FTS5.

    - FTS5 query built from hybrid.ts buildFtsQuery (AND-joined quoted tokens)
    - bm25() ranking from SQLite FTS5 (negative scores, lower = better)
    - Score conversion from hybrid.ts bm25RankToScore: 1 / (1 + max(0, rank))
    """
    import sqlite3

    tokens = re.findall(r"[A-Za-z0-9_]+", query)
    if not tokens:
        return []
    fts_query = " AND ".join(f'"{t}"' for t in tokens)

    db = sqlite3.connect(":memory:")
    db.execute("CREATE VIRTUAL TABLE mem_fts USING fts5(chunk_id, text)")
    for i, chunk in enumerate(chunks):
        db.execute(
            "INSERT INTO mem_fts(chunk_id, text) VALUES (?, ?)", (str(i), chunk.text)
        )

    try:
        rows = db.execute(
            "SELECT chunk_id, bm25(mem_fts) AS rank "
            "FROM mem_fts WHERE mem_fts MATCH ? "
            "ORDER BY rank ASC LIMIT ?",
            (fts_query, limit),
        ).fetchall()
    except sqlite3.OperationalError:

        db.close()
        return []
    results: list[tuple[int, float]] = []
    for row in rows:
        idx = int(row[0])
        rank = float(row[1])
        text_score = 1.0 / (1.0 + max(0.0, rank))
        results.append((idx, text_score))

    db.close()
    return results


# ---------------------------------------------------------------------------
# Vector search
# ---------------------------------------------------------------------------


def vector_search(
    query_embedding: list[float],
    chunk_embeddings: list[list[float]],
    limit: int,
) -> list[tuple[int, float]]:
    """Vector similarity search over pre-computed chunk embeddings.

    Returns list of (chunk_index, cosine_score) sorted by score descending.
    """
    scored = [
        (i, cosine_similarity(query_embedding, emb))
        for i, emb in enumerate(chunk_embeddings)
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:limit]


# ---------------------------------------------------------------------------
# Hybrid search — port of OpenClaw mergeHybridResults (hybrid.ts)
# ---------------------------------------------------------------------------


def hybrid_search(
    query: str,
    chunks: list[MemoryChunk],
    chunk_embeddings: list[list[float]],
    query_embedding: list[float],
    top_k: int = 20,
    vector_weight: float = 0.7,
    text_weight: float = 0.3,
    candidate_multiplier: int = 4,
) -> list[tuple[MemoryChunk, float]]:
    """Hybrid BM25 + vector search, matching OpenClaw's mergeHybridResults.

    Retrieves ``top_k * candidate_multiplier`` candidates from each source,
    merges with weighted scores, and returns the top ``top_k`` results.
    """
    candidate_limit = top_k * candidate_multiplier

    vec_results = vector_search(query_embedding, chunk_embeddings, candidate_limit)
    bm25_results = bm25_search(query, chunks, candidate_limit)

    merged: dict[int, dict] = {}

    for idx, vec_score in vec_results:
        merged[idx] = {"vector_score": vec_score, "text_score": 0.0}

    for idx, txt_score in bm25_results:
        if idx in merged:
            merged[idx]["text_score"] = txt_score
        else:
            merged[idx] = {"vector_score": 0.0, "text_score": txt_score}

    results: list[tuple[int, float]] = []
    for idx, scores in merged.items():
        final = (
            vector_weight * scores["vector_score"] + text_weight * scores["text_score"]
        )
        results.append((idx, final))

    results.sort(key=lambda x: x[1], reverse=True)

    return [(chunks[idx], score) for idx, score in results[:top_k] if idx < len(chunks)]
