"""PropMem: proposition-based memory with entity-centric retrieval.

Extends OpenClaw's chunk-and-search with proposition extraction,
entity-filtered retrieval, and CoT answer generation.
See PROPMEM.md for design details.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from openai import OpenAI

from agents_memory.locomo import extract_dialogues, format_as_markdown
from agents_memory.openclaw import (
    MemoryChunk,
    chunk_markdown,
    cosine_similarity,
    embed_texts,
    hybrid_search,
    vector_search,
)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

EXTRACT_PROMPT = """\
Extract all facts from this conversation session. For each fact, specify \
which speaker it is about.

Session date: {date}
Speakers: {speaker_a} and {speaker_b}

Conversation:
{turns_text}

Extract EVERY fact mentioned including:
- Personal details, preferences, opinions, decisions
- Events, activities, plans (with temporal details)
- Relationships, feelings, reactions
- Names, places, objects, tools, systems mentioned
- Career, education, family, professional background
- Technical values, parameters, configurations, specifications
- Recommendations, procedures, troubleshooting steps

Return a JSON object:
{{"facts": [
  {{"entity": "{speaker_a}", "fact": "self-contained fact about {speaker_a}"}},
  {{"entity": "{speaker_b}", "fact": "self-contained fact about {speaker_b}"}}
]}}

Rules:
- Each fact must be a COMPLETE sentence starting with the entity name
- Include ALL facts, even minor ones
- Resolve pronouns to actual names
- Convert ALL relative dates to ABSOLUTE dates using the session date. \
Examples: "last week" in a session dated 2023-05-15 → "the week of May 8, 2023"; \
"three months ago" → "approximately February 2023"; "yesterday" → "May 14, 2023"; \
"next Tuesday" → "May 23, 2023". Always include the computed absolute date in the fact.
- Preserve exact technical values (numbers, codes, formulas, URLs) verbatim
- Be specific and factual, not interpretive
- One atomic fact per entry — do NOT combine multiple facts"""

ANSWER_PROMPT = """\
Answer the question using ONLY the evidence below.

{entity_section}

{chunks_section}

Question: {question}

Rules:
1. Reason step by step from the evidence
2. CRITICAL: The conversation context discusses MULTIPLE people. \
If the question asks about {entity_name}, ONLY use facts about {entity_name}. \
Do NOT attribute another person's experiences, activities, or opinions to \
{entity_name}.
3. ONLY answer if the specific fact is DIRECTLY stated in the evidence for \
the person asked about. Do NOT guess or construct answers from vague evidence.
4. For "Would..." questions, infer from {entity_name}'s stated facts only
5. Answer must be a DIRECT listing of facts — NO full sentences, NO "The user...", \
NO explanations. Use commas to separate multiple items. Copy exact words from \
the evidence.
6. For dates, resolve relative references to absolute dates using timestamps
7. Answer in the SAME LANGUAGE as the question
8. If the answer is not clearly about {entity_name} in the evidence, say "None"
9. If NO propositions about {entity_name} are found, answer "None" immediately \
without reasoning.

Return JSON: {{"reasoning": "brief thought process", \
"answer": "direct fact listing, comma-separated"}}"""

ANSWER_PROMPT_INFERENTIAL = """\
Answer this inference question using the evidence below. This question \
requires you to REASON from known facts to draw a logical conclusion.

{entity_section}

{chunks_section}

Question: {question}

Rules:
1. This is an INFERENCE question — you must reason from {entity_name}'s known \
facts, interests, personality, and circumstances
2. Think step by step:
   a) What relevant facts do we know about {entity_name}?
   b) What can we logically infer from these facts?
   c) What is the most reasonable conclusion?
3. ONLY use facts about {entity_name}, not about other people
4. ALWAYS provide an answer — NEVER say "None". These questions always have \
an answer that can be inferred from the evidence
5. For "Would..." questions: reason from {entity_name}'s stated preferences, \
personality, and life circumstances
6. Give a direct answer — no full sentences. Prefer: "Yes", "No", "Likely yes", \
"Likely no", or a brief factual phrase with key details.
7. Answer in the SAME LANGUAGE as the question

Return JSON: {{"reasoning": "step by step inference from known facts", \
"answer": "direct answer, no filler"}}"""

# --- Single-user variants (user-assistant conversations like LongMemEval) ---

ANSWER_PROMPT_SINGLE_USER = """\
Answer the question using ONLY the evidence below. The evidence comes from \
a conversation between a user and an AI assistant.

{entity_section}

{chunks_section}

Question: {question}

Rules:
1. Reason step by step from the evidence
2. The question is about the USER's information — preferences, experiences, \
facts shared during conversation. The assistant's responses provide context.
3. Answer if the fact is stated or clearly implied in the evidence
4. Answer concisely but completely. Copy exact words from the evidence \
where possible. Use commas to separate multiple items.
5. For dates, resolve relative references to absolute dates using timestamps
6. Answer in the SAME LANGUAGE as the question
7. If the answer cannot be found in the evidence, say "None"

Return JSON: {{"reasoning": "brief thought process", \
"answer": "concise but complete answer"}}"""

ANSWER_PROMPT_SINGLE_USER_INFERENTIAL = """\
Answer this inference question using the evidence below. The evidence comes \
from a conversation between a user and an AI assistant.

{entity_section}

{chunks_section}

Question: {question}

Rules:
1. This is an INFERENCE question — reason from the user's known facts, \
interests, personality, and circumstances
2. Think step by step:
   a) What relevant facts do we know about the user?
   b) What can we logically infer from these facts?
   c) What is the most reasonable conclusion?
3. ALWAYS provide an answer — NEVER say "None". These questions always have \
an answer that can be inferred from the evidence
4. For "Would..." questions: reason from the user's stated preferences, \
personality, and life circumstances
5. Give a direct answer. Prefer: "Yes", "No", "Likely yes", "Likely no", \
or a brief factual phrase with key details.
6. Answer in the SAME LANGUAGE as the question

Return JSON: {{"reasoning": "step by step inference from known facts", \
"answer": "direct answer, no filler"}}"""

# --- v4: unified question classifier ---

CLASSIFY_PROMPT = """\
Given this question and the known entities, classify it.

Entities: {entities}
Question: {question}

Return JSON:
{{"entity": "<exact name from list or null>", "is_inferential": true/false, \
"is_temporal": true/false}}

Rules:
- entity: Match pronouns/aliases to entity names. null if unclear.
- is_inferential: Requires reasoning, not direct fact recall \
(would/could/likely/prefer)
- is_temporal: Involves time ordering, recency, dates \
(before/after/when/first/last/latest)"""

# --- v4: unified answer prompt ---

ANSWER_PROMPT_V4 = """\
Answer the question using ONLY the evidence below.

{entity_section}

{chunks_section}

Question: {question}
{classification_info}

Rules:
1. Reason step by step from the evidence
2. {entity_constraint}
3. {answer_style}
4. Answer must be a DIRECT listing of facts — NO full sentences, NO "The user...", \
NO explanations. Use commas to separate multiple items. Copy exact words from evidence.
5. Resolve relative dates to absolute dates using timestamps
6. Answer in the SAME LANGUAGE as the question
{none_rule}

Return JSON: {{"reasoning": "...", "answer": "..."}}"""


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class Proposition:
    """An atomic fact about an entity, extracted from a conversation session."""

    text: str
    entity: str
    date: str
    session_id: str
    date_ordinal: int = 0  # days since epoch, for temporal boosting


def _parse_date_ordinal(date_str: str) -> int:
    """Parse a date string to an ordinal (days since epoch) for temporal ranking.

    Tries common formats, falls back to regex extraction. Returns 0 on failure.
    """
    if not date_str:
        return 0
    from datetime import datetime

    # Common formats from LoCoMo / LongMemEval
    for fmt in (
        "%B %d, %Y",       # "January 15, 2023"
        "%b %d, %Y",       # "Jan 15, 2023"
        "%Y-%m-%d",         # "2023-01-15"
        "%m/%d/%Y",         # "01/15/2023"
        "%d %B %Y",         # "15 January 2023"
        "%B %Y",            # "January 2023"
    ):
        try:
            return datetime.strptime(date_str.strip(), fmt).toordinal()
        except ValueError:
            continue

    # Regex fallback: extract year-month-day components
    m = re.search(r"(\d{4})[/-](\d{1,2})[/-](\d{1,2})", date_str)
    if m:
        try:
            return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3))).toordinal()
        except ValueError:
            pass

    # Year-month only
    m = re.search(r"(\d{4})[/-](\d{1,2})", date_str)
    if m:
        try:
            return datetime(int(m.group(1)), int(m.group(2)), 1).toordinal()
        except ValueError:
            pass

    return 0


# ---------------------------------------------------------------------------
# PropMemSystem
# ---------------------------------------------------------------------------


@dataclass
class PropMemSystem:
    """PropMem: proposition-based memory with entity-centric retrieval.

    Ingestion:
      1. Chunk + embed raw markdown (OpenClaw) — fallback retrieval
      2. Extract atomic propositions per entity per session (LLM)
      3. Embed all propositions

    Per question:
      1. Identify target entity (string matching)
      2. Retrieve entity-filtered propositions (vector + BM25)
      3. Retrieve raw chunks (broader context)
      4. CoT answer with structured evidence
    """

    embedding_model: str = "text-embedding-3-small"
    top_k_props: int = 30
    top_k_chunks: int = 3

    # Ablation flags (all True = v3 behaviour)
    use_propositions: bool = True
    use_chunks: bool = True
    use_entity_filter: bool = True
    use_clustering: bool = True
    use_bm25: bool = True

    # v4 feature flags (all False = v3 behaviour)
    use_llm_classifier: bool = False
    use_temporal_boost: bool = False
    use_knowledge_updates: bool = False

    # Storage — populated during ingestion
    propositions: list[Proposition] = field(default_factory=list)
    proposition_embeddings: list[list[float]] = field(default_factory=list)
    chunks: list[MemoryChunk] = field(default_factory=list)
    chunk_embeddings: list[list[float]] = field(default_factory=list)
    entity_names: list[str] = field(default_factory=list)
    _embedding_cache: dict[str, list[float]] = field(default_factory=dict)

    # Cluster data for topic-based retrieval (used when entity=None)
    _cluster_centroids: Any = field(default=None, init=False, repr=False)
    _cluster_to_indices: dict[int, list[int]] = field(
        default_factory=dict, init=False
    )

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest_conversation(
        self,
        conv: dict,
        llm_client: OpenAI | None = None,
        llm_model: str | None = None,
    ) -> dict:
        """Ingest a LoCoMo conversation with proposition extraction."""
        if llm_client is None:
            llm_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        if not llm_model:
            llm_model = os.environ.get("LLM_MODEL", "gpt-4.1")

        conversation = conv["conversation"]
        speaker_a = conversation.get("speaker_a", "User A")
        speaker_b = conversation.get("speaker_b", "User B")
        self.entity_names = [speaker_a, speaker_b]

        # 1. Raw chunks (OpenClaw) for fallback
        dialogues = extract_dialogues(conv)
        markdown = format_as_markdown(dialogues)
        if self.use_chunks:
            self.chunks = chunk_markdown(markdown, tokens=400, overlap=80)
            if self.chunks:
                self.chunk_embeddings = embed_texts(
                    [c.text for c in self.chunks], model=self.embedding_model
                )

        # 2. Extract propositions per session
        if self.use_propositions:
            session_keys = sorted(
                [
                    k
                    for k in conversation.keys()
                    if k.startswith("session_") and not k.endswith("_date_time")
                ],
                key=lambda x: int(x.split("_")[1]),
            )

            # Build extraction tasks
            extraction_tasks = []
            for sk in session_keys:
                session_num = sk.split("_")[1]
                date_key = f"session_{session_num}_date_time"
                date = conversation.get(date_key, "")
                turns = conversation[sk]

                if not isinstance(turns, list) or not turns:
                    continue

                extraction_tasks.append((turns, date, sk, speaker_a, speaker_b))

            # Parallel proposition extraction (10 concurrent LLM calls)
            if extraction_tasks:
                with ThreadPoolExecutor(max_workers=10) as pool:
                    futures = {
                        pool.submit(
                            self._extract_propositions,
                            turns, date, sk, sa, sb, llm_client, llm_model,
                        ): sk
                        for turns, date, sk, sa, sb in extraction_tasks
                    }
                    for future in as_completed(futures):
                        try:
                            props = future.result()
                            self.propositions.extend(props)
                        except Exception as err:
                            print(f"  Extraction error ({futures[future]}): {err}")

            # 3. Embed propositions
            if self.propositions:
                self.proposition_embeddings = embed_texts(
                    [p.text for p in self.propositions], model=self.embedding_model
                )

                # Parse date ordinals for temporal boosting
                if self.use_temporal_boost:
                    for p in self.propositions:
                        p.date_ordinal = _parse_date_ordinal(p.date)

        # 4. Build clusters for topic-based retrieval (fallback when entity=None)
        n_clusters = 0
        if self.use_clustering and self.proposition_embeddings:
            n_clusters = self._cluster_propositions()

        return {
            "num_turns": len(dialogues),
            "num_chunks": len(self.chunks),
            "num_propositions": len(self.propositions),
            "num_clusters": n_clusters,
            "entities": self.entity_names,
        }

    @staticmethod
    def _recover_partial_json(content: str) -> list[dict]:
        """Extract complete {"entity": ..., "fact": ...} objects from truncated JSON."""
        pattern = r'\{\s*"entity"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"\s*,\s*"fact"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"\s*\}'
        items = []
        for m in re.finditer(pattern, content):
            entity = m.group(1).replace('\\"', '"').replace("\\n", " ")
            fact = m.group(2).replace('\\"', '"').replace("\\n", " ")
            if entity and fact:
                items.append({"entity": entity, "fact": fact})
        return items

    def _extract_propositions(
        self,
        turns: list[dict],
        date: str,
        session_id: str,
        speaker_a: str,
        speaker_b: str,
        client: OpenAI,
        model: str,
    ) -> list[Proposition]:
        """Extract atomic propositions from a session using LLM."""
        turns_text = "\n".join(
            f"{t.get('speaker', '?')}: {t.get('text', '')}" for t in turns
        )

        prompt = EXTRACT_PROMPT.format(
            date=date,
            speaker_a=speaker_a,
            speaker_b=speaker_b,
            turns_text=turns_text,
        )

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                max_tokens=8192,
                temperature=0.1,
            )
            content = response.choices[0].message.content or ""
            finish_reason = response.choices[0].finish_reason

            # Try normal JSON parse first
            items = []
            try:
                parsed = json.loads(content)
                if isinstance(parsed, dict):
                    items = parsed.get("facts", parsed.get("propositions", []))
                    if not items:
                        for v in parsed.values():
                            if isinstance(v, list):
                                items = v
                                break
                elif isinstance(parsed, list):
                    items = parsed
            except json.JSONDecodeError:
                # Truncated JSON (finish_reason="length") — recover what we can
                items = self._recover_partial_json(content)
                if items:
                    print(
                        f"  Partial recovery ({session_id}): {len(items)} facts from truncated JSON"
                    )

            propositions = []
            for item in items:
                if isinstance(item, dict):
                    entity = item.get("entity", "")
                    fact = item.get("fact", "")
                    if entity and fact:
                        propositions.append(
                            Proposition(
                                text=fact,
                                entity=entity,
                                date=date,
                                session_id=session_id,
                            )
                        )

            if finish_reason == "length" and propositions:
                print(
                    f"  Truncated output ({session_id}): recovered {len(propositions)} facts"
                )

            return propositions

        except Exception as err:
            print(f"  Proposition extraction error ({session_id}): {err}")
            return []

    # ------------------------------------------------------------------
    # Question type detection
    # ------------------------------------------------------------------

    @staticmethod
    def _is_inferential(question: str) -> bool:
        """Detect inferential questions that require reasoning from evidence."""
        q_lower = question.lower().strip()
        # "Would/Could/Should X..." patterns
        if q_lower.startswith(("would ", "could ", "should ")):
            return True
        # "What would/might/could..." patterns
        if re.match(r"^what\s+(would|might|could)\b", q_lower):
            return True
        # "Do you think...", "How would...", "Is it likely...", "Is X the type..."
        if re.match(
            r"^(do you think|how would|is it likely|is \w+ the type)\b", q_lower
        ):
            return True
        # "...likely..." patterns
        if " likely " in q_lower:
            return True
        # "...might..." in question context
        if " might " in q_lower:
            return True
        # "...prefer..." in question form
        if " prefer " in q_lower and "?" in question:
            return True
        return False

    # ------------------------------------------------------------------
    # Entity identification
    # ------------------------------------------------------------------

    def _identify_entity(self, question: str) -> str | None:
        """Identify which entity the question is about via string matching."""
        q_lower = question.lower()

        # Check full names
        for name in self.entity_names:
            if name.lower() in q_lower:
                return name

        # Check first names and word-boundary partial matches (min 4 chars)
        for name in self.entity_names:
            for part in name.split():
                if len(part) >= 4:
                    pattern = r"\b" + re.escape(part.lower()) + r"\b"
                    if re.search(pattern, q_lower):
                        return name

        return None

    # ------------------------------------------------------------------
    # Proposition retrieval (vector + BM25, entity-filtered)
    # ------------------------------------------------------------------

    def _embed_query(self, text: str) -> list[float]:
        if text in self._embedding_cache:
            return self._embedding_cache[text]
        emb = embed_texts([text], model=self.embedding_model)[0]
        self._embedding_cache[text] = emb
        return emb

    # ------------------------------------------------------------------
    # Embedding clusters (topic-based retrieval for entity=None)
    # ------------------------------------------------------------------

    def _cluster_propositions(self) -> int:
        """K-means cluster proposition embeddings for topic-based retrieval.

        Used when entity matching fails (single-user conversations).
        Clusters serve as implicit topics — no extra LLM calls or embeddings.

        Returns number of clusters created.
        """
        n = len(self.proposition_embeddings)
        if n < 10:
            return 0

        embeddings = np.array(self.proposition_embeddings, dtype=np.float32)

        # Number of clusters: sqrt(n), capped at [5, 40]
        k = min(max(5, int(n**0.5)), 40)

        # K-means++ initialization (deterministic seed)
        rng = np.random.default_rng(42)
        centroid_indices = [rng.integers(n)]
        for _ in range(1, k):
            dists = np.min(
                np.stack(
                    [
                        np.sum((embeddings - embeddings[ci]) ** 2, axis=1)
                        for ci in centroid_indices
                    ]
                ),
                axis=0,
            )
            probs = dists / (dists.sum() + 1e-10)
            centroid_indices.append(int(rng.choice(n, p=probs)))
        centroids = embeddings[centroid_indices].copy()

        # K-means iterations
        for _ in range(30):
            # Squared distances: ||X - C||^2 via expansion
            # dists[i,j] = ||C[i]||^2 - 2*C[i]·X[j] + ||X[j]||^2
            c_sq = np.sum(centroids**2, axis=1, keepdims=True)
            x_sq = np.sum(embeddings**2, axis=1, keepdims=True).T
            dists = c_sq - 2.0 * (centroids @ embeddings.T) + x_sq
            assignments = np.argmin(dists, axis=0)

            new_centroids = np.zeros_like(centroids)
            for i in range(k):
                mask = assignments == i
                if np.any(mask):
                    new_centroids[i] = embeddings[mask].mean(axis=0)
                else:
                    new_centroids[i] = centroids[i]

            if np.allclose(centroids, new_centroids, atol=1e-6):
                break
            centroids = new_centroids

        self._cluster_centroids = centroids
        self._cluster_to_indices = {}
        for i, c in enumerate(assignments.tolist()):
            self._cluster_to_indices.setdefault(c, []).append(i)

        return k

    def _get_cluster_indices(
        self, query_embedding: list[float], top_clusters: int = 5
    ) -> list[int]:
        """Return proposition indices from the most relevant clusters.

        Cosine similarity between query and cluster centroids determines
        which clusters are relevant — no LLM call needed.
        """
        if self._cluster_centroids is None:
            return list(range(len(self.propositions)))

        query = np.array(query_embedding, dtype=np.float32)
        centroids = self._cluster_centroids

        # Cosine similarity against centroids
        c_norms = np.linalg.norm(centroids, axis=1)
        q_norm = np.linalg.norm(query)
        sims = (centroids @ query) / (c_norms * q_norm + 1e-10)

        top_k = min(top_clusters, len(centroids))
        top_ids = np.argsort(sims)[-top_k:][::-1]

        indices = []
        for cid in top_ids:
            indices.extend(self._cluster_to_indices.get(int(cid), []))
        return indices

    # ------------------------------------------------------------------
    # Proposition retrieval (vector + BM25, entity/cluster-filtered)
    # ------------------------------------------------------------------

    def _retrieve_propositions(
        self,
        question: str,
        entity: str | None = None,
        top_k: int = 30,
        is_temporal: bool = False,
    ) -> list[tuple[Proposition, float]]:
        """Hybrid vector + BM25 search over propositions, optionally entity-filtered."""
        if not self.propositions or not self.proposition_embeddings:
            return []

        # Determine which propositions to search
        if entity and self.use_entity_filter:
            indices = [
                i
                for i, p in enumerate(self.propositions)
                if p.entity.lower() == entity.lower()
            ]
            # Use entity-specific results even if few — don't pollute with other entities
        else:
            # For large proposition sets, cluster filtering narrows to relevant
            # topics. For small sets (< 500), direct vector+BM25 search is
            # effective enough and cluster pre-filtering risks excluding
            # relevant propositions.
            n_props = len(self.propositions)
            if (
                self.use_clustering
                and n_props >= 500
                and self._cluster_centroids is not None
            ):
                query_emb = self._embed_query(question)
                indices = self._get_cluster_indices(query_emb, top_clusters=5)
            else:
                indices = list(range(n_props))

        # Vector search
        query_emb = self._embed_query(question)
        vec_scores: dict[int, float] = {}
        for idx in indices:
            score = cosine_similarity(query_emb, self.proposition_embeddings[idx])
            vec_scores[idx] = score

        # Merge with BM25 if enabled
        if self.use_bm25:
            bm25_scores = self._bm25_propositions(question, indices)
            # 0.8 vector, 0.2 BM25 — propositions are ~25 words,
            # too short for BM25 to be highly discriminative
            merged: dict[int, float] = {}
            for idx in indices:
                vs = vec_scores.get(idx, 0.0)
                bs = bm25_scores.get(idx, 0.0)
                merged[idx] = 0.8 * vs + 0.2 * bs
        else:
            merged = vec_scores

        # Temporal boosting: blend relevance with recency for time-sensitive queries
        if self.use_temporal_boost and is_temporal:
            max_ord = max(
                (self.propositions[i].date_ordinal for i in indices), default=0
            )
            if max_ord > 0:
                for idx in merged:
                    recency = self.propositions[idx].date_ordinal / max_ord
                    merged[idx] = 0.85 * merged[idx] + 0.15 * recency

        # Sort, deduplicate by normalized text, and return top-k
        ranked = sorted(merged.items(), key=lambda x: x[1], reverse=True)
        seen_texts: set[str] = set()
        deduped: list[tuple[Proposition, float]] = []
        for idx, score in ranked:
            norm = re.sub(r"[^\w\s]", "", self.propositions[idx].text.strip().lower())
            if norm not in seen_texts:
                seen_texts.add(norm)
                deduped.append((self.propositions[idx], score))
            if len(deduped) >= top_k:
                break
        return deduped

    def _bm25_propositions(self, query: str, indices: list[int]) -> dict[int, float]:
        """BM25 search over a subset of propositions."""
        tokens = re.findall(r"[A-Za-z0-9_]+", query)
        if not tokens:
            return {}

        fts_query_and = " AND ".join(f'"{t}"' for t in tokens)
        fts_query_or = " OR ".join(f'"{t}"' for t in tokens)

        db = sqlite3.connect(":memory:")
        db.execute("CREATE VIRTUAL TABLE prop_fts USING fts5(prop_id, text)")
        for idx in indices:
            db.execute(
                "INSERT INTO prop_fts(prop_id, text) VALUES (?, ?)",
                (str(idx), self.propositions[idx].text),
            )

        try:
            # Try AND first for precise matches
            rows = db.execute(
                "SELECT prop_id, bm25(prop_fts) AS rank "
                "FROM prop_fts WHERE prop_fts MATCH ? "
                "ORDER BY rank ASC LIMIT ?",
                (fts_query_and, len(indices)),
            ).fetchall()
            # Fall back to OR if AND returns nothing
            if not rows:
                rows = db.execute(
                    "SELECT prop_id, bm25(prop_fts) AS rank "
                    "FROM prop_fts WHERE prop_fts MATCH ? "
                    "ORDER BY rank ASC LIMIT ?",
                    (fts_query_or, len(indices)),
                ).fetchall()
        except sqlite3.OperationalError:
            db.close()
            return {}

        scores: dict[int, float] = {}
        for row in rows:
            idx = int(row[0])
            rank = float(row[1])
            scores[idx] = 1.0 / (1.0 + max(0.0, rank))

        db.close()
        return scores

    # ------------------------------------------------------------------
    # Chunk retrieval (OpenClaw hybrid search, fallback)
    # ------------------------------------------------------------------

    def _retrieve_chunks(
        self, question: str, top_k: int = 10, entity: str | None = None
    ) -> list[tuple[MemoryChunk, float]]:
        """Hybrid BM25+vector or vector-only search over raw chunks."""
        if not self.chunks or not self.chunk_embeddings:
            return []
        query_emb = self._embed_query(question)
        if self.use_bm25:
            results = hybrid_search(
                query=question,
                chunks=self.chunks,
                chunk_embeddings=self.chunk_embeddings,
                query_embedding=query_emb,
                top_k=top_k,
                vector_weight=0.7,
                text_weight=0.3,
            )
        else:
            vec_results = vector_search(
                query_emb, self.chunk_embeddings, top_k
            )
            results = [
                (self.chunks[idx], score) for idx, score in vec_results
            ]
        if entity:
            filtered = [(c, s) for c, s in results if entity.lower() in c.text.lower()]
            if filtered:
                return filtered[:top_k]
        return results[:top_k]

    # ------------------------------------------------------------------
    # Answer generation
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_answer(
        question: str,
        propositions: list[tuple[Proposition, float]],
        chunks: list[tuple[MemoryChunk, float]],
        entity: str | None,
        client: OpenAI,
        model: str,
        is_inferential: bool = False,
        single_user: bool = False,
    ) -> str:
        """Generate answer using CoT over propositions + chunks."""
        # Build entity section
        if entity and propositions:
            prop_lines = []
            for p, _score in propositions:
                prop_lines.append(f"- [{p.date}] {p.text}")
            entity_section = f"Known facts about {entity}:\n" + "\n".join(prop_lines)
        elif propositions:
            prop_lines = []
            for p, _score in propositions:
                prop_lines.append(f"- [{p.date}] {p.text}")
            entity_section = "Known facts from conversation:\n" + "\n".join(prop_lines)
        else:
            entity_section = "Known facts: (none extracted)"

        # Build chunk section
        if chunks:
            chunk_text = "\n---\n".join(c.text for c, _s in chunks)
            chunks_section = f"Additional conversation context:\n{chunk_text}"
        else:
            chunks_section = ""

        # Select prompt based on conversation type
        if single_user:
            prompt_template = (
                ANSWER_PROMPT_SINGLE_USER_INFERENTIAL
                if is_inferential
                else ANSWER_PROMPT_SINGLE_USER
            )
            prompt = prompt_template.format(
                entity_section=entity_section,
                chunks_section=chunks_section,
                question=question,
            )
        else:
            entity_name = entity if entity else "the person asked about"
            prompt_template = (
                ANSWER_PROMPT_INFERENTIAL if is_inferential else ANSWER_PROMPT
            )
            prompt = prompt_template.format(
                entity_section=entity_section,
                chunks_section=chunks_section,
                question=question,
                entity_name=entity_name,
            )

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                max_tokens=500,
                temperature=0.1,
            )
            content = response.choices[0].message.content or ""
            result = json.loads(content)
            return result.get("answer", "").strip()
        except (json.JSONDecodeError, Exception) as err:
            print(f"  Answer error: {err}")
            return ""

    # ------------------------------------------------------------------
    # v4: unified question classifier
    # ------------------------------------------------------------------

    def _classify_question(
        self,
        question: str,
        client: OpenAI,
        model: str,
    ) -> dict:
        """Classify question via LLM: entity, is_inferential, is_temporal.

        Falls back to v3 heuristics on error.
        """
        entities_str = ", ".join(self.entity_names) if self.entity_names else "unknown"
        prompt = CLASSIFY_PROMPT.format(entities=entities_str, question=question)

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                max_tokens=80,
                temperature=0,
            )
            content = response.choices[0].message.content or ""
            result = json.loads(content)

            # Validate entity against known names
            raw_entity = result.get("entity")
            entity = None
            if raw_entity and isinstance(raw_entity, str):
                for name in self.entity_names:
                    if name.lower() == raw_entity.lower():
                        entity = name
                        break

            return {
                "entity": entity,
                "is_inferential": bool(result.get("is_inferential", False)),
                "is_temporal": bool(result.get("is_temporal", False)),
            }
        except Exception as err:
            print(f"  Classifier error (falling back to heuristics): {err}")
            return {
                "entity": self._identify_entity(question),
                "is_inferential": self._is_inferential(question),
                "is_temporal": False,
            }

    # ------------------------------------------------------------------
    # v4: knowledge update handling (contradiction detection)
    # ------------------------------------------------------------------

    def _apply_knowledge_updates(
        self,
        props: list[tuple[Proposition, float]],
    ) -> list[tuple[Proposition, float]]:
        """Penalize older propositions contradicted by newer same-entity props.

        For same-entity propositions with cosine similarity >0.85 (same topic),
        the older one gets a 30% score penalty. O(n^2) only within top-k
        retrieved props (30 items max = 450 dot products).
        """
        if len(props) < 2:
            return props

        # Build local embeddings for the retrieved props
        texts = [p.text for p, _ in props]
        embs = embed_texts(texts, model=self.embedding_model)

        penalties: dict[int, float] = {}
        for i in range(len(props)):
            for j in range(i + 1, len(props)):
                pi, si = props[i]
                pj, sj = props[j]
                if pi.entity.lower() != pj.entity.lower():
                    continue
                sim = cosine_similarity(embs[i], embs[j])
                if sim > 0.85:
                    # Penalize the older one
                    if pi.date_ordinal < pj.date_ordinal:
                        penalties[i] = penalties.get(i, 0) + 0.30
                    elif pj.date_ordinal < pi.date_ordinal:
                        penalties[j] = penalties.get(j, 0) + 0.30

        if not penalties:
            return props

        result = []
        for idx, (p, score) in enumerate(props):
            penalty = min(penalties.get(idx, 0), 0.60)  # cap total penalty
            result.append((p, score * (1.0 - penalty)))
        result.sort(key=lambda x: x[1], reverse=True)
        return result

    # ------------------------------------------------------------------
    # v4: unified answer generation
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_answer_v4(
        question: str,
        propositions: list[tuple[Proposition, float]],
        chunks: list[tuple[MemoryChunk, float]],
        entity: str | None,
        client: OpenAI,
        model: str,
        is_inferential: bool = False,
        is_temporal: bool = False,
        single_user: bool = False,
    ) -> str:
        """Generate answer using the unified v4 prompt template."""
        # Build entity section
        if entity and propositions:
            prop_lines = [f"- [{p.date}] {p.text}" for p, _ in propositions]
            entity_section = f"Known facts about {entity}:\n" + "\n".join(prop_lines)
        elif propositions:
            prop_lines = [f"- [{p.date}] {p.text}" for p, _ in propositions]
            entity_section = "Known facts from conversation:\n" + "\n".join(prop_lines)
        else:
            entity_section = "Known facts: (none extracted)"

        # Build chunk section
        if chunks:
            chunk_text = "\n---\n".join(c.text for c, _ in chunks)
            chunks_section = f"Additional conversation context:\n{chunk_text}"
        else:
            chunks_section = ""

        # Dynamic sections based on classification
        if entity and not single_user:
            entity_constraint = (
                f"ONLY use facts about {entity}. Do NOT attribute another "
                f"person's experiences to {entity}."
            )
        elif single_user:
            entity_constraint = (
                "The question is about the USER's information. "
                "The assistant's responses provide context."
            )
        else:
            entity_constraint = "Use facts about the person asked about."

        if is_inferential:
            answer_style = (
                "This is an INFERENCE question — reason from known facts, "
                "interests, personality, and circumstances. "
                "For 'Would...' questions: reason from stated preferences. "
                "Give a direct answer: 'Yes', 'No', 'Likely yes', 'Likely no', "
                "or a brief factual phrase."
            )
            none_rule = (
                "7. ALWAYS provide an answer — NEVER say 'None'. "
                "These questions always have an answer that can be inferred."
            )
        else:
            answer_style = (
                "ONLY answer if the specific fact is DIRECTLY stated in evidence. "
                "Do NOT guess or construct answers from vague evidence."
            )
            none_rule = (
                "7. If the answer is not clearly stated in the evidence, "
                "say 'None'"
            )

        classification_info = ""
        if is_temporal:
            classification_info = (
                "Note: This is a TEMPORAL question. "
                "Pay attention to dates and time ordering."
            )

        prompt = ANSWER_PROMPT_V4.format(
            entity_section=entity_section,
            chunks_section=chunks_section,
            question=question,
            classification_info=classification_info,
            entity_constraint=entity_constraint,
            answer_style=answer_style,
            none_rule=none_rule,
        )

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                max_tokens=500,
                temperature=0.1,
            )
            content = response.choices[0].message.content or ""
            result = json.loads(content)
            return result.get("answer", "").strip()
        except (json.JSONDecodeError, Exception) as err:
            print(f"  Answer v4 error: {err}")
            return ""

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def answer_question(
        self,
        question: str,
        llm_client: OpenAI | None = None,
        llm_model: str | None = None,
    ) -> dict[str, Any]:
        """Answer with entity-centric proposition retrieval + CoT.

        Pipeline:
          1. Identify target entity (string matching or LLM classifier)
          2. Retrieve entity-filtered propositions
          3. Retrieve raw chunks (broader context)
          4. CoT answer with structured evidence
        """
        if not question.strip():
            return {
                "answer": "None",
                "entity": None,
                "num_propositions": 0,
                "num_chunks": 0,
            }

        if llm_client is None:
            llm_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        if not llm_model:
            llm_model = os.environ.get("LLM_MODEL", "gpt-4.1")

        # Step 1: Identify entity, question type, and conversation type
        _generic = {"user", "assistant", "system", "bot", "ai"}
        single_user = all(n.lower() in _generic for n in self.entity_names)

        if self.use_llm_classifier:
            classification = self._classify_question(
                question, llm_client, llm_model
            )
            entity = classification["entity"]
            inferential = classification["is_inferential"]
            is_temporal = classification["is_temporal"]
        else:
            entity = self._identify_entity(question)
            inferential = self._is_inferential(question)
            is_temporal = False

        # Step 2: Retrieve entity-filtered propositions
        # Increase top_k when chunks are disabled to compensate
        prop_k = self.top_k_props
        if not self.use_chunks:
            prop_k = int(prop_k * 1.5)

        props = []
        if self.use_propositions:
            props = self._retrieve_propositions(
                question, entity=entity, top_k=prop_k, is_temporal=is_temporal
            )

        # Knowledge update handling: penalize older contradicted propositions
        if self.use_knowledge_updates and props and entity:
            props = self._apply_knowledge_updates(props)

        # Step 3: Retrieve raw chunks (more chunks for single-user since
        # entity filtering doesn't apply and chunks provide broader context)
        chunk_k = self.top_k_chunks
        if single_user:
            chunk_k *= 2
        if not self.use_propositions:
            chunk_k = max(chunk_k, 10)

        chunks = []
        if self.use_chunks:
            chunks = self._retrieve_chunks(question, top_k=chunk_k, entity=entity)

        # Step 4: Generate answer (route to appropriate prompt)
        if self.use_llm_classifier:
            answer = self._generate_answer_v4(
                question,
                props,
                chunks,
                entity,
                llm_client,
                llm_model,
                is_inferential=inferential,
                is_temporal=is_temporal,
                single_user=single_user,
            )
        else:
            answer = self._generate_answer(
                question,
                props,
                chunks,
                entity,
                llm_client,
                llm_model,
                is_inferential=inferential,
                single_user=single_user,
            )

        # Clean common prefixes
        for prefix in ("Answer:", "A:", "answer:"):
            if answer.startswith(prefix):
                answer = answer[len(prefix) :].strip()

        return {
            "answer": answer,
            "entity": entity,
            "num_propositions": len(props),
            "num_chunks": len(chunks),
            "is_inferential": inferential,
        }
