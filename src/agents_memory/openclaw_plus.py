"""OpenClawPlus: proposition-based memory with entity-centric retrieval.

Extends OpenClaw's chunk-and-search with proposition extraction,
entity-filtered retrieval, and CoT answer generation.
See OPENCLAW_PLUS.md for design details.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI

from agents_memory.openclaw import (
    MemoryChunk,
    chunk_markdown,
    cosine_similarity,
    embed_texts,
    hybrid_search,
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
- Include temporal context ("last week", "in 2016", "recently", etc.)
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


# ---------------------------------------------------------------------------
# OpenClawPlusSystem
# ---------------------------------------------------------------------------


@dataclass
class OpenClawPlusSystem:
    """OpenClawPlus: proposition-based memory with entity-centric retrieval.

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

    # Storage — populated during ingestion
    propositions: list[Proposition] = field(default_factory=list)
    proposition_embeddings: list[list[float]] = field(default_factory=list)
    chunks: list[MemoryChunk] = field(default_factory=list)
    chunk_embeddings: list[list[float]] = field(default_factory=list)
    entity_names: list[str] = field(default_factory=list)
    _embedding_cache: dict[str, list[float]] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    @staticmethod
    def _format_as_markdown(dialogues: list[dict]) -> str:
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
        from agents_memory.locomo import extract_dialogues

        dialogues = extract_dialogues(conv)
        markdown = self._format_as_markdown(dialogues)
        self.chunks = chunk_markdown(markdown, tokens=400, overlap=80)
        if self.chunks:
            self.chunk_embeddings = embed_texts(
                [c.text for c in self.chunks], model=self.embedding_model
            )

        # 2. Extract propositions per session
        session_keys = sorted(
            [
                k
                for k in conversation.keys()
                if k.startswith("session_") and not k.endswith("_date_time")
            ],
            key=lambda x: int(x.split("_")[1]),
        )

        for sk in session_keys:
            session_num = sk.split("_")[1]
            date_key = f"session_{session_num}_date_time"
            date = conversation.get(date_key, "")
            turns = conversation[sk]

            if not isinstance(turns, list) or not turns:
                continue

            props = self._extract_propositions(
                turns, date, sk, speaker_a, speaker_b, llm_client, llm_model
            )
            self.propositions.extend(props)

        # 3. Embed propositions
        if self.propositions:
            self.proposition_embeddings = embed_texts(
                [p.text for p in self.propositions], model=self.embedding_model
            )

        return {
            "num_turns": len(dialogues),
            "num_chunks": len(self.chunks),
            "num_propositions": len(self.propositions),
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

    def _retrieve_propositions(
        self,
        question: str,
        entity: str | None = None,
        top_k: int = 30,
    ) -> list[tuple[Proposition, float]]:
        """Hybrid vector + BM25 search over propositions, optionally entity-filtered."""
        if not self.propositions or not self.proposition_embeddings:
            return []

        # Determine which propositions to search
        if entity:
            indices = [
                i
                for i, p in enumerate(self.propositions)
                if p.entity.lower() == entity.lower()
            ]
            # Use entity-specific results even if few — don't pollute with other entities
        else:
            indices = list(range(len(self.propositions)))

        # Vector search
        query_emb = self._embed_query(question)
        vec_scores: dict[int, float] = {}
        for idx in indices:
            score = cosine_similarity(query_emb, self.proposition_embeddings[idx])
            vec_scores[idx] = score

        # BM25 search over propositions
        bm25_scores = self._bm25_propositions(question, indices)

        # Merge scores (0.8 vector, 0.2 BM25 — propositions are ~25 words,
        # too short for BM25 to be highly discriminative)
        merged: dict[int, float] = {}
        for idx in indices:
            vs = vec_scores.get(idx, 0.0)
            bs = bm25_scores.get(idx, 0.0)
            merged[idx] = 0.8 * vs + 0.2 * bs

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
        """Hybrid BM25+vector search over raw chunks."""
        if not self.chunks or not self.chunk_embeddings:
            return []
        query_emb = self._embed_query(question)
        results = hybrid_search(
            query=question,
            chunks=self.chunks,
            chunk_embeddings=self.chunk_embeddings,
            query_embedding=query_emb,
            top_k=top_k,
            vector_weight=0.7,
            text_weight=0.3,
        )
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
                prop_lines.append(f"- [{p.entity}] {p.text}")
            entity_section = "Known facts:\n" + "\n".join(prop_lines)
        else:
            entity_section = "Known facts: (none extracted)"

        # Build chunk section
        if chunks:
            chunk_text = "\n---\n".join(c.text for c, _s in chunks)
            chunks_section = f"Additional conversation context:\n{chunk_text}"
        else:
            chunks_section = ""

        entity_name = entity if entity else "the person asked about"

        prompt_template = ANSWER_PROMPT_INFERENTIAL if is_inferential else ANSWER_PROMPT
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
          1. Identify target entity (string matching)
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

        # Step 1: Identify entity
        entity = self._identify_entity(question)

        # Step 2: Retrieve entity-filtered propositions
        props = self._retrieve_propositions(
            question, entity=entity, top_k=self.top_k_props
        )

        # Step 3: Retrieve raw chunks (broader context for temporal/multi-hop)
        chunks = self._retrieve_chunks(question, top_k=self.top_k_chunks, entity=entity)

        # Step 4: Generate answer (route to inferential prompt if needed)
        inferential = self._is_inferential(question)
        answer = self._generate_answer(
            question,
            props,
            chunks,
            entity,
            llm_client,
            llm_model,
            is_inferential=inferential,
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
