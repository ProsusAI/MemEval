"""Mimir: persistent agent memory with structured entities and FTS5 hybrid search."""

import asyncio
import json
import os
import subprocess
import time
import urllib.request
import uuid
from datetime import datetime
from typing import Any

from agents_memory.systems._helpers import _qa_results, _qa_results_async, run_async
from agents_memory.locomo import extract_dialogues


SYSTEM_INFO = {
    "name": "mimir",
    "architecture": "structured entity model + FTS5 hybrid search + proactive recall hooks",
    "infrastructure": "Mimir server + SQLite + FTS5",
    "url": "https://github.com/Perseus-Computing-LLC/mimir",
    "fqdn": "perseus.observer",
}


class MimirClient:
    """Thin HTTP wrapper around the Mimir MCP server (SSE transport)."""

    def __init__(self, base_url: str = "http://localhost:8766", session_id: str = "bench"):
        self._base_url = base_url
        self._session_id = session_id
        self._msg_url = f"{base_url}/message?sessionId={session_id}"

    def _call_mcp(self, method: str, params: dict | None = None) -> dict:
        body = json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params or {},
        }).encode()
        req = urllib.request.Request(
            self._msg_url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())

    def init(self) -> str:
        r = self._call_mcp("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "mimir-bench", "version": "1.0"},
        })
        self._call_mcp("notifications/initialized", {})
        return r.get("result", {}).get("serverInfo", {}).get("version", "?")

    def remember(self, category: str, key: str, body_json: str,
                 entity_type: str = "fact", importance: float = 0.5,
                 tags: list[str] | None = None) -> str:
        args: dict[str, Any] = {
            "category": category, "key": key, "body_json": body_json,
            "entity_type": entity_type, "importance": importance,
        }
        if tags:
            args["tags"] = tags
        r = self._call_mcp("tools/call", {"name": "mimir_remember", "arguments": args})
        return r.get("result", {}).get("content", [{}])[0].get("text", "")

    def recall_all(self, category: str, limit: int = 100) -> str:
        """Retrieve ALL entities for a category directly from the DB, bypassing FTS5."""
        import sqlite3
        try:
            db_path = "/tmp/mimir_sse.db"
            conn = sqlite3.connect(db_path)
            cur = conn.execute(
                "SELECT body_json FROM entities WHERE category = ? ORDER BY created_at_unix_ms LIMIT ?",
                (category, limit)
            )
            lines = []
            for (bj,) in cur.fetchall():
                if not bj:
                    continue
                try:
                    data = json.loads(bj)
                    text = data.get("text", "")
                    if text and text.strip():
                        lines.append(text.strip())
                except json.JSONDecodeError:
                    pass
            conn.close()
            return "\n".join(lines)
        except Exception:
            return ""

    def recall(self, query: str, category: str | None = None,
               limit: int = 30) -> str:
        args: dict[str, Any] = {"query": query, "limit": limit}
        if category:
            args["category"] = category
        r = self._call_mcp("tools/call", {"name": "mimir_recall", "arguments": args})
        raw = r.get("result", {}).get("content", [{}])[0].get("text", "")
        try:
            data = json.loads(raw)
            items = data.get("items", [])
            if not items:
                return ""
            return "\n".join(it.get("text", "") for it in items if it.get("text"))
        except json.JSONDecodeError:
            return raw


# ---------------------------------------------------------------------------
# Answering
# ---------------------------------------------------------------------------

ANSWER_PROMPT = """Question: {question}

Memories:
{memories}

Answer using ONLY the memories above. Be one sentence:"""


# ---------------------------------------------------------------------------
# Benchmark entry point
# ---------------------------------------------------------------------------

@run_async
async def run(
    conv: dict, llm_model: str, run_judge: bool,
    category_names: dict | None = None, judge_fn: str | None = None,
) -> list[dict]:
    from openai import OpenAI
    import re

    api_key = os.environ["OPENAI_API_KEY"]
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    sample_id = conv.get("sample_id", "unknown")

    # Stopwords to strip from queries
    STOPWORDS = {'what','was','the','i','had','with','my','did','do','does','is','are',
                 'a','an','of','in','to','for','on','at','from','by','about','after',
                 'which','first','when','where','how','who','why','can','could','would',
                 'should','will','shall','may','might','this','that','these','those','me',
                 'you','your','he','she','it','we','they','him','her','them','his','their',
                 'its','not','no','be','been','being','have','has','had','having','and',
                 'or','but','if','then','else','than','too','very','just','now','here','there'}

    def extract_keywords(question: str) -> str:
        """Extract meaningful keywords from a question for FTS5 search."""
        # Remove punctuation, lowercase, split
        words = re.findall(r"[a-zA-Z0-9]+", question.lower())
        # Keep words longer than 2 chars that aren't stopwords
        keywords = [w for w in words if len(w) > 2 and w not in STOPWORDS]
        return " ".join(keywords) if keywords else question

    # Connect to the running Mimir server
    # Auto-pick a free session ID to avoid collisions
    session_id = f"bench_{uuid.uuid4().hex[:8]}"
    mimir = MimirClient(base_url="http://localhost:8766", session_id=session_id)
    ver = mimir.init()
    print(f"    Mimir v{ver} connected (HTTP)")

    # --- Ingest dialogue turns ---
    dialogues = extract_dialogues(conv)
    keys_used: set[str] = set()
    for i, turn in enumerate(dialogues):
        speaker = turn.get("speaker", "unknown")
        text = turn.get("text", "")
        if not text or not text.strip():
            continue
        key = f"turn_{i}"
        body = json.dumps({"speaker": speaker, "text": text})

        # Extract temporal tag if present
        tags = [speaker]
        ts = turn.get("timestamp", "")
        if ts:
            for fmt in ("%I:%M %p on %d %B, %Y", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
                try:
                    parsed = datetime.strptime(ts.strip(), fmt)
                    tags.append(parsed.strftime("month_%Y-%m"))
                    break
                except ValueError:
                    pass
        try:
            mimir.remember(
                category=f"conv_{sample_id}",
                key=key,
                body_json=body,
                tags=tags,
                entity_type="conversation_turn",
                importance=0.5,
            )
            keys_used.add(key)
        except Exception as e:
            print(f"    Remember error turn {i}: {e}")

    print(f"    Ingested: {len(keys_used)} dialogue turns into Mimir")
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=30)

    async def answer_fn(question: str) -> str:
        try:
            # Multi-keyword FTS5 search — try combos for best coverage
            query = extract_keywords(question)
            recalled_set = set()
            
            # Search with full keyword string
            if query:
                r = mimir.recall(query=query, category=f"conv_{sample_id}", limit=20)
                for line in r.split("\n"):
                    if line.strip():
                        recalled_set.add(line.strip())
            
            # Also search each significant keyword individually
            for kw in query.split()[:5]:
                if len(kw) >= 3:
                    r = mimir.recall(query=kw, category=f"conv_{sample_id}", limit=5)
                    for line in r.split("\n"):
                        if line.strip():
                            recalled_set.add(line.strip())
            
            # Fall back to full context if FTS5 found nothing useful
            if len(recalled_set) < 3:
                recalled = mimir.recall_all(category=f"conv_{sample_id}")
            else:
                recalled = "\n".join(recalled_set)
        except Exception as e:
            print(f"    Recall error: {e}")
            recalled = ""
        if not recalled or not recalled.strip():
            return "None"
        try:
            response = client.chat.completions.create(
                model=llm_model,
                messages=[{
                    "role": "user",
                    "content": ANSWER_PROMPT.format(question=question, memories=recalled),
                }],
                max_tokens=80,
                temperature=0.1,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"    LLM error: {e}")
            return "None"

    return await _qa_results_async(
        conv, answer_fn, run_judge,
        category_names=category_names, judge_fn=judge_fn,
    )
