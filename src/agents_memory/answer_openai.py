"""OpenAI-based short-answer agent + EM utilities.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field

from openai import AsyncOpenAI, OpenAI

ANSWER_PROMPT_SHORT = """\
You are answering questions using ONLY the provided memories from a conversation.
The memories may mention MULTIPLE people. Names matter: do NOT treat different names as the same person.
Memories may include metadata prefixes like "[Speaker]" and "(Session N)". Treat these prefixes as metadata unless the question asks about the speaker/session.

Rules:
1. Give the SHORTEST answer possible (usually 1-8 words).
2. Use EXACT words from the memories when possible.
3. NO full sentences, NO explanations, NO extra punctuation.
4. For dates/times/numbers, copy the format from the memories.
5. If the answer is not supported by the memories, output exactly: None
"""


ANSWER_PROMPT_JSON_F1 = """\
You are answering questions using ONLY the provided memories from a conversation.
The memories may mention MULTIPLE people. Names matter: do NOT treat different names as the same person.
Memories may include metadata prefixes like "[Speaker]" and "(Session N)". Treat these prefixes as metadata unless the question asks about the speaker/session.

Return a JSON object: {"answer": "..."}.

Rules:
1. Be concise but COMPLETE. If the question asks for multiple items, list ALL items (comma-separated).
2. Use EXACT words from the memories when possible.
3. For dates/times/numbers, copy the format from the memories.
4. If the answer is not supported by the memories, answer must be exactly: "None".
5. Do NOT include any extra keys or any text outside the JSON object.
"""


def _max_token_kwargs(model: str, max_tokens: int) -> dict[str, int]:
    # GPT-5.x and o-series use max_completion_tokens; older use max_tokens.
    if model.startswith("gpt-5") or model.startswith("o"):
        return {"max_completion_tokens": max_tokens}
    return {"max_tokens": max_tokens}


_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]")


def normalize_for_em(text: str) -> str:
    text = text.strip().lower()
    text = _PUNCT_RE.sub("", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def exact_match(predicted: str, ground_truth: str) -> int:
    return 1 if normalize_for_em(predicted) == normalize_for_em(ground_truth) else 0


def clean_short_answer(text: str) -> str:
    text = text.strip()
    for prefix in ("Answer:", "A:", "answer:"):
        if text.startswith(prefix):
            text = text[len(prefix) :].strip()
    return text


@dataclass
class OpenAIAnswerAgent:
    model: str = "gpt-4.1"
    api_key: str | None = None
    temperature: float = 0.0
    seed: int | None = None
    prompt_style: str = "short"  # "short" | "json_f1"
    _client: OpenAI = field(init=False)
    _cache: dict[str, str] = field(default_factory=dict, init=False)
    _seed_supported: bool = field(default=True, init=False)

    def __post_init__(self) -> None:
        self._client = OpenAI(api_key=self.api_key or os.environ.get("OPENAI_API_KEY"))

    def answer(self, question: str, memory_text: str, max_tokens: int = 64) -> str:
        key = hashlib.sha256(
            f"{self.model}\n{self.prompt_style}\n{question}\n{memory_text}".encode()
        ).hexdigest()
        if key in self._cache:
            return self._cache[key]
        if self.prompt_style == "json_f1":
            system_prompt = ANSWER_PROMPT_JSON_F1
            response_format = {"type": "json_object"}
        else:
            system_prompt = ANSWER_PROMPT_SHORT
            response_format = None
        kwargs: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"Memories:\n{memory_text}\n\nQuestion: {question}",
                },
            ],
            "temperature": self.temperature,
            **_max_token_kwargs(self.model, max_tokens),
        }
        if response_format is not None:
            kwargs["response_format"] = response_format
        if self.seed is not None and self._seed_supported:
            kwargs["seed"] = int(self.seed)
        try:
            resp = self._client.chat.completions.create(**kwargs)
        except Exception:
            # Some models/endpoints may not support `seed`. Retry once without it.
            if "seed" in kwargs:
                self._seed_supported = False
                kwargs.pop("seed", None)
                resp = self._client.chat.completions.create(**kwargs)
            else:
                raise
        out_raw = resp.choices[0].message.content or ""
        if self.prompt_style == "json_f1":
            try:
                obj = json.loads(out_raw)
                out = str(obj.get("answer", "")).strip()
            except Exception:
                out = out_raw
        else:
            out = out_raw
        out = clean_short_answer(out)
        self._cache[key] = out
        return out


@dataclass
class AsyncOpenAIAnswerAgent:
    model: str = "gpt-4.1"
    api_key: str | None = None
    temperature: float = 0.0
    seed: int | None = None
    prompt_style: str = "short"  # "short" | "json_f1"
    _client: AsyncOpenAI = field(init=False)
    _cache: dict[str, str] = field(default_factory=dict, init=False)
    _seed_supported: bool = field(default=True, init=False)

    def __post_init__(self) -> None:
        self._client = AsyncOpenAI(
            api_key=self.api_key or os.environ.get("OPENAI_API_KEY")
        )

    async def answer(
        self, question: str, memory_text: str, max_tokens: int = 64
    ) -> str:
        key = hashlib.sha256(
            f"{self.model}\n{self.prompt_style}\n{question}\n{memory_text}".encode()
        ).hexdigest()
        if key in self._cache:
            return self._cache[key]
        if self.prompt_style == "json_f1":
            system_prompt = ANSWER_PROMPT_JSON_F1
            response_format = {"type": "json_object"}
        else:
            system_prompt = ANSWER_PROMPT_SHORT
            response_format = None
        kwargs: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"Memories:\n{memory_text}\n\nQuestion: {question}",
                },
            ],
            "temperature": self.temperature,
            **_max_token_kwargs(self.model, max_tokens),
        }
        if response_format is not None:
            kwargs["response_format"] = response_format
        if self.seed is not None and self._seed_supported:
            kwargs["seed"] = int(self.seed)
        try:
            resp = await self._client.chat.completions.create(**kwargs)
        except Exception:
            if "seed" in kwargs:
                self._seed_supported = False
                kwargs.pop("seed", None)
                resp = await self._client.chat.completions.create(**kwargs)
            else:
                raise
        out_raw = resp.choices[0].message.content or ""
        if self.prompt_style == "json_f1":
            try:
                obj = json.loads(out_raw)
                out = str(obj.get("answer", "")).strip()
            except Exception:
                out = out_raw
        else:
            out = out_raw
        out = clean_short_answer(out)
        self._cache[key] = out
        return out
