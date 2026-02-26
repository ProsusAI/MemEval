"""SimpleMem: multi-round retrieval with parallel processing."""

import os
from datetime import datetime

import numpy as np
from openai import OpenAI

from agents_memory.locomo import extract_dialogues
from agents_memory.systems._helpers import _qa_results

SYSTEM_INFO = {
    "architecture": "multi-round retrieval with parallel processing",
    "infrastructure": "SimpleMem library",
}

# ---------------------------------------------------------------------------
# Monkey-patch SimpleMem's EmbeddingModel to use OpenAI text-embedding-3-small
# instead of the default local Qwen3 SentenceTransformers model, so that all
# benchmark systems share the same embedding model for a fair comparison.
# ---------------------------------------------------------------------------
from simplemem.utils.embedding import EmbeddingModel  # noqa: E402

_embed_client: OpenAI | None = None


def _get_embed_client() -> OpenAI:
    global _embed_client
    if _embed_client is None:
        _embed_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _embed_client


def _openai_init(self, model_name=None, use_optimization=True):
    """Skip local model loading; set dimension for text-embedding-3-small."""
    self.model_name = "text-embedding-3-small"
    self.dimension = 1536
    self.model = None
    self.model_type = "openai"
    self.supports_query_prompt = False


def _openai_encode(self, texts, **kwargs):
    """Use OpenAI text-embedding-3-small instead of local SentenceTransformers."""
    if isinstance(texts, str):
        texts = [texts]
    response = _get_embed_client().embeddings.create(
        input=texts, model="text-embedding-3-small"
    )
    return np.array([d.embedding for d in response.data])


EmbeddingModel.__init__ = _openai_init
EmbeddingModel.encode = _openai_encode


def run(conv: dict, llm_model: str, run_judge: bool) -> list[dict]:
    from simplemem import SimpleMemConfig, SimpleMemSystem, set_config
    from simplemem.models.memory_entry import Dialogue

    config = SimpleMemConfig(
        openai_api_key=os.environ["OPENAI_API_KEY"],
        llm_model=llm_model,
    )
    set_config(config)

    dialogues = extract_dialogues(conv)
    memory = SimpleMemSystem(clear_db=True)

    dialogue_objects = [
        Dialogue(
            dialogue_id=i + 1,
            speaker=d["speaker"],
            content=d["text"],
            timestamp=d["timestamp"] or datetime.now().isoformat(),
        )
        for i, d in enumerate(dialogues)
    ]

    memory.add_dialogues(dialogue_objects)
    memory.finalize()
    print(f"    Dialogues ingested: {len(dialogue_objects)}")

    return _qa_results(
        conv,
        lambda q: memory.ask(q),
        run_judge,
    )
