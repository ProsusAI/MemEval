"""Monkey-patch OpenAI client to count LLM calls and tokens."""

import threading
from collections import defaultdict

import openai.resources.chat.completions

_lock = threading.Lock()
_stats = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
_stats_by_model: dict[str, dict[str, int]] = defaultdict(
    lambda: {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
)
_original_create = None
_original_async_create = None


def _record_usage(
    prompt_tokens: int, completion_tokens: int, model: str | None = None
) -> None:
    """Record token usage in both aggregate and per-model stats."""
    with _lock:
        _stats["prompt_tokens"] += prompt_tokens
        _stats["completion_tokens"] += completion_tokens
        if model:
            _stats_by_model[model]["prompt_tokens"] += prompt_tokens
            _stats_by_model[model]["completion_tokens"] += completion_tokens


def _record_call(model: str | None = None) -> None:
    """Record a single LLM call in both aggregate and per-model stats."""
    with _lock:
        _stats["calls"] += 1
        if model:
            _stats_by_model[model]["calls"] += 1


class _StreamWrapper:
    """Wrap a streaming response to capture usage from the final chunk."""

    def __init__(self, stream, model: str | None = None):
        self._stream = stream
        self._model = model

    def __iter__(self):
        for chunk in self._stream:
            if chunk.usage:
                _record_usage(
                    chunk.usage.prompt_tokens or 0,
                    chunk.usage.completion_tokens or 0,
                    self._model,
                )
            yield chunk

    def __enter__(self):
        self._stream.__enter__()
        return self

    def __exit__(self, *args):
        return self._stream.__exit__(*args)


class _AsyncStreamWrapper:
    """Wrap an async streaming response to capture usage from the final chunk."""

    def __init__(self, stream, model: str | None = None):
        self._stream = stream
        self._model = model

    async def __aiter__(self):
        async for chunk in self._stream:
            if chunk.usage:
                _record_usage(
                    chunk.usage.prompt_tokens or 0,
                    chunk.usage.completion_tokens or 0,
                    self._model,
                )
            yield chunk

    async def __aenter__(self):
        await self._stream.__aenter__()
        return self

    async def __aexit__(self, *args):
        return await self._stream.__aexit__(*args)


def _tracked_create(*args, **kwargs):
    model = kwargs.get("model")
    is_stream = kwargs.get("stream", False)

    # For streaming: request usage in final chunk
    if is_stream:
        kwargs.setdefault("stream_options", {})["include_usage"] = True

    response = _original_create(*args, **kwargs)

    _record_call(model)

    if is_stream:
        return _StreamWrapper(response, model)

    if hasattr(response, "usage") and response.usage:
        _record_usage(
            response.usage.prompt_tokens or 0,
            response.usage.completion_tokens or 0,
            model,
        )

    return response


async def _tracked_async_create(*args, **kwargs):
    model = kwargs.get("model")
    is_stream = kwargs.get("stream", False)

    # For streaming: request usage in final chunk
    if is_stream:
        kwargs.setdefault("stream_options", {})["include_usage"] = True

    response = await _original_async_create(*args, **kwargs)

    _record_call(model)

    if is_stream:
        return _AsyncStreamWrapper(response, model)

    if hasattr(response, "usage") and response.usage:
        _record_usage(
            response.usage.prompt_tokens or 0,
            response.usage.completion_tokens or 0,
            model,
        )

    return response


def start():
    """Patch OpenAI client (sync and async) to start tracking."""
    global _original_create, _original_async_create
    if _original_create is None:
        _original_create = openai.resources.chat.completions.Completions.create
        openai.resources.chat.completions.Completions.create = _tracked_create
    if _original_async_create is None:
        _original_async_create = (
            openai.resources.chat.completions.AsyncCompletions.create
        )
        openai.resources.chat.completions.AsyncCompletions.create = (
            _tracked_async_create
        )


def get_stats():
    """Return current aggregate stats dict."""
    with _lock:
        return {
            "calls": _stats["calls"],
            "prompt_tokens": _stats["prompt_tokens"],
            "completion_tokens": _stats["completion_tokens"],
            "total_tokens": _stats["prompt_tokens"] + _stats["completion_tokens"],
        }


def get_stats_by_model() -> dict[str, dict[str, int]]:
    """Return per-model stats dict.

    Returns:
        {"gpt-4.1": {"calls": N, "prompt_tokens": N, ...}, ...}
    """
    with _lock:
        result = {}
        for model, s in _stats_by_model.items():
            result[model] = {
                "calls": s["calls"],
                "prompt_tokens": s["prompt_tokens"],
                "completion_tokens": s["completion_tokens"],
                "total_tokens": s["prompt_tokens"] + s["completion_tokens"],
            }
        return result


def reset():
    """Reset counters to zero."""
    with _lock:
        _stats["calls"] = 0
        _stats["prompt_tokens"] = 0
        _stats["completion_tokens"] = 0
        _stats_by_model.clear()
