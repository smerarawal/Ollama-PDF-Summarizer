"""Thin wrapper around the Ollama Python client.

Adds: the same generation options on every call, retries, token counting and
friendly errors when the server is off or a model has not been pulled.
"""

from __future__ import annotations

import threading
import time

import ollama

from . import config


class OllamaUnavailable(RuntimeError):
    """Raised with a message that tells the user how to fix the problem."""


class Stats:
    """Thread-safe counters for tokens and calls (calls may run in parallel)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self.prompt_tokens = 0
            self.completion_tokens = 0
            self.calls = 0

    def add(self, prompt_tokens: int, completion_tokens: int) -> None:
        with self._lock:
            self.prompt_tokens += prompt_tokens or 0
            self.completion_tokens += completion_tokens or 0
            self.calls += 1

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "calls": self.calls,
            }


STATS = Stats()


def _options() -> dict:
    return {"temperature": config.TEMPERATURE, "num_ctx": config.NUM_CTX}


def _messages(prompt: str, system: str | None) -> list[dict]:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return messages


def _model_names(listed) -> list[str]:
    """Names of installed models; works for old (dict) and new (object) clients."""
    models = listed["models"] if isinstance(listed, dict) else listed.models
    names = []
    for item in models:
        if isinstance(item, dict):
            name = item.get("model") or item.get("name")
        else:
            name = getattr(item, "model", None) or getattr(item, "name", None)
        if name:
            names.append(name)
    return names


def check_ollama(models: list[str] | None = None) -> None:
    """Raise OllamaUnavailable if the server is down or a model is missing."""
    try:
        installed = _model_names(ollama.list())
    except Exception as exc:
        raise OllamaUnavailable(
            "Cannot reach Ollama. Start it with `ollama serve` and try again."
        ) from exc

    missing = [
        m
        for m in (models or [])
        if not any(name == m or name.startswith(m + ":") for name in installed)
    ]
    if missing:
        pulls = " and ".join(f"`ollama pull {m}`" for m in missing)
        raise OllamaUnavailable(f"Model not installed yet. Run {pulls}.")


def chat(
    prompt: str,
    model: str | None = None,
    system: str | None = None,
    retries: int = 2,
) -> str:
    """Send one prompt, return the reply text. Retries with a short back-off."""
    model = model or config.SUMMARY_MODEL
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = ollama.chat(
                model=model, messages=_messages(prompt, system), options=_options()
            )
            STATS.add(response.get("prompt_eval_count"), response.get("eval_count"))
            return response["message"]["content"].strip()
        except Exception as exc:  # network blips, model still loading, ...
            last_error = exc
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
    raise OllamaUnavailable(f"Ollama call failed after {retries + 1} tries: {last_error}")


def stream_chat(prompt: str, model: str | None = None, system: str | None = None):
    """Yield the reply piece by piece (for st.write_stream)."""
    model = model or config.SUMMARY_MODEL
    try:
        stream = ollama.chat(
            model=model,
            messages=_messages(prompt, system),
            options=_options(),
            stream=True,
        )
        for part in stream:
            piece = part["message"]["content"]
            if piece:
                yield piece
            if part.get("done"):
                STATS.add(part.get("prompt_eval_count"), part.get("eval_count"))
    except Exception as exc:
        raise OllamaUnavailable(f"Ollama streaming call failed: {exc}") from exc
