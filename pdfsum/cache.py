"""Tiny JSON cache so a document is never summarized twice with the same settings."""

from __future__ import annotations

import hashlib
import json
import os
import re

from . import config


def file_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def cache_key(file_hash_: str, strategy: str, model: str, chunk_size: int) -> str:
    raw = f"{file_hash_}-{strategy}-{model}-{chunk_size}"
    return re.sub(r"[^A-Za-z0-9._-]", "_", raw)


def _path(key: str) -> str:
    return os.path.join(config.CACHE_DIR, f"{key}.json")


def load(key: str) -> dict | None:
    try:
        with open(_path(key), "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def save(key: str, payload: dict) -> None:
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    with open(_path(key), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
