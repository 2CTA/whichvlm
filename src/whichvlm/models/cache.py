from __future__ import annotations

import json
import time

from whichvlm.models.cache_format import (
    cache_expired,
    cache_snapshot_metadata,
    read_cache_payload,
)
from whichvlm.utils import cache_dir

CACHE_DIR = cache_dir()
CACHE_FILE = CACHE_DIR / "models.json"
DEFAULT_TTL_SECONDS = 6 * 3600
CACHE_SCHEMA_VERSION = 2
DEFAULT_SOURCE = {"name": "huggingface"}


def ensure_cache_dir() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def load_cache(*, allow_stale: bool = False) -> list[dict] | None:
    payload = read_cache_payload(CACHE_FILE)
    if payload is None:
        return None

    try:
        if cache_expired(
            payload["cached_at"], DEFAULT_TTL_SECONDS, allow_stale=allow_stale
        ):
            return None
        models = payload["models"]
        return models if isinstance(models, list) else None
    except (KeyError, TypeError):
        return None


def cache_snapshot() -> dict | None:
    payload = read_cache_payload(CACHE_FILE)
    if payload is None:
        return None
    return cache_snapshot_metadata(
        payload,
        default_ttl_seconds=DEFAULT_TTL_SECONDS,
        item_key="models",
        item_count_key="model_count",
        default_source=DEFAULT_SOURCE,
    )


def save_cache(models: list[dict], *, source: dict | None = None) -> None:
    ensure_cache_dir()
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "cached_at": time.time(),
        "ttl_seconds": DEFAULT_TTL_SECONDS,
        "source": source or DEFAULT_SOURCE,
        "models": models,
    }
    CACHE_FILE.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
