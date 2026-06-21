from __future__ import annotations

import json
import time

from whichvlm.utils import cache_dir

CACHE_DIR = cache_dir()
CACHE_FILE = CACHE_DIR / "models.json"
DEFAULT_TTL_SECONDS = 6 * 3600  # 6 hours


def _ensure_cache_dir() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def load_cache() -> list[dict] | None:
    if not CACHE_FILE.exists():
        return None

    try:
        data = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return None
        cached_at = data["cached_at"]
        if time.time() - cached_at > DEFAULT_TTL_SECONDS:
            return None
        models = data["models"]
        return models if isinstance(models, list) else None
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def save_cache(models: list[dict]) -> None:
    _ensure_cache_dir()
    data = {
        "cached_at": time.time(),
        "models": models,
    }
    CACHE_FILE.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
