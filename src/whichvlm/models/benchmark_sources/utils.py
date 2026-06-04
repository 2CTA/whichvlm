from __future__ import annotations


def _walk(obj, depth: int = 0):
    """Yield every dict encountered while recursively walking a JSON tree."""
    if depth > 12:
        return
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from _walk(v, depth + 1)
    elif isinstance(obj, list):
        for item in obj:
            yield from _walk(item, depth + 1)
