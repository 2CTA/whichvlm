from __future__ import annotations

import os
import re
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path

import click


def current_version() -> str:
    try:
        return version("whichvlm")
    except PackageNotFoundError:
        return "unknown"


_SHORTHAND_RE = re.compile(r"^(\d+(?:\.\d+)?)\s*([kmb])$", re.IGNORECASE)
_MULTIPLIERS = {"k": 1024, "m": 1024 * 1024, "b": 1024 * 1024 * 1024}


def parse_context_length(value: str) -> int:
    """Parse a context length string, supporting shorthand like 64k or 128K.

    Accepts plain integers (e.g. "4096") or shorthand with a suffix:
      k/K = x1,024  (64k -> 65536)
      m/M = x1,048,576
      b/B = x1,073,741,824

    Returns the integer context length. Raises ValueError on bad input.
    """
    value = value.strip()
    match = _SHORTHAND_RE.match(value)
    if match:
        number = float(match.group(1))
        suffix = match.group(2).lower()
        context_length = int(number * _MULTIPLIERS[suffix])
        if context_length <= 0:
            raise ValueError(f"Context length must be positive, got {value!r}")
        return context_length
    try:
        context_length = int(value)
    except ValueError:
        raise ValueError(
            f"Invalid context length {value!r}. "
            "Use a plain integer (4096) or shorthand (64k, 128k)."
        )
    if context_length <= 0:
        raise ValueError(f"Context length must be positive, got {context_length}")
    return context_length


class ContextLengthType(click.ParamType):
    """Click parameter type that accepts integers or shorthand like 64k."""

    name = "context_length"

    def convert(self, value, param, ctx):
        if isinstance(value, int):
            return value
        try:
            return parse_context_length(str(value))
        except ValueError as e:
            self.fail(str(e), param, ctx)


CONTEXT_LENGTH = ContextLengthType()


def cache_dir() -> Path:
    """Return the whichvlm cache directory, respecting XDG_CACHE_HOME."""
    base = os.environ.get("XDG_CACHE_HOME")
    if base and Path(base).is_absolute():
        return Path(base) / "whichvlm"
    return Path.home() / ".cache" / "whichvlm"
