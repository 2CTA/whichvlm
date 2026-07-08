from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from models.benchmark_sources.aa_index import (
    AA_LEADERBOARD_URL,
    ExtractionFailed,
    canonical_name,
    decode_rsc_blob,
    extract_aa_pairs_from_html,
    fetch_aa_index_scores,
    get_aa_curated_fallback,
)


def rsc_page(records: list[dict]) -> str:
    fragment = ",".join(
        '{"slug":"x","name":%s,"reasoningModel":false,'
        '"intelligenceIndex":%s,"codingIndex":1.0}'
        % (json.dumps(r["name"]), r["index"])
        for r in records
    )
    chunk = json.dumps("3:[" + fragment + "]\n")
    return (
        "<!DOCTYPE html><html><body>"
        "<script>self.__next_f.push([0])</script>"
        f"<script>self.__next_f.push([1,{chunk}])</script>"
        "</body></html>"
    )


def test_canonical_name_strips_variants_and_separators():
    assert canonical_name("Qwen3 14B (Reasoning)") == "qwen3 14b"
    assert canonical_name("Qwen3-14B") == "qwen3 14b"

    assert canonical_name("GLM-5 (Non-reasoning)") == "glm 5"
    assert canonical_name("DeepSeek V4 Pro (Reasoning, Max Effort)") == (
        "deepseek v4 pro"
    )


def test_decode_rsc_blob_unescapes_chunks():
    page = rsc_page([{"name": "Qwen3 14B (Reasoning)", "index": 33.0}])
    blob = decode_rsc_blob(page)
    assert '"name":"Qwen3 14B (Reasoning)"' in blob
    assert '"intelligenceIndex":33.0' in blob


def test_extract_pairs_from_rsc_html():
    page = rsc_page(
        [
            {"name": "Qwen3 14B (Reasoning)", "index": 33.0},
            {"name": "Qwen3 14B (Non-reasoning)", "index": 30.0},
            {"name": "GLM-5 (Reasoning)", "index": 50.0},
        ]
    )
    pairs = dict(extract_aa_pairs_from_html(page))
    assert pairs["Qwen3 14B (Reasoning)"] == 33.0
    assert pairs["GLM-5 (Reasoning)"] == 50.0

    assert len(pairs) == 3


def test_extract_pairs_returns_empty_on_legacy_or_garbage_html():
    assert extract_aa_pairs_from_html("<html>no rsc here</html>") == []


def run_fetch(html: str) -> dict[str, float]:
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == AA_LEADERBOARD_URL
        return httpx.Response(200, text=html)

    async def go() -> dict[str, float]:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            return await fetch_aa_index_scores(client)

    return asyncio.run(go())


def test_fetch_maps_canonical_names_and_merges_over_fallback():
    page = rsc_page([{"name": "Qwen3 14B (Reasoning)", "index": 55.0}])
    scores = run_fetch(page)

    fallback = get_aa_curated_fallback()

    assert set(fallback).issubset(set(scores))

    assert scores["Qwen/Qwen3-14B"] > fallback["Qwen/Qwen3-14B"]


def test_fetch_raises_when_no_records_found():
    with pytest.raises(ExtractionFailed):
        run_fetch("<html><body>nothing to see</body></html>")
