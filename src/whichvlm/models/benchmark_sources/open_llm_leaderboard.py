from __future__ import annotations

import io

import httpx

from whichvlm.models.http import get_with_retries

LEADERBOARD_PARQUET_URL = (
    "https://huggingface.co/api/datasets/open-llm-leaderboard/contents"
    "/parquet/default/train/0.parquet"
)
LEADERBOARD_ROWS_URL = "https://datasets-server.huggingface.co/rows"
LEADERBOARD_DATASET = "open-llm-leaderboard/contents"


LB_AVG_MAX = 52
ARCHIVE_SOURCE_MAX_NORMALIZED = 78.0


async def fetch_leaderboard_parquet(client: httpx.AsyncClient) -> dict[str, float]:

    import pyarrow.parquet as pq

    resp = await get_with_retries(
        client, LEADERBOARD_PARQUET_URL, follow_redirects=True
    )
    resp.raise_for_status()
    table = pq.read_table(
        io.BytesIO(resp.content),
        columns=["fullname", "Average ⬆️"],
    )
    d = table.to_pydict()
    scores: dict[str, float] = {}
    for i in range(len(d["fullname"])):
        name = d["fullname"][i]
        avg = d["Average ⬆️"][i]
        if name and avg and avg > 0:
            scores[name] = normalize_leaderboard_avg(avg)
    return scores


async def fetch_leaderboard_api(client: httpx.AsyncClient) -> dict[str, float]:

    scores: dict[str, float] = {}
    offset = 0

    while True:
        resp = await get_with_retries(
            client,
            LEADERBOARD_ROWS_URL,
            params={
                "dataset": LEADERBOARD_DATASET,
                "config": "default",
                "split": "train",
                "offset": str(offset),
                "length": "100",
            },
        )
        resp.raise_for_status()
        data = resp.json()
        rows = data.get("rows", [])
        if not rows:
            break

        for r in rows:
            row = r.get("row", {})
            name = row.get("fullname")
            avg = row.get("Average ⬆️")
            if name and avg and avg > 0:
                scores[name] = normalize_leaderboard_avg(avg)

        offset += len(rows)
        total = data.get("num_rows_total", 0)
        if total and offset >= total:
            break

    return scores


def normalize_leaderboard_avg(avg: float) -> float:

    score = avg / LB_AVG_MAX * ARCHIVE_SOURCE_MAX_NORMALIZED
    return max(0.0, min(ARCHIVE_SOURCE_MAX_NORMALIZED, round(score, 1)))


async def fetch_leaderboard_with_fallback(
    client: httpx.AsyncClient,
) -> dict[str, float]:

    try:
        return await fetch_leaderboard_parquet(client)
    except ImportError:
        return await fetch_leaderboard_api(client)
