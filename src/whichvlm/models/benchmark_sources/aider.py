from __future__ import annotations

import logging
import re

import httpx

from whichvlm.models.http import get_with_retries

logger = logging.getLogger(__name__)

AIDER_POLYGLOT_YML_URL = (
    "https://raw.githubusercontent.com/Aider-AI/aider/main/"
    "aider/website/_data/polyglot_leaderboard.yml"
)


PG_MIN = 0.0
PG_MAX = 90.0

AIDER_NAME_TO_HF_IDS: dict[str, list[str]] = {
    "deepseek-r1": ["deepseek-ai/DeepSeek-R1"],
    "deepseek-r1-0528": ["deepseek-ai/DeepSeek-R1-0528"],
    "deepseek-v3": ["deepseek-ai/DeepSeek-V3"],
    "deepseek-v3-0324": ["deepseek-ai/DeepSeek-V3-0324"],
    "deepseek-v3.1": ["deepseek-ai/DeepSeek-V3.1"],
    "deepseek-v3.2": ["deepseek-ai/DeepSeek-V3.2"],
    "deepseek-v4-pro": ["deepseek-ai/DeepSeek-V4-Pro"],
    "deepseek-v4-flash": ["deepseek-ai/DeepSeek-V4-Flash"],
    "qwen3-coder-30b-a3b-instruct": ["Qwen/Qwen3-Coder-30B-A3B-Instruct"],
    "qwen3-coder-next": ["Qwen/Qwen3-Coder-Next"],
    "qwen2.5-coder-32b-instruct": ["Qwen/Qwen2.5-Coder-32B-Instruct"],
    "qwen3-32b": ["Qwen/Qwen3-32B"],
    "qwen3.6-27b": ["Qwen/Qwen3.6-27B"],
    "llama-3.3-70b-instruct": ["meta-llama/Llama-3.3-70B-Instruct"],
    "llama-4-maverick": ["meta-llama/Llama-4-Maverick-17B-128E-Instruct"],
    "gemma-3-27b-it": ["google/gemma-3-27b-it"],
    "gemma-4-31b": ["google/gemma-4-31b-it"],
    "mistral-large-2411": ["mistralai/Mistral-Large-Instruct-2411"],
    "devstral-small": ["mistralai/Devstral-Small-2505"],
    "gpt-oss-120b": ["openai/gpt-oss-120b"],
    "gpt-oss-20b": ["openai/gpt-oss-20b"],
    "glm-4.5": ["zai-org/GLM-4.5"],
    "glm-4.6": ["zai-org/GLM-4.6"],
    "glm-5": ["zai-org/GLM-5"],
    "glm-5.1": ["zai-org/GLM-5.1"],
    "kimi-k2-instruct": ["moonshotai/Kimi-K2-Instruct"],
    "phi-4": ["microsoft/phi-4"],
    "qwq-32b": ["Qwen/QwQ-32B"],
}


PASS_RATE_RE = re.compile(r"pass_rate[_-]?2[:\s]+(\d+(?:\.\d+)?)", re.IGNORECASE)
MODEL_RE = re.compile(r"^- model[:\s]+(.+)$", re.MULTILINE)


def normalize(pass_rate: float) -> float:
    if not isinstance(pass_rate, (int, float)):
        return 0.0
    span = PG_MAX - PG_MIN
    normalized = (pass_rate - PG_MIN) / span * 100.0
    return max(0.0, min(100.0, round(normalized, 1)))


def parse_yaml_lite(text: str) -> list[tuple[str, float]]:

    out: list[tuple[str, float]] = []

    records = re.split(r"\n(?=-\s+\w)", text)
    for rec in records:
        m_model = re.search(r"^\s*model[:\s]+(.+?)$", rec, re.MULTILINE | re.IGNORECASE)
        m_rate = re.search(r"pass_rate_2[:\s]+(\d+(?:\.\d+)?)", rec, re.IGNORECASE)
        if not m_model or not m_rate:
            continue
        name = m_model.group(1).strip().strip("\"'")

        name = name.split("/", 1)[-1].strip().lower()
        try:
            rate = float(m_rate.group(1))
        except ValueError:
            continue
        if rate <= 0:
            continue
        out.append((name, rate))
    return out


async def fetch_aider_polyglot_scores(client: httpx.AsyncClient) -> dict[str, float]:

    scores: dict[str, float] = {}
    resp = await get_with_retries(client, AIDER_POLYGLOT_YML_URL)
    resp.raise_for_status()
    pairs = parse_yaml_lite(resp.text)
    if not pairs:
        logger.debug("Coding benchmark: 0 records parsed")
        return {}
    best_by_name: dict[str, float] = {}
    for name, rate in pairs:
        cur = best_by_name.get(name)
        if cur is None or rate > cur:
            best_by_name[name] = rate
    for name, rate in best_by_name.items():
        ids = AIDER_NAME_TO_HF_IDS.get(name)
        if not ids:
            continue
        normalized = normalize(rate)
        if normalized <= 0:
            continue
        for hf_id in ids:
            if scores.get(hf_id, 0.0) < normalized:
                scores[hf_id] = normalized
    logger.debug(f"Coding benchmark: {len(scores)} mapped scores")
    return scores
