from __future__ import annotations

import json
import logging
import re

import httpx

from models.http import get_with_retries

logger = logging.getLogger(__name__)

NEXT_DATA_RE = re.compile(
    r'<script id="__NEXT_DATA__"[^>]*>(?P<json>.*?)</script>', re.DOTALL
)


class ExtractionFailed(Exception):
    pass


def walk_json_dicts(obj, depth: int = 0):
    if depth > 12:
        return
    if isinstance(obj, dict):
        yield obj
        for value in obj.values():
            yield from walk_json_dicts(value, depth + 1)
    elif isinstance(obj, list):
        for item in obj:
            yield from walk_json_dicts(item, depth + 1)


AA_NAME_TO_HF_IDS: dict[str, list[str]] = {
    "Kimi K2": ["moonshotai/Kimi-K2-Instruct", "moonshotai/Kimi-K2-Base"],
    "Kimi K2-Thinking": ["moonshotai/Kimi-K2-Thinking"],
    "DeepSeek V3": ["deepseek-ai/DeepSeek-V3", "deepseek-ai/DeepSeek-V3-0324"],
    "DeepSeek V3.1": ["deepseek-ai/DeepSeek-V3.1"],
    "DeepSeek V3.2": ["deepseek-ai/DeepSeek-V3.2"],
    "DeepSeek V3.2-Exp": ["deepseek-ai/DeepSeek-V3.2-Exp"],
    "DeepSeek V4 Pro": ["deepseek-ai/DeepSeek-V4-Pro"],
    "DeepSeek V4 Flash": ["deepseek-ai/DeepSeek-V4-Flash"],
    "DeepSeek R1": ["deepseek-ai/DeepSeek-R1"],
    "DeepSeek R1-0528": ["deepseek-ai/DeepSeek-R1-0528"],
    "DeepSeek R1-Distill 32B": ["deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"],
    "DeepSeek R1-Distill 14B": ["deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"],
    "DeepSeek R1-Distill 8B": ["deepseek-ai/DeepSeek-R1-Distill-Llama-8B"],
    "QwQ 32B": ["Qwen/QwQ-32B"],
    "Qwen3 4B Thinking": ["Qwen/Qwen3-4B-Thinking-2507"],
    "MiMo V2.5": ["XiaomiMiMo/MiMo-V2.5"],
    "MiMo V2.5 Pro": ["XiaomiMiMo/MiMo-V2.5-Pro"],
    "MiMo V2 Flash": ["XiaomiMiMo/MiMo-V2-Flash"],
    "GLM-4.5": ["zai-org/GLM-4.5", "zai-org/GLM-4.5-Air"],
    "GLM-4.6": ["zai-org/GLM-4.6"],
    "GLM-4.7": ["zai-org/GLM-4.7"],
    "GLM-4.7-Flash": ["zai-org/GLM-4.7-Flash"],
    "GLM-5": ["zai-org/GLM-5", "zai-org/GLM-5-FP8"],
    "GLM-5.1": ["zai-org/GLM-5.1", "zai-org/GLM-5.1-FP8"],
    "gpt-oss-20b": ["openai/gpt-oss-20b"],
    "gpt-oss-120b": ["openai/gpt-oss-120b"],
    "Qwen3-Next 80B-A3B": ["Qwen/Qwen3-Next-80B-A3B-Instruct"],
    "Qwen3.5 397B-A17B": ["Qwen/Qwen3.5-397B-A17B"],
    "Qwen3 235B-A22B": ["Qwen/Qwen3-235B-A22B"],
    "Qwen3 32B": ["Qwen/Qwen3-32B"],
    "Qwen3 14B": ["Qwen/Qwen3-14B"],
    "Qwen3 8B": ["Qwen/Qwen3-8B"],
    "Qwen3-VL 235B-A22B": ["Qwen/Qwen3-VL-235B-A22B-Instruct"],
    "Llama 3.3 70B": ["meta-llama/Llama-3.3-70B-Instruct"],
    "Llama 4 Scout": ["meta-llama/Llama-4-Scout-17B-16E-Instruct"],
    "Llama 4 Maverick": ["meta-llama/Llama-4-Maverick-17B-128E-Instruct"],
    "Gemma 3 27B": ["google/gemma-3-27b-it"],
    "Gemma 3 12B": ["google/gemma-3-12b-it"],
    "Gemma 4 31B": ["google/gemma-4-31b-it"],
    "Gemma 4 26B-A4B": ["google/gemma-4-26b-a4b-it"],
    "Mistral Large 2": ["mistralai/Mistral-Large-Instruct-2411"],
    "Devstral Small": ["mistralai/Devstral-Small-2505"],
    "Phi-4": ["microsoft/phi-4"],
    "Command A": ["CohereForAI/c4ai-command-a-03-2025"],
    "Command R+": [
        "CohereForAI/c4ai-command-r-plus-08-2024",
        "CohereForAI/c4ai-command-r-plus",
    ],
    "MiniMax-M2": ["MiniMaxAI/MiniMax-M2"],
    "MiniMax-M2.5": ["MiniMaxAI/MiniMax-M2.5"],
    "Nemotron 3 Super 120B-A12B": ["nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"],
    "Nemotron 3 Nano 30B-A3B": [
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8",
    ],
}


AA_INDEX_MIN = 12.5
AA_INDEX_MAX = 56.2

AA_LEADERBOARD_URL = "https://artificialanalysis.ai/leaderboards/models"


AA_INDEX_FALLBACK_2026_05_14: dict[str, float] = {
    "moonshotai/Kimi-K2-Thinking": 50.0,
    "moonshotai/Kimi-K2-Instruct": 47.0,
    "XiaomiMiMo/MiMo-V2.5-Pro": 54.0,
    "XiaomiMiMo/MiMo-V2.5": 49.0,
    "deepseek-ai/DeepSeek-V4-Pro": 52.0,
    "deepseek-ai/DeepSeek-V4-Flash": 47.0,
    "deepseek-ai/DeepSeek-V3.2": 45.0,
    "deepseek-ai/DeepSeek-V3.2-Exp": 44.0,
    "deepseek-ai/DeepSeek-V3.1": 42.0,
    "deepseek-ai/DeepSeek-V3-0324": 40.0,
    "deepseek-ai/DeepSeek-V3": 38.0,
    "deepseek-ai/DeepSeek-R1-0528": 48.0,
    "deepseek-ai/DeepSeek-R1": 43.0,
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": 32.0,
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B": 26.0,
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": 20.0,
    "Qwen/QwQ-32B": 36.0,
    "Qwen/Qwen3-4B-Thinking-2507": 22.0,
    "zai-org/GLM-5.1": 51.0,
    "zai-org/GLM-5": 50.0,
    "zai-org/GLM-5-FP8": 50.0,
    "zai-org/GLM-5.1-FP8": 51.0,
    "zai-org/GLM-4.7-Flash": 42.0,
    "zai-org/GLM-4.6": 40.0,
    "zai-org/GLM-4.5": 38.0,
    "zai-org/GLM-4.5-Air": 36.0,
    "Qwen/Qwen3.6-27B": 46.0,
    "Qwen/Qwen3.5-397B-A17B": 45.0,
    "Qwen/Qwen3-Next-80B-A3B-Instruct": 42.0,
    "Qwen/Qwen3-235B-A22B": 41.0,
    "Qwen/Qwen3-Coder-30B-A3B-Instruct": 38.0,
    "Qwen/Qwen3-32B": 37.0,
    "Qwen/Qwen3-14B": 33.0,
    "Qwen/Qwen3-8B": 30.0,
    "Qwen/Qwen3-4B-Instruct-2507": 28.0,
    "Qwen/Qwen3-4B": 26.0,
    "Qwen/Qwen3-1.7B": 20.0,
    "Qwen/Qwen3-0.6B": 16.0,
    "meta-llama/Llama-3.1-8B-Instruct": 22.0,
    "meta-llama/Meta-Llama-3-8B-Instruct": 20.0,
    "google/gemma-2-9b-it": 23.0,
    "microsoft/Phi-4-mini-instruct": 24.0,
    "mistralai/Mistral-7B-Instruct-v0.3": 20.0,
    "Qwen/Qwen2.5-7B-Instruct": 22.0,
    "Qwen/Qwen2.5-14B-Instruct": 26.0,
    "Qwen/Qwen2.5-32B-Instruct": 30.0,
    "Qwen/Qwen3-30B-A3B": 32.0,
    "openai/gpt-oss-120b": 41.0,
    "openai/gpt-oss-20b": 34.0,
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct": 38.0,
    "meta-llama/Llama-4-Scout-17B-16E-Instruct": 34.0,
    "meta-llama/Llama-3.3-70B-Instruct": 33.0,
    "google/gemma-4-31b-it": 38.0,
    "google/gemma-4-26b-a4b-it": 36.0,
    "google/gemma-3-27b-it": 33.0,
    "google/gemma-3-12b-it": 30.0,
    "microsoft/phi-4": 33.0,
    "mistralai/Mistral-Large-Instruct-2411": 35.0,
    "mistralai/Mistral-Small-3.2-24B-Instruct-2506": 32.0,
    "mistralai/Mistral-Small-3.1-24B-Instruct-2503": 30.0,
    "mistralai/Devstral-Small-2505": 33.0,
    "MiniMaxAI/MiniMax-M2.5": 40.0,
    "stepfun-ai/Step-3.5-Flash": 38.0,
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16": 36.0,
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16": 33.0,
    "allenai/Olmo-3-7B-Instruct": 22.0,
    "allenai/Olmo-3-1025-7B": 22.0,
    "ibm-granite/granite-4.0-h-small": 30.0,
    "ibm-granite/granite-4.0-h-tiny": 22.0,
    "ibm-granite/granite-3.3-8b-instruct": 23.0,
    "ibm-granite/granite-3.3-2b-instruct": 17.0,
    "mistralai/Codestral-22B-v0.1": 28.0,
}


def normalize_aa_index(index: float) -> float:
    if not isinstance(index, (int, float)):
        return 0.0
    span = AA_INDEX_MAX - AA_INDEX_MIN
    normalized = (index - AA_INDEX_MIN) / span * 100.0
    return max(0.0, min(100.0, round(normalized, 1)))


RSC_CHUNK_RE = re.compile(r'self\.__next_f\.push\(\[\d+,(?P<s>"(?:[^"\\]|\\.)*")\]\)')


AA_RECORD_RE = re.compile(
    r'"name":"(?P<name>(?:[^"\\]|\\.)*)"'
    r'(?:(?!"name":").)*?'
    r'"intelligenceIndex":(?P<idx>-?\d+(?:\.\d+)?)',
    re.DOTALL,
)


PAREN_RE = re.compile(r"\([^)]*\)")


def canonical_name(name: str) -> str:
    name = PAREN_RE.sub("", name)
    name = name.lower().replace("-", " ").replace("_", " ")
    return re.sub(r"\s+", " ", name).strip()


AA_CANON_TO_HF_IDS: dict[str, list[str]] = {}
for display_name, hf_ids in AA_NAME_TO_HF_IDS.items():
    AA_CANON_TO_HF_IDS.setdefault(canonical_name(display_name), []).extend(hf_ids)


def decode_rsc_blob(html: str) -> str:
    parts: list[str] = []
    for m in RSC_CHUNK_RE.finditer(html):
        try:
            parts.append(json.loads(m.group("s")))
        except (ValueError, json.JSONDecodeError):
            continue
    return "".join(parts)


def extract_aa_pairs_from_html(html: str) -> list[tuple[str, float]]:
    blob = decode_rsc_blob(html)
    if not blob:
        return []
    pairs: list[tuple[str, float]] = []
    for m in AA_RECORD_RE.finditer(blob):
        try:
            name = json.loads('"' + m.group("name") + '"').strip()
            score = float(m.group("idx"))
        except (ValueError, json.JSONDecodeError):
            continue
        if name and score > 0:
            pairs.append((name, score))
    return pairs


def extract_aa_pairs(payload: dict) -> list[tuple[str, float]]:
    pairs: list[tuple[str, float]] = []
    for node in walk_json_dicts(payload):
        name = None
        score = None
        for name_key in ("model_name", "modelName", "name", "displayName"):
            v = node.get(name_key)
            if isinstance(v, str) and v.strip():
                name = v.strip()
                break
        for score_key in (
            "intelligence_index",
            "intelligenceIndex",
            "aa_index",
            "aaIndex",
            "score",
        ):
            v = node.get(score_key)
            if isinstance(v, (int, float)):
                score = float(v)
                break
        if name and score is not None and score > 0:
            pairs.append((name, score))
    return pairs


async def fetch_aa_index_scores(client: httpx.AsyncClient) -> dict[str, float]:
    resp = await get_with_retries(client, AA_LEADERBOARD_URL)
    resp.raise_for_status()

    pairs = extract_aa_pairs_from_html(resp.text)

    if not pairs:
        match = NEXT_DATA_RE.search(resp.text)
        if match:
            try:
                pairs = extract_aa_pairs(json.loads(match.group("json")))
            except (ValueError, json.JSONDecodeError):
                pairs = []
    if not pairs:
        raise ExtractionFailed(
            "frontier index: no (name, score) pairs found "
            "(neither RSC __next_f nor __NEXT_DATA__ matched)"
        )

    best_by_name: dict[str, float] = {}
    for name, score in pairs:
        current = best_by_name.get(name)
        if current is None or score > current:
            best_by_name[name] = score

    live: dict[str, float] = {}
    for name, score in best_by_name.items():
        hf_ids = AA_NAME_TO_HF_IDS.get(name) or AA_CANON_TO_HF_IDS.get(
            canonical_name(name)
        )
        if not hf_ids:
            continue
        normalized = normalize_aa_index(score)
        if normalized <= 0:
            continue
        for hf_id in hf_ids:
            if normalized > live.get(hf_id, 0.0):
                live[hf_id] = normalized
    if not live:
        raise ExtractionFailed("frontier index: live fetch returned 0 mapped scores")

    scores = get_aa_curated_fallback()
    for hf_id, normalized in live.items():
        if normalized > scores.get(hf_id, 0.0):
            scores[hf_id] = normalized
    logger.debug(f"frontier index: {len(live)} live + {len(scores)} merged scores")
    return scores


def get_aa_curated_fallback() -> dict[str, float]:
    result: dict[str, float] = {}
    for hf_id, raw in AA_INDEX_FALLBACK_2026_05_14.items():
        normalized = normalize_aa_index(raw)
        if normalized > 0:
            result[hf_id] = normalized
    return result
