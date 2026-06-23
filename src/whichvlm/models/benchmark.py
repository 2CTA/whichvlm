from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import statistics
import time
from dataclasses import dataclass

import httpx

from whichvlm.models.cache_format import (
    cache_expired,
    cache_snapshot_metadata,
    read_cache_payload,
)
from whichvlm.utils import cache_dir, current_version

# Benchmark merge layer. Blends current and fallback score sources.
logger = logging.getLogger(__name__)

CACHE_DIR = cache_dir()
BENCHMARK_CACHE = CACHE_DIR / "benchmark.json"
DEFAULT_TTL_SECONDS = 24 * 3600
BENCHMARK_CACHE_SCHEMA_VERSION = 2
BENCHMARK_SOURCE_PROVENANCE = {
    "name": "benchmark_index",
    "sources": [
        "aa_index",
        "livebench",
        "vision",
        "chatbot_arena",
        "aider_polyglot",
        "open_llm_leaderboard",
    ],
}

VLM_BENCHMARK_ID_RE = re.compile(
    r"(?:^|[-_/])(vl|vision|multimodal|llava|pixtral|image)(?:[-_/]|$)|"
    r"internvl\d*|"
    r"qwen.*vl|deepseek[-_]?vl|glm-?4v|glm-?4\.5v",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class BenchmarkEvidence:
    # Evidence record. Carries score plus trust level for one match.
    score: float | None
    confidence: float
    source: str


def looks_like_vlm_id(model_id: str) -> bool:
    return bool(VLM_BENCHMARK_ID_RE.search(model_id))


def benchmark_confidence(model_id: str, source: str, default: float) -> float:
    if not looks_like_vlm_id(model_id):
        return default
    vlm_confidence = {
        "direct": 0.88,
        "variant": 0.48,
        "base_model": 0.52,
        "line_interp": default * 0.75,
        "self_reported": 0.30,
    }
    return vlm_confidence.get(source, default)


def load_benchmark_cache(*, allow_stale: bool = False) -> dict[str, float] | None:
    # Cache read. Reuses merged benchmark scores until ttl expires.
    payload = read_cache_payload(BENCHMARK_CACHE)
    if payload is None:
        return None
    try:
        if cache_expired(
            payload["cached_at"], DEFAULT_TTL_SECONDS, allow_stale=allow_stale
        ):
            logger.debug("Benchmark cache expired")
            return None
        scores = payload["scores"]
        if not isinstance(scores, dict):
            return None
        return scores
    except (KeyError, TypeError) as e:
        logger.debug(f"Benchmark cache corrupted: {e}")
        return None


def benchmark_cache_snapshot() -> dict | None:
    payload = read_cache_payload(BENCHMARK_CACHE)
    if payload is None:
        return None
    return cache_snapshot_metadata(
        payload,
        default_ttl_seconds=DEFAULT_TTL_SECONDS,
        item_key="scores",
        item_count_key="score_count",
        default_source=BENCHMARK_SOURCE_PROVENANCE,
    )


def save_benchmark_cache(scores: dict[str, float]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": BENCHMARK_CACHE_SCHEMA_VERSION,
        "cached_at": time.time(),
        "ttl_seconds": DEFAULT_TTL_SECONDS,
        "source": BENCHMARK_SOURCE_PROVENANCE,
        "scores": scores,
    }
    BENCHMARK_CACHE.write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )
    logger.debug(f"Saved {len(scores)} benchmark scores to cache")


LINEAGE_REGEX_CACHE = None


def lineage_regex_table():
    # Regex cache. Compiles family lineage rules once for generation math.
    global LINEAGE_REGEX_CACHE
    if LINEAGE_REGEX_CACHE is not None:
        return LINEAGE_REGEX_CACHE
    from whichvlm.constants import MODEL_LINEAGE_VERSIONS

    out = {}
    for family, entries in MODEL_LINEAGE_VERSIONS.items():
        compiled = [(re.compile(pat), idx) for pat, idx in entries]
        max_idx = max(idx for _, idx in entries)
        out[family] = (compiled, max_idx)
    LINEAGE_REGEX_CACHE = out
    return out


def lineage_recency_factor(model_id: str) -> float:

    if not model_id:
        return 1.0
    lower = model_id.lower()
    families = lineage_regex_table()
    best_factor = 1.0
    for family, (patterns, max_idx) in families.items():
        for regex, idx in patterns:
            if regex.search(lower):
                gens_old = max(0, max_idx - idx)
                factor = max(0.55, 1.0 - 0.12 * gens_old)
                if factor < best_factor:
                    best_factor = factor
                break
    return best_factor


def apply_lineage_recency_demotion(
    combined: dict[str, float],
    frozen: dict[str, float],
    current: dict[str, float],
) -> dict[str, float]:

    if not combined:
        return combined
    out: dict[str, float] = {}
    for k, v in combined.items():
        if k in current:
            out[k] = v
            continue
        factor = lineage_recency_factor(k)
        out[k] = round(v * factor, 1)
    return out


async def fetch_benchmark_scores() -> dict[str, float]:
    # Main fetch. Pulls every benchmark source and merges trust tiers.
    from whichvlm.models.benchmark_sources import (
        fetch_aa_index_scores,
        fetch_aider_polyglot_scores,
        fetch_arena_scores,
        fetch_leaderboard_with_fallback,
        fetch_vision_scores,
        get_aa_curated_fallback,
        get_livebench_data,
    )

    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        client.headers["User-Agent"] = f"whichvlm/{current_version()}"
        leaderboard_task = asyncio.create_task(fetch_leaderboard_with_fallback(client))
        arena_task = asyncio.create_task(fetch_arena_scores(client))
        aa_task = asyncio.create_task(fetch_aa_index_scores(client))
        aider_task = asyncio.create_task(fetch_aider_polyglot_scores(client))
        vision_task = asyncio.create_task(fetch_vision_scores(client))

        (
            lb_result,
            arena_result,
            aa_result,
            aider_result,
            vision_result,
        ) = await asyncio.gather(
            leaderboard_task,
            arena_task,
            aa_task,
            aider_task,
            vision_task,
            return_exceptions=True,
        )


    frozen: dict[str, float] = {}
    current: dict[str, float] = {}


    if isinstance(lb_result, BaseException):
        logger.warning(f"Leaderboard fetch failed: {lb_result}")
    else:
        frozen.update(lb_result)
        logger.debug(f"Leaderboard: {len(lb_result)} scores (frozen)")


    if isinstance(arena_result, BaseException):
        logger.warning(f"secondary benchmark fetch failed, using fallback: {arena_result}")
    else:
        for k, v in arena_result.items():
            if frozen.get(k, 0.0) < v:
                frozen[k] = v
        logger.debug(f"secondary benchmark: {len(arena_result)} scores (frozen)")


    livebench_result = get_livebench_data()
    for k, v in livebench_result.items():
        if current.get(k, 0.0) < v:
            current[k] = v
    logger.debug(f"vision index: {len(livebench_result)} scores (current)")


    if isinstance(aa_result, BaseException):
        logger.warning(f"capability index fetch failed, using fallback: {aa_result}")
        aa_result = get_aa_curated_fallback()

    for k, v in aa_result.items():
        if current.get(k, 0.0) < v:
            current[k] = v
    logger.debug(f"capability index: {len(aa_result)} scores (current)")


    if isinstance(aider_result, BaseException):
        logger.warning(f"coding benchmark fetch failed: {aider_result}")
    else:
        for k, v in aider_result.items():
            if current.get(k, 0.0) < v * 0.85:
                current[k] = v * 0.85
        logger.debug(f"coding benchmark: {len(aider_result)} scores (current, 0.85x)")


    if isinstance(vision_result, BaseException):
        logger.warning(f"Vision fetch failed: {vision_result}")
    else:
        for k, v in vision_result.items():
            if looks_like_vlm_id(k) or current.get(k, 0.0) < v:
                current[k] = v
        logger.debug(f"Vision: {len(vision_result)} scores (current)")


    combined: dict[str, float] = dict(frozen)
    combined.update(current)


    combined = apply_lineage_recency_demotion(combined, frozen, current)

    logger.debug(f"Combined: {len(combined)} benchmark scores")
    return combined


def extract_params_b_from_id(model_id: str) -> float | None:
    lower = model_id.lower()
    matches = re.findall(r"(\d+(?:\.\d+)?)b(?:-a\d+(?:\.\d+)?b)?", lower)
    if not matches:
        return None
    try:
        return max(float(v) for v in matches)
    except ValueError:
        return None


def extract_model_lines(model_id: str) -> list[str]:

    if "/" not in model_id:
        return []
    lower = model_id.lower()


    stripped = re.sub(r"-(gguf|awq|gptq|fp8|fp16|bf16|mxfp4|nvfp4)$", "", lower)
    stripped = re.sub(r"-\d{4}(-hf)?$", "", stripped)

    lines: list[str] = []


    cleaned = re.sub(
        r"-\d+(\.\d+)?b(-a\d+b)?(-[a-z][-a-z0-9]*)*$",
        "",
        stripped,
    )
    if cleaned != stripped and "/" in cleaned:
        lines.append(cleaned)


    for line in list(lines) + ([stripped] if not lines else []):
        broader = re.sub(r"(\d+)\.\d+$", r"\1", line)
        if broader != line and broader not in lines:
            lines.append(broader)

    return lines


def interpolate_line_score(
    bucket: list[tuple[float | None, float]],
    params_b: float | None,
) -> tuple[float, float]:

    if not bucket:
        return 0.0, 0.0

    valid = [(p, s) for p, s in bucket if p is not None]
    if not valid:
        vals = [s for _, s in bucket]
        return statistics.median(vals), 0.25

    if params_b is None or params_b <= 0:
        vals = [s for _, s in valid]
        return statistics.median(vals), 0.30

    weighted: list[tuple[float, float, float]] = []
    for p, s in valid:
        assert p is not None
        dist = abs(math.log2(max(params_b, 0.1) / max(p, 0.1)))
        w = 1.0 / (0.35 + dist)
        weighted.append((w, s, dist))

    score = sum(w * s for w, s, _ in weighted) / sum(w for w, _, _ in weighted)
    nearest = min(d for _, _, d in weighted)
    if nearest <= 0.15:
        conf = 0.45
    elif nearest <= 0.50:
        conf = 0.34
    else:
        conf = 0.26
    return score, conf


def build_score_index(
    scores: dict[str, float],
) -> tuple[dict[str, float], dict[str, float]]:

    ci_index: dict[str, float] = {}
    line_index: dict[str, float] = {}

    for key, val in scores.items():
        lk = key.lower()
        if lk not in ci_index or val > ci_index[lk]:
            ci_index[lk] = val

        lines = extract_model_lines(key)
        if not lines and "/" in key:

            lines = [lk]
        for line in lines:
            if line not in line_index or val > line_index[line]:
                line_index[line] = val

    return ci_index, line_index


def build_line_bucket_index(
    scores: dict[str, float],
) -> dict[str, list[tuple[float | None, float]]]:

    buckets: dict[str, list[tuple[float | None, float]]] = {}
    for key, val in scores.items():
        params_b = extract_params_b_from_id(key)
        lines = extract_model_lines(key)
        if not lines and "/" in key:
            lines = [key.lower()]
        for line in lines:
            buckets.setdefault(line, []).append((params_b, val))
    return buckets


def lookup_score(
    candidate: str, scores: dict[str, float], ci_index: dict[str, float]
) -> float | None:
    if candidate in scores:
        return scores[candidate]
    lc = candidate.lower()
    if lc in ci_index:
        return ci_index[lc]
    return None


REPO_SUFFIXES = ("-GGUF", "-gguf", "-AWQ", "-GPTQ", "-FP8", "-fp8", "-BF16", "-bf16")


def benchmark_id_candidates(model_id: str) -> list[str]:
    candidates = [model_id]


    for suffix in REPO_SUFFIXES:
        if model_id.endswith(suffix):
            candidates.append(model_id[: -len(suffix)])
            break


    base = candidates[-1]
    if base.endswith("-Instruct"):
        candidates.append(base[: -len("-Instruct")])
    else:
        candidates.append(base + "-Instruct")

    return candidates


def append_if_missing(candidates: list[str], candidate: str) -> None:
    if candidate and candidate not in candidates:
        candidates.append(candidate)


def strip_repo_suffix(model_id: str) -> str:
    for suffix in REPO_SUFFIXES:
        if model_id.endswith(suffix):
            return model_id[: -len(suffix)]
    return model_id


def generate_score_name_candidates(
    model_id: str, scores: dict[str, float]
) -> list[str]:

    stripped = strip_repo_suffix(model_id)
    repo_name = stripped.rsplit("/", 1)[-1]
    model_names = [repo_name]

    explicit_candidates: list[str] = []
    if "_" in repo_name:
        org, name = repo_name.split("_", 1)
        if org and name:
            append_if_missing(explicit_candidates, f"{org}/{name}")
            append_if_missing(model_names, name)

    score_candidates: list[str] = []
    wanted_names = {name.lower() for name in model_names if name}
    for score_id in scores:
        score_name = score_id.rsplit("/", 1)[-1].lower()
        if score_name in wanted_names:
            append_if_missing(score_candidates, score_id)

    return explicit_candidates + [
        candidate
        for candidate in score_candidates
        if candidate not in explicit_candidates
    ]


def lookup_benchmark(
    model_id: str,
    base_model: str | None,
    scores: dict[str, float],
    ci_index: dict[str, float] | None = None,
    line_index: dict[str, float] | None = None,
) -> tuple[float, bool] | None:

    evidence = lookup_benchmark_evidence(
        model_id,
        base_model,
        scores,
        ci_index=ci_index,
        line_index=line_index,
    )
    if evidence.score is None:
        return None
    return evidence.score, evidence.source == "direct"


def params_compatible(actual_b: float | None, ref_id: str) -> bool:

    if actual_b is None or actual_b <= 0:
        return True
    ref_b = extract_params_b_from_id(ref_id)
    if ref_b is None or ref_b <= 0:
        return True
    ratio = actual_b / ref_b
    return 0.5 <= ratio <= 2.0


def lookup_benchmark_evidence(
    model_id: str,
    base_model: str | None,
    scores: dict[str, float],
    ci_index: dict[str, float] | None = None,
    line_index: dict[str, float] | None = None,
    line_bucket_index: dict[str, list[tuple[float | None, float]]] | None = None,
    self_reported_score: float | None = None,
    actual_params_b: float | None = None,
) -> BenchmarkEvidence:
    # Evidence lookup. Returns score plus how that score was inferred.

    if ci_index is None or line_index is None:
        ci_index, line_index = build_score_index(scores)
    if line_bucket_index is None:
        line_bucket_index = build_line_bucket_index(scores)


    direct_result = lookup_score(model_id, scores, ci_index)
    if direct_result is not None:
        return BenchmarkEvidence(
            score=direct_result,
            confidence=benchmark_confidence(model_id, "direct", 1.0),
            source="direct",
        )


    variant_candidates = benchmark_id_candidates(model_id)[1:]
    for candidate in generate_score_name_candidates(model_id, scores):
        append_if_missing(variant_candidates, candidate)
    for candidate in variant_candidates:
        result = lookup_score(candidate, scores, ci_index)
        if result is not None:
            if not params_compatible(actual_params_b, candidate):
                continue
            return BenchmarkEvidence(
                score=result,
                confidence=benchmark_confidence(model_id, "variant", 0.55),
                source="variant",
            )


    if base_model:
        for candidate in benchmark_id_candidates(base_model):
            result = lookup_score(candidate, scores, ci_index)
            if result is not None:
                if not params_compatible(actual_params_b, candidate):
                    continue
                return BenchmarkEvidence(
                    score=result,
                    confidence=benchmark_confidence(model_id, "base_model", 0.60),
                    source="base_model",
                )


    size_hint = (
        actual_params_b
        or extract_params_b_from_id(model_id)
        or extract_params_b_from_id(base_model or "")
    )
    for mid in (model_id, base_model):
        if mid:
            for line in extract_model_lines(mid):
                if line in line_bucket_index:
                    score, conf = interpolate_line_score(
                        line_bucket_index[line], size_hint
                    )
                    if score > 0:
                        return BenchmarkEvidence(
                            score=score,
                            confidence=benchmark_confidence(
                                model_id, "line_interp", conf
                            ),
                            source="line_interp",
                        )
                if line in line_index:
                    return BenchmarkEvidence(
                        score=line_index[line],
                        confidence=benchmark_confidence(
                            model_id, "line_interp", 0.22
                        ),
                        source="line_interp",
                    )


    if (
        self_reported_score is not None
        and isinstance(self_reported_score, (int, float))
        and self_reported_score > 0
    ):
        return BenchmarkEvidence(
            score=float(self_reported_score),
            confidence=benchmark_confidence(model_id, "self_reported", 0.40),
            source="self_reported",
        )

    return BenchmarkEvidence(score=None, confidence=0.0, source="none")
