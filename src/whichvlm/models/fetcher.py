"""HuggingFace Hub API model fetcher."""

from __future__ import annotations

import asyncio
import logging
import re
import statistics

import httpx

from whichvlm.constants import QUANT_BYTES_PER_WEIGHT
from whichvlm.data.vlm_inventory import known_vlm_model_ids
from whichvlm.models.http import get_with_retries
from whichvlm.models.package_graph import (
    artifact_from_dict,
    artifact_to_dict,
    build_artifacts,
    build_components,
    build_lineage,
    component_from_dict,
    component_to_dict,
    infer_variant_kind,
    is_projector_filename,
    lineage_from_dict,
    lineage_to_dict,
    looks_quantized_repo_name,
)
from whichvlm.models.types import GGUFVariant, ModelInfo

logger = logging.getLogger(__name__)

HF_API_BASE = "https://huggingface.co/api"
_GGUF_SPLIT_RE = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$", re.IGNORECASE)
_GENERAL_EVAL_KEYWORDS = (
    "mmlu",
    "gpqa",
    "gsm8k",
    "hellaswag",
    "arc",
    "bbh",
    "ifeval",
    "truthfulqa",
    "ceval",
    "cmmlu",
)
_VLM_PIPELINE_TAGS = (
    "image-text-to-text",
    "visual-question-answering",
    "image-to-text",
)
_VLM_VARIANT_FILTERS = (None, "gguf", "mlx", "awq", "gptq", "bnb", "fp8")
_HF_MODEL_EXPAND = (
    "config",
    "safetensors",
    "gguf",
    "cardData",
    "siblings",
    "evalResults",
)
_HF_MODEL_DETAIL_EXPAND = (
    *_HF_MODEL_EXPAND,
    "downloads",
    "likes",
    "createdAt",
    "lastModified",
)
_MODEL_LIST_CONCURRENCY = 8
_MODEL_DETAIL_CONCURRENCY = 6
_OFFICIAL_MODEL_ORGS = frozenset(
    {
        "Qwen",
        "meta-llama",
        "google",
        "mistralai",
        "deepseek-ai",
        "microsoft",
        "nvidia",
        "apple",
        "CohereForAI",
        "openai",
        "zai-org",
        "moonshotai",
        "MiniMaxAI",
        "XiaomiMiMo",
        "OpenGVLab",
        "Salesforce",
        "BAAI",
    }
)


def _extract_published_at(data: dict) -> str | None:
    """APIレスポンスから公開日時候補を取り出す。"""
    created = data.get("createdAt")
    if isinstance(created, str) and created:
        return created
    modified = data.get("lastModified")
    if isinstance(modified, str) and modified:
        return modified
    return None


def _normalize_eval_value(raw: object) -> float | None:
    """Convert eval value to a comparable 0-100 score."""
    if not isinstance(raw, (int, float)):
        return None
    value = float(raw)
    if value <= 0:
        return None
    if value <= 1.0:
        value *= 100.0
    if value > 100.0:
        return None
    return value


def _is_general_eval_entry(entry: dict) -> bool:
    """Keep eval entries that are broadly useful for general chat quality."""
    data = entry.get("data")
    if not isinstance(data, dict):
        return False

    notes = str(data.get("notes", "")).lower()
    # ツール利用前提の数値はローカル推論の比較軸として混ざりやすいため除外する。
    if "with tools" in notes:
        return False

    dataset = data.get("dataset")
    dataset_id = ""
    task_id = ""
    if isinstance(dataset, dict):
        dataset_id = str(dataset.get("id", "")).lower()
        task_id = str(dataset.get("task_id", "")).lower()
    filename = str(entry.get("filename", "")).lower()

    return any(
        k in dataset_id or k in task_id or k in filename for k in _GENERAL_EVAL_KEYWORDS
    )


def _extract_hf_eval_score(data: dict) -> float | None:
    """Extract conservative aggregate score from HF evalResults."""
    eval_results = data.get("evalResults")
    if not isinstance(eval_results, list) or not eval_results:
        return None

    values: list[float] = []
    for entry in eval_results:
        if not isinstance(entry, dict):
            continue
        if not _is_general_eval_entry(entry):
            continue
        data_obj = entry.get("data")
        if not isinstance(data_obj, dict):
            continue
        normalized = _normalize_eval_value(data_obj.get("value"))
        if normalized is not None:
            values.append(normalized)

    if not values:
        return None
    return round(statistics.median(values), 1)


def _extract_size_hint_from_id(model_id: str | None) -> int | None:
    """Extract parameter size hint (in params) from model ID like 27B or 30B-A3B."""
    if not model_id:
        return None
    lower = model_id.lower()
    matches = re.findall(r"(\d+(?:\.\d+)?)b(?:-a\d+(?:\.\d+)?b)?", lower)
    if not matches:
        return None
    try:
        max_b = max(float(m) for m in matches)
    except ValueError:
        return None
    if max_b <= 0:
        return None
    return int(max_b * 1e9)


def _extract_active_size_hint_from_id(model_id: str | None) -> int | None:
    """Extract MoE active parameter hint from names like 35B-A3B."""
    if not model_id:
        return None
    lower = model_id.lower()
    matches = re.findall(r"\d+(?:\.\d+)?b[-_]?a(\d+(?:\.\d+)?)b", lower)
    if not matches:
        return None
    try:
        max_b = max(float(m) for m in matches)
    except ValueError:
        return None
    if max_b <= 0:
        return None
    return int(max_b * 1e9)


def _extract_tags(data: dict) -> list[str]:
    raw_tags = data.get("tags", [])
    if not isinstance(raw_tags, list):
        return []
    return [str(tag) for tag in raw_tags if isinstance(tag, str)]


def _extract_base_models(card_data: dict) -> list[str]:
    raw = card_data.get("base_model")
    if isinstance(raw, str) and raw:
        return [raw]
    if isinstance(raw, list):
        return [v for v in raw if isinstance(v, str) and v]
    return []


def _extract_access(data: dict) -> str:
    gated = data.get("gated")
    if gated is True:
        return "gated"
    if gated is False:
        return "ungated"
    if isinstance(gated, str):
        value = gated.lower()
        if value in {"true", "auto", "manual"}:
            return "gated"
        if value in {"false", "none"}:
            return "ungated"
    return "unknown"


def _is_official_model(model_id: str) -> bool:
    org = model_id.split("/", 1)[0] if "/" in model_id else ""
    return org in _OFFICIAL_MODEL_ORGS


def _infer_repo_quantization(model_id: str, tags: list[str]) -> str | None:
    tag_set = {tag.lower() for tag in tags}
    if "gguf" in tag_set:
        return "GGUF"
    if "mlx" in tag_set:
        return "MLX"

    haystack = " ".join([model_id, *tags]).lower()
    patterns = (
        ("AWQ", r"(^|[-_/\s])awq($|[-_/\s])"),
        ("GPTQ", r"(^|[-_/\s])gptq($|[-_/\s])"),
        ("BNB_4BIT", r"(bnb[-_/]?4bit|nf4|4bit)"),
        ("INT8", r"(int8|8bit)"),
        ("FP8", r"(^|[-_/\s])fp8($|[-_/\s])"),
        ("MXFP4", r"(^|[-_/\s])mxfp4($|[-_/\s])"),
        ("NVFP4", r"(^|[-_/\s])nvfp4($|[-_/\s])"),
        ("MLX", r"(^|[-_/\s])mlx($|[-_/\s])"),
        ("GGUF", r"(^|[-_/\s])gguf($|[-_/\s])"),
    )
    for quant, pattern in patterns:
        if re.search(pattern, haystack):
            return quant
    return None


def _infer_model_format(
    model_id: str,
    tags: list[str],
    gguf_variants: list[GGUFVariant],
    data: dict,
) -> str:
    haystack = " ".join([model_id, *tags]).lower()
    if gguf_variants or "gguf" in haystack:
        return "gguf"
    if re.search(r"(^|[-_/])mlx($|[-_/])", haystack):
        return "mlx"
    if looks_quantized_repo_name(model_id):
        return "quantized"
    config = data.get("config", {}) or {}
    if isinstance(config, dict) and config.get("quantization_config"):
        return "quantized"
    if data.get("safetensors"):
        return "safetensors"
    return "unknown"


def _lookup_curated_count(mapping: dict[str, int], model_id: str) -> int | None:
    value = mapping.get(model_id)
    if value is not None:
        return value

    model_id_folded = model_id.casefold()
    for key, value in mapping.items():
        if key.casefold() == model_id_folded:
            return value
    return None


def _resolve_moe_active_params(
    total_params: int,
    *model_refs: str | None,
) -> int | None:
    """Resolve active params from curated data or A*B naming hints."""
    for ref in model_refs:
        if not ref:
            continue
        active = _lookup_curated_count(_KNOWN_MOE_ACTIVE_PARAMS, ref)
        if active and active > 0:
            return active

    for ref in model_refs:
        active = _extract_active_size_hint_from_id(ref)
        if active and active > 0 and (total_params <= 0 or active < total_params):
            return active
    return None


def _normalize_param_count(
    extracted: int,
    model_id: str,
    base_model: str | None,
) -> int:
    """Normalize parameter count when metadata is inconsistent."""
    authoritative = _lookup_curated_count(_AUTHORITATIVE_PARAM_COUNTS, model_id)
    if authoritative and authoritative > 0:
        return authoritative
    known = _lookup_curated_count(_KNOWN_PARAM_COUNTS, model_id)
    if extracted <= 0:
        return known or extracted
    if known and extracted < int(known * 0.35):
        return known

    hints = [
        h
        for h in (
            _extract_size_hint_from_id(model_id),
            _extract_size_hint_from_id(base_model),
        )
        if h is not None
    ]
    if not hints:
        return extracted

    hinted = max(hints)
    if looks_quantized_repo_name(model_id):
        # Quantized derivatives sometimes report compressed safetensors counts.
        if extracted < int(hinted * 0.70):
            return hinted
    elif extracted < int(hinted * 0.35):
        return hinted

    return extracted


def _extract_quant_type(filename: str) -> str:
    """Extract quantization type from GGUF filename."""
    # Common patterns: model-Q4_K_M.gguf, model.Q4_K_M.gguf
    patterns = [
        r"[.-](Q\d+_K_[SMLA])",
        r"[.-](Q\d+_\d+)",
        r"[.-](Q\d+_K)",
        r"[.-](IQ\d+_\w+)",
        r"[.-](MXFP4|NVFP4)",
        r"[.-](F16|FP16|BF16|F32)",
    ]
    upper = filename.upper()
    for pattern in patterns:
        m = re.search(pattern, upper)
        if m:
            return m.group(1)
    return "unknown"


def _estimate_gguf_size(param_count: int, quant_type: str) -> int:
    """Estimate GGUF file size from parameter count and quantization type."""
    bpw = QUANT_BYTES_PER_WEIGHT.get(quant_type.upper(), 0.5625)  # default Q4_K_M
    return int(param_count * bpw)


# Curated MoE active-parameter counts. Used when HF config lacks the
# `num_local_experts` / `num_experts_per_tok` keys that whichvlm reads.
# Without this, frontier MoEs are scored as dense models which over-counts
# their VRAM cost and under-counts their inference speed.
_KNOWN_MOE_ACTIVE_PARAMS: dict[str, int] = {
    "meta-llama/Llama-4-Scout-17B-16E-Instruct": 17_000_000_000,
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct": 17_000_000_000,
    "Qwen/Qwen3-Next-80B-A3B-Instruct": 3_000_000_000,
    "Qwen/Qwen3-30B-A3B": 3_000_000_000,
    "Qwen/Qwen3-Coder-30B-A3B-Instruct": 3_000_000_000,
    "Qwen/Qwen3-235B-A22B": 22_000_000_000,
    "Qwen/Qwen3.5-397B-A17B": 17_000_000_000,
    "deepseek-ai/DeepSeek-V3": 37_000_000_000,
    "deepseek-ai/DeepSeek-V3-0324": 37_000_000_000,
    "deepseek-ai/DeepSeek-V3.1": 37_000_000_000,
    "deepseek-ai/DeepSeek-V3.2": 37_000_000_000,
    "deepseek-ai/DeepSeek-V3.2-Exp": 37_000_000_000,
    "deepseek-ai/DeepSeek-R1": 37_000_000_000,
    "deepseek-ai/DeepSeek-R1-0528": 37_000_000_000,
    "deepseek-ai/DeepSeek-V4-Pro": 49_000_000_000,
    "deepseek-ai/DeepSeek-V4-Flash": 13_000_000_000,
    "zai-org/GLM-4.5": 32_000_000_000,
    "zai-org/GLM-4.5-Air": 12_000_000_000,
    "zai-org/GLM-4.6": 32_000_000_000,
    "zai-org/GLM-4.7": 32_000_000_000,
    "zai-org/GLM-4.7-Flash": 12_000_000_000,
    "zai-org/GLM-5": 40_000_000_000,
    "zai-org/GLM-5-FP8": 40_000_000_000,
    "zai-org/GLM-5.1": 40_000_000_000,
    "zai-org/GLM-5.1-FP8": 40_000_000_000,
    "moonshotai/Kimi-K2-Instruct": 32_000_000_000,
    "moonshotai/Kimi-K2-Thinking": 32_000_000_000,
    "MiniMaxAI/MiniMax-M2": 10_000_000_000,
    "MiniMaxAI/MiniMax-M2.5": 10_000_000_000,
    "XiaomiMiMo/MiMo-V2.5": 15_000_000_000,
    "XiaomiMiMo/MiMo-V2.5-Pro": 42_000_000_000,
    "XiaomiMiMo/MiMo-V2-Flash": 15_000_000_000,
    "google/gemma-4-26B-A4B-it": 3_800_000_000,
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16": 3_000_000_000,
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8": 3_000_000_000,
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16": 12_000_000_000,
    # OpenAI gpt-oss MoE family — 5B active for 20b/120b.
    "openai/gpt-oss-20b": 3_600_000_000,
    "openai/gpt-oss-120b": 5_100_000_000,
}

# Hardcoded parameter counts for frontier models that HF's API leaves with
# missing safetensors/gguf/config metadata. Used as a last-resort fallback
# inside :func:`_extract_param_count` so these models still enter the cache
# and become rankable. Maintain only entries that lack a size hint in the
# model ID itself (those are handled by :func:`_extract_size_hint_from_id`).
_KNOWN_PARAM_COUNTS: dict[str, int] = {
    "microsoft/phi-4": 14_700_000_000,
    "microsoft/Phi-4-mini-instruct": 3_800_000_000,
    "microsoft/Phi-4-multimodal-instruct": 5_600_000_000,
    "microsoft/Phi-4-reasoning": 14_700_000_000,
    "microsoft/Phi-4-reasoning-plus": 14_700_000_000,
    "openai/gpt-oss-20b": 20_000_000_000,
    "openai/gpt-oss-120b": 120_000_000_000,
    # IBM Granite 4.0 family
    "ibm-granite/granite-4.0-h-small": 32_000_000_000,
    "ibm-granite/granite-4.0-h-tiny": 7_000_000_000,
    "ibm-granite/granite-3.3-8b-instruct": 8_000_000_000,
    "ibm-granite/granite-3.3-2b-instruct": 2_000_000_000,
    # AllenAI Olmo-3
    "allenai/Olmo-3-7B-Instruct": 7_000_000_000,
    "allenai/Olmo-3-1025-7B": 7_000_000_000,
    # Llama 4 MoE totals — repo names advertise the *active* size (17B) but
    # the total weight footprint is much larger (16 / 128 experts × shared
    # backbone). Without this override the cache scores them as 17B dense
    # models, which lets them appear in 12-16 GB rankings they can't run.
    "meta-llama/Llama-4-Scout-17B-16E-Instruct": 109_000_000_000,
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct": 400_000_000_000,
    "deepseek-ai/DeepSeek-R1": 671_000_000_000,
    "deepseek-ai/DeepSeek-R1-0528": 671_000_000_000,
    "deepseek-ai/DeepSeek-V3": 671_000_000_000,
    "deepseek-ai/DeepSeek-V3-0324": 671_000_000_000,
    "deepseek-ai/DeepSeek-V3.1": 671_000_000_000,
    "deepseek-ai/DeepSeek-V3.2": 685_000_000_000,
    "deepseek-ai/DeepSeek-V4-Pro": 1_600_000_000_000,
    "deepseek-ai/DeepSeek-V4-Flash": 284_000_000_000,
    "moonshotai/Kimi-K2-Instruct": 1_026_000_000_000,
    "moonshotai/Kimi-K2-Thinking": 1_026_000_000_000,
    "XiaomiMiMo/MiMo-V2.5": 310_000_000_000,
    "XiaomiMiMo/MiMo-V2.5-Pro": 1_020_000_000_000,
    "XiaomiMiMo/MiMo-V2-Flash": 309_000_000_000,
    "zai-org/GLM-4.5": 355_000_000_000,
    "zai-org/GLM-4.5-Air": 106_000_000_000,
    "zai-org/GLM-4.6": 355_000_000_000,
    "zai-org/GLM-4.7": 355_000_000_000,
    "zai-org/GLM-4.7-Flash": 30_000_000_000,
    "zai-org/GLM-5": 744_000_000_000,
    "zai-org/GLM-5-FP8": 744_000_000_000,
    "zai-org/GLM-5.1": 744_000_000_000,
    "zai-org/GLM-5.1-FP8": 744_000_000_000,
    "MiniMaxAI/MiniMax-M2": 230_000_000_000,
    "MiniMaxAI/MiniMax-M2.5": 230_000_000_000,
    "stepfun-ai/Step-3.5-Flash": 30_000_000_000,
}

# Curated counts that should win even when the HF API exposes safetensors
# metadata. Some mixed-precision MoEs publish compressed checkpoint tensor
# counts that are useful for storage inspection but understate the model-card
# capacity used for ranking, GGUF synthesis, and VRAM planning.
_AUTHORITATIVE_PARAM_COUNTS: dict[str, int] = {
    "deepseek-ai/DeepSeek-V4-Pro": 1_600_000_000_000,
    "deepseek-ai/DeepSeek-V4-Flash": 284_000_000_000,
}


def _extract_param_count(model_data: dict) -> int:
    """Extract parameter count from model data.

    Resolution order:
      1. authoritative model-card overrides for known mixed-precision MoEs
      2. safetensors metadata (most reliable when present)
      3. gguf metadata
      4. config (estimated from hidden_size + num_layers + vocab_size)
      5. name-based size hint (e.g. ``Qwen/Qwen3-32B`` → 32B)
      6. ``_KNOWN_PARAM_COUNTS`` lookup (for models like ``microsoft/phi-4``
         that have neither indexed metadata nor a size in the repo name)

    Returns 0 if none of the above succeed (caller drops the model).
    """
    model_id = model_data.get("id", "") or ""
    authoritative = _lookup_curated_count(_AUTHORITATIVE_PARAM_COUNTS, model_id)
    if authoritative and authoritative > 0:
        return authoritative

    safetensors = model_data.get("safetensors")
    if safetensors and isinstance(safetensors, dict):
        params = safetensors.get("total")
        if params:
            return int(params)
        parameters = safetensors.get("parameters")
        if isinstance(parameters, dict):
            total = sum(parameters.values())
            if total > 0:
                return total

    gguf_meta = model_data.get("gguf", {}) or {}
    if isinstance(gguf_meta, dict):
        total = gguf_meta.get("total")
        if total and total > 0:
            return int(total)

    config = model_data.get("config", {}) or {}
    hidden = config.get("hidden_size", 0)
    layers = config.get("num_hidden_layers", 0)
    vocab = config.get("vocab_size", 0)
    if hidden and layers and vocab:
        return 12 * layers * hidden * hidden + vocab * hidden * 2

    # Fall back to ID-based hints — these are the recourse when HF doesn't
    # index safetensors metadata for a repo (e.g. Qwen3-32B, phi-4, Mistral
    # Small 3.2 24B). Without this branch these popular models silently
    # disappear from the ranker.
    #
    # ``_KNOWN_PARAM_COUNTS`` is checked *before* the name hint because it
    # is curated: for Llama-4-Scout-17B-16E (16-expert MoE) the name hint
    # gives 17B (the active size) but the actual VRAM footprint is 109B.
    known = _lookup_curated_count(_KNOWN_PARAM_COUNTS, model_id)
    if known and known > 0:
        return known
    name_hint = _extract_size_hint_from_id(model_id)
    if name_hint and name_hint > 0:
        return name_hint

    return 0


def _extract_architecture(config: dict) -> str:
    arch_list = config.get("architectures", [])
    if arch_list:
        arch = arch_list[0].lower()
        for name in [
            "llama",
            "qwen2",
            "mistral",
            "mixtral",
            "gemma",
            "phi",
            "starcoder",
            "command",
            "deepseek",
        ]:
            if name in arch:
                return name
        return arch.replace("forcausallm", "").replace("forconditionalgeneration", "")
    model_type = config.get("model_type", "")
    return model_type.lower()


def _parse_model(data: dict) -> ModelInfo | None:
    model_id = data.get("id", "")
    if not model_id:
        return None

    config = data.get("config", {}) or {}
    card_data = data.get("cardData", {}) or {}
    tags = _extract_tags(data)

    # Base model from card data. Keep the first pointer for compatibility,
    # while preserving all parents for VLM package and merge lineage.
    base_models = _extract_base_models(card_data)
    base_model = base_models[0] if base_models else None

    param_count = _extract_param_count(data)
    param_count = _normalize_param_count(param_count, model_id, base_model)
    if param_count == 0:
        return None

    # MoE detection. HF model configs use a variety of keys for the
    # expert-count field — try the common ones before giving up.
    num_experts = 0
    for k in (
        "num_local_experts",
        "num_experts",
        "n_routed_experts",
        "moe_num_experts",
        "num_moe_experts",
        "n_local_experts",
    ):
        v = config.get(k, 0)
        if isinstance(v, int) and v > num_experts:
            num_experts = v
    experts_per_tok = 0
    for k in (
        "num_experts_per_tok",
        "moe_topk",
        "moe_top_k",
        "num_experts_per_token",
        "top_k",
    ):
        v = config.get(k, 0)
        if isinstance(v, int) and v > experts_per_tok:
            experts_per_tok = v

    # Known-frontier MoE registry and A*B naming hints: when HF config lacks
    # expert metadata, fall back to release-card counts or model IDs such as
    # Qwen3.6-35B-A3B. The A3B suffix means 3B active params per token.
    known_moe_active = _resolve_moe_active_params(param_count, model_id, base_model)
    is_moe = num_experts > 0 or known_moe_active is not None
    active_params = None
    if is_moe:
        if known_moe_active is not None:
            active_params = known_moe_active
        elif num_experts > 0:
            ept = experts_per_tok if experts_per_tok > 0 else 2
            active_ratio = ept / num_experts
            expert_fraction = 0.6  # ~60% of MoE weight lives in experts
            active_params = int(
                param_count * (1 - expert_fraction + expert_fraction * active_ratio)
            )

    quant_sizes: dict[str, int] = {}
    quant_first_filename: dict[str, str] = {}
    projector_files: list[tuple[str, int | None]] = []
    siblings = data.get("siblings", []) or []
    for sib in siblings:
        fname = sib.get("rfilename", "")
        if not isinstance(fname, str) or fname.startswith("."):
            continue
        size = sib.get("size", 0)
        file_size = size if isinstance(size, int) and size > 0 else None
        if is_projector_filename(fname):
            projector_files.append((fname, file_size))
            continue
        if not fname.endswith(".gguf"):
            continue
        quant = _extract_quant_type(fname)
        if quant == "unknown":
            continue
        if file_size is None:
            file_size = 0

        # 分割GGUFは量子化ごとに合算して1候補として扱う。
        quant_sizes[quant] = quant_sizes.get(quant, 0) + file_size
        if quant not in quant_first_filename or _GGUF_SPLIT_RE.search(
            quant_first_filename[quant]
        ):
            quant_first_filename[quant] = fname

    gguf_variants = []
    for quant, total_size in quant_sizes.items():
        if total_size <= 0:
            total_size = _estimate_gguf_size(param_count, quant)
        gguf_variants.append(
            GGUFVariant(
                filename=quant_first_filename[quant],
                quant_type=quant,
                file_size_bytes=total_size,
            )
        )

    model_format = _infer_model_format(model_id, tags, gguf_variants, data)
    quantization_type = _infer_repo_quantization(model_id, tags)
    is_official = _is_official_model(model_id)
    variant_kind = infer_variant_kind(
        model_id=model_id,
        base_models=base_models,
        model_format=model_format,
        is_official=is_official,
        tags=tags,
        card_data=card_data,
    )
    access = _extract_access(data)
    lineage = build_lineage(base_models, tags, card_data)
    artifacts = build_artifacts(
        model_id,
        model_format=model_format,
        quantization_type=quantization_type,
        access=access,
        variant_kind=variant_kind,
        gguf_variants=gguf_variants,
        parameter_count=param_count,
        projector_files=projector_files,
    )
    components = build_components(
        model_id,
        parameter_count=param_count,
        quantization_type=quantization_type,
        pipeline_tag=data.get("pipeline_tag"),
        tags=tags,
        lineage=lineage,
    )

    architecture = _extract_architecture(config)
    gguf_meta = data.get("gguf", {}) or {}
    if not architecture and isinstance(gguf_meta, dict):
        architecture = gguf_meta.get("architecture", "")

    context_length = config.get("max_position_embeddings") or config.get(
        "max_sequence_length"
    )
    if not context_length and isinstance(gguf_meta, dict):
        context_length = gguf_meta.get("context_length")

    benchmark_scores: dict[str, float] = {}
    eval_score = _extract_hf_eval_score(data)
    if eval_score is not None:
        benchmark_scores["hf_eval"] = eval_score

    return ModelInfo(
        id=model_id,
        family_id=model_id,  # will be set by grouper
        name=model_id.split("/")[-1],
        parameter_count=param_count,
        parameter_count_active=active_params,
        architecture=architecture,
        is_moe=is_moe,
        context_length=context_length,
        license=card_data.get("license"),
        published_at=_extract_published_at(data),
        downloads=data.get("downloads", 0),
        likes=data.get("likes", 0),
        gguf_variants=gguf_variants,
        benchmark_scores=benchmark_scores,
        base_model=base_model,
        hf_pipeline_tag=data.get("pipeline_tag"),
        tags=tags,
        access=access,
        is_official=is_official,
        model_format=model_format,
        variant_kind=variant_kind,
        quantization_type=quantization_type,
        variant_of=base_model,
        base_models=base_models,
        artifacts=artifacts,
        components=components,
        lineage=lineage,
    )


def _model_list_params(
    *,
    pipeline_tag: str,
    sort: str,
    limit: int,
    filter_value: str | None = None,
) -> dict:
    params = {
        "pipeline_tag": pipeline_tag,
        "sort": sort,
        "limit": str(limit),
        "expand[]": list(_HF_MODEL_EXPAND),
    }
    if filter_value:
        params["filter"] = filter_value
    return params


async def _fetch_model_list(
    client: httpx.AsyncClient,
    params: dict,
    label: str,
    *,
    required: bool,
    semaphore: asyncio.Semaphore,
) -> list[dict]:
    async with semaphore:
        try:
            resp = await get_with_retries(client, f"{HF_API_BASE}/models", params=params)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list):
                return data
            raise ValueError(f"HF API returned {type(data).__name__} for {label}")
        except (httpx.HTTPError, ValueError) as e:
            if required:
                raise
            logger.debug("%s fetch skipped: %s", label, e)
            return []


async def _fetch_model_detail(
    client: httpx.AsyncClient,
    model_id: str,
    semaphore: asyncio.Semaphore,
) -> dict | None:
    async with semaphore:
        try:
            resp = await get_with_retries(
                client,
                f"{HF_API_BASE}/models/{model_id}",
                params={"expand[]": list(_HF_MODEL_DETAIL_EXPAND)},
            )
            if resp.status_code >= 400:
                logger.debug(
                    "Seed fetch skipped %s: HTTP %s", model_id, resp.status_code
                )
                return None
            data = resp.json()
            return data if isinstance(data, dict) else None
        except (httpx.HTTPError, ValueError) as e:
            logger.debug("Seed fetch failed for %s: %s", model_id, e)
            return None


def _append_parsed_models(
    models: list[ModelInfo],
    seen_ids: set[str],
    data_list: list[dict],
) -> None:
    for data in data_list:
        model_id = data.get("id")
        if not isinstance(model_id, str) or model_id in seen_ids:
            continue
        model = _parse_model(data)
        if model:
            models.append(model)
            seen_ids.add(model.id)


async def fetch_models(
    limit: int = 300, include_vision: bool = True
) -> list[ModelInfo]:
    models: list[ModelInfo] = []
    seen_ids: set[str] = set()

    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        list_queries: list[tuple[dict, str, bool]] = []
        if include_vision:
            for pipeline_tag in _VLM_PIPELINE_TAGS:
                for filter_value in _VLM_VARIANT_FILTERS:
                    label = f"VLM {pipeline_tag}/{filter_value or 'all'}"
                    list_queries.append(
                        (
                            _model_list_params(
                                pipeline_tag=pipeline_tag,
                                sort="downloads",
                                limit=limit,
                                filter_value=filter_value,
                            ),
                            label,
                            False,
                        )
                    )

        list_queries.extend(
            [
                (
                    _model_list_params(
                        pipeline_tag="text-generation",
                        sort="downloads",
                        limit=limit,
                    ),
                    "text-generation downloads",
                    True,
                ),
                (
                    _model_list_params(
                        pipeline_tag="text-generation",
                        sort="downloads",
                        limit=limit,
                        filter_value="gguf",
                    ),
                    "text-generation GGUF downloads",
                    True,
                ),
                (
                    _model_list_params(
                        pipeline_tag="text-generation",
                        sort="lastModified",
                        limit=limit,
                        filter_value="gguf",
                    ),
                    "text-generation recent GGUF",
                    True,
                ),
                (
                    _model_list_params(
                        pipeline_tag="text-generation",
                        sort="trending",
                        limit=limit,
                    ),
                    "text-generation trending",
                    False,
                ),
                (
                    _model_list_params(
                        pipeline_tag="text-generation",
                        sort="trending",
                        limit=limit,
                        filter_value="gguf",
                    ),
                    "text-generation trending GGUF",
                    False,
                ),
            ]
        )

        list_semaphore = asyncio.Semaphore(_MODEL_LIST_CONCURRENCY)
        batches = await asyncio.gather(
            *(
                _fetch_model_list(
                    client,
                    params,
                    label,
                    required=required,
                    semaphore=list_semaphore,
                )
                for params, label, required in list_queries
            )
        )
        for batch in batches:
            _append_parsed_models(models, seen_ids, batch)

        if include_vision:
            seed_ids = [
                model_id for model_id in known_vlm_model_ids() if model_id not in seen_ids
            ]
            detail_semaphore = asyncio.Semaphore(_MODEL_DETAIL_CONCURRENCY)
            seed_data = await asyncio.gather(
                *(
                    _fetch_model_detail(client, model_id, detail_semaphore)
                    for model_id in seed_ids
                )
            )
            _append_parsed_models(
                models,
                seen_ids,
                [data for data in seed_data if data is not None],
            )

    logger.debug(f"Fetched {len(models)} models total")
    return models


def models_to_dicts(models: list[ModelInfo]) -> list[dict]:
    """Serialize models to dicts for caching."""
    result = []
    for m in models:
        result.append(
            {
                "id": m.id,
                "family_id": m.family_id,
                "name": m.name,
                "parameter_count": m.parameter_count,
                "parameter_count_active": m.parameter_count_active,
                "architecture": m.architecture,
                "is_moe": m.is_moe,
                "context_length": m.context_length,
                "license": m.license,
                "published_at": m.published_at,
                "downloads": m.downloads,
                "likes": m.likes,
                "gguf_variants": [
                    {
                        "filename": v.filename,
                        "quant_type": v.quant_type,
                        "file_size_bytes": v.file_size_bytes,
                    }
                    for v in m.gguf_variants
                ],
                "benchmark_scores": m.benchmark_scores,
                "base_model": m.base_model,
                "hf_pipeline_tag": m.hf_pipeline_tag,
                "tags": m.tags,
                "access": m.access,
                "is_official": m.is_official,
                "model_format": m.model_format,
                "variant_kind": m.variant_kind,
                "quantization_type": m.quantization_type,
                "variant_of": m.variant_of,
                "base_models": m.base_models,
                "artifacts": [artifact_to_dict(a) for a in m.artifacts],
                "components": [component_to_dict(c) for c in m.components],
                "lineage": lineage_to_dict(m.lineage),
            }
        )
    return result


def dicts_to_models(data: list[dict]) -> list[ModelInfo]:
    """Deserialize models from cached dicts."""
    models = []
    for d in data:
        base_model = d.get("base_model")
        param_count = _normalize_param_count(
            d["parameter_count"],
            d["id"],
            base_model,
        )
        active_params = _resolve_moe_active_params(
            param_count,
            d["id"],
            base_model,
            d.get("name"),
            d.get("architecture"),
        )
        if active_params is None:
            active_params = d.get("parameter_count_active")
        gguf_variants = [
            GGUFVariant(
                filename=v["filename"],
                quant_type=v["quant_type"],
                file_size_bytes=v["file_size_bytes"],
            )
            for v in d.get("gguf_variants", [])
        ]
        tags = [str(t) for t in d.get("tags", []) if isinstance(t, str)]
        base_models = [
            str(v) for v in d.get("base_models", []) if isinstance(v, str)
        ]
        if not base_models and base_model:
            base_models = [base_model]
        lineage = lineage_from_dict(d.get("lineage"), base_model)
        artifacts = [
            artifact_from_dict(v) for v in d.get("artifacts", []) if isinstance(v, dict)
        ]
        components = [
            component_from_dict(v)
            for v in d.get("components", [])
            if isinstance(v, dict)
        ]
        model_format = d.get(
            "model_format",
            _infer_model_format(d["id"], tags, gguf_variants, d),
        )
        quantization_type = d.get("quantization_type")
        variant_kind = d.get("variant_kind", "base")
        access = d.get("access", "unknown")
        if not artifacts:
            artifacts = build_artifacts(
                d["id"],
                model_format=model_format,
                quantization_type=quantization_type,
                access=access,
                variant_kind=variant_kind,
                gguf_variants=gguf_variants,
                parameter_count=param_count,
            )
        if not components:
            components = build_components(
                d["id"],
                parameter_count=param_count,
                quantization_type=quantization_type,
                pipeline_tag=d.get("hf_pipeline_tag"),
                tags=tags,
                lineage=lineage,
            )
        models.append(
            ModelInfo(
                id=d["id"],
                family_id=d.get("family_id", d["id"]),
                name=d["name"],
                parameter_count=param_count,
                parameter_count_active=active_params,
                architecture=d.get("architecture", ""),
                is_moe=d.get("is_moe", False) or active_params is not None,
                context_length=d.get("context_length"),
                license=d.get("license"),
                published_at=d.get("published_at"),
                downloads=d.get("downloads", 0),
                likes=d.get("likes", 0),
                gguf_variants=gguf_variants,
                benchmark_scores=d.get("benchmark_scores", {}),
                base_model=base_model,
                hf_pipeline_tag=d.get("hf_pipeline_tag"),
                tags=tags,
                access=access,
                is_official=d.get("is_official", _is_official_model(d["id"])),
                model_format=model_format,
                variant_kind=variant_kind,
                quantization_type=quantization_type,
                variant_of=d.get("variant_of", base_model),
                base_models=base_models,
                artifacts=artifacts,
                components=components,
                lineage=lineage,
            )
        )
    return models


async def fetch_model_published_at(model_ids: list[str]) -> dict[str, str]:
    """Fetch published timestamps for specific model IDs."""
    unique_ids = sorted({m for m in model_ids if m})
    if not unique_ids:
        return {}

    async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
        tasks = [
            client.get(
                f"{HF_API_BASE}/models/{model_id}",
                params={"expand[]": ["createdAt", "lastModified"]},
            )
            for model_id in unique_ids
        ]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

    result: dict[str, str] = {}
    for model_id, resp in zip(unique_ids, responses, strict=False):
        if isinstance(resp, Exception):
            logger.debug("Failed to fetch model detail for %s: %s", model_id, resp)
            continue
        if resp.status_code >= 400:
            logger.debug(
                "Failed to fetch model detail for %s: HTTP %s",
                model_id,
                resp.status_code,
            )
            continue
        try:
            data = resp.json()
        except ValueError:
            continue
        published_at = _extract_published_at(data)
        if published_at:
            result[model_id] = published_at
    return result
