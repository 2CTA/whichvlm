from __future__ import annotations

import math
import re

from whichvlm.constants import (
    MODEL_GENERATION_BONUS_MAX,
    MODEL_GENERATION_PENALTY_MAX,
    MODEL_LINEAGE_VERSIONS,
    QUANT_BYTES_PER_WEIGHT,
    QUANT_PREFERENCE_ORDER,
)
from whichvlm.engine.compatibility import check_compatibility
from whichvlm.engine.performance import estimate_speed_uncertainty, estimate_tok_per_sec
from whichvlm.engine.quantization import (
    effective_quant_type,
    infer_non_gguf_quant_type,
    quant_quality_penalty,
)
from whichvlm.engine.types import CompatibilityResult
from whichvlm.engine.workload import VisionWorkload
from whichvlm.hardware.types import HardwareInfo, has_backend, infer_backend_capabilities
from whichvlm.models.benchmark import (
    BenchmarkEvidence,
    build_line_bucket_index,
    build_score_index,
    lookup_benchmark_evidence,
)
from whichvlm.models.types import GGUFVariant, ModelInfo

# Ranking core. Expands variants, scores fit, and orders final picks.

LINEAGE_REGEX: dict[str, list[tuple[re.Pattern[str], int]]] = {
    family: [(re.compile(pat), idx) for pat, idx in entries]
    for family, entries in MODEL_LINEAGE_VERSIONS.items()
}
LINEAGE_FAMILY_MAX: dict[str, int] = {
    family: max(idx for _, idx in entries) for family, entries in LINEAGE_REGEX.items()
}
MULTI_GPU_SPEED_FACTOR = 0.70


def family_selection_key(
    result: CompatibilityResult,
    require_direct_top: bool,
) -> tuple[float]:
    # Family sort key. Keeps final ordering close to the shown score.
    if require_direct_top and result.benchmark_status == "direct":
        direct_bonus = 5.0
    else:
        direct_bonus = 0.0
    cpu_penalty = -6.0 if result.fit_type == "cpu_only" else 0.0
    ctx_penalty = -20.0 if not result.context_fits else 0.0
    return (result.quality_score + direct_bonus + cpu_penalty + ctx_penalty,)


def partial_offload_quality_factor(model: ModelInfo, offload_ratio: float) -> float:

    ratio = max(0.0, min(1.0, offload_ratio))
    if ratio >= 0.75:
        factor = 0.42
    elif ratio >= 0.60:
        factor = 0.52
    elif ratio >= 0.40:
        factor = 0.62
    elif ratio >= 0.25:
        factor = 0.76
    else:
        factor = 0.86


    if model.is_moe and model.parameter_count_active:
        active_ratio = (
            model.parameter_count_active / model.parameter_count
            if model.parameter_count > 0
            else 1.0
        )
        active_ratio = max(0.0, min(1.0, active_ratio))
        active_set_fits = ratio <= max(0.0, 1.0 - active_ratio)
        if active_set_fits:
            if ratio >= 0.75:
                factor = max(factor, 0.66)
            elif ratio >= 0.60:
                factor = max(factor, 0.70)
            elif ratio >= 0.40:
                factor = max(factor, 0.76)
            elif ratio >= 0.25:
                factor = max(factor, 0.82)
            else:
                factor = max(factor, 0.88)
        else:
            factor = min(0.76, factor + 0.08)

    return factor


SOURCE_WEIGHTS: dict[str, float] = {
    "direct": 0.62,
    "base_model": 0.55,
    "variant": 0.50,
    "line_interp": 0.40,
    "self_reported": 0.30,
    "none": 0.0,
}


SYNTHETIC_QUANTS = ("Q3_K_M", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0")
PREQUANTIZED_REPO_RE = re.compile(
    r"-(awq|gptq|bnb|fp8|fp16|bf16|mxfp4|nvfp4|int4|int8|4bit|8bit|gguf)$",
    re.IGNORECASE,
)


def synthesize_variants_for_official_repo(
    model: ModelInfo, quant_filter_upper: str | None
) -> list[GGUFVariant]:
    # Synthetic GGUF layer. Makes safetensors-only repos rank like real quants.
    if "vision" in detect_specializations(model):
        return []

    org = model.id.split("/", 1)[0] if "/" in model.id else ""
    if org not in OFFICIAL_ORGS:
        return []
    if PREQUANTIZED_REPO_RE.search(model.id):
        return []
    out: list[GGUFVariant] = []
    for quant in SYNTHETIC_QUANTS:
        if quant_filter_upper and quant != quant_filter_upper:
            continue
        bpw = QUANT_BYTES_PER_WEIGHT.get(quant, 0.5625)
        out.append(
            GGUFVariant(
                filename=f"{model.name}.{quant}.gguf",
                quant_type=quant,
                file_size_bytes=int(model.parameter_count * bpw),
            )
        )
    return out


def iter_candidate_variants(
    model: ModelInfo,
    quant_filter: str | None = None,
) -> list[GGUFVariant | None]:
    quant_filter_upper = quant_filter.upper() if quant_filter else None

    if not model.gguf_variants:
        synthetic = synthesize_variants_for_official_repo(model, quant_filter_upper)
        if synthetic:
            return synthetic
        quant_type = effective_quant_type(model, None)
        if quant_filter_upper and quant_type != quant_filter_upper:
            return []
        return [None]

    candidates: list[GGUFVariant] = model.gguf_variants
    if quant_filter_upper:
        candidates = [
            v for v in candidates if v.quant_type.upper() == quant_filter_upper
        ]
        if not candidates:
            return []
    else:


        EXTREME_QUANTS = {
            "Q2_K",
            "Q2_0",
            "Q1_0",
            "TQ2_0",
            "TQ1_0",
            "IQ3_XXS",
            "IQ2_XXS",
            "IQ2_S",
            "IQ2_M",
            "IQ1_M",
            "IQ1_S",
        }
        filtered = [
            v for v in candidates if v.quant_type.upper() not in EXTREME_QUANTS
        ]
        if filtered:
            candidates = filtered

    def variant_sort_key(v: GGUFVariant) -> int:
        try:
            return QUANT_PREFERENCE_ORDER.index(v.quant_type.upper())
        except ValueError:
            return len(QUANT_PREFERENCE_ORDER)

    candidates = sorted(candidates, key=variant_sort_key)

    return candidates


OFFICIAL_ORGS = frozenset(
    {
        "Qwen",
        "meta-llama",
        "google",
        "mistralai",
        "deepseek-ai",
        "microsoft",
        "nvidia",
        "01-ai",
        "tiiuae",
        "apple",
        "CohereForAI",
        "bigcode",


        "openai",
        "zai-org",
        "moonshotai",
        "MiniMaxAI",
        "XiaomiMiMo",
        "allenai",
        "ibm-granite",
        "stepfun-ai",
    }
)


EXCLUDED_ORGS = frozenset(
    {
        "openai-community",
        "distilbert",
        "facebook",
        "EleutherAI",
        "trl-internal-testing",
        "hmellor",
        "HuggingFaceH4",
        "transformersbook",
        "togethercomputer",
    }
)

BENCHMARK_ASSET_ORGS = frozenset(
    {
        "Civitai",
    }
)

EXCLUDED_NAME_PATTERNS = (
    "tiny-",
    "-tiny",
    "tiny_",
    "_tiny",
    "test-only",
    "debug-",
    "playground",
    "-fixture",
    "for-testing",
    "tiny-random",
    "ci-",
)


DUBIOUS_DERIVATIVE_PATTERNS = (
    "heretic",
    "abliterat",
    "uncensored",
    "obliterat",
    "abliter",
    "horror",
    "erotic",
    "nsfw",
    "rp-",
    "-rp",
    "roleplay",
    "darkidol",
    "darkforest",
    "tiefigh",
    "smaug",
    "personalityengine",
    "lexi",
    "violence",
    "violet",
    "schizo",
    "dark-",
    "twilight",
    "celeste",
    "midnight-rose",
    "moistral",
    "stheno",
    "fimbulvetr",
    "wizard-vicuna",
    "kunoichi",
    "crack",
)


def derivative_name_penalty(model_id: str) -> float:

    if not model_id:
        return 0.0
    lower = model_id.lower()
    name = lower.split("/", 1)[1] if "/" in lower else lower
    for pat in DUBIOUS_DERIVATIVE_PATTERNS:
        if pat in name:
            return -10.0
    return 0.0


def is_excluded_model(model_id: str) -> bool:

    if not model_id:
        return True
    org = model_id.split("/", 1)[0] if "/" in model_id else ""
    if org in EXCLUDED_ORGS:
        return True
    lower = model_id.lower()
    name = lower.split("/", 1)[1] if "/" in lower else lower
    for pat in EXCLUDED_NAME_PATTERNS:
        if pat in name:
            return True
    return False


def generation_bonus(model_id: str) -> float:

    if not model_id:
        return 0.0
    lower = model_id.lower()
    best_bonus = 0.0
    for family, patterns in LINEAGE_REGEX.items():
        for regex, idx in patterns:
            if regex.search(lower):
                top = LINEAGE_FAMILY_MAX[family]
                if top <= 1:
                    contribution = 0.0
                else:

                    norm = (idx - 1) / (top - 1)
                    span = MODEL_GENERATION_BONUS_MAX + MODEL_GENERATION_PENALTY_MAX
                    contribution = norm * span - MODEL_GENERATION_PENALTY_MAX
                if abs(contribution) > abs(best_bonus):
                    best_bonus = contribution
                break
    return best_bonus


def detect_specializations(model: ModelInfo) -> set[str]:
    lower = " ".join(
        [model.id, model.hf_pipeline_tag or "", *model.tags, model.architecture]
    ).lower()
    tags: set[str] = set()
    if re.search(r"(coder|codegen|starcoder|program|coding)", lower):
        tags.add("coding")
    if re.search(
        r"(^|[-_/])(vl|vision|multimodal|llava|image)([-_/]|$)|"
        r"image-text-to-text|visual-question-answering|image-to-text|internvl|pixtral",
        lower,
    ):
        tags.add("vision")
    if re.search(r"(^|[-_/])math([-_/]|$)", lower):
        tags.add("math")
    return tags


def matches_profile(model: ModelInfo, task_profile: str) -> bool:
    profile = task_profile.lower()
    tags = detect_specializations(model)
    if profile == "any":
        return True
    if profile == "general":
        return len(tags) == 0
    return profile in tags


def effective_params_b(model: ModelInfo) -> float:

    if model.is_moe and model.parameter_count_active:
        return model.parameter_count_active / 1e9
    return model.parameter_count / 1e9


def knowledge_capacity_b(model: ModelInfo) -> float:

    return model.parameter_count / 1e9


def passes_evidence_filter(source: str, evidence_filter: str) -> bool:
    mode = evidence_filter.lower()
    if mode == "strict":
        return source == "direct"
    if mode == "base":
        return source in {"direct", "variant", "base_model"}
    return True


def is_gguf_only_backend(hardware: HardwareInfo) -> bool:
    if not hardware.gpus:
        return True
    if hardware.os == "darwin":
        return False


    has_linux_nvidia = hardware.os == "linux" and any(
        g.vendor == "nvidia" for g in hardware.gpus
    )
    return not has_linux_nvidia


def model_artifact_backends(model: ModelInfo) -> set[str]:
    backends: set[str] = set()
    for artifact in model.artifacts:
        backends.update(b.lower() for b in artifact.backend_support)
    if backends:
        return backends

    quant = (model.quantization_type or infer_non_gguf_quant_type(model.id)).upper()
    if model.gguf_variants or model.model_format == "gguf" or quant == "GGUF":
        return {"metal", "cuda", "vulkan", "cpu"}
    if model.model_format == "mlx" or quant == "MLX":
        return {"mlx", "metal"}
    if quant in {"AWQ", "GPTQ", "BNB_4BIT", "FP8", "MXFP4", "NVFP4"}:
        return {"cuda"}
    if model.model_format in {"safetensors", "unknown"} or quant in {
        "FP16",
        "BF16",
        "INT8",
    }:
        return {"cuda", "mps", "cpu"}
    return backends


def hardware_backend_names(hardware: HardwareInfo) -> set[str]:
    backends = {
        capability.name.lower()
        for capability in hardware.backend_capabilities
        if capability.available
    }
    backends.add("cpu")
    for gpu in hardware.gpus:
        capabilities = gpu.backend_capabilities or infer_backend_capabilities(
            gpu, hardware.os
        )
        backends.update(c.name.lower() for c in capabilities if c.available)
    return backends


def model_backend_compatible(
    model: ModelInfo,
    variant: GGUFVariant | None,
    hardware: HardwareInfo,
    hardware_backends: set[str] | None = None,
    model_backends: set[str] | None = None,
) -> bool:
    if hardware_backends is None:
        hardware_backends = hardware_backend_names(hardware)
    if variant is not None:
        return bool(hardware_backends & {"metal", "cuda", "vulkan", "cpu"})
    if model_backends is None:
        model_backends = model_artifact_backends(model)
    if not model_backends:
        return True
    return bool(model_backends & hardware_backends)


def backend_priority_bonus(
    model: ModelInfo,
    variant: GGUFVariant | None,
    hardware: HardwareInfo,
    model_backends: set[str] | None = None,
) -> float:
    if not hardware.gpus:
        return -4.0
    best_gpu = max(hardware.gpus, key=lambda g: g.vram_bytes)
    if model_backends is None:
        model_backends = model_artifact_backends(model)

    if best_gpu.vendor == "apple" and hardware.os == "darwin":
        if has_backend(best_gpu, "mlx") and "mlx" in model_backends:
            return 16.0
        if variant is not None or "metal" in model_backends:
            return 10.0
        if "mps" in model_backends:
            return 1.0
        return -4.0
    if best_gpu.vendor == "nvidia":
        if has_backend(best_gpu, "cuda") and (
            {"cuda", "awq", "gptq", "fp8"} & model_backends or variant is None
        ):
            return 5.0
        if variant is not None:
            return 3.0
    if best_gpu.vendor == "amd":
        if has_backend(best_gpu, "rocm") and "cuda" not in model_backends:
            return 2.0
        if variant is not None or "vulkan" in model_backends:
            return 2.5
    if variant is not None or "vulkan" in model_backends:
        return 1.5
    return 0.0


def compute_quality_score(
    model: ModelInfo,
    variant: GGUFVariant | None,
    tok_per_sec: float,
    fit_type: str,
    offload_ratio: float = 0.0,
    family_downloads: int = 0,
    family_likes: int = 0,
    benchmark_avg: float | None = None,
    benchmark_source: str = "none",
) -> float:

    params_b = model.parameter_count / 1e9
    if model.is_moe and model.parameter_count_active:
        effective_b = model.parameter_count_active / 1e9
    else:
        effective_b = params_b

    if effective_b <= 0:
        return 0.0


    size_basis_b = params_b
    size_score = 4.2 * math.log2(max(size_basis_b, 0.5)) + 9
    size_score = min(size_score, 35)

    has_benchmark = benchmark_avg is not None and benchmark_avg > 0
    is_direct = benchmark_source == "direct"
    is_self_reported = benchmark_source == "self_reported"
    is_inherited = benchmark_source in {"variant", "base_model", "line_interp"}

    bench_weight = SOURCE_WEIGHTS.get(benchmark_source, 0.0)
    benchmark_score = 0.0
    if has_benchmark:
        raw = min(100.0, benchmark_avg)
        benchmark_score = raw * bench_weight


    quant_penalty = quant_quality_penalty(model, variant)
    quality_core = (benchmark_score + size_score) * (1 - quant_penalty)


    if not has_benchmark:
        quality_core *= 0.55
    elif is_self_reported:
        quality_core *= 0.55
    elif is_inherited:
        quality_core *= 0.78


    if fit_type == "partial_offload":
        quality_core *= partial_offload_quality_factor(model, offload_ratio)
    elif fit_type == "cpu_only":
        quality_core *= 0.50


    required_speed = (
        8.0
        if fit_type == "full_gpu"
        else (4.0 if fit_type == "partial_offload" else 1.5)
    )
    if tok_per_sec > 0:
        if tok_per_sec < required_speed:
            speed_score = -8.0 * (1 - (tok_per_sec / required_speed))
        else:
            speed_score = min(8.0, math.log2(tok_per_sec / required_speed + 1.0) * 3.2)
    else:
        if fit_type == "partial_offload":
            if offload_ratio >= 0.70:
                speed_score = -24.0
            elif offload_ratio >= 0.40:
                speed_score = -18.0
            else:
                speed_score = -12.0
        else:
            speed_score = -8.0


    downloads = max(model.downloads, family_downloads)
    likes = max(model.likes, family_likes)
    pop_score_raw = 0.0
    if downloads > 0:
        pop_score_raw += min(1.0, math.log10(max(downloads, 1)) / 6 * 1.0)
    if likes > 0:
        pop_score_raw += min(1.0, math.log10(max(likes, 1)) / 4 * 1.0)

    if is_direct:
        pop_weight = 0.0
    elif is_self_reported:
        pop_weight = 0.4
    elif has_benchmark:
        pop_weight = 0.2
    else:
        pop_weight = 0.6
    pop_score = pop_score_raw * pop_weight


    source_bonus_raw = 0.0
    org = model.id.split("/")[0] if "/" in model.id else ""
    if org in BENCHMARK_ASSET_ORGS:
        source_bonus_raw = -5.0
    elif org in OFFICIAL_ORGS:
        source_bonus_raw = 5.0
    elif model.base_model:
        base_org = model.base_model.split("/")[0] if "/" in model.base_model else ""
        if base_org in OFFICIAL_ORGS:
            source_bonus_raw = 2.5

    if is_direct:
        source_weight = 0.2
    elif is_self_reported:
        source_weight = 0.5
    elif has_benchmark:
        source_weight = 0.4
    else:
        source_weight = 0.6
    source_bonus = source_bonus_raw * source_weight


    gen_bonus = generation_bonus(model.id)


    if not has_benchmark or is_self_reported:
        gen_bonus *= 1.5
    elif is_direct:
        gen_bonus *= 0.6


    derivative_penalty = derivative_name_penalty(model.id)

    return max(
        0.0,
        min(
            100.0,
            quality_core
            + speed_score
            + pop_score
            + source_bonus
            + gen_bonus
            + derivative_penalty,
        ),
    )


def rank_models(
    models: list[ModelInfo],
    hardware: HardwareInfo,
    context_length: int = 4096,
    top_n: int = 10,
    quant_filter: str | None = None,
    min_speed: float | None = None,
    benchmark_scores: dict[str, float] | None = None,
    task_profile: str = "general",
    require_direct_top: bool = True,
    min_params_b: float | None = None,
    evidence_filter: str = "any",
    fit_filter: str = "any",
    vision_workload: VisionWorkload | None = None,
) -> list[CompatibilityResult]:
    # Main rank pass. Scores every candidate against hardware and evidence.

    results: list[CompatibilityResult] = []
    gguf_only_backend = is_gguf_only_backend(hardware)
    if vision_workload is None and task_profile.lower() == "vision":
        vision_workload = VisionWorkload(context_length=context_length)


    family_max_downloads: dict[str, int] = {}
    family_max_likes: dict[str, int] = {}


    family_dominant_params: dict[str, int] = {}
    family_dominant_downloads: dict[str, int] = {}
    for m in models:
        fid = m.family_id
        family_max_downloads[fid] = max(family_max_downloads.get(fid, 0), m.downloads)
        family_max_likes[fid] = max(family_max_likes.get(fid, 0), m.likes)
        if m.parameter_count and m.downloads >= family_dominant_downloads.get(fid, -1):
            family_dominant_downloads[fid] = m.downloads
            family_dominant_params[fid] = m.parameter_count

    seen_families: set[str] = set()

    sorted_models = sorted(models, key=lambda m: m.downloads, reverse=True)

    if benchmark_scores:
        bench_ci_index, bench_line_index = build_score_index(benchmark_scores)
        bench_line_buckets = build_line_bucket_index(benchmark_scores)
    else:
        bench_ci_index, bench_line_index = {}, {}
        bench_line_buckets = {}

    best_gpu = None
    for gpu in hardware.gpus:
        if best_gpu is None or gpu.vram_bytes > best_gpu.vram_bytes:
            best_gpu = gpu
    hardware_backends = hardware_backend_names(hardware)

    for model in sorted_models:
        if is_excluded_model(model.id):
            continue
        if not matches_profile(model, task_profile):
            continue
        if min_params_b is not None and knowledge_capacity_b(model) < min_params_b:
            continue

        candidates = iter_candidate_variants(model, quant_filter)
        if not candidates:
            continue

        fid = model.family_id
        model_backends = model_artifact_backends(model)

        self_reported = None
        if isinstance(model.benchmark_scores, dict):
            v = model.benchmark_scores.get("hf_eval")
            if isinstance(v, (int, float)) and v > 0:
                self_reported = float(v)

        bench_evidence = BenchmarkEvidence(score=None, confidence=0.0, source="none")
        if benchmark_scores or self_reported is not None:
            actual_params_b = (
                (model.parameter_count or 0) / 1e9 if model.parameter_count else None
            )
            bench_evidence = lookup_benchmark_evidence(
                model.id,
                model.base_model,
                benchmark_scores or {},
                ci_index=bench_ci_index,
                line_index=bench_line_index,
                line_bucket_index=bench_line_buckets,
                self_reported_score=self_reported,
                actual_params_b=actual_params_b,
            )


            if bench_evidence.source in ("variant", "base_model", "line_interp"):
                dom_params = family_dominant_params.get(model.family_id)
                if dom_params and model.parameter_count and dom_params > 0:
                    ratio = model.parameter_count / dom_params
                    if ratio < 0.5 or ratio > 2.0:
                        bench_evidence = BenchmarkEvidence(
                            score=None, confidence=0.0, source="none"
                        )
        if not passes_evidence_filter(bench_evidence.source, evidence_filter):
            continue


        best_for_model: CompatibilityResult | None = None
        for variant in candidates:
            if gguf_only_backend and variant is None and "mlx" not in model_backends:
                continue
            if not model_backend_compatible(
                model,
                variant,
                hardware,
                hardware_backends=hardware_backends,
                model_backends=model_backends,
            ):
                continue
            compat = check_compatibility(
                model,
                variant,
                hardware,
                context_length,
                vision_workload=vision_workload,
            )
            if not compat.can_run:
                continue
            if fit_filter == "full_gpu" and compat.fit_type != "full_gpu":
                continue

            tok_per_sec = estimate_tok_per_sec(
                model, variant, best_gpu, compat.fit_type
            )
            if compat.uses_multi_gpu:
                tok_per_sec *= MULTI_GPU_SPEED_FACTOR
            if min_speed is not None and tok_per_sec < min_speed:
                continue

            bench_avg = None
            if bench_evidence.score is not None:
                if bench_evidence.source in {"direct", "self_reported"}:
                    bench_avg = bench_evidence.score
                else:


                    confidence = max(0.0, min(1.0, bench_evidence.confidence))
                    bench_avg = bench_evidence.score * (0.75 + 0.25 * confidence)

            compat.estimated_tok_per_sec = tok_per_sec
            (
                compat.speed_confidence,
                compat.speed_range_tok_per_sec,
                compat.speed_notes,
            ) = estimate_speed_uncertainty(
                model,
                variant,
                best_gpu,
                compat.fit_type,
                tok_per_sec,
            )
            if compat.uses_multi_gpu:
                compat.speed_confidence = "low"
                if tok_per_sec > 0:
                    compat.speed_range_tok_per_sec = (
                        round(tok_per_sec * 0.35, 1),
                        round(tok_per_sec * 2.0, 1),
                    )
                compat.speed_notes.append(
                    "Multi-GPU speed depends on layer/tensor split mode, "
                    "PCIe/NVLink bandwidth, and backend support; this estimate "
                    "does not assume ideal scaling."
                )
            compat.quality_score = compute_quality_score(
                model,
                variant,
                tok_per_sec,
                compat.fit_type,
                offload_ratio=compat.offload_ratio,
                family_downloads=family_max_downloads.get(fid, 0),
                family_likes=family_max_likes.get(fid, 0),
                benchmark_avg=bench_avg,
                benchmark_source=bench_evidence.source,
            )
            compat.quality_score = min(
                100.0,
                compat.quality_score
                + backend_priority_bonus(
                    model,
                    variant,
                    hardware,
                    model_backends=model_backends,
                ),
            )
            if bench_evidence.score is None:
                compat.benchmark_status = "none"
            elif bench_evidence.source == "direct":
                compat.benchmark_status = "direct"
            elif bench_evidence.source == "self_reported":
                compat.benchmark_status = "self_reported"
            else:
                compat.benchmark_status = "estimated"
            compat.benchmark_source = bench_evidence.source
            compat.benchmark_confidence = bench_evidence.confidence

            if (
                best_for_model is None
                or compat.quality_score > best_for_model.quality_score
            ):
                best_for_model = compat

        if best_for_model is None:
            continue

        family_key = model.family_id
        if family_key in seen_families:
            existing = next(
                (r for r in results if r.model.family_id == family_key), None
            )
            if existing and family_selection_key(
                best_for_model,
                require_direct_top,
            ) > family_selection_key(existing, require_direct_top):
                results.remove(existing)
                results.append(best_for_model)
            continue

        seen_families.add(family_key)
        results.append(best_for_model)

    if require_direct_top:
        results.sort(
            key=lambda r: family_selection_key(r, require_direct_top),
            reverse=True,
        )
    else:
        results.sort(
            key=lambda r: family_selection_key(r, require_direct_top), reverse=True
        )


    if any(r.quality_score >= 30 for r in results):
        results = [r for r in results if r.quality_score >= 20]


    if any(r.estimated_tok_per_sec >= 5.0 for r in results):
        results = [r for r in results if r.estimated_tok_per_sec >= 1.5]

    return results[:top_n]
