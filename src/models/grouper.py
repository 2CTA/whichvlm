from __future__ import annotations

import re

from data.vlm_inventory import canonical_vlm_family_id
from models.package_graph import merge_family_graph
from models.types import ModelFamily, ModelInfo


def normalize_name(model_id: str) -> str:
    canonical = canonical_vlm_family_id(model_id)
    if canonical:
        return canonical

    name = model_id.lower()

    if "/" in name:
        name = name.split("/", 1)[1]

    name = re.sub(r"^(qwen_|meta-llama_|google_)", "", name)

    suffixes = [
        r"-gguf$",
        r"-gptq$",
        r"-awq$",
        r"-instruct$",
        r"-chat$",
        r"-it$",
        r"-hf$",
        r"-fp8$",
        r"-fp16$",
        r"-bf16$",
        r"-mxfp4$",
        r"-nvfp4$",
        r"-\d+bit$",
        r"-\d{4}$",
    ]
    for _ in range(3):
        prev = name
        for suffix in suffixes:
            name = re.sub(suffix, "", name)
        if name == prev:
            break

    name = re.sub(r"-\d+\.\d+(-\d+(?:\.\d+)?b(?:-a\d+b)?)$", r"\1", name)

    m = re.match(r"^(.+?)-(\d+(?:\.\d+)?b(?:-a\d+b)?)$", name)
    if m:
        series, size = m.group(1), m.group(2)
        series = re.sub(r"(\d+)\.\d+$", r"\1", series)
        name = f"{series}-{size}"
    else:
        name = re.sub(r"(\d+)\.\d+$", r"\1", name)

    return name


def group_models(models: list[ModelInfo]) -> list[ModelFamily]:
    base_model_groups: dict[str, list[ModelInfo]] = {}
    ungrouped: list[ModelInfo] = []

    for model in models:
        if model.base_model:
            key = model.base_model.lower()
            base_model_groups.setdefault(key, []).append(model)
        else:
            ungrouped.append(model)

    name_groups: dict[str, list[ModelInfo]] = {}
    for model in ungrouped:
        key = normalize_name(model.id)
        name_groups.setdefault(key, []).append(model)

    merged_base: dict[str, list[ModelInfo]] = {}
    for key, group in base_model_groups.items():
        norm_key = normalize_name(key)
        merged_base.setdefault(norm_key, []).extend(group)

    for norm_key, group in list(merged_base.items()):
        if norm_key in name_groups:
            group.extend(name_groups.pop(norm_key))

    base_model_groups = merged_base

    families: list[ModelFamily] = []

    for group_key, group in list(base_model_groups.items()) + list(name_groups.items()):
        if not group:
            continue

        referenced_as_base: set[str] = {m.base_model for m in group if m.base_model}
        referenced_candidates = [m for m in group if m.id in referenced_as_base]
        if referenced_candidates:
            base_candidates = referenced_candidates
        else:
            base_candidates = [
                m for m in group if not m.gguf_variants or m.base_model is None
            ]
            if not base_candidates:
                base_candidates = group

        base = max(base_candidates, key=lambda m: m.downloads)
        variants = [m for m in group if m.id != base.id]

        family_id = normalize_name(base.id)
        base.family_id = family_id
        for v in variants:
            v.family_id = family_id

        family_artifacts, family_components, family_lineage = merge_family_graph(group)
        base.artifacts = family_artifacts or base.artifacts
        base.components = family_components or base.components
        base.lineage = family_lineage

        best_bench: dict[str, float] = {}
        for m in group:
            for k, score in m.benchmark_scores.items():
                if k not in best_bench or score > best_bench[k]:
                    best_bench[k] = score

        families.append(
            ModelFamily(
                family_id=family_id,
                display_name=base.name,
                base_model=base,
                variants=variants,
                best_benchmark=best_bench,
                artifacts=family_artifacts,
                components=family_components,
                lineage=family_lineage,
            )
        )

    return families
