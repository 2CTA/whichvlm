"""Curated seed data for VLM family grouping.

This is deliberately small. It gives the fetcher/grouper stable family anchors
without making ranking claims or trying to solve the full model inventory.
"""

from __future__ import annotations

from typing import Final


VLM_FAMILY_SEEDS: Final[dict[str, dict[str, tuple[str, ...]]]] = {
    "qwen-vl": {
        "canonical": (
            "Qwen/Qwen2-VL-7B-Instruct",
            "Qwen/Qwen2-VL-72B-Instruct",
            "Qwen/Qwen2.5-VL-3B-Instruct",
            "Qwen/Qwen2.5-VL-7B-Instruct",
            "Qwen/Qwen2.5-VL-32B-Instruct",
            "Qwen/Qwen2.5-VL-72B-Instruct",
            "Qwen/Qwen3-VL-4B-Instruct",
            "Qwen/Qwen3-VL-8B-Instruct",
            "Qwen/Qwen3-VL-30B-A3B-Instruct",
            "Qwen/Qwen3-VL-32B-Instruct",
            "Qwen/Qwen3-VL-235B-A22B-Instruct",
        ),
        "aliases": ("qwen-vl", "qwen2-vl", "qwen2.5-vl", "qwen3-vl"),
    },
    "internvl": {
        "canonical": (
            "OpenGVLab/InternVL2_5-78B",
            "OpenGVLab/InternVL2_5-8B",
            "OpenGVLab/InternVL3-8B",
            "OpenGVLab/InternVL3-14B",
            "OpenGVLab/InternVL3-38B",
            "OpenGVLab/InternVL3-78B",
        ),
        "aliases": ("internvl",),
    },
    "gemma-multimodal": {
        "canonical": (
            "google/gemma-3-4b-it",
            "google/gemma-3-12b-it",
            "google/gemma-3-27b-it",
        ),
        "aliases": ("gemma-3", "gemma3"),
    },
    "llama-vision": {
        "canonical": (
            "meta-llama/Llama-3.2-11B-Vision-Instruct",
            "meta-llama/Llama-3.2-90B-Vision-Instruct",
        ),
        "aliases": ("llama-vision", "llama-3.2-vision"),
    },
    "pixtral": {
        "canonical": (
            "mistralai/Pixtral-12B-2409",
            "mistralai/Pixtral-Large-Instruct-2411",
        ),
        "aliases": ("pixtral",),
    },
    "phi-vision": {
        "canonical": (
            "microsoft/Phi-3.5-vision-instruct",
            "microsoft/Phi-4-multimodal-instruct",
            "microsoft/Phi-4-reasoning-vision-15B",
        ),
        "aliases": ("phi-vision", "phi-3.5-vision", "phi-4-multimodal"),
    },
    "deepseek-vl": {
        "canonical": (
            "deepseek-ai/deepseek-vl-7b-chat",
            "deepseek-ai/deepseek-vl2",
        ),
        "aliases": ("deepseek-vl", "deepseek-vl2"),
    },
    "glm-vision": {
        "canonical": (
            "zai-org/GLM-4.5V",
            "THUDM/glm-4v-9b",
        ),
        "aliases": ("glm-4v", "glm-4.5v", "glm-vision"),
    },
    "llava": {
        "canonical": (
            "liuhaotian/llava-v1.5-7b",
            "liuhaotian/llava-v1.5-13b",
            "llava-hf/llava-1.5-7b-hf",
        ),
        "aliases": ("llava", "bakllava", "llava-next", "llava-onevision"),
    },
}

VLM_SEED_MODEL_IDS: Final[tuple[str, ...]] = (
    "mistral-community/pixtral-12b",
    "THUDM/glm-4v-9b",
)


def normalize_vlm_match_text(value: str) -> str:
    return value.casefold().replace("_", "-")


def canonical_vlm_family_id(model_id: str) -> str | None:
    value = normalize_vlm_match_text(model_id)
    name = value.split("/", 1)[1] if "/" in value else value
    for family_id, data in VLM_FAMILY_SEEDS.items():
        if any(
            value == normalize_vlm_match_text(repo_id) for repo_id in data["canonical"]
        ):
            return family_id
        if any(alias in name for alias in data["aliases"]):
            return family_id
    return None


def known_vlm_model_ids() -> tuple[str, ...]:
    ids: list[str] = []
    for data in VLM_FAMILY_SEEDS.values():
        ids.extend(data["canonical"])
    ids.extend(VLM_SEED_MODEL_IDS)
    return tuple(dict.fromkeys(ids))
