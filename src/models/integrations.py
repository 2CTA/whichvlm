from __future__ import annotations

import re
from dataclasses import dataclass

from models.types import ModelCapabilities


@dataclass(frozen=True)
class IntegrationProfile:
    integration_id: str
    capability_names: tuple[str, ...]
    pipeline_tags: tuple[str, ...]
    tag_patterns: tuple[str, ...]
    component_roles: tuple[str, ...]
    workload_tasks: tuple[str, ...]
    runtime_backends: tuple[str, ...]


INTEGRATION_PROFILES: tuple[IntegrationProfile, ...] = (
    IntegrationProfile(
        integration_id="vision-language",
        capability_names=("image",),
        pipeline_tags=(
            "image-text-to-text",
            "visual-question-answering",
            "image-to-text",
        ),
        tag_patterns=(
            r"(^|[-_/\s])(vl|vision|multimodal|llava|image)([-_/\s]|$)",
            r"image-text-to-text|visual-question-answering|image-to-text",
            r"qwen.*vl|internvl|pixtral|deepseek[-_]vl",
            r"paligemma|idefics|mllama|phi3v|phi3_v|glm4v",
            r"xgenmm|fuyu|kosmos|instructblip|blip|florence",
        ),
        component_roles=("language", "vision_encoder", "projector", "processor"),
        workload_tasks=("image_qa", "general_multimodal"),
        runtime_backends=("transformers", "llama.cpp", "mlx", "vllm", "sglang"),
    ),
    IntegrationProfile(
        integration_id="document-ocr",
        capability_names=("ocr", "document"),
        pipeline_tags=(),
        tag_patterns=(
            r"(^|[-_/\s])(ocr|docvqa|document)([-_/\s]|$)",
            r"text[-_ ]?recognition",
        ),
        component_roles=("language", "vision_encoder", "projector", "processor"),
        workload_tasks=("ocr", "document"),
        runtime_backends=("transformers", "llama.cpp", "mlx", "vllm", "sglang"),
    ),
)

VISUAL_COMPONENT_ROLES = frozenset({"vision_encoder", "video_encoder", "projector"})
AUDIO_COMPONENT_ROLES = frozenset({"audio_encoder"})
VIDEO_PIPELINE_TAGS = ("video-text-to-text",)
AUDIO_PIPELINE_TAGS = ("audio-text-to-text", "automatic-speech-recognition")


def _append_unique(values: list[str], candidates: tuple[str, ...]) -> None:
    for candidate in candidates:
        if candidate not in values:
            values.append(candidate)


def _matches_profile(
    profile: IntegrationProfile,
    model_id: str,
    pipeline_tag: object,
    tags: list[str],
    architecture: str = "",
) -> bool:
    if pipeline_tag in profile.pipeline_tags:
        return True
    haystack = " ".join(
        [model_id, str(pipeline_tag or ""), *tags, architecture]
    ).lower()
    return any(re.search(pattern, haystack) for pattern in profile.tag_patterns)


def matching_profiles_for_data(
    model_id: str,
    pipeline_tag: object,
    tags: list[str],
    architecture: str = "",
) -> list[IntegrationProfile]:
    return [
        profile
        for profile in INTEGRATION_PROFILES
        if _matches_profile(profile, model_id, pipeline_tag, tags, architecture)
    ]


def capability_names_for_data(
    model_id: str,
    pipeline_tag: object,
    tags: list[str],
    architecture: str = "",
) -> list[str]:
    capabilities: list[str] = []
    for profile in matching_profiles_for_data(
        model_id,
        pipeline_tag,
        tags,
        architecture,
    ):
        _append_unique(capabilities, profile.capability_names)
    return capabilities


def enabled_capability_names(capabilities: ModelCapabilities) -> list[str]:
    names = []
    for name in (
        "image",
        "video",
        "audio",
        "ocr",
        "document",
        "chart",
        "multi_image",
        "tool_use",
    ):
        if getattr(capabilities, name):
            names.append(name)
    return names


def capabilities_for_data(
    model_id: str,
    pipeline_tag: object,
    tags: list[str],
    architecture: str = "",
    supported_languages: list[str] | None = None,
) -> ModelCapabilities:
    names = set(capability_names_for_data(model_id, pipeline_tag, tags, architecture))
    image = "image" in names or bool(names & {"ocr", "document", "chart"})
    return ModelCapabilities(
        image=image,
        video="video" in names,
        audio="audio" in names,
        ocr="ocr" in names,
        document="document" in names,
        chart="chart" in names,
        multi_image="multi_image" in names,
        tool_use="tool_use" in names,
        supported_languages=supported_languages or [],
    )


def component_roles_for_capabilities(capabilities: ModelCapabilities) -> list[str]:
    capability_names = set(enabled_capability_names(capabilities))
    roles: list[str] = []
    for profile in INTEGRATION_PROFILES:
        if capability_names & set(profile.capability_names):
            _append_unique(roles, profile.component_roles)
    return roles


def integration_ids_for_capabilities(capabilities: ModelCapabilities) -> list[str]:
    capability_names = set(enabled_capability_names(capabilities))
    return [
        profile.integration_id
        for profile in INTEGRATION_PROFILES
        if capability_names & set(profile.capability_names)
    ]


def runtime_backends_for_capabilities(capabilities: ModelCapabilities) -> list[str]:
    capability_names = set(enabled_capability_names(capabilities))
    backends: list[str] = []
    for profile in INTEGRATION_PROFILES:
        if capability_names & set(profile.capability_names):
            _append_unique(backends, profile.runtime_backends)
    return backends


def pipeline_tags_for_capabilities(capabilities: tuple[str, ...]) -> tuple[str, ...]:
    tags: list[str] = []
    for profile in INTEGRATION_PROFILES:
        if set(capabilities) & set(profile.capability_names):
            _append_unique(tags, profile.pipeline_tags)
    return tuple(tags)


def discovery_pipeline_tags() -> tuple[str, ...]:
    tags = list(pipeline_tags_for_capabilities(("image",)))
    _append_unique(tags, VIDEO_PIPELINE_TAGS)
    _append_unique(tags, ("audio-text-to-text",))
    return tuple(tags)


def specialization_tags_for_capabilities(capabilities: ModelCapabilities) -> set[str]:
    tags: set[str] = set()
    if capabilities.image or capabilities.video:
        tags.add("vision")
    if capabilities.ocr:
        tags.add("ocr")
    if capabilities.document:
        tags.add("document")
    if capabilities.chart:
        tags.add("chart")
    if capabilities.video:
        tags.add("video")
    if capabilities.audio:
        tags.add("audio")
    return tags


def specialization_tags_for_data(
    model_id: str,
    pipeline_tag: object,
    tags: list[str],
    architecture: str = "",
) -> set[str]:
    return specialization_tags_for_capabilities(
        capabilities_for_data(model_id, pipeline_tag, tags, architecture)
    )


def has_visual_input(capabilities: ModelCapabilities) -> bool:
    return capabilities.image or capabilities.video


def has_audio_input(capabilities: ModelCapabilities) -> bool:
    return capabilities.audio


def pipeline_tag_has_visual_input(pipeline_tag: object) -> bool:
    return pipeline_tag in {
        *pipeline_tags_for_capabilities(("image",)),
        *VIDEO_PIPELINE_TAGS,
    }


def pipeline_tag_has_audio_input(pipeline_tag: object) -> bool:
    return pipeline_tag in AUDIO_PIPELINE_TAGS
