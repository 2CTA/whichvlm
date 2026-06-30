from __future__ import annotations

import re
from dataclasses import dataclass

from whichvlm.models.types import ModelCapabilities


@dataclass(frozen=True)
class IntegrationProfile:
    integration_id: str
    capability_names: tuple[str, ...]
    pipeline_tags: tuple[str, ...]
    tag_patterns: tuple[str, ...]
    component_roles: tuple[str, ...]


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
            r"qwen.*vl|internvl|pixtral|deepseek-vl",
        ),
        component_roles=("language", "vision_encoder", "projector", "processor"),
    ),
    IntegrationProfile(
        integration_id="document-ocr",
        capability_names=("image", "ocr", "document"),
        pipeline_tags=(),
        tag_patterns=(
            r"(^|[-_/\s])(ocr|docvqa|document)([-_/\s]|$)",
            r"text[-_ ]?recognition",
        ),
        component_roles=("language", "vision_encoder", "projector", "processor"),
    ),
    IntegrationProfile(
        integration_id="video-language",
        capability_names=("image", "video"),
        pipeline_tags=("video-text-to-text",),
        tag_patterns=(r"video|onevision",),
        component_roles=("language", "video_encoder", "projector", "processor"),
    ),
    IntegrationProfile(
        integration_id="audio-language",
        capability_names=("audio",),
        pipeline_tags=("audio-text-to-text",),
        tag_patterns=(r"audio|speech|whisper",),
        component_roles=("language", "audio_encoder", "processor"),
    ),
)


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


def capability_names_for_data(
    model_id: str,
    pipeline_tag: object,
    tags: list[str],
    architecture: str = "",
) -> list[str]:
    capabilities: list[str] = []
    for profile in INTEGRATION_PROFILES:
        if _matches_profile(profile, model_id, pipeline_tag, tags, architecture):
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
    return ModelCapabilities(
        image="image" in names,
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


def pipeline_tags_for_capabilities(capabilities: tuple[str, ...]) -> tuple[str, ...]:
    tags: list[str] = []
    for profile in INTEGRATION_PROFILES:
        if set(capabilities) & set(profile.capability_names):
            _append_unique(tags, profile.pipeline_tags)
    return tuple(tags)
