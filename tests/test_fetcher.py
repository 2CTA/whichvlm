import asyncio

import models.fetcher as fetcher_mod
from models.fetcher import (
    extract_hf_eval_score,
    extract_published_at,
    normalize_param_count,
    parse_model,
    dicts_to_models,
    fetch_models,
    inventory_source_provenance,
    models_to_dicts,
)
from models.types import ModelArtifact, ModelInfo


def test_normalize_param_count_for_quantized_repo_uses_size_hint():
    corrected = normalize_param_count(
        extracted=5_233_828_308,
        model_id="ISTA-DASLab/gemma-3-27b-it-GPTQ-4b-128g",
        base_model="google/gemma-3-27b-it",
    )
    assert corrected == 27_000_000_000


def test_normalize_param_count_keeps_reasonable_value():
    kept = normalize_param_count(
        extracted=11_765_788_416,
        model_id="community-quants/gemma-3-12b-it-GGUF",
        base_model="google/gemma-3-12b-it",
    )
    assert kept == 11_765_788_416


def test_normalize_param_count_with_no_hint_keeps_original():
    kept = normalize_param_count(
        extracted=3_820_000_000,
        model_id="microsoft/Phi-3-mini-4k-instruct-gguf",
        base_model=None,
    )
    assert kept == 3_820_000_000


def test_dicts_to_models_normalizes_cached_parameter_count():
    models = dicts_to_models(
        [
            {
                "id": "ISTA-DASLab/gemma-3-27b-it-GPTQ-4b-128g",
                "family_id": "gemma-3-27b",
                "name": "gemma-3-27b-it-GPTQ-4b-128g",
                "parameter_count": 5_233_828_308,
                "downloads": 1,
                "likes": 1,
                "gguf_variants": [],
                "benchmark_scores": {},
                "base_model": "google/gemma-3-27b-it",
            }
        ]
    )
    assert len(models) == 1
    assert models[0].parameter_count == 27_000_000_000


def test_dicts_to_models_refreshes_cached_deepseek_v4_flash_counts():
    models = dicts_to_models(
        [
            {
                "id": "deepseek-ai/DeepSeek-V4-Flash",
                "family_id": "deepseek-v4-flash",
                "name": "DeepSeek-V4-Flash",
                "parameter_count": 158_069_433_298,
                "parameter_count_active": 10_000_000_000,
                "downloads": 1,
                "likes": 1,
                "gguf_variants": [],
                "benchmark_scores": {},
            }
        ]
    )

    assert len(models) == 1
    assert models[0].parameter_count == 284_000_000_000
    assert models[0].parameter_count_active == 13_000_000_000
    assert models[0].is_moe is True


def test_dicts_to_models_uses_case_insensitive_curated_active_params():
    models = dicts_to_models(
        [
            {
                "id": "google/gemma-4-26B-A4B-it",
                "family_id": "gemma-4-26b-a4b",
                "name": "gemma-4-26B-A4B-it",
                "parameter_count": 26_544_131_376,
                "parameter_count_active": None,
                "downloads": 1,
                "likes": 1,
                "gguf_variants": [],
                "benchmark_scores": {},
            }
        ]
    )

    assert len(models) == 1
    assert models[0].parameter_count_active == 3_800_000_000
    assert models[0].is_moe is True


def test_dicts_to_models_recovers_a3b_active_params_from_cached_qwen_model():
    models = dicts_to_models(
        [
            {
                "id": "Qwen/Qwen3.6-35B-A3B",
                "family_id": "qwen3.6-35b-a3b",
                "name": "Qwen3.6-35B-A3B",
                "parameter_count": 35_951_822_704,
                "parameter_count_active": None,
                "architecture": "qwen3_5moe",
                "downloads": 1,
                "likes": 1,
                "gguf_variants": [],
                "benchmark_scores": {},
            }
        ]
    )

    assert len(models) == 1
    assert models[0].parameter_count_active == 3_000_000_000
    assert models[0].is_moe is True


def test_dicts_to_models_recovers_a3b_active_params_from_base_model():
    models = dicts_to_models(
        [
            {
                "id": "local/Qwen36-GGUF",
                "family_id": "qwen36-gguf",
                "name": "Qwen36-GGUF",
                "parameter_count": 34_660_610_688,
                "parameter_count_active": None,
                "architecture": "qwen35moe",
                "downloads": 1,
                "likes": 1,
                "gguf_variants": [],
                "benchmark_scores": {},
                "base_model": "Qwen/Qwen3.6-35B-A3B",
            }
        ]
    )

    assert len(models) == 1
    assert models[0].parameter_count_active == 3_000_000_000
    assert models[0].is_moe is True


def test_dicts_to_models_refreshes_stale_xiaomi_moe_cache_counts():
    models = dicts_to_models(
        [
            {
                "id": "XiaomiMiMo/MiMo-V2.5-Pro",
                "family_id": "mimo-v2.5-pro",
                "name": "MiMo-V2.5-Pro",
                "parameter_count": 58_000_000_000,
                "parameter_count_active": 11_000_000_000,
                "downloads": 1,
                "likes": 1,
                "gguf_variants": [],
                "benchmark_scores": {},
            }
        ]
    )

    assert len(models) == 1
    assert models[0].parameter_count == 1_020_000_000_000
    assert models[0].parameter_count_active == 42_000_000_000
    assert models[0].is_moe is True


def test_dicts_to_models_recovers_missing_known_parameter_count():
    models = dicts_to_models(
        [
            {
                "id": "zai-org/GLM-5",
                "family_id": "glm-5",
                "name": "GLM-5",
                "parameter_count": 0,
                "parameter_count_active": None,
                "downloads": 1,
                "likes": 1,
                "gguf_variants": [],
                "benchmark_scores": {},
            }
        ]
    )

    assert len(models) == 1
    assert models[0].parameter_count == 744_000_000_000
    assert models[0].parameter_count_active == 40_000_000_000
    assert models[0].is_moe is True


def test_dicts_to_models_restores_capability_components_from_architecture():
    models = dicts_to_models(
        [
            {
                "id": "org/Archived-7B",
                "family_id": "archived-7b",
                "name": "Archived-7B",
                "parameter_count": 7_000_000_000,
                "architecture": "qwen2vl",
                "downloads": 1,
                "likes": 1,
                "gguf_variants": [],
                "benchmark_scores": {},
            }
        ]
    )

    assert models[0].capabilities.image is True
    assert {component.role for component in models[0].components} >= {
        "vision_encoder",
        "projector",
        "processor",
    }


def test_parse_model_uses_current_glm5_and_xiaomi_active_counts():
    glm = parse_model(
        {
            "id": "zai-org/GLM-5",
            "config": {"architectures": ["GlmForCausalLM"]},
            "safetensors": {"total": 753_864_139_008},
            "siblings": [],
            "cardData": {},
        }
    )
    mimo = parse_model(
        {
            "id": "XiaomiMiMo/MiMo-V2-Flash",
            "config": {"architectures": ["LlamaForCausalLM"]},
            "safetensors": {"total": 309_785_318_400},
            "siblings": [],
            "cardData": {},
        }
    )

    assert glm is not None
    assert glm.parameter_count_active == 40_000_000_000
    assert mimo is not None
    assert mimo.parameter_count_active == 15_000_000_000


def test_parse_model_recovers_qwen36_a3b_active_params_from_name():
    parsed = parse_model(
        {
            "id": "Qwen/Qwen3.6-35B-A3B",
            "config": {
                "architectures": ["Qwen3_5MoeForConditionalGeneration"],
                "model_type": "qwen3_5_moe",
            },
            "safetensors": {"total": 35_951_822_704},
            "siblings": [],
            "cardData": {},
        }
    )

    assert parsed is not None
    assert parsed.parameter_count == 35_951_822_704
    assert parsed.parameter_count_active == 3_000_000_000
    assert parsed.is_moe is True


def test_models_cache_roundtrip_keeps_published_at():
    models = [
        ModelInfo(
            id="Qwen/Qwen3-8B-AWQ",
            family_id="qwen3-8b",
            name="Qwen3-8B-AWQ",
            parameter_count=8_000_000_000,
            published_at="2025-09-17T12:34:56.000Z",
            downloads=123_456,
            likes=789,
        )
    ]
    cached = models_to_dicts(models)
    restored = dicts_to_models(cached)
    assert len(restored) == 1
    assert restored[0].published_at == "2025-09-17T12:34:56.000Z"
    assert restored[0].downloads == 123_456


def test_models_cache_roundtrip_keeps_architecture_metadata():
    model = ModelInfo(
        id="Qwen/Qwen2.5-VL-7B-Instruct",
        family_id="qwen2.5-vl-7b",
        name="Qwen2.5-VL-7B-Instruct",
        parameter_count=7_000_000_000,
        layer_count=28,
        hidden_size=3584,
        intermediate_size=18944,
        attention_heads=28,
        kv_heads=4,
        head_dim=128,
        dtype="bfloat16",
        vision_layer_count=32,
        vision_hidden_size=1280,
        vision_intermediate_size=3420,
        vision_attention_heads=16,
        projector_hidden_size=3584,
        patch_size=14,
        spatial_merge_size=2,
        image_token_strategy="full",
    )

    restored = dicts_to_models(models_to_dicts([model]))

    assert restored[0].layer_count == 28
    assert restored[0].intermediate_size == 18944
    assert restored[0].kv_heads == 4
    assert restored[0].vision_hidden_size == 1280
    assert restored[0].vision_intermediate_size == 3420
    assert restored[0].vision_attention_heads == 16
    assert restored[0].patch_size == 14
    assert restored[0].spatial_merge_size == 2


def test_models_cache_roundtrip_keeps_vlm_package_graph():
    models = [
        ModelInfo(
            id="community/Qwen2.5-VL-7B-MLX",
            family_id="qwen2.5-vl-7b",
            name="Qwen2.5-VL-7B-MLX",
            parameter_count=7_000_000_000,
            base_model="Qwen/Qwen2.5-VL-7B-Instruct",
            base_models=["Qwen/Qwen2.5-VL-7B-Instruct"],
            artifacts=[
                ModelArtifact(
                    repo_id="community/Qwen2.5-VL-7B-MLX",
                    format="mlx",
                    quantization="MLX",
                    access="ungated",
                    backend_support=["mlx", "metal"],
                    source_kind="mlx_variant",
                )
            ],
        )
    ]

    restored = dicts_to_models(models_to_dicts(models))

    assert restored[0].base_models == ["Qwen/Qwen2.5-VL-7B-Instruct"]
    assert restored[0].artifacts[0].format == "mlx"
    assert restored[0].artifacts[0].backend_support == ["mlx", "metal"]
    assert restored[0].lineage.variant_of == "Qwen/Qwen2.5-VL-7B-Instruct"


def test_extract_published_at_prefers_created_at():
    value = extract_published_at(
        {
            "createdAt": "2025-01-01T00:00:00.000Z",
            "lastModified": "2026-01-01T00:00:00.000Z",
        }
    )
    assert value == "2025-01-01T00:00:00.000Z"


def test_extract_published_at_falls_back_to_last_modified():
    value = extract_published_at(
        {
            "lastModified": "2026-01-01T00:00:00.000Z",
        }
    )
    assert value == "2026-01-01T00:00:00.000Z"


def test_parse_model_keeps_split_gguf_as_single_variant():
    parsed = parse_model(
        {
            "id": "org/Test-8B-GGUF",
            "config": {
                "architectures": ["LlamaForCausalLM"],
            },
            "safetensors": {"total": 8_000_000_000},
            "siblings": [
                {
                    "rfilename": "model-Q4_K_M-00001-of-00002.gguf",
                    "size": 2_000_000_000,
                },
                {
                    "rfilename": "model-Q4_K_M-00002-of-00002.gguf",
                    "size": 2_500_000_000,
                },
                {"rfilename": "model-Q8_0.gguf", "size": 8_000_000_000},
            ],
            "cardData": {},
        }
    )
    assert parsed is not None
    q4 = [v for v in parsed.gguf_variants if v.quant_type == "Q4_K_M"]
    q8 = [v for v in parsed.gguf_variants if v.quant_type == "Q8_0"]
    assert len(q4) == 1
    assert len(q8) == 1
    assert q4[0].file_size_bytes == 4_500_000_000


def test_extract_hf_eval_score_uses_general_datasets_and_median():
    score = extract_hf_eval_score(
        {
            "evalResults": [
                {
                    "filename": ".eval_results/mmlu-pro.yaml",
                    "data": {"dataset": {"id": "TIGER-Lab/MMLU-Pro"}, "value": 48.3},
                },
                {
                    "filename": ".eval_results/gsm8k.yaml",
                    "data": {"dataset": {"id": "openai/gsm8k"}, "value": 84.5},
                },
                {
                    "filename": ".eval_results/hle_medium_with_tools.yaml",
                    "data": {
                        "dataset": {"id": "cais/hle"},
                        "value": 99.0,
                        "notes": "Reasoning: medium, With tools",
                    },
                },
                {
                    "filename": ".eval_results/swe_bench.yaml",
                    "data": {
                        "dataset": {"id": "SWE-bench/SWE-bench_Verified"},
                        "value": 53.2,
                    },
                },
            ]
        }
    )

    assert score == 66.4


def test_parse_model_extracts_hf_eval_benchmark_score():
    parsed = parse_model(
        {
            "id": "meta-llama/Llama-3.1-8B-Instruct",
            "config": {"architectures": ["LlamaForCausalLM"]},
            "safetensors": {"total": 8_000_000_000},
            "siblings": [],
            "cardData": {},
            "evalResults": [
                {
                    "filename": ".eval_results/mmlu-pro.yaml",
                    "data": {"dataset": {"id": "TIGER-Lab/MMLU-Pro"}, "value": 48.3},
                },
                {
                    "filename": ".eval_results/gsm8k.yaml",
                    "data": {"dataset": {"id": "openai/gsm8k"}, "value": 84.5},
                },
            ],
        }
    )
    assert parsed is not None
    assert parsed.benchmark_scores.get("hf_eval") == 66.4


def test_parse_model_builds_vlm_package_metadata():
    parsed = parse_model(
        {
            "id": "Qwen/Qwen2.5-VL-7B-Instruct",
            "pipeline_tag": "image-text-to-text",
            "tags": ["vision-language", "safetensors"],
            "gated": False,
            "config": {"architectures": ["Qwen2VLForConditionalGeneration"]},
            "safetensors": {"total": 7_000_000_000},
            "siblings": [],
            "cardData": {"license": "apache-2.0"},
        }
    )

    assert parsed is not None
    assert parsed.hf_pipeline_tag == "image-text-to-text"
    assert parsed.access == "ungated"
    assert parsed.is_official is True
    assert parsed.model_format == "safetensors"
    assert parsed.variant_kind == "official"
    assert parsed.tags == ["vision-language", "safetensors"]
    assert parsed.capabilities.image is True
    assert parsed.artifacts[0].format == "safetensors"
    assert parsed.artifacts[0].access == "ungated"
    assert parsed.components[0].role == "language"
    assert {c.role for c in parsed.components} >= {
        "vision_encoder",
        "projector",
        "processor",
    }
    assert parsed.lineage.base_model_ids == []


def test_parse_model_uses_integration_registry_for_ocr_capabilities():
    parsed = parse_model(
        {
            "id": "org/DocVQA-OCR-7B",
            "pipeline_tag": "image-to-text",
            "tags": ["document", "ocr", "safetensors"],
            "config": {"architectures": ["Qwen2VLForConditionalGeneration"]},
            "safetensors": {"total": 7_000_000_000},
            "siblings": [],
            "cardData": {},
        }
    )

    assert parsed is not None
    assert parsed.capabilities.image is True
    assert parsed.capabilities.ocr is True
    assert parsed.capabilities.document is True
    assert {component.role for component in parsed.components} >= {
        "language",
        "vision_encoder",
        "projector",
        "processor",
    }


def test_parse_model_detects_transformers_vlm_from_architecture():
    parsed = parse_model(
        {
            "id": "org/ConfigOnly-3B",
            "tags": ["transformers", "safetensors"],
            "config": {
                "architectures": ["PaliGemmaForConditionalGeneration"],
                "model_type": "paligemma",
            },
            "safetensors": {"total": 3_000_000_000},
            "siblings": [],
            "cardData": {},
        }
    )

    assert parsed is not None
    assert parsed.hf_pipeline_tag is None
    assert parsed.architecture == "paligemma"
    assert parsed.capabilities.image
    assert {c.role for c in parsed.components} >= {
        "vision_encoder",
        "projector",
        "processor",
    }


def test_inventory_discovery_uses_registered_vision_pipelines():
    provenance = inventory_source_provenance(include_vision=True)

    assert provenance["pipeline_tags"][:3] == [
        "image-text-to-text",
        "visual-question-answering",
        "image-to-text",
    ]
    assert "video-text-to-text" in provenance["pipeline_tags"]
    assert "audio-text-to-text" in provenance["pipeline_tags"]
    assert "automatic-speech-recognition" not in provenance["pipeline_tags"]


def test_parse_model_extracts_architecture_metadata():
    parsed = parse_model(
        {
            "id": "Qwen/Qwen2.5-VL-7B-Instruct",
            "pipeline_tag": "image-text-to-text",
            "tags": ["vision-language", "safetensors"],
            "config": {
                "architectures": ["Qwen2VLForConditionalGeneration"],
                "num_hidden_layers": 28,
                "hidden_size": 3584,
                "intermediate_size": 18944,
                "num_attention_heads": 28,
                "num_key_value_heads": 4,
                "head_dim": 128,
                "torch_dtype": "bfloat16",
                "vision_feature_select_strategy": "full",
                "vision_config": {
                    "num_hidden_layers": 32,
                    "hidden_size": 1280,
                    "intermediate_size": 3420,
                    "num_attention_heads": 16,
                    "patch_size": 14,
                    "spatial_merge_size": 2,
                },
            },
            "safetensors": {"total": 7_000_000_000},
            "siblings": [],
            "cardData": {},
        }
    )

    assert parsed is not None
    assert parsed.layer_count == 28
    assert parsed.hidden_size == 3584
    assert parsed.intermediate_size == 18944
    assert parsed.attention_heads == 28
    assert parsed.kv_heads == 4
    assert parsed.head_dim == 128
    assert parsed.dtype == "bfloat16"
    assert parsed.vision_layer_count == 32
    assert parsed.vision_hidden_size == 1280
    assert parsed.vision_intermediate_size == 3420
    assert parsed.vision_attention_heads == 16
    assert parsed.patch_size == 14
    assert parsed.spatial_merge_size == 2
    assert parsed.image_token_strategy == "full"


def test_parse_model_marks_community_gguf_relationship():
    parsed = parse_model(
        {
            "id": "community/Qwen2.5-VL-7B-GGUF",
            "pipeline_tag": "image-text-to-text",
            "tags": ["gguf", "vision-language"],
            "gated": "auto",
            "config": {"architectures": ["Qwen2VLForConditionalGeneration"]},
            "safetensors": {"total": 7_000_000_000},
            "siblings": [
                {"rfilename": "qwen2.5-vl-7b-q4_k_m.gguf", "size": 4_500_000_000}
            ],
            "cardData": {"base_model": "Qwen/Qwen2.5-VL-7B-Instruct"},
        }
    )

    assert parsed is not None
    assert parsed.access == "gated"
    assert parsed.is_official is False
    assert parsed.model_format == "gguf"
    assert parsed.variant_kind == "gguf_variant"
    assert parsed.quantization_type == "GGUF"
    assert parsed.variant_of == "Qwen/Qwen2.5-VL-7B-Instruct"
    assert parsed.base_model == "Qwen/Qwen2.5-VL-7B-Instruct"
    assert parsed.base_models == ["Qwen/Qwen2.5-VL-7B-Instruct"]
    assert parsed.artifacts[0].format == "gguf"
    assert parsed.artifacts[0].access == "gated"
    assert "metal" in parsed.artifacts[0].backend_support
    assert parsed.lineage.variant_of == "Qwen/Qwen2.5-VL-7B-Instruct"
    assert parsed.lineage.is_merged is False


def test_parse_model_records_mmproj_artifact_separately_from_gguf_variants():
    parsed = parse_model(
        {
            "id": "community/LLaVA-7B-GGUF",
            "pipeline_tag": "image-text-to-text",
            "tags": ["gguf", "vision-language"],
            "gated": False,
            "config": {"architectures": ["LlavaForConditionalGeneration"]},
            "safetensors": {"total": 7_000_000_000},
            "siblings": [
                {"rfilename": "llava-7b-q4_k_m.gguf", "size": 4_500_000_000},
                {"rfilename": "mmproj-llava-7b-f16.gguf", "size": 400_000_000},
            ],
            "cardData": {"base_model": "liuhaotian/llava-v1.5-7b"},
        }
    )

    assert parsed is not None
    assert len(parsed.gguf_variants) == 1
    assert parsed.gguf_variants[0].filename == "llava-7b-q4_k_m.gguf"
    projector_artifacts = [
        artifact for artifact in parsed.artifacts if artifact.source_kind == "mmproj"
    ]
    assert len(projector_artifacts) == 1
    assert projector_artifacts[0].filename == "mmproj-llava-7b-f16.gguf"
    assert projector_artifacts[0].format == "adapter"


def test_parse_model_preserves_multi_parent_merged_lineage():
    parsed = parse_model(
        {
            "id": "community/Fused-VL-Model",
            "pipeline_tag": "image-text-to-text",
            "tags": ["vision-language", "merge"],
            "gated": False,
            "config": {"architectures": ["Qwen2VLForConditionalGeneration"]},
            "safetensors": {"total": 8_000_000_000},
            "siblings": [],
            "cardData": {
                "base_model": [
                    "Qwen/Qwen2.5-VL-7B-Instruct",
                    "openai/clip-vit-large-patch14",
                ],
                "base_model_relation": "merged",
            },
        }
    )

    assert parsed is not None
    assert parsed.base_model == "Qwen/Qwen2.5-VL-7B-Instruct"
    assert parsed.base_models == [
        "Qwen/Qwen2.5-VL-7B-Instruct",
        "openai/clip-vit-large-patch14",
    ]
    assert parsed.variant_kind == "merged_model"
    assert parsed.lineage.is_merged is True
    assert parsed.lineage.merged_parent_ids == parsed.base_models
    assert parsed.components[0].role == "merged_checkpoint"


def test_deepseek_v4_flash_uses_model_card_counts_over_hf_tensor_metadata():
    parsed = parse_model(
        {
            "id": "deepseek-ai/DeepSeek-V4-Flash",
            "config": {
                "architectures": ["DeepseekV4ForCausalLM"],
                "model_type": "deepseek_v4",
                "quantization_config": {"quant_method": "fp8"},
            },
            "safetensors": {
                "total": 158_069_433_298,
                "parameters": {
                    "BF16": 1_415_259_264,
                    "F8_E8M0": 8_858_737_664,
                    "F8_E4M3": 6_023_020_544,
                    "I8": 141_733_920_768,
                },
            },
            "siblings": [],
            "cardData": {},
        }
    )

    assert parsed is not None
    assert parsed.parameter_count == 284_000_000_000
    assert parsed.parameter_count_active == 13_000_000_000


def test_fetch_models_backfills_explicit_vlm_seed_details(monkeypatch):
    seed_id = "Qwen/Qwen3-VL-235B-A22B-Instruct"
    seen_urls: list[str] = []

    class Response:
        def __init__(self, data, status_code: int = 200):
            self.data = data
            self.status_code = status_code

        def raise_for_status(self):
            if self.status_code >= 400:
                raise AssertionError(self.status_code)

        def json(self):
            return self.data

    async def fake_get(client, url, params=None):
        seen_urls.append(url)
        if url.endswith("/models"):
            return Response([])
        if url.endswith(f"/models/{seed_id}"):
            return Response(
                {
                    "id": seed_id,
                    "pipeline_tag": "image-text-to-text",
                    "tags": ["vision-language"],
                    "config": {"architectures": ["Qwen2VLForConditionalGeneration"]},
                    "safetensors": {"total": 235_000_000_000},
                    "siblings": [],
                    "cardData": {},
                    "downloads": 10,
                    "likes": 1,
                }
            )
        return Response({}, status_code=404)

    monkeypatch.setattr(fetcher_mod, "get_with_retries", fake_get)
    monkeypatch.setattr(fetcher_mod, "known_vlm_model_ids", lambda: (seed_id,))

    models = asyncio.run(fetch_models(limit=1, include_vision=True))

    assert [model.id for model in models] == [seed_id]
    assert any(url.endswith(f"/models/{seed_id}") for url in seen_urls)
