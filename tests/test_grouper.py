from models.grouper import normalize_name, group_models
from models.types import ModelArtifact, ModelInfo


def make_model(
    id: str, base_model: str | None = None, downloads: int = 100
) -> ModelInfo:
    return ModelInfo(
        id=id,
        family_id=id,
        name=id.split("/")[-1],
        parameter_count=7_000_000_000,
        downloads=downloads,
        base_model=base_model,
    )


def test_group_by_base_model():
    base = make_model("meta/Llama-3-8B", downloads=1000)
    gguf = make_model(
        "user/Llama-3-8B-GGUF", base_model="meta/Llama-3-8B", downloads=500
    )
    families = group_models([base, gguf])
    assert len(families) == 1
    assert families[0].base_model.id == "meta/Llama-3-8B"
    assert [variant.id for variant in families[0].variants] == ["user/Llama-3-8B-GGUF"]


def test_group_by_name_normalization():
    base = make_model("org/model-v1", downloads=1000)
    gguf = make_model("org/model-v1-GGUF", downloads=200)
    families = group_models([base, gguf])
    assert len(families) == 1
    assert families[0].base_model.id == "org/model-v1"
    assert [variant.id for variant in families[0].variants] == ["org/model-v1-GGUF"]


def test_fp4_suffixes_normalize_to_base_family():
    base = normalize_name("openai/gpt-oss-20b")
    assert normalize_name("openai/gpt-oss-20b-MXFP4") == base
    assert normalize_name("openai/gpt-oss-20b-NVFP4") == base


def test_ungrouped_models_separate():
    m1 = make_model("org/alpha", downloads=100)
    m2 = make_model("org/beta", downloads=200)
    families = group_models([m1, m2])
    assert len(families) == 2


def test_empty_input():
    families = group_models([])
    assert families == []


def test_family_id_set():
    base = make_model("meta/Llama-3-8B", downloads=1000)
    gguf = make_model(
        "user/Llama-3-8B-GGUF", base_model="meta/Llama-3-8B", downloads=500
    )
    families = group_models([base, gguf])
    for family in families:
        assert family.family_id
        assert family.base_model.family_id == family.family_id


def test_vlm_variants_group_into_logical_model_package():
    base = ModelInfo(
        id="Qwen/Qwen2.5-VL-7B-Instruct",
        family_id="base",
        name="Qwen2.5-VL-7B-Instruct",
        parameter_count=7_000_000_000,
        hf_pipeline_tag="image-text-to-text",
        tags=["vision-language"],
        downloads=1000,
        artifacts=[
            ModelArtifact(
                repo_id="Qwen/Qwen2.5-VL-7B-Instruct",
                format="safetensors",
                access="ungated",
                backend_support=["cuda", "mps", "cpu"],
                source_kind="official",
            )
        ],
    )
    gguf = ModelInfo(
        id="community/Qwen2.5-VL-7B-GGUF",
        family_id="gguf",
        name="Qwen2.5-VL-7B-GGUF",
        parameter_count=7_000_000_000,
        base_model=base.id,
        base_models=[base.id],
        downloads=500,
        artifacts=[
            ModelArtifact(
                repo_id="community/Qwen2.5-VL-7B-GGUF",
                format="gguf",
                quantization="Q4_K_M",
                access="gated",
                backend_support=["metal", "cuda", "vulkan", "cpu"],
                source_kind="gguf_variant",
            )
        ],
    )
    mlx = ModelInfo(
        id="community/Qwen2.5-VL-7B-MLX",
        family_id="mlx",
        name="Qwen2.5-VL-7B-MLX",
        parameter_count=7_000_000_000,
        base_model=base.id,
        base_models=[base.id],
        downloads=200,
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

    families = group_models([gguf, mlx, base])

    assert len(families) == 1
    family = families[0]
    assert family.base_model.id == base.id
    assert {a.format for a in family.artifacts} == {"safetensors", "gguf", "mlx"}
    assert any(a.access == "gated" for a in family.artifacts)
    assert family.lineage.base_model_ids == [base.id]
    assert family.lineage.is_merged is False


def test_vlm_inventory_groups_known_family_aliases_without_base_model():
    official = make_model("llava-hf/llava-1.5-7b-hf", downloads=500)
    original = make_model("liuhaotian/llava-v1.5-7b", downloads=1000)

    families = group_models([official, original])

    assert len(families) == 1
    assert families[0].family_id == "llava"
    assert {families[0].base_model.id, *(v.id for v in families[0].variants)} == {
        "llava-hf/llava-1.5-7b-hf",
        "liuhaotian/llava-v1.5-7b",
    }
