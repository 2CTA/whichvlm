from engine.quantization import (
    effective_quant_type,
    estimate_weight_bytes,
    infer_non_gguf_quant_type,
)
from engine.vram import estimate_vram
from models.types import ModelComponent, ModelInfo


def make_model(model_id: str, params: int = 14_000_000_000) -> ModelInfo:
    return ModelInfo(
        id=model_id,
        family_id=model_id,
        name=model_id.split("/")[-1],
        parameter_count=params,
    )


def test_infer_non_gguf_awq():
    model = make_model("Qwen/Qwen2.5-14B-Instruct-AWQ")
    assert infer_non_gguf_quant_type(model.id) == "AWQ"
    assert effective_quant_type(model, None) == "AWQ"


def test_estimate_weight_bytes_for_awq():
    model = make_model("Qwen/Qwen2.5-14B-Instruct-AWQ", params=10_000_000_000)
    assert estimate_weight_bytes(model, None) == 5_000_000_000


def test_awq_vram_is_lower_than_fp16_fallback():
    awq = make_model("Qwen/Qwen2.5-14B-Instruct-AWQ")
    fp16 = make_model("Qwen/Qwen2.5-14B-Instruct")
    assert estimate_vram(awq, None, context_length=4096) < estimate_vram(
        fp16, None, context_length=4096
    )


def test_infer_mxfp4():
    model = make_model("openai/gpt-oss-20b-MXFP4")
    assert infer_non_gguf_quant_type(model.id) == "MXFP4"
    assert effective_quant_type(model, None) == "MXFP4"


def test_infer_nvfp4():
    model = make_model("nvidia/Llama-3.1-8B-Instruct-NVFP4")
    assert infer_non_gguf_quant_type(model.id) == "NVFP4"
    assert effective_quant_type(model, None) == "NVFP4"


def test_fp4_patterns_do_not_false_match_plain_ids():
    plain = make_model("meta-llama/Llama-3.1-8B-Instruct")
    assert infer_non_gguf_quant_type(plain.id) == "FP16"


def test_estimate_weight_bytes_for_fp4_formats():
    mxfp4 = make_model("openai/gpt-oss-20b-MXFP4", params=20_000_000_000)
    nvfp4 = make_model("nvidia/model-NVFP4", params=20_000_000_000)
    assert estimate_weight_bytes(mxfp4, None) == int(20_000_000_000 * 0.53125)
    assert estimate_weight_bytes(nvfp4, None) == int(20_000_000_000 * 0.5625)


def test_fp4_vram_is_lower_than_fp16_fallback():
    mxfp4 = make_model("openai/gpt-oss-20b-MXFP4")
    fp16 = make_model("openai/gpt-oss-20b")
    assert estimate_vram(mxfp4, None, context_length=4096) < estimate_vram(
        fp16, None, context_length=4096
    )


def test_extract_quant_type_parses_fp4_gguf_filenames():
    from models.fetcher import extract_quant_type

    assert extract_quant_type("gpt-oss-20b-MXFP4.gguf") == "MXFP4"
    assert extract_quant_type("model.NVFP4.gguf") == "NVFP4"


def test_estimate_weight_bytes_uses_component_quantization():
    model = make_model("org/Test-VL-7B", params=7_000_000_000)
    model.components = [
        ModelComponent(
            role="language",
            repo_id=model.id,
            parameter_count=6_000_000_000,
            quantization="AWQ",
        ),
        ModelComponent(
            role="vision_encoder",
            repo_id=model.id,
            parameter_count=900_000_000,
            quantization="FP16",
        ),
        ModelComponent(role="processor", repo_id=model.id),
    ]

    expected = int(6_000_000_000 * 0.5 + 900_000_000 * 2.0 + 100_000_000 * 2.0)
    assert estimate_weight_bytes(model, None) == expected
