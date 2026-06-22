from __future__ import annotations

from whichvlm.engine.ranker import (
    SOURCE_WEIGHTS,
    derivative_name_penalty,
    generation_bonus,
    is_excluded_model,
    rank_models,
)
from whichvlm.hardware.types import GPUInfo, HardwareInfo
from whichvlm.models.types import GGUFVariant, ModelInfo


def hw(
    vram_gb: int = 24,
    bandwidth_gbps: float = 1000.0,
    vendor: str = "nvidia",
    os_name: str = "linux",
    with_gpu: bool = True,
) -> HardwareInfo:
    gpus = []
    if with_gpu:
        gpus = [
            GPUInfo(
                name="Test GPU",
                vendor=vendor,
                vram_bytes=vram_gb * 1024**3,
                compute_capability=(8, 9) if vendor == "nvidia" else None,
                memory_bandwidth_gbps=bandwidth_gbps,
            )
        ]
    return HardwareInfo(
        gpus=gpus,
        cpu_name="Test CPU",
        cpu_cores=8,
        has_avx2=True,
        ram_bytes=64 * 1024**3,
        disk_free_bytes=500 * 1024**3,
        os=os_name,
    )


def gguf(quant: str, size_gb: float) -> GGUFVariant:
    return GGUFVariant(
        filename=f"model-{quant}.gguf",
        quant_type=quant,
        file_size_bytes=int(size_gb * 1e9),
    )


def test_q1_0_only_repo_is_severely_penalized_when_no_quant_filter():

    weak_quant = ModelInfo(
        id="fixture-org/Weak-Quant-8B-GGUF",
        family_id="weak-quant-8b",
        name="Weak-Quant-8B",
        parameter_count=8_000_000_000,
        downloads=48_000,
        likes=0,
        gguf_variants=[gguf("Q1_0", 4.6)],
        benchmark_scores={"hf_eval": 88.0},
    )
    normal = ModelInfo(
        id="community-quants/Qwen3-8B-GGUF",
        family_id="qwen3-8b",
        name="Qwen3-8B",
        parameter_count=8_000_000_000,
        downloads=58_000,
        likes=20,
        base_model="Qwen/Qwen3-8B",
        gguf_variants=[gguf("Q4_K_M", 4.7)],
    )
    results = rank_models(
        [weak_quant, normal],
        hw(),
        top_n=5,
        benchmark_scores={"Qwen/Qwen3-8B-Instruct": 65.0},
    )
    ids = [r.model.id for r in results]
    assert "community-quants/Qwen3-8B-GGUF" in ids

    sc_weak_quant = next(
        (
            r.quality_score
            for r in results
            if r.model.id == "fixture-org/Weak-Quant-8B-GGUF"
        ),
        0.0,
    )
    sc_normal = next(
        (
            r.quality_score
            for r in results
            if r.model.id == "community-quants/Qwen3-8B-GGUF"
        ),
        0.0,
    )
    assert sc_normal > sc_weak_quant

    assert sc_weak_quant < 25


def test_q1_0_returned_when_explicitly_requested_via_quant_filter():

    model = ModelInfo(
        id="fixture-org/Weak-Quant-8B-GGUF",
        family_id="weak-quant-8b",
        name="Weak-Quant-8B",
        parameter_count=8_000_000_000,
        downloads=48_000,
        likes=0,
        gguf_variants=[gguf("Q1_0", 4.6)],
    )
    results = rank_models(
        [model], hw(), top_n=5, benchmark_scores={}, quant_filter="Q1_0"
    )
    assert len(results) == 1
    assert results[0].gguf_variant is not None
    assert results[0].gguf_variant.quant_type == "Q1_0"


def test_q1_q2_quality_penalty_is_severe():

    from whichvlm.constants import QUANT_QUALITY_PENALTY

    assert QUANT_QUALITY_PENALTY["Q1_0"] >= 0.50
    assert QUANT_QUALITY_PENALTY["Q2_0"] >= 0.40
    assert QUANT_QUALITY_PENALTY["TQ1_0"] >= 0.50
    assert QUANT_QUALITY_PENALTY["IQ1_S"] >= 0.50
    assert QUANT_QUALITY_PENALTY["IQ1_M"] >= 0.45
    assert QUANT_QUALITY_PENALTY["IQ2_M"] >= 0.25


def test_excluded_orgs_never_rank():

    gpt2 = ModelInfo(
        id="openai-community/gpt2",
        family_id="gpt2",
        name="gpt2",
        parameter_count=124_000_000,
        downloads=16_000_000,
        likes=1000,
    )
    opt = ModelInfo(
        id="facebook/opt-125m",
        family_id="opt-125m",
        name="opt-125m",
        parameter_count=125_000_000,
        downloads=9_000_000,
    )
    tiny = ModelInfo(
        id="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
        family_id="tiny-q2",
        name="tiny",
        parameter_count=1_000_000,
        downloads=6_000_000,
    )

    real = ModelInfo(
        id="community-quants/gemma-3-12b-it-GGUF",
        family_id="gemma-3-12b",
        name="gemma-3-12b",
        parameter_count=12_200_000_000,
        downloads=70_000,
        likes=10,
        gguf_variants=[gguf("Q4_K_M", 7.0)],
    )
    results = rank_models([gpt2, opt, tiny, real], hw(), top_n=10, benchmark_scores={})
    ids = [r.model.id for r in results]
    assert "openai-community/gpt2" not in ids
    assert "facebook/opt-125m" not in ids
    assert "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5" not in ids
    assert "community-quants/gemma-3-12b-it-GGUF" in ids


def test_is_excluded_model_helper():
    assert is_excluded_model("openai-community/gpt2")
    assert is_excluded_model("facebook/opt-125m")
    assert is_excluded_model("EleutherAI/pythia-70m-deduped")
    assert is_excluded_model("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")
    assert is_excluded_model("hmellor/tiny-random-LlamaForCausalLM")
    assert not is_excluded_model("Qwen/Qwen3-8B")
    assert not is_excluded_model("meta-llama/Llama-4-Maverick-17B-128E-Instruct")


def test_generation_bonus_newer_wins_over_legacy_in_same_family():
    new = generation_bonus("Qwen/Qwen3.6-27B")
    mid = generation_bonus("Qwen/Qwen3-8B")
    old = generation_bonus("Qwen/Qwen2.5-7B-Instruct")
    ancient = generation_bonus("Qwen/Qwen-7B")
    assert new > mid > old > ancient


def test_generation_bonus_covers_vlm_families():
    assert generation_bonus("Qwen/Qwen3-VL-8B-Instruct") > generation_bonus(
        "Qwen/Qwen2.5-VL-7B-Instruct"
    )
    assert generation_bonus("OpenGVLab/InternVL3-8B") > generation_bonus(
        "OpenGVLab/InternVL2_5-8B"
    )
    assert generation_bonus("llava-hf/llava-onevision-qwen2-7b") > generation_bonus(
        "llava-hf/llava-1.5-7b-hf"
    )


def test_lineage_covers_llama_deepseek_gemma_phi():
    assert generation_bonus("meta-llama/Llama-4-Scout") > generation_bonus(
        "meta-llama/Llama-3.1-8B-Instruct"
    )
    assert generation_bonus("deepseek-ai/DeepSeek-V4-Pro") > generation_bonus(
        "deepseek-ai/DeepSeek-V2.5"
    )
    assert generation_bonus("google/gemma-4-31b-it") > generation_bonus(
        "google/gemma-2-27b-it"
    )
    assert generation_bonus("microsoft/phi-4") > generation_bonus(
        "microsoft/Phi-3-mini-4k-instruct"
    )


def test_lineage_covers_t5_variants_without_gemma_collision():
    assert generation_bonus("google/t5gemma-4b") > generation_bonus(
        "google/flan-t5-xl"
    )
    assert generation_bonus("google/t5-gemma-4b") == generation_bonus(
        "google/t5gemma-4b"
    )
    assert generation_bonus("google/t5_gemma-4b") == generation_bonus(
        "google/t5gemma-4b"
    )
    assert generation_bonus("openai/gpt5-test") == 0.0


def test_unknown_family_gets_zero_bonus():
    assert generation_bonus("random-org/random-model-7b") == 0.0


def test_derivative_penalty_for_heretic_uncensored():
    assert derivative_name_penalty("derivative-fixtures/gemma-3-12b-it-heretic-v2") < 0
    assert (
        derivative_name_penalty(
            "derivative-fixtures/Qwen3.6-27B-Abliterated-Heretic-Uncensored-GGUF"
        )
        < 0
    )
    assert derivative_name_penalty("derivative-fixtures/gemma-4-E4B-it-OBLITERATED") < 0
    assert derivative_name_penalty("dealignai/Qwen3.5-VL-9B-JANG_4S-CRACK") < 0
    assert derivative_name_penalty("community-quants/Qwen3-32B-GGUF") == 0.0


def test_civitai_benchmark_repo_ranks_below_provider_backed_converter():
    converter = ModelInfo(
        id="community-quants/Qwen3-8B-Instruct-GGUF",
        family_id="qwen3-8b",
        name="Qwen3-8B-Instruct-GGUF",
        parameter_count=8_000_000_000,
        downloads=100,
        base_model="Qwen/Qwen3-8B-Instruct",
        gguf_variants=[gguf("Q4_K_M", 4.5)],
    )
    benchmark_repo = ModelInfo(
        id="Civitai/Qwen3-8B-Bench-FP8",
        family_id="qwen3-8b",
        name="Qwen3-8B-Bench-FP8",
        parameter_count=8_000_000_000,
        downloads=200,
        base_model="Qwen/Qwen3-8B-Instruct",
        gguf_variants=[gguf("Q4_K_M", 4.5)],
    )

    results = rank_models(
        [benchmark_repo, converter],
        hw(),
        top_n=1,
        benchmark_scores={"Qwen/Qwen3-8B-Instruct": 70.0},
    )
    assert results[0].model.id == "community-quants/Qwen3-8B-Instruct-GGUF"


def test_self_reported_evidence_does_not_outrank_direct_leaderboard():

    self_reported = ModelInfo(
        id="fixture-org/Self-Reported-8B",
        family_id="self-reported-8b",
        name="Self-Reported",
        parameter_count=8_000_000_000,
        downloads=20_000,
        likes=10,
        gguf_variants=[gguf("Q4_K_M", 4.5)],
        benchmark_scores={"hf_eval": 91.0},
    )
    direct_hit = ModelInfo(
        id="trusted-org/Real-Bench-8B",
        family_id="real-bench-8b",
        name="Real-Bench",
        parameter_count=8_000_000_000,
        downloads=50_000,
        likes=200,
        gguf_variants=[gguf("Q4_K_M", 4.5)],
    )
    results = rank_models(
        [self_reported, direct_hit],
        hw(),
        top_n=5,
        benchmark_scores={"trusted-org/Real-Bench-8B": 70.0},
    )
    assert len(results) == 2
    assert results[0].model.id == "trusted-org/Real-Bench-8B"
    assert results[0].benchmark_status == "direct"
    assert results[1].benchmark_status == "self_reported"


def test_self_reported_outranks_only_when_there_is_nothing_else():
    only_self_reported = ModelInfo(
        id="some-org/Only-Self-Reported-8B",
        family_id="only-sr-8b",
        name="Only-SR",
        parameter_count=8_000_000_000,
        downloads=20_000,
        gguf_variants=[gguf("Q4_K_M", 4.5)],
        benchmark_scores={"hf_eval": 90.0},
    )
    results = rank_models(
        [only_self_reported],
        hw(),
        top_n=5,
        benchmark_scores={},
    )
    assert len(results) == 1
    assert results[0].benchmark_status == "self_reported"


    assert 0 < results[0].quality_score < 60


def test_source_weights_ordering():

    assert SOURCE_WEIGHTS["direct"] > SOURCE_WEIGHTS["base_model"]
    assert SOURCE_WEIGHTS["base_model"] > SOURCE_WEIGHTS["variant"]
    assert SOURCE_WEIGHTS["variant"] > SOURCE_WEIGHTS["line_interp"]
    assert SOURCE_WEIGHTS["line_interp"] > SOURCE_WEIGHTS["self_reported"]
    assert SOURCE_WEIGHTS["self_reported"] > 0
    assert SOURCE_WEIGHTS["none"] == 0.0


def test_strict_evidence_filter_excludes_self_reported():

    self_reported = ModelInfo(
        id="some-org/Self-Reported-8B",
        family_id="sr-8b",
        name="SR",
        parameter_count=8_000_000_000,
        downloads=20_000,
        gguf_variants=[gguf("Q4_K_M", 4.5)],
        benchmark_scores={"hf_eval": 95.0},
    )
    direct_hit = ModelInfo(
        id="trusted-org/Real-Bench-8B",
        family_id="real-bench-8b",
        name="Real-Bench",
        parameter_count=8_000_000_000,
        downloads=50_000,
        gguf_variants=[gguf("Q4_K_M", 4.5)],
    )
    results = rank_models(
        [self_reported, direct_hit],
        hw(),
        top_n=5,
        benchmark_scores={"trusted-org/Real-Bench-8B": 70.0},
        evidence_filter="strict",
    )
    ids = [r.model.id for r in results]
    assert ids == ["trusted-org/Real-Bench-8B"]


def test_official_org_safetensors_gets_q4km_synthesis():

    model = ModelInfo(
        id="Qwen/Qwen3.6-27B",
        family_id="qwen3.6-27b",
        name="Qwen3.6-27B",
        parameter_count=27_000_000_000,
        downloads=50_000,
        gguf_variants=[],
    )
    results = rank_models(
        [model],
        hw(vram_gb=24),
        top_n=1,
        benchmark_scores={"Qwen/Qwen3.6-27B": 84.0},
    )
    assert results
    chosen = results[0]
    assert chosen.fit_type == "full_gpu"
    assert chosen.gguf_variant is not None


    assert chosen.gguf_variant.quant_type in {
        "Q3_K_M",
        "Q4_K_M",
        "Q5_K_M",
        "Q6_K",
        "Q8_0",
    }


def test_prequantized_repo_skips_synthesis():

    model = ModelInfo(
        id="Qwen/Qwen2.5-14B-Instruct-AWQ",
        family_id="qwen2.5-14b-awq",
        name="Qwen2.5-14B-Instruct-AWQ",
        parameter_count=14_000_000_000,
        downloads=10_000,
    )
    q4_filtered = rank_models(
        [model],
        hw(vram_gb=24),
        top_n=5,
        benchmark_scores={"Qwen/Qwen2.5-14B-Instruct-AWQ": 70.0},
        quant_filter="Q4_K_M",
    )
    assert q4_filtered == []


def test_newer_generation_beats_older_at_same_size():

    new_gen = ModelInfo(
        id="Qwen/Qwen3-8B",
        family_id="qwen3-8b",
        name="Qwen3-8B",
        parameter_count=8_000_000_000,
        downloads=10_000_000,
        gguf_variants=[gguf("Q4_K_M", 4.5)],
    )
    old_gen = ModelInfo(
        id="Qwen/Qwen2.5-7B-Instruct",
        family_id="qwen2.5-7b",
        name="Qwen2.5-7B-Instruct",
        parameter_count=7_000_000_000,
        downloads=12_000_000,
        gguf_variants=[gguf("Q4_K_M", 4.0)],
    )
    results = rank_models(
        [new_gen, old_gen],
        hw(vram_gb=24),
        top_n=2,
        benchmark_scores={
            "Qwen/Qwen3-8B": 56.0,
            "Qwen/Qwen2.5-7B-Instruct": 35.0,
        },
    )
    assert [r.model.id for r in results][0] == "Qwen/Qwen3-8B"


def test_speed_estimator_differs_by_quant_and_backend():

    from whichvlm.engine.performance import estimate_tok_per_sec
    from whichvlm.hardware.types import GPUInfo

    model = ModelInfo(
        id="t/x",
        family_id="t/x",
        name="x",
        parameter_count=8_000_000_000,
        downloads=0,
    )
    q4 = GGUFVariant(
        filename="x.Q4_K_M.gguf", quant_type="Q4_K_M", file_size_bytes=int(8e9 * 0.5625)
    )
    f16 = GGUFVariant(
        filename="x.F16.gguf", quant_type="F16", file_size_bytes=int(8e9 * 2.0)
    )
    cuda = GPUInfo(
        name="t-nv",
        vendor="nvidia",
        vram_bytes=24 * 1024**3,
        memory_bandwidth_gbps=1000.0,
    )
    metal = GPUInfo(
        name="t-apple",
        vendor="apple",
        vram_bytes=24 * 1024**3,
        memory_bandwidth_gbps=1000.0,
    )
    q4_cuda = estimate_tok_per_sec(model, q4, cuda, "full_gpu")
    f16_cuda = estimate_tok_per_sec(model, f16, cuda, "full_gpu")
    q4_metal = estimate_tok_per_sec(model, q4, metal, "full_gpu")
    assert q4_cuda > f16_cuda
    assert q4_cuda > q4_metal


def test_vlm_speed_estimate_is_discounted_for_image_prefill():
    from whichvlm.engine.performance import (
        estimate_speed_uncertainty,
        estimate_tok_per_sec,
    )
    from whichvlm.hardware.types import GPUInfo

    text = ModelInfo(
        id="Qwen/Qwen2.5-7B-Instruct",
        family_id="qwen2.5-7b",
        name="Qwen2.5-7B-Instruct",
        parameter_count=7_000_000_000,
        downloads=0,
    )
    vlm = ModelInfo(
        id="Qwen/Qwen2.5-VL-7B-Instruct",
        family_id="qwen-vl",
        name="Qwen2.5-VL-7B-Instruct",
        parameter_count=7_000_000_000,
        hf_pipeline_tag="image-text-to-text",
        downloads=0,
    )
    gpu = GPUInfo(
        name="t-nv",
        vendor="nvidia",
        vram_bytes=24 * 1024**3,
        memory_bandwidth_gbps=1000.0,
    )

    text_speed = estimate_tok_per_sec(text, None, gpu, "full_gpu")
    vlm_speed = estimate_tok_per_sec(vlm, None, gpu, "full_gpu")
    confidence, _, notes = estimate_speed_uncertainty(
        vlm,
        None,
        gpu,
        "full_gpu",
        vlm_speed,
    )

    assert vlm_speed < text_speed
    assert confidence == "medium"
    assert any("image prefill" in note for note in notes)


def test_vram_kv_cache_scales_with_context():

    from whichvlm.engine.vram import estimate_kv_cache

    model = ModelInfo(
        id="t/x",
        family_id="t/x",
        name="x",
        parameter_count=32_000_000_000,
        downloads=0,
    )
    kv_4k = estimate_kv_cache(model, 4096)
    kv_32k = estimate_kv_cache(model, 32768)
    assert kv_32k > kv_4k * 7

    assert kv_32k > 2 * 1024**3
