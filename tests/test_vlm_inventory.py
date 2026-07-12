from data.vlm_inventory import canonical_vlm_family_id, known_vlm_model_ids


def test_known_vlm_inventory_families_have_stable_ids():
    assert canonical_vlm_family_id("Qwen/Qwen2.5-VL-7B-Instruct") == "qwen-vl"
    assert canonical_vlm_family_id("qwen2vl") == "qwen-vl"
    assert canonical_vlm_family_id("Qwen/Qwen2-Audio-7B-Instruct") == "qwen2-audio"
    assert canonical_vlm_family_id("OpenGVLab/InternVL3-8B") == "internvl"
    assert canonical_vlm_family_id("llava-hf/llava-1.5-7b-hf") == "llava"
    assert canonical_vlm_family_id("unknown/Plain-Text-7B") is None


def test_known_vlm_model_ids_exposes_seed_repos():
    ids = known_vlm_model_ids()
    assert "Qwen/Qwen3-VL-235B-A22B-Instruct" in ids
    assert "Qwen/Qwen2-Audio-7B-Instruct" in ids
    assert "OpenGVLab/InternVL3-78B" in ids
    assert "mistralai/Pixtral-12B-2409" in ids
    assert "mistral-community/pixtral-12b" in ids
    assert "microsoft/Phi-4-multimodal-instruct" in ids
