import base64
import mimetypes
import os
import statistics
import time
from pathlib import Path

import pytest


def _benchmarks_enabled() -> bool:
    return os.getenv("WHICHVLM_REAL_HARDWARE_BENCHMARKS") == "1"


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value else default


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


def _all_backends(hw) -> set[str]:
    names = {c.name.lower() for c in hw.backend_capabilities if c.available}
    for gpu in hw.gpus:
        names.update(c.name.lower() for c in gpu.backend_capabilities if c.available)
    names.add("cpu")
    return names


@pytest.mark.real_hardware
def test_hardware_detection_benchmark():
    if not _benchmarks_enabled():
        pytest.skip("set WHICHVLM_REAL_HARDWARE_BENCHMARKS=1")

    from whichvlm.hardware.detector import detect_hardware

    repeats = _int_env("WHICHVLM_DETECT_REPEATS", 3)
    max_seconds = _float_env("WHICHVLM_DETECT_MAX_SECONDS", 8.0)
    durations: list[float] = []
    snapshots = []

    for _ in range(repeats):
        start = time.perf_counter()
        snapshots.append(detect_hardware())
        durations.append(time.perf_counter() - start)

    hw = snapshots[-1]
    median_seconds = statistics.median(durations)
    assert median_seconds <= max_seconds, (
        f"median detection {median_seconds:.2f}s exceeded {max_seconds:.2f}s "
        f"over {repeats} runs: {durations}"
    )
    assert hw.cpu_cores > 0
    assert hw.ram_bytes > 0

    expected_backend = os.getenv("WHICHVLM_EXPECT_BACKEND")
    if expected_backend:
        backends = _all_backends(hw)
        assert expected_backend.lower() in backends, (
            f"expected backend {expected_backend!r}, detected {sorted(backends)}"
        )


def _image_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _llama_cpp_handler(llama_chat_format, model_id: str, mmproj_path: str):
    handler_name = os.getenv("WHICHVLM_GGUF_VLM_HANDLER")
    if handler_name:
        handler_names = [handler_name]
    else:
        lower = model_id.lower()
        handler_names = []
        if "qwen" in lower and "vl" in lower:
            handler_names.extend(["Qwen25VLChatHandler", "Qwen2VLChatHandler"])
        if "llava" in lower:
            handler_names.extend(["Llava16ChatHandler", "Llava15ChatHandler"])
        if "minicpm" in lower:
            handler_names.extend(["MiniCPMv26ChatHandler", "MiniCPMVChatHandler"])
        handler_names.extend(["Llava16ChatHandler", "Llava15ChatHandler"])

    for name in dict.fromkeys(handler_names):
        cls = getattr(llama_chat_format, name, None)
        if cls is not None:
            return cls(clip_model_path=mmproj_path)
    pytest.fail(f"llama-cpp-python has none of these handlers: {handler_names}")


@pytest.mark.real_hardware
def test_gguf_mmproj_vlm_generation_benchmark():
    if not _benchmarks_enabled():
        pytest.skip("set WHICHVLM_REAL_HARDWARE_BENCHMARKS=1")

    repo = os.getenv("WHICHVLM_GGUF_VLM_REPO")
    model_file = os.getenv("WHICHVLM_GGUF_VLM_MODEL_FILE")
    mmproj_file = os.getenv("WHICHVLM_GGUF_VLM_MMPROJ_FILE")
    image_path = os.getenv("WHICHVLM_BENCH_IMAGE")
    if not all((repo, model_file, mmproj_file, image_path)):
        pytest.skip(
            "set WHICHVLM_GGUF_VLM_REPO, WHICHVLM_GGUF_VLM_MODEL_FILE, "
            "WHICHVLM_GGUF_VLM_MMPROJ_FILE, and WHICHVLM_BENCH_IMAGE"
        )

    from huggingface_hub import hf_hub_download
    from llama_cpp import Llama, llama_chat_format

    image = Path(image_path).expanduser()
    assert image.is_file(), f"missing benchmark image: {image}"

    max_load_seconds = _float_env("WHICHVLM_GGUF_MAX_LOAD_SECONDS", 180.0)
    min_tok_s = _float_env("WHICHVLM_GGUF_MIN_TOKENS_PER_SECOND", 0.2)
    max_tokens = _int_env("WHICHVLM_GGUF_MAX_TOKENS", 24)
    n_ctx = _int_env("WHICHVLM_GGUF_CONTEXT", 2048)
    n_gpu_layers = _int_env("WHICHVLM_GGUF_N_GPU_LAYERS", -1)

    model_path = hf_hub_download(repo_id=repo, filename=model_file)
    mmproj_path = hf_hub_download(repo_id=repo, filename=mmproj_file)
    handler = _llama_cpp_handler(llama_chat_format, repo, mmproj_path)

    start_load = time.perf_counter()
    llm = Llama(
        model_path=model_path,
        chat_handler=handler,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        verbose=False,
    )
    load_seconds = time.perf_counter() - start_load
    assert load_seconds <= max_load_seconds, (
        f"load {load_seconds:.2f}s exceeded {max_load_seconds:.2f}s"
    )

    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": _image_data_url(image)}},
                    {"type": "text", "text": "Describe the image in one sentence."},
                ],
            }
        ]
        start_gen = time.perf_counter()
        response = llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.0,
        )
        gen_seconds = time.perf_counter() - start_gen
    finally:
        close = getattr(llm, "close", None)
        if close:
            close()

    content = response["choices"][0]["message"]["content"].strip()
    completion_tokens = response.get("usage", {}).get("completion_tokens")
    if not completion_tokens:
        completion_tokens = max(1, len(content.split()))
    tok_s = completion_tokens / max(gen_seconds, 1e-6)

    assert content
    assert tok_s >= min_tok_s, (
        f"{tok_s:.2f} tok/s below {min_tok_s:.2f} tok/s "
        f"({completion_tokens} tokens in {gen_seconds:.2f}s)"
    )
