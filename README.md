# whichvlm

Find local vision-language models that fit your machine.

`whichvlm` detects GPU, CPU, RAM, Apple Metal/MLX readiness, model formats, quantized variants, GGUF projectors, and model lineage. It ranks VLM candidates for local inference instead of treating every Hugging Face repo as one plain text model.

 Shared infrastructure is adapted with attribution in `LICENSE`, `NOTICE`, and `UPSTREAM_NOTES.md`.

## Install

Use Python 3.11 or newer.

```bash
cd ~/Downloads/Github/whichvlm
uv sync
uv run whichvlm --help
```

For the shell environment used during local development on this machine:

```bash
source "$HOME/Documents/uv_global_venv/bin/activate"
```

For editable development without `uv run`:

```bash
uv pip install -e ".[dev]"
whichvlm --help
```

## Use

Rank VLMs for the current machine:

```bash
uv run whichvlm
```

Simulate Apple Silicon or a discrete GPU:

```bash
uv run whichvlm --gpu "Apple M3 Max"
uv run whichvlm --gpu "RTX 4090" --vram-headroom 10%
```

Return machine-readable output:

```bash
uv run whichvlm --json --top 5
```

Change the VLM workload estimate:

```bash
uv run whichvlm --image-count 2 --image-size 896 --context-length 8192
```

Only show full GPU fits:

```bash
uv run whichvlm --gpu-only
uv run whichvlm --fit full-gpu
```

## Run A Model

VLM runners require an image path.

```bash
uv run whichvlm run Qwen/Qwen2.5-VL-7B-Instruct --image ./image.jpg
uv run whichvlm snippet Qwen/Qwen2.5-VL-7B-Instruct --image ./image.jpg
```

Runtime support is intentionally guarded:

- Transformers VLMs use `AutoProcessor` and image/text chat templates.
- GGUF VLMs require a concrete GGUF file plus an `mmproj` or projector artifact.
- MLX VLMs require a concrete MLX model package.
- Text-only GGUF and Transformers paths remain available for inherited core behavior.

## What It Models

`whichvlm` tracks a VLM as a package graph:

- `ModelArtifact`: repo, file format, quantization, access, backend support, source kind, filename.
- `ModelComponent`: language tower, vision encoder, projector, processor, tokenizer, merged checkpoint, adapter.
- `ModelLineage`: base models, merged parents, variant relation, and fused/merged status.

The ranker is VLM-aware but conservative. Vision memory includes language weights, KV cache, activation memory, estimated vision encoder/projector overhead, image-token expansion, and prefill scratch. These estimates are useful for ranking. They are not final benchmark-quality measurements.

## Data Sources

Model metadata comes from Hugging Face API queries, local cache, and curated VLM seeds.

The fetcher prioritizes:

- `image-text-to-text`
- `visual-question-answering`
- `image-to-text`
- GGUF, MLX, AWQ, GPTQ, BNB, and FP8 variants
- text-generation only as backbone or variant discovery

Benchmark evidence is graded as direct, variant, base model, interpolated, self-reported, or absent. Vision scores lead the `vision` profile. Text benchmarks are fallback evidence.

## Development

Run the full suite:

```bash
uv run pytest -q
```

Run focused tests:

```bash
uv run pytest -q tests/test_runtime.py tests/test_fetcher.py tests/test_ranker.py
```

Compile-check source and tests:

```bash
uv run python -m compileall -q src tests
```

The source layout is under `src/whichvlm`. Tests live under `tests`. Avoid importing private CLI helpers in new tests; prefer runtime, ranker, fetcher, or output APIs.

## Real Hardware Benchmarks

These are opt-in. They are skipped unless explicitly enabled because they use real hardware, downloads, and runtime dependencies.

Run the same detection benchmark on every target machine:

```bash
WHICHVLM_REAL_HARDWARE_BENCHMARKS=1 \
WHICHVLM_EXPECT_BACKEND=metal \
uv run pytest -q tests/test_real_hardware_benchmarks.py::test_hardware_detection_benchmark
```

Run the same GGUF+mmproj VLM benchmark on every target machine:

```bash
WHICHVLM_REAL_HARDWARE_BENCHMARKS=1 \
WHICHVLM_GGUF_VLM_REPO="owner/model-gguf" \
WHICHVLM_GGUF_VLM_MODEL_FILE="model-q4_k_m.gguf" \
WHICHVLM_GGUF_VLM_MMPROJ_FILE="mmproj-model-f16.gguf" \
WHICHVLM_GGUF_VLM_HANDLER="Llava16ChatHandler" \
WHICHVLM_BENCH_IMAGE="./image.jpg" \
uv run pytest -q tests/test_real_hardware_benchmarks.py::test_gguf_mmproj_vlm_generation_benchmark
```

Useful thresholds:

- `WHICHVLM_DETECT_MAX_SECONDS`, default `8`
- `WHICHVLM_GGUF_MAX_LOAD_SECONDS`, default `180`
- `WHICHVLM_GGUF_MIN_TOKENS_PER_SECOND`, default `0.2`

## Current Limits

The model inventory is not complete.

Multimodal benchmark calibration is not final.

GGUF VLM and MLX VLM runners are only as reliable as the concrete artifacts and runtime handlers discovered for a model package.

ANE is detected as information only. It is not scored until there is a concrete VLM runtime path.
