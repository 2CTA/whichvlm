# Upstream Notes

`whichvlm` is a VLM-specific sibling of `compatible multimodal source`. The reusable core was
ported and adapted under the original MIT license and notices. Keep copied or
substantially adapted code attributed through `LICENSE`, `NOTICE`, and these
notes.

## Audit Buckets

Keep identical reusable core where possible:

- hardware detection and simulated hardware
- GPU/CPU/RAM data structures
- cache and HTTP retry utilities
- output formatting and JSON shape
- compatibility result abstractions
- most low-level tests and error handling utilities

Adapt for VLM behavior:

- package graph data: artifacts, components, lineage, gated status, and variants
- Hugging Face fetch order: VLM tasks first, text-generation only as support
- default profile and workload: `vision` plus `VisionWorkload`
- backend priority: MLX/Metal, concrete GGUF Metal/CUDA, CUDA Transformers, CPU
- memory estimates: language tower plus vision encoder/projector/image prefill

Intentionally diverge:

- A repo is not assumed to equal one runnable model. `whichvlm` groups official
  checkpoints, community quantizations, MLX/GGUF variants, adapters, and merged
  parents into logical VLM packages.
- Synthetic GGUF assumptions from text LLMs do not apply to VLMs unless a
  concrete multimodal GGUF artifact is discovered.
- Runtime snippets require image input for VLM models and fail clearly when a
  ranked artifact has no implemented runner.
- Text benchmark scores remain fallback evidence until multimodal benchmark
  sources and confidence rules are fully implemented.

## Audit Method

The current sync pass used non-mutating module inspection and file comparison
against the sibling `compatible multimodal source` checkout. Generic infrastructure fixes should be
pulled forward when they do not change the VLM contract. Product behavior should
follow `whichvlm`'s VLM package graph, vision workload, backend priorities, model
inventory, and runtime constraints.

## 2026-06-16 14:00 checkpoint
- Added runtime-path checkpoint notes for llama.cpp and MLX codegen coverage and guardrail evolution.
