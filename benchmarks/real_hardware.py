from __future__ import annotations

import argparse
import base64
import mimetypes
import statistics
import time
from pathlib import Path


def detection(args: argparse.Namespace) -> None:
    from whichvlm.hardware.detector import detect_hardware

    durations: list[float] = []
    snapshots = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        snapshots.append(detect_hardware())
        durations.append(time.perf_counter() - start)

    hw = snapshots[-1]
    median_seconds = statistics.median(durations)
    if median_seconds > args.max_seconds:
        raise SystemExit(
            f"median detection {median_seconds:.2f}s exceeded "
            f"{args.max_seconds:.2f}s over {args.repeats} runs: {durations}"
        )
    if hw.cpu_cores <= 0 or hw.ram_bytes <= 0:
        raise SystemExit("hardware detection returned missing CPU/RAM data")

    if args.expect_backend:
        backends = available_backends(hw)
        expected = args.expect_backend.lower()
        if expected not in backends:
            raise SystemExit(f"expected backend {expected!r}, detected {sorted(backends)}")

    print(
        f"detection median={median_seconds:.2f}s "
        f"runs={','.join(f'{d:.2f}' for d in durations)}"
    )


def available_backends(hw) -> set[str]:
    names = {c.name.lower() for c in hw.backend_capabilities if c.available}
    for gpu in hw.gpus:
        names.update(c.name.lower() for c in gpu.backend_capabilities if c.available)
    names.add("cpu")
    return names


def image_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def llama_cpp_handler(llama_chat_format, model_id: str, mmproj_path: str, handler: str | None):
    if handler:
        names = [handler]
    else:
        lower = model_id.lower()
        names = []
        if "qwen" in lower and "vl" in lower:
            names.extend(["Qwen25VLChatHandler", "Qwen2VLChatHandler"])
        if "llava" in lower:
            names.extend(["Llava16ChatHandler", "Llava15ChatHandler"])
        if "minicpm" in lower:
            names.extend(["MiniCPMv26ChatHandler", "MiniCPMVChatHandler"])
        names.extend(["Llava16ChatHandler", "Llava15ChatHandler"])

    for name in dict.fromkeys(names):
        cls = getattr(llama_chat_format, name, None)
        if cls is not None:
            return cls(clip_model_path=mmproj_path)
    raise SystemExit(f"llama-cpp-python has none of these handlers: {names}")


def gguf_mmproj(args: argparse.Namespace) -> None:
    from huggingface_hub import hf_hub_download
    from llama_cpp import Llama, llama_chat_format

    image = args.image.expanduser()
    if not image.is_file():
        raise SystemExit(f"missing benchmark image: {image}")

    model_path = hf_hub_download(repo_id=args.repo, filename=args.model_file)
    mmproj_path = hf_hub_download(repo_id=args.repo, filename=args.mmproj_file)
    handler = llama_cpp_handler(
        llama_chat_format, args.repo, mmproj_path, args.handler
    )

    start_load = time.perf_counter()
    llm = Llama(
        model_path=model_path,
        chat_handler=handler,
        n_ctx=args.context,
        n_gpu_layers=args.gpu_layers,
        verbose=False,
    )
    load_seconds = time.perf_counter() - start_load
    if load_seconds > args.max_load_seconds:
        raise SystemExit(
            f"load {load_seconds:.2f}s exceeded {args.max_load_seconds:.2f}s"
        )

    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_data_url(image)}},
                    {"type": "text", "text": "Describe the image in one sentence."},
                ],
            }
        ]
        start_gen = time.perf_counter()
        response = llm.create_chat_completion(
            messages=messages,
            max_tokens=args.max_tokens,
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

    if not content:
        raise SystemExit("empty model response")
    if tok_s < args.min_tokens_per_second:
        raise SystemExit(
            f"{tok_s:.2f} tok/s below {args.min_tokens_per_second:.2f} tok/s "
            f"({completion_tokens} tokens in {gen_seconds:.2f}s)"
        )

    print(
        f"gguf-mmproj load={load_seconds:.2f}s generation={gen_seconds:.2f}s "
        f"tokens={completion_tokens} tok/s={tok_s:.2f}"
    )


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(required=True)

    detect = sub.add_parser("detection")
    detect.add_argument("--repeats", type=int, default=3)
    detect.add_argument("--max-seconds", type=float, default=8.0)
    detect.add_argument("--expect-backend")
    detect.set_defaults(func=detection)

    gguf = sub.add_parser("gguf-mmproj")
    gguf.add_argument("--repo", required=True)
    gguf.add_argument("--model-file", required=True)
    gguf.add_argument("--mmproj-file", required=True)
    gguf.add_argument("--image", type=Path, required=True)
    gguf.add_argument("--handler")
    gguf.add_argument("--context", type=int, default=2048)
    gguf.add_argument("--gpu-layers", type=int, default=-1)
    gguf.add_argument("--max-tokens", type=int, default=24)
    gguf.add_argument("--max-load-seconds", type=float, default=180.0)
    gguf.add_argument("--min-tokens-per-second", type=float, default=0.2)
    gguf.set_defaults(func=gguf_mmproj)
    return p


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
