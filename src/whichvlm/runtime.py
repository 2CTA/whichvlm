from __future__ import annotations

from whichvlm.engine.quantization import infer_non_gguf_quant_type
from whichvlm.models.package_graph import is_projector_filename, is_vision_model
from whichvlm.models.types import GGUFVariant, ModelArtifact, ModelInfo

# Runtime layer. Chooses script shape for transformers, GGUF, or MLX.

class RuntimeUnsupportedError(ValueError):
    pass


TransformersProfile = tuple[str, str, tuple[str, ...]]


def is_vlm_model(model: ModelInfo) -> bool:
    # VLM check. Detects image-capable models from tags and components.
    if is_vision_model(model.id, model.hf_pipeline_tag, model.tags):
        return True
    return any(
        component.role in {"vision_encoder", "projector", "processor"}
        for component in model.components
    )


def requires_image(model: ModelInfo) -> bool:
    return is_vlm_model(model)


def resolve_model_deps(
    model: ModelInfo,
    variant: GGUFVariant | None,
) -> tuple[list[str], str]:
    # Dependency planner. Returns pip deps plus runtime family label.
    vlm_model = is_vlm_model(model)
    if variant:
        deps = ["llama-cpp-python", "huggingface-hub", "psutil"]
        if vlm_model:
            deps.append("pillow")
        return deps, "gguf_vlm" if vlm_model else "gguf"

    if vlm_model:
        if is_mlx_model(model):
            return ["mlx-vlm", "pillow"], "mlx_vlm"
        return [
            "transformers",
            "torch",
            "torchvision",
            "accelerate",
            "pillow",
            "psutil",
            *transformers_quant_deps(model),
        ], "transformers_vlm"

    base = ["transformers", "torch", "accelerate", "psutil"]
    return [*base, *transformers_quant_deps(model)], "transformers"


def transformers_quant_deps(model: ModelInfo) -> list[str]:
    qt = transformers_quant_type(model)
    if qt == "AWQ":
        return ["autoawq"]
    if qt == "GPTQ":
        return ["auto-gptq"]
    if qt in {"BNB_4BIT", "INT8"}:
        return ["bitsandbytes"]
    return []


def generate_run_script(
    model: ModelInfo,
    variant: GGUFVariant | None,
    context_length: int,
    cpu_only: bool,
    image_path: str | None = None,
) -> str:
    # Script builder. Emits the runnable snippet for one chosen path.
    vlm_model = is_vlm_model(model)
    if vlm_model:
        if image_path is None:
            raise RuntimeUnsupportedError("VLM runners require --image PATH.")
        if variant:
            projector = find_projector_artifact(model)
            if projector is None or projector.filename is None:
                raise RuntimeUnsupportedError(
                    "GGUF VLM runtime requires an mmproj/projector artifact in "
                    "the model package metadata."
                )
            return generate_llama_cpp_vlm_script(
                model,
                variant,
                projector,
                context_length,
                cpu_only,
                image_path,
            )
        if is_mlx_model(model):
            return generate_mlx_vlm_script(model, image_path)
        return generate_transformers_vlm_script(model, image_path, cpu_only)

    if variant:
        return generate_llama_cpp_text_script(model, variant, context_length, cpu_only)
    return generate_transformers_text_script(model, cpu_only)


def is_mlx_model(model: ModelInfo) -> bool:
    if model.model_format == "mlx":
        return True
    if (model.quantization_type or "").upper() == "MLX":
        return True
    return any(artifact.format == "mlx" for artifact in model.artifacts)


def find_projector_artifact(model: ModelInfo) -> ModelArtifact | None:
    # Projector lookup. Finds the mmproj file VLM GGUF runners need.
    for artifact in model.artifacts:
        if artifact.source_kind == "mmproj" and artifact.filename:
            return artifact
    for artifact in model.artifacts:
        if artifact.filename and is_projector_filename(artifact.filename):
            return artifact
    return None


def model_family_text(model: ModelInfo) -> str:
    return " ".join(
        value.lower()
        for value in (model.id, model.family_id, model.name, model.architecture)
        if value
    )


def transformers_quant_type(model: ModelInfo) -> str:
    return (model.quantization_type or infer_non_gguf_quant_type(model.id)).upper()


def transformers_text_profile() -> TransformersProfile:
    return "AutoModelForCausalLM", "AutoTokenizer", ()


def transformers_vlm_profile(model: ModelInfo) -> TransformersProfile:
    family = model_family_text(model)
    if "qwen" in family and "vl" in family:
        model_class = (
            "Qwen2_5_VLForConditionalGeneration"
            if "2.5" in family or "2-5" in family
            else "Qwen2VLForConditionalGeneration"
        )
        return (
            model_class,
            "AutoProcessor",
            (
                "min_pixels=256 * 28 * 28",
                "max_pixels=1280 * 28 * 28",
            ),
        )
    if "llama-3.2" in family or "mllama" in family:
        return "MllamaForConditionalGeneration", "AutoProcessor", ()
    if "llava" in family:
        return "LlavaForConditionalGeneration", "AutoProcessor", ()
    return "AutoModelForImageTextToText", "AutoProcessor", ()


def transformers_import_names(
    model_class: str,
    processor_class: str,
    extra: tuple[str, ...] = (),
) -> str:
    return ", ".join(sorted({model_class, processor_class, *extra}))


def processor_kwargs_lines(processor_kwargs: tuple[str, ...]) -> str:
    if not processor_kwargs:
        return ""
    return ",\n        " + ",\n        ".join(processor_kwargs)


def quantization_config_lines(model: ModelInfo) -> str:
    qt = transformers_quant_type(model)
    if qt == "BNB_4BIT":
        return '''\
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch_dtype,
)
model_kwargs["quantization_config"] = quantization_config
'''
    if qt == "INT8":
        return '''\
quantization_config = BitsAndBytesConfig(load_in_8bit=True)
model_kwargs["quantization_config"] = quantization_config
'''
    return ""


def quantization_import_names(model: ModelInfo) -> tuple[str, ...]:
    if transformers_quant_type(model) in {"BNB_4BIT", "INT8"}:
        return ("BitsAndBytesConfig",)
    return ()


def llama_decode_metrics_block() -> str:
    return '''\
process = psutil.Process()


def print_decode_metrics(started_at, first_token_at, token_count):
    finished_at = time.perf_counter()
    ttft = (first_token_at or finished_at) - started_at
    decode_seconds = max(finished_at - (first_token_at or finished_at), 1e-6)
    print(
        f"[metrics] ttft={ttft:.2f}s decode={token_count / decode_seconds:.2f} tok/s "
        f"rss={process.memory_info().rss / 1024**3:.2f}GB"
    )

'''


def transformers_runtime_setup(quantization_lines: str) -> str:
    return f'''\
offload_folder = tempfile.mkdtemp(prefix="whichvlm_transformers_offload_")
process = psutil.Process()


def cuda_memory_limits():
    if not torch.cuda.is_available():
        return None
    return {{
        index: f"{{int(torch.cuda.mem_get_info(index)[0] * 0.9 / 1024**2)}}MiB"
        for index in range(torch.cuda.device_count())
    }}


def print_decode_metrics(started_at, first_token_at, output_text):
    finished_at = time.perf_counter()
    token_count = len(tokenizer(output_text, add_special_tokens=False).input_ids)
    ttft = (first_token_at or finished_at) - started_at
    decode_seconds = max(finished_at - (first_token_at or finished_at), 1e-6)
    gpu_peak = ""
    if torch.cuda.is_available():
        gpu_peak = (
            f" gpu={{torch.cuda.max_memory_allocated() / 1024**3:.2f}}GB"
            f" reserved={{torch.cuda.max_memory_reserved() / 1024**3:.2f}}GB"
        )
    print(
        f"[metrics] ttft={{ttft:.2f}}s decode={{token_count / decode_seconds:.2f}} tok/s "
        f"rss={{process.memory_info().rss / 1024**3:.2f}}GB{{gpu_peak}}"
    )


torch_dtype = (
    torch.float32
    if device_map == "cpu"
    else torch.bfloat16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else torch.float16
)
model_kwargs = dict(
    device_map=device_map,
    torch_dtype=torch_dtype,
    trust_remote_code=True,
    offload_folder=offload_folder,
    offload_state_dict=True,
    attn_implementation="sdpa",
    max_memory=cuda_memory_limits(),
)
{quantization_lines}
'''


def generate_llama_cpp_text_script(
    model: ModelInfo,
    variant: GGUFVariant,
    context_length: int,
    cpu_only: bool,
) -> str:
    n_gpu = 0 if cpu_only else -1
    metrics = llama_decode_metrics_block()
    return f'''\
import psutil
import time

from huggingface_hub import hf_hub_download
from llama_cpp import Llama

{metrics}
print("Downloading {model.id} ({variant.quant_type})...")
model_path = hf_hub_download(repo_id="{model.id}", filename="{variant.filename}")
load_started_at = time.perf_counter()
print("Loading model...")
llm = Llama(
    model_path=model_path,
    n_ctx={context_length},
    n_gpu_layers={n_gpu},
    verbose=False,
)
print(f"Loaded in {{time.perf_counter() - load_started_at:.2f}}s")
print("Ready! Type 'exit' to quit.\\n")
messages = []
while True:
    try:
        text = input("> ")
    except (KeyboardInterrupt, EOFError):
        break
    if text.strip().lower() in ("exit", "quit", "q"):
        break
    if not text.strip():
        continue
    messages.append({{"role": "user", "content": text}})
    started_at = time.perf_counter()
    response = llm.create_chat_completion(messages=messages, stream=True)
    full = ""
    first_token_at = None
    token_count = 0
    for chunk in response:
        delta = chunk["choices"][0].get("delta", {{}})
        content = delta.get("content", "")
        if content:
            if first_token_at is None:
                first_token_at = time.perf_counter()
            print(content, end="", flush=True)
            full += content
            token_count += 1
    print()
    print_decode_metrics(started_at, first_token_at, token_count)
    messages.append({{"role": "assistant", "content": full}})
print("\\nBye!")
'''


def generate_llama_cpp_vlm_script(
    model: ModelInfo,
    variant: GGUFVariant,
    projector: ModelArtifact,
    context_length: int,
    cpu_only: bool,
    image_path: str,
) -> str:
    n_gpu = 0 if cpu_only else -1
    metrics = llama_decode_metrics_block()
    return f'''\
import base64
import mimetypes
import psutil
import time

from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from llama_cpp import llama_chat_format

model_id = "{model.id}"
model_filename = "{variant.filename}"
projector_filename = "{projector.filename}"
image_path = {image_path!r}
{metrics}


def image_data_url(path):
    mime = mimetypes.guess_type(path)[0] or "image/png"
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")
    return f"data:{{mime}};base64,{{encoded}}"


def chat_handler(model_id, mmproj_path):
    lower = model_id.lower()
    preferred = []
    if "qwen" in lower and "vl" in lower:
        preferred.extend(["Qwen25VLChatHandler", "Qwen2VLChatHandler"])
    if "llava" in lower:
        preferred.extend(["Llava16ChatHandler", "Llava15ChatHandler"])
    if "minicpm" in lower:
        preferred.extend(["MiniCPMv26ChatHandler", "MiniCPMVChatHandler"])
    preferred.extend(["Llava16ChatHandler", "Llava15ChatHandler"])

    seen = set()
    for name in preferred:
        if name in seen:
            continue
        seen.add(name)
        cls = getattr(llama_chat_format, name, None)
        if cls is not None:
            return cls(clip_model_path=mmproj_path)
    raise SystemExit(
        "llama-cpp-python does not expose a compatible multimodal chat handler "
        f"for {{model_id}}. Install a newer llama-cpp-python or use Transformers/MLX."
    )


print(f"Downloading {{model_id}}...")
model_path = hf_hub_download(repo_id=model_id, filename=model_filename)
mmproj_path = hf_hub_download(repo_id=model_id, filename=projector_filename)
handler = chat_handler(model_id, mmproj_path)

load_started_at = time.perf_counter()
print("Loading model...")
llm = Llama(
    model_path=model_path,
    chat_handler=handler,
    n_ctx={context_length},
    n_gpu_layers={n_gpu},
    verbose=False,
)
print(f"Loaded in {{time.perf_counter() - load_started_at:.2f}}s")

print("Ready! Type 'exit' to quit.\\n")
image_url = image_data_url(image_path)
while True:
    try:
        text = input("> ")
    except (KeyboardInterrupt, EOFError):
        break
    if text.strip().lower() in ("exit", "quit", "q"):
        break
    if not text.strip():
        continue
    messages = [
        {{
            "role": "user",
            "content": [
                {{"type": "image_url", "image_url": {{"url": image_url}}}},
                {{"type": "text", "text": text}},
            ],
        }}
    ]
    started_at = time.perf_counter()
    response = llm.create_chat_completion(messages=messages, stream=True)
    first_token_at = None
    token_count = 0
    for chunk in response:
        delta = chunk["choices"][0].get("delta", {{}})
        content = delta.get("content", "")
        if content:
            if first_token_at is None:
                first_token_at = time.perf_counter()
            print(content, end="", flush=True)
            token_count += 1
    print()
    print_decode_metrics(started_at, first_token_at, token_count)
print("\\nBye!")
'''


def generate_transformers_text_script(model: ModelInfo, cpu_only: bool) -> str:
    device_map = '"cpu"' if cpu_only else '"auto"'
    model_class, tokenizer_class, _ = transformers_text_profile()
    imports = transformers_import_names(
        model_class, tokenizer_class, ("TextIteratorStreamer", *quantization_import_names(model))
    )
    runtime_setup = transformers_runtime_setup(quantization_config_lines(model))
    return f'''\
import shutil
import tempfile
import time

import psutil
import torch
from threading import Thread
from transformers import {imports}

model_id = "{model.id}"
device_map = {device_map}
{runtime_setup}
try:
    print(f"Loading {{model_id}}...")
    load_started_at = time.perf_counter()
    tokenizer = {tokenizer_class}.from_pretrained(model_id, trust_remote_code=True)
    model = {model_class}.from_pretrained(model_id, **model_kwargs)
    model.eval()
    print(f"Loaded in {{time.perf_counter() - load_started_at:.2f}}s")
    print("Ready! Type 'exit' to quit.\\n")
    messages = []
    while True:
        try:
            text = input("> ")
        except (KeyboardInterrupt, EOFError):
            break
        if text.strip().lower() in ("exit", "quit", "q"):
            break
        if not text.strip():
            continue
        messages.append({{"role": "user", "content": text}})
        inputs = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            return_dict=True,
            add_generation_prompt=True,
        ).to(model.device)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        streamer = TextIteratorStreamer(
            tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        def run_generate():
            with torch.inference_mode():
                model.generate(**inputs, max_new_tokens=512, streamer=streamer)

        started_at = time.perf_counter()
        thread = Thread(target=run_generate)
        thread.start()
        full = ""
        first_token_at = None
        for text in streamer:
            if first_token_at is None:
                first_token_at = time.perf_counter()
            print(text, end="", flush=True)
            full += text
        thread.join()
        print()
        print_decode_metrics(started_at, first_token_at, full)
        messages.append({{"role": "assistant", "content": full}})
    print("\\nBye!")
finally:
    try:
        del model
    except NameError:
        pass
    shutil.rmtree(offload_folder, ignore_errors=True)
'''


def generate_transformers_vlm_script(
    model: ModelInfo,
    image_path: str,
    cpu_only: bool,
) -> str:
    device_map = '"cpu"' if cpu_only else '"auto"'
    model_class, processor_class, processor_extra_args = transformers_vlm_profile(model)
    imports = transformers_import_names(
        model_class, processor_class, ("TextIteratorStreamer", *quantization_import_names(model))
    )
    processor_arg_lines = processor_kwargs_lines(processor_extra_args)
    runtime_setup = transformers_runtime_setup(quantization_config_lines(model))
    return f'''\
import shutil
import tempfile
import time

import psutil
import torch
from PIL import Image
from PIL import ImageOps
from threading import Thread
from transformers import {imports}

model_id = "{model.id}"
image_path = {image_path!r}
device_map = {device_map}
{runtime_setup}
try:
    print(f"Loading {{model_id}}...")
    load_started_at = time.perf_counter()
    processor = {processor_class}.from_pretrained(
        model_id,
        trust_remote_code=True{processor_arg_lines},
    )
    tokenizer = processor.tokenizer
    model = {model_class}.from_pretrained(model_id, **model_kwargs)
    model.eval()
    image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
    print(f"Loaded in {{time.perf_counter() - load_started_at:.2f}}s")
    print("Ready! Type 'exit' to quit.\\n")
    while True:
        try:
            text = input("> ")
        except (KeyboardInterrupt, EOFError):
            break
        if text.strip().lower() in ("exit", "quit", "q"):
            break
        if not text.strip():
            continue
        messages = [
            {{
                "role": "user",
                "content": [
                    {{"type": "image", "image": image}},
                    {{"type": "text", "text": text}},
                ],
            }}
        ]
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        streamer = TextIteratorStreamer(
            tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        def run_generate():
            with torch.inference_mode():
                model.generate(**inputs, max_new_tokens=512, streamer=streamer)

        started_at = time.perf_counter()
        thread = Thread(target=run_generate)
        thread.start()
        full = ""
        first_token_at = None
        for text in streamer:
            if first_token_at is None:
                first_token_at = time.perf_counter()
            print(text, end="", flush=True)
            full += text
        thread.join()
        print()
        print_decode_metrics(started_at, first_token_at, full)
    print("\\nBye!")
finally:
    try:
        del model
    except NameError:
        pass
    shutil.rmtree(offload_folder, ignore_errors=True)
'''


def generate_mlx_vlm_script(model: ModelInfo, image_path: str) -> str:
    return f'''\
from mlx_vlm import generate, load

try:
    from mlx_vlm.prompt_utils import apply_chat_template
except ImportError:
    apply_chat_template = None

model_id = "{model.id}"
image_path = {image_path!r}

print(f"Loading {{model_id}}...")
model, processor = load(model_id)
print("Ready! Type 'exit' to quit.\\n")

while True:
    try:
        text = input("> ")
    except (KeyboardInterrupt, EOFError):
        break
    if text.strip().lower() in ("exit", "quit", "q"):
        break
    if not text.strip():
        continue
    if apply_chat_template is not None:
        prompt = apply_chat_template(
            processor,
            getattr(model, "config", None),
            text,
            num_images=1,
        )
    else:
        prompt = text
    output = generate(
        model,
        processor,
        prompt,
        [image_path],
        max_tokens=512,
        verbose=False,
    )
    print(output)
print("\\nBye!")
'''
