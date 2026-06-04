"""Runtime dependency and snippet generation for local VLM execution."""

from __future__ import annotations

from whichvlm.engine.quantization import infer_non_gguf_quant_type
from whichvlm.models.package_graph import is_projector_filename, is_vision_model
from whichvlm.models.types import GGUFVariant, ModelArtifact, ModelInfo


class RuntimeUnsupportedError(ValueError):
    pass


def is_vlm_model(model: ModelInfo) -> bool:
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
    if variant:
        deps = ["llama-cpp-python", "huggingface-hub"]
        if is_vlm_model(model):
            deps.append("pillow")
        return deps, (
            "gguf_vlm" if is_vlm_model(model) else "gguf"
        )

    if is_vlm_model(model):
        if _is_mlx_model(model):
            return ["mlx-vlm", "pillow"], "mlx_vlm"
        return ["transformers", "torch", "accelerate", "pillow"], "transformers_vlm"

    qt = infer_non_gguf_quant_type(model.id)
    base = ["transformers", "torch", "accelerate"]
    if qt == "AWQ":
        return [*base, "autoawq"], "transformers"
    if qt == "GPTQ":
        return [*base, "auto-gptq"], "transformers"
    return base, "transformers"


def generate_run_script(
    model: ModelInfo,
    variant: GGUFVariant | None,
    context_length: int,
    cpu_only: bool,
    image_path: str | None = None,
) -> str:
    if is_vlm_model(model):
        if image_path is None:
            raise RuntimeUnsupportedError("VLM runners require --image PATH.")
        if variant:
            projector = _find_projector_artifact(model)
            if projector is None or projector.filename is None:
                raise RuntimeUnsupportedError(
                    "GGUF VLM runtime requires an mmproj/projector artifact in "
                    "the model package metadata."
                )
            return _generate_llama_cpp_vlm_script(
                model,
                variant,
                projector,
                context_length,
                cpu_only,
                image_path,
            )
        if _is_mlx_model(model):
            return _generate_mlx_vlm_script(model, image_path)
        return _generate_transformers_vlm_script(model, image_path, cpu_only)

    if variant:
        return _generate_llama_cpp_text_script(model, variant, context_length, cpu_only)
    return _generate_transformers_text_script(model, cpu_only)


def _is_mlx_model(model: ModelInfo) -> bool:
    if model.model_format == "mlx":
        return True
    if (model.quantization_type or "").upper() == "MLX":
        return True
    return any(artifact.format == "mlx" for artifact in model.artifacts)


def _find_projector_artifact(model: ModelInfo) -> ModelArtifact | None:
    for artifact in model.artifacts:
        if artifact.source_kind == "mmproj" and artifact.filename:
            return artifact
    for artifact in model.artifacts:
        if artifact.filename and is_projector_filename(artifact.filename):
            return artifact
    return None


def _generate_llama_cpp_text_script(
    model: ModelInfo,
    variant: GGUFVariant,
    context_length: int,
    cpu_only: bool,
) -> str:
    n_gpu = 0 if cpu_only else -1
    return f'''\
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

print("Downloading {model.id} ({variant.quant_type})...")
model_path = hf_hub_download(repo_id="{model.id}", filename="{variant.filename}")
print("Loading model...")
llm = Llama(
    model_path=model_path,
    n_ctx={context_length},
    n_gpu_layers={n_gpu},
    verbose=False,
)
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
    response = llm.create_chat_completion(messages=messages, stream=True)
    full = ""
    for chunk in response:
        delta = chunk["choices"][0].get("delta", {{}})
        content = delta.get("content", "")
        if content:
            print(content, end="", flush=True)
            full += content
    print()
    messages.append({{"role": "assistant", "content": full}})
print("\\nBye!")
'''


def _generate_llama_cpp_vlm_script(
    model: ModelInfo,
    variant: GGUFVariant,
    projector: ModelArtifact,
    context_length: int,
    cpu_only: bool,
    image_path: str,
) -> str:
    n_gpu = 0 if cpu_only else -1
    return f'''\
import base64
import mimetypes

from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from llama_cpp import llama_chat_format

model_id = "{model.id}"
model_filename = "{variant.filename}"
projector_filename = "{projector.filename}"
image_path = {image_path!r}


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

print("Loading model...")
llm = Llama(
    model_path=model_path,
    chat_handler=handler,
    n_ctx={context_length},
    n_gpu_layers={n_gpu},
    verbose=False,
)

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
    response = llm.create_chat_completion(messages=messages, stream=True)
    for chunk in response:
        delta = chunk["choices"][0].get("delta", {{}})
        content = delta.get("content", "")
        if content:
            print(content, end="", flush=True)
    print()
print("\\nBye!")
'''


def _generate_transformers_text_script(model: ModelInfo, cpu_only: bool) -> str:
    device_map = '"cpu"' if cpu_only else '"auto"'
    dtype = "torch.float32" if cpu_only else '"auto"'
    return f'''\
import shutil
import tempfile
import torch
from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

model_id = "{model.id}"
offload_folder = tempfile.mkdtemp(prefix="whichvlm_transformers_offload_")
try:
    print(f"Loading {{model_id}}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map={device_map},
        torch_dtype={dtype},
        trust_remote_code=True,
        offload_folder=offload_folder,
    )
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
        streamer = TextIteratorStreamer(
            tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        thread = Thread(
            target=model.generate,
            kwargs=dict(**inputs, max_new_tokens=512, streamer=streamer),
        )
        thread.start()
        full = ""
        for text in streamer:
            print(text, end="", flush=True)
            full += text
        thread.join()
        print()
        messages.append({{"role": "assistant", "content": full}})
    print("\\nBye!")
finally:
    try:
        del model
    except NameError:
        pass
    shutil.rmtree(offload_folder, ignore_errors=True)
'''


def _generate_transformers_vlm_script(
    model: ModelInfo,
    image_path: str,
    cpu_only: bool,
) -> str:
    device_map = '"cpu"' if cpu_only else '"auto"'
    dtype = "torch.float32" if cpu_only else '"auto"'
    return f'''\
import shutil
import tempfile
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

model_id = "{model.id}"
image_path = {image_path!r}
offload_folder = tempfile.mkdtemp(prefix="whichvlm_transformers_offload_")
try:
    print(f"Loading {{model_id}}...")
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        device_map={device_map},
        torch_dtype={dtype},
        trust_remote_code=True,
        offload_folder=offload_folder,
    )
    image = Image.open(image_path).convert("RGB")
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
        outputs = model.generate(**inputs, max_new_tokens=512)
        print(processor.decode(outputs[0], skip_special_tokens=True))
    print("\\nBye!")
finally:
    try:
        del model
    except NameError:
        pass
    shutil.rmtree(offload_folder, ignore_errors=True)
'''


def _generate_mlx_vlm_script(model: ModelInfo, image_path: str) -> str:
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
