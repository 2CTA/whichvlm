from __future__ import annotations

import os
import platform
import subprocess
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace

from whichvlm.data.vlm_inventory import canonical_vlm_family_id
from whichvlm.engine.quantization import infer_non_gguf_quant_type
from whichvlm.hardware.types import HardwareInfo
from whichvlm.models.package_graph import is_projector_filename, is_vision_model
from whichvlm.models.types import GGUFVariant, ModelArtifact, ModelInfo

# Runtime layer. Chooses script shape for local run backends.


class RuntimeUnsupportedError(ValueError):
    pass


@dataclass(frozen=True)
class RuntimeRequest:
    model: ModelInfo
    artifact: GGUFVariant | None
    context_length: int
    cpu_only: bool
    image_path: str | None = None
    hardware: HardwareInfo | None = None
    script_path: str | None = None


@dataclass(frozen=True)
class ServeRequest:
    model: ModelInfo
    artifact: GGUFVariant | None
    context_length: int
    cpu_only: bool
    hardware: HardwareInfo | None
    host: str
    port: int


@dataclass(frozen=True)
class CompatibilityRule:
    backend: str
    families: frozenset[str]
    artifact_formats: frozenset[str]
    operating_systems: frozenset[str]
    accelerators: frozenset[str]


TRANSFORMERS_VLM_FAMILIES = frozenset(
    {
        "qwen-vl",
        "gemma-multimodal",
        "llama-vision",
        "pixtral",
        "phi-vision",
        "llava",
    }
)
VLLM_VLM_FAMILIES = frozenset(
    {
        "qwen-vl",
        "gemma-multimodal",
        "llama-vision",
        "pixtral",
        "phi-vision",
        "llava",
        "internvl",
        "deepseek-vl",
        "glm-vision",
    }
)
SGLANG_VLM_FAMILIES = VLLM_VLM_FAMILIES
ALL_OSES = frozenset({"linux", "darwin", "windows"})

COMPATIBILITY_MATRIX = (
    CompatibilityRule(
        "llama.cpp",
        frozenset(),
        frozenset({"gguf"}),
        ALL_OSES,
        frozenset({"cpu", "cuda", "vulkan", "metal"}),
    ),
    CompatibilityRule(
        "mlx",
        frozenset(),
        frozenset({"mlx"}),
        frozenset({"darwin"}),
        frozenset({"mlx", "metal"}),
    ),
    CompatibilityRule(
        "transformers",
        TRANSFORMERS_VLM_FAMILIES,
        frozenset({"transformers"}),
        ALL_OSES,
        frozenset({"cpu", "cuda", "rocm", "mps"}),
    ),
    CompatibilityRule(
        "vllm",
        VLLM_VLM_FAMILIES,
        frozenset({"transformers"}),
        frozenset({"linux"}),
        frozenset({"cuda"}),
    ),
    CompatibilityRule(
        "sglang",
        SGLANG_VLM_FAMILIES,
        frozenset({"transformers"}),
        frozenset({"linux"}),
        frozenset({"cuda"}),
    ),
)


class Backend(ABC):
    name: str
    can_serve = False

    @abstractmethod
    def supports(
        self,
        model: ModelInfo,
        artifact: GGUFVariant | None,
        hardware: HardwareInfo | None,
    ) -> bool: ...

    @abstractmethod
    def dependencies(
        self,
        model: ModelInfo,
        artifact: GGUFVariant | None,
    ) -> list[str]: ...

    def build_command(self, request: RuntimeRequest) -> list[str]:
        assert request.script_path is not None
        return uv_command(
            self.dependencies(request.model, request.artifact),
            [request.script_path],
        )

    def run(self, request: RuntimeRequest) -> int:
        script = self.generate_script(request)
        fd, script_path = tempfile.mkstemp(suffix=".py", prefix="whichvlm_run_")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(script)
            result = subprocess.run(
                self.build_command(replace(request, script_path=script_path))
            )
            return result.returncode
        finally:
            os.unlink(script_path)

    @abstractmethod
    def generate_script(self, request: RuntimeRequest) -> str: ...

    def serve_dependencies(
        self,
        model: ModelInfo,
        artifact: GGUFVariant | None,
    ) -> list[str]:
        return self.dependencies(model, artifact)

    def serve(self, request: ServeRequest) -> int:
        raise RuntimeUnsupportedError(f"{self.name} does not support serve.")


def uv_command(deps: list[str], command: list[str]) -> list[str]:
    cmd = ["uv", "run", "--no-project"]
    for dep in deps:
        cmd.extend(["--with", dep])
    return [*cmd, *command]


def model_family_keys(model: ModelInfo) -> set[str]:
    keys = {model.family_id, model.architecture}
    for model_id in [model.id, model.base_model, model.variant_of, *model.base_models]:
        if model_id:
            family = canonical_vlm_family_id(model_id)
            if family:
                keys.add(family)
    return {key for key in keys if key}


def artifact_format(model: ModelInfo, artifact: GGUFVariant | None) -> str:
    if artifact:
        return "gguf"
    if is_mlx_model(model):
        return "mlx"
    return "transformers"


def hardware_accelerators(hardware: HardwareInfo | None) -> set[str]:
    if hardware is None:
        return {"cpu", "cuda", "rocm", "mps", "mlx", "metal", "vulkan"}
    names = {
        capability.name.lower()
        for capability in hardware.backend_capabilities
        if capability.available
    }
    for gpu in hardware.gpus:
        names.update(
            capability.name.lower()
            for capability in gpu.backend_capabilities
            if capability.available
        )
    names.add("cpu")
    return names


def hardware_os(hardware: HardwareInfo | None) -> str:
    if hardware is not None:
        return hardware.os.lower()
    return platform.system().lower()


def matrix_supports(
    backend: str,
    model: ModelInfo,
    artifact: GGUFVariant | None,
    hardware: HardwareInfo | None,
) -> bool:
    families = model_family_keys(model)
    fmt = artifact_format(model, artifact)
    os_name = hardware_os(hardware)
    accelerators = hardware_accelerators(hardware)
    return any(
        rule.backend == backend
        and fmt in rule.artifact_formats
        and os_name in rule.operating_systems
        and bool(accelerators & rule.accelerators)
        and (not rule.families or bool(families & rule.families))
        for rule in COMPATIBILITY_MATRIX
    )


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
    backend_name: str | None = None,
    hardware: HardwareInfo | None = None,
) -> tuple[list[str], str]:
    backend = select_backend(model, variant, hardware, backend_name)
    script_type = backend.name
    if script_type == "llama.cpp":
        script_type = "gguf_vlm" if is_vlm_model(model) else "gguf"
    elif script_type == "mlx":
        script_type = "mlx_vlm"
    elif script_type == "transformers" and is_vlm_model(model):
        script_type = "transformers_vlm"
    return backend.dependencies(model, variant), script_type


def generate_run_script(
    model: ModelInfo,
    variant: GGUFVariant | None,
    context_length: int,
    cpu_only: bool,
    image_path: str | None = None,
    backend_name: str | None = None,
    hardware: HardwareInfo | None = None,
) -> str:
    backend = select_backend(model, variant, hardware, backend_name)
    request = RuntimeRequest(
        model=model,
        artifact=variant,
        context_length=context_length,
        cpu_only=cpu_only,
        image_path=image_path,
        hardware=hardware,
    )
    return backend.generate_script(request)


def run_request(request: RuntimeRequest, backend_name: str | None = None) -> int:
    backend = select_backend(
        request.model,
        request.artifact,
        request.hardware,
        backend_name,
    )
    return backend.run(request)


def serve_request(request: ServeRequest, backend_name: str | None = None) -> int:
    backend = select_serve_backend(
        request.model,
        request.artifact,
        request.hardware,
        backend_name,
    )
    return backend.serve(request)


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


class LlamaCppBackend(Backend):
    name = "llama.cpp"
    can_serve = True

    def supports(
        self,
        model: ModelInfo,
        artifact: GGUFVariant | None,
        hardware: HardwareInfo | None,
    ) -> bool:
        return artifact is not None and matrix_supports(
            self.name, model, artifact, hardware
        )

    def dependencies(
        self,
        model: ModelInfo,
        artifact: GGUFVariant | None,
    ) -> list[str]:
        deps = ["llama-cpp-python", "huggingface-hub"]
        if is_vlm_model(model):
            deps.append("pillow")
        return deps

    def generate_script(self, request: RuntimeRequest) -> str:
        assert request.artifact is not None
        if is_vlm_model(request.model):
            if request.image_path is None:
                raise RuntimeUnsupportedError("VLM runners require --image PATH.")
            projector = find_projector_artifact(request.model)
            if projector is None or projector.filename is None:
                raise RuntimeUnsupportedError(
                    "GGUF VLM runtime requires an mmproj/projector artifact in "
                    "the model package metadata."
                )
            return generate_llama_cpp_vlm_script(
                request.model,
                request.artifact,
                projector,
                request.context_length,
                request.cpu_only,
                request.image_path,
            )
        return generate_llama_cpp_text_script(
            request.model,
            request.artifact,
            request.context_length,
            request.cpu_only,
        )

    def serve_dependencies(
        self,
        model: ModelInfo,
        artifact: GGUFVariant | None,
    ) -> list[str]:
        return ["llama-cpp-python[server]", "huggingface-hub"]

    def serve(self, request: ServeRequest) -> int:
        assert request.artifact is not None
        projector = None
        if is_vlm_model(request.model):
            projector = find_projector_artifact(request.model)
            if projector is None or projector.filename is None:
                raise RuntimeUnsupportedError(
                    "GGUF VLM server requires an mmproj/projector artifact in "
                    "the model package metadata."
                )
        script = generate_llama_cpp_serve_script(
            request.model,
            request.artifact,
            projector,
            request.context_length,
            request.cpu_only,
            request.host,
            request.port,
        )
        fd, script_path = tempfile.mkstemp(suffix=".py", prefix="whichvlm_serve_")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(script)
            result = subprocess.run(
                uv_command(self.serve_dependencies(request.model, request.artifact), [script_path])
            )
            return result.returncode
        finally:
            os.unlink(script_path)


class MLXBackend(Backend):
    name = "mlx"

    def supports(
        self,
        model: ModelInfo,
        artifact: GGUFVariant | None,
        hardware: HardwareInfo | None,
    ) -> bool:
        return artifact is None and matrix_supports(
            self.name, model, artifact, hardware
        )

    def dependencies(
        self,
        model: ModelInfo,
        artifact: GGUFVariant | None,
    ) -> list[str]:
        return ["mlx-vlm", "pillow"]

    def generate_script(self, request: RuntimeRequest) -> str:
        if request.image_path is None:
            raise RuntimeUnsupportedError("VLM runners require --image PATH.")
        return generate_mlx_vlm_script(request.model, request.image_path)


class TransformersBackend(Backend):
    name = "transformers"

    def supports(
        self,
        model: ModelInfo,
        artifact: GGUFVariant | None,
        hardware: HardwareInfo | None,
    ) -> bool:
        if artifact is not None or is_mlx_model(model):
            return False
        if not is_vlm_model(model):
            return True
        return matrix_supports(self.name, model, artifact, hardware)

    def dependencies(
        self,
        model: ModelInfo,
        artifact: GGUFVariant | None,
    ) -> list[str]:
        if is_vlm_model(model):
            return ["transformers", "torch", "torchvision", "accelerate", "pillow"]

        qt = infer_non_gguf_quant_type(model.id)
        base = ["transformers", "torch", "accelerate"]
        if qt == "AWQ":
            return [*base, "autoawq"]
        if qt == "GPTQ":
            return [*base, "auto-gptq"]
        return base

    def generate_script(self, request: RuntimeRequest) -> str:
        if is_vlm_model(request.model):
            if request.image_path is None:
                raise RuntimeUnsupportedError("VLM runners require --image PATH.")
            return generate_transformers_vlm_script(
                request.model,
                request.image_path,
                request.cpu_only,
            )
        return generate_transformers_text_script(request.model, request.cpu_only)


class VLLMBackend(Backend):
    name = "vllm"
    can_serve = True

    def supports(
        self,
        model: ModelInfo,
        artifact: GGUFVariant | None,
        hardware: HardwareInfo | None,
    ) -> bool:
        return (
            hardware is not None
            and artifact is None
            and is_vlm_model(model)
            and matrix_supports(self.name, model, artifact, hardware)
        )

    def dependencies(
        self,
        model: ModelInfo,
        artifact: GGUFVariant | None,
    ) -> list[str]:
        return ["vllm"]

    def generate_script(self, request: RuntimeRequest) -> str:
        if request.image_path is None:
            raise RuntimeUnsupportedError("VLM runners require --image PATH.")
        return generate_vllm_vlm_script(
            request.model,
            request.context_length,
            request.image_path,
        )

    def serve(self, request: ServeRequest) -> int:
        result = subprocess.run(
            uv_command(
                self.serve_dependencies(request.model, request.artifact),
                [
                    "vllm",
                    "serve",
                    request.model.id,
                    "--host",
                    request.host,
                    "--port",
                    str(request.port),
                    "--max-model-len",
                    str(request.context_length),
                    "--trust-remote-code",
                ],
            )
        )
        return result.returncode


class SGLangBackend(Backend):
    name = "sglang"
    can_serve = True

    def supports(
        self,
        model: ModelInfo,
        artifact: GGUFVariant | None,
        hardware: HardwareInfo | None,
    ) -> bool:
        return (
            hardware is not None
            and artifact is None
            and is_vlm_model(model)
            and matrix_supports(self.name, model, artifact, hardware)
        )

    def dependencies(
        self,
        model: ModelInfo,
        artifact: GGUFVariant | None,
    ) -> list[str]:
        return ["sglang"]

    def generate_script(self, request: RuntimeRequest) -> str:
        if request.image_path is None:
            raise RuntimeUnsupportedError("VLM runners require --image PATH.")
        return generate_sglang_vlm_script(
            request.model,
            request.context_length,
            request.image_path,
        )

    def serve(self, request: ServeRequest) -> int:
        result = subprocess.run(
            uv_command(
                self.serve_dependencies(request.model, request.artifact),
                [
                    "python",
                    "-m",
                    "sglang.launch_server",
                    "--model-path",
                    request.model.id,
                    "--host",
                    request.host,
                    "--port",
                    str(request.port),
                    "--context-length",
                    str(request.context_length),
                    "--trust-remote-code",
                ],
            )
        )
        return result.returncode


AUTO_BACKENDS: tuple[Backend, ...] = (
    LlamaCppBackend(),
    MLXBackend(),
    TransformersBackend(),
)
EXPLICIT_BACKENDS: tuple[Backend, ...] = (
    *AUTO_BACKENDS,
    VLLMBackend(),
    SGLangBackend(),
)
SERVE_AUTO_BACKENDS: tuple[Backend, ...] = (
    LlamaCppBackend(),
    VLLMBackend(),
    SGLangBackend(),
)


def normalize_backend_name(name: str) -> str:
    value = name.lower().replace("_", "-")
    if value in {"llama-cpp", "llamacpp", "gguf"}:
        return "llama.cpp"
    return value


def select_backend(
    model: ModelInfo,
    artifact: GGUFVariant | None,
    hardware: HardwareInfo | None = None,
    backend_name: str | None = None,
) -> Backend:
    if backend_name and backend_name != "auto":
        target = normalize_backend_name(backend_name)
        for backend in EXPLICIT_BACKENDS:
            if backend.name == target:
                if backend.supports(model, artifact, hardware):
                    return backend
                raise RuntimeUnsupportedError(
                    f"{backend_name} does not support {model.id} on this hardware."
                )
        raise RuntimeUnsupportedError(f"Unknown backend: {backend_name}")

    for backend in AUTO_BACKENDS:
        if backend.supports(model, artifact, hardware):
            return backend

    raise RuntimeUnsupportedError(
        f"No supported run backend for {model.id}. "
        "Try --backend vllm or --backend sglang on Linux/CUDA for supported VLMs."
    )


def select_serve_backend(
    model: ModelInfo,
    artifact: GGUFVariant | None,
    hardware: HardwareInfo | None = None,
    backend_name: str | None = None,
) -> Backend:
    if backend_name and backend_name != "auto":
        target = normalize_backend_name(backend_name)
        for backend in EXPLICIT_BACKENDS:
            if backend.name != target:
                continue
            if not backend.can_serve:
                raise RuntimeUnsupportedError(
                    f"{backend_name} does not support serve; use run instead."
                )
            if backend.supports(model, artifact, hardware):
                return backend
            raise RuntimeUnsupportedError(
                f"{backend_name} cannot serve {model.id} on this hardware."
            )
        raise RuntimeUnsupportedError(f"Unknown backend: {backend_name}")

    for backend in SERVE_AUTO_BACKENDS:
        if backend.supports(model, artifact, hardware):
            return backend

    raise RuntimeUnsupportedError(
        f"No supported serve backend for {model.id}. "
        "Use a GGUF artifact for llama.cpp or --backend vllm/sglang on Linux/CUDA."
    )


def generate_llama_cpp_text_script(
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


def generate_llama_cpp_vlm_script(
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


def llama_cpp_server_chat_format(model_id: str) -> str:
    value = model_id.lower()
    if "qwen" in value and "vl" in value:
        return "qwen2-vl"
    if "minicpm" in value:
        return "minicpm-v-2.6"
    return "llava-1-5"


def generate_llama_cpp_serve_script(
    model: ModelInfo,
    variant: GGUFVariant,
    projector: ModelArtifact | None,
    context_length: int,
    cpu_only: bool,
    host: str,
    port: int,
) -> str:
    n_gpu = 0 if cpu_only else -1
    projector_filename = projector.filename if projector else None
    chat_format = llama_cpp_server_chat_format(model.id)
    return f'''\
import subprocess
import sys

from huggingface_hub import hf_hub_download

model_id = "{model.id}"
model_filename = "{variant.filename}"
projector_filename = {projector_filename!r}

print(f"Downloading {{model_id}}...")
model_path = hf_hub_download(repo_id=model_id, filename=model_filename)
cmd = [
    sys.executable,
    "-m",
    "llama_cpp.server",
    "--model",
    model_path,
    "--n_ctx",
    "{context_length}",
    "--n_gpu_layers",
    "{n_gpu}",
    "--host",
    "{host}",
    "--port",
    "{port}",
]
if projector_filename is not None:
    mmproj_path = hf_hub_download(repo_id=model_id, filename=projector_filename)
    cmd.extend(
        [
            "--clip_model_path",
            mmproj_path,
            "--chat_format",
            "{chat_format}",
        ]
    )
raise SystemExit(subprocess.run(cmd).returncode)
'''


def generate_transformers_text_script(model: ModelInfo, cpu_only: bool) -> str:
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
    model.eval()
    print("Ready! Type 'exit' to quit.\\n")
    messages = []

    def generate_stream(**kwargs):
        with torch.inference_mode():
            model.generate(**kwargs)

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
            target=generate_stream,
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
    shutil.rmtree(offload_folder, ignore_errors=True)
'''


def generate_transformers_vlm_script(
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
    model.eval()
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
        with torch.inference_mode():
            outputs = model.generate(**inputs, max_new_tokens=512)
        print(processor.decode(outputs[0], skip_special_tokens=True))
    print("\\nBye!")
finally:
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


def generate_vllm_vlm_script(
    model: ModelInfo,
    context_length: int,
    image_path: str,
) -> str:
    return f'''\
import base64
import mimetypes

from vllm import LLM, SamplingParams

model_id = "{model.id}"
image_path = {image_path!r}


def image_data_url(path):
    mime = mimetypes.guess_type(path)[0] or "image/png"
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")
    return f"data:{{mime}};base64,{{encoded}}"


print(f"Loading {{model_id}} with vLLM...")
llm = LLM(
    model=model_id,
    trust_remote_code=True,
    max_model_len={context_length},
)
sampling = SamplingParams(max_tokens=512)
image_url = image_data_url(image_path)
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
                {{"type": "image_url", "image_url": {{"url": image_url}}}},
                {{"type": "text", "text": text}},
            ],
        }}
    ]
    outputs = llm.chat(messages, sampling_params=sampling)
    print(outputs[0].outputs[0].text)
print("\\nBye!")
'''


def generate_sglang_vlm_script(
    model: ModelInfo,
    context_length: int,
    image_path: str,
) -> str:
    return f'''\
from sglang import Engine

model_id = "{model.id}"
image_path = {image_path!r}

print(f"Loading {{model_id}} with SGLang...")
engine = Engine(
    model_path=model_id,
    trust_remote_code=True,
    context_length={context_length},
)
try:
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
        response = engine.generate(
            prompt=text,
            image_data=image_path,
            sampling_params={{"max_new_tokens": 512}},
        )
        print(response["text"])
    print("\\nBye!")
finally:
    engine.shutdown()
'''
