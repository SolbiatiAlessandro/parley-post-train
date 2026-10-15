"""Serve Qwen3.5-4B on one Modal GPU with vLLM's OpenAI-compatible server.

Deploy:  modal deploy modal_qwen.py
The container starts on the first request and shuts down after 10 idle
minutes, so the GPU is billed only while a game is using it.
The API key comes from the local .env file (VLLM_API_KEY); vLLM reads it
from the environment, so it never appears on a command line or in logs.
"""

import subprocess
from pathlib import Path

import modal

MODEL = "Qwen/Qwen3.5-4B"
SERVED_NAME = "qwen3.5-4b"
VLLM_VERSION = "0.31.0"
GPU = "L40S"

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(f"vllm=={VLLM_VERSION}", "huggingface_hub[hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/cache/hf"})
)

hf_cache = modal.Volume.from_name("parley-hf-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("parley-vllm-cache", create_if_missing=True)

app = modal.App("parley-qwen")


@app.function(
    image=image,
    gpu=GPU,
    volumes={"/cache/hf": hf_cache, "/root/.cache/vllm": vllm_cache},
    secrets=[modal.Secret.from_dotenv(Path(__file__).parent)],
    timeout=2 * 60 * 60,
    scaledown_window=10 * 60,
    max_containers=1,
)
@modal.concurrent(max_inputs=32)
@modal.web_server(port=8000, startup_timeout=20 * 60)
def serve():
    subprocess.Popen(
        [
            "vllm", "serve", MODEL,
            "--served-model-name", SERVED_NAME,
            "--host", "0.0.0.0", "--port", "8000",
            "--max-model-len", "32768",
            "--language-model-only",
            "--reasoning-parser", "qwen3",
            "--gpu-memory-utilization", "0.90",
        ]
    )
