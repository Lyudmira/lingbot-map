# LingBot-Map: GPU inference and demo (viser / GLB export).
# Matches README Quick Start: PyTorch 2.9 + CUDA 12.8, FlashInfer for paged KV attention.
#
# Use the **devel** base (not runtime): flashinfer-python JIT-compiles some CUDA on first use and needs nvcc.
#
# Build (repository root):
#   docker build -t lingbot-map:latest .
#
# Other CUDA / PyTorch tags (host driver must support the image CUDA user-mode stack):
#   docker build -t lingbot-map:cu126 --build-arg PYTORCH_BASE=pytorch/pytorch:2.9.1-cuda12.6-cudnn9-devel \
#     --build-arg FLASHINFER_EXTRA_INDEX=https://flashinfer.ai/whl/cu126/torch2.9/ .
# Pick the FlashInfer index to match your CUDA major line; see https://docs.flashinfer.ai/installation.html
#
# Example: interactive viewer (map checkpoints + images; viser on 8080)
#   docker run --rm -it --gpus all -p 8080:8080 \
#     -v /path/to/checkpoints:/workspace/lingbot-map/checkpoints:ro \
#     -v /path/to/images:/data/images:ro \
#     -v lingbot-output:/workspace/lingbot-map/output \
#     lingbot-map:latest \
#     python demo.py \
#       --model_path /workspace/lingbot-map/checkpoints/lingbot-map.pt \
#       --image_folder /data/images --viewer --no_export_glb
#
# Headless GLB export (no port publish):
#   docker run --rm -it --gpus all \
#     -v ... (same mounts) \
#     lingbot-map:latest \
#     python demo.py --model_path ... --image_folder /data/images
#
# Build needs outbound HTTPS to pypi.org and flashinfer.ai. Host needs NVIDIA Container Toolkit and a
# driver new enough for the CUDA version in PYTORCH_BASE.
#
# Adapting to other machines (complexity):
# - Same arch (linux/amd64), driver OK: no Dockerfile edits; build and run.
# - Older driver: choose an older PYTORCH_BASE tag (lower CUDA) + matching FLASHINFER_EXTRA_INDEX.
# - ARM / Jetson: official pytorch tags differ; you may need a community base or conda-forge; expect more trial.
# - Air-gapped build: pull base + wheels on a connected host, docker save, transfer, docker load; or a private registry.

ARG PYTORCH_BASE=pytorch/pytorch:2.9.1-cuda12.8-cudnn9-devel
FROM ${PYTORCH_BASE}

ARG FLASHINFER_EXTRA_INDEX=https://flashinfer.ai/whl/cu128/torch2.9/

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FLASHINFER_WORKSPACE_BASE=/workspace/lingbot-map

# OpenCV (opencv-python) and some GUI-adjacent stacks expect basic X / GLib libs even for headless decode.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/lingbot-map

COPY pyproject.toml .
COPY lingbot_map ./lingbot_map
COPY demo.py .

RUN pip install --no-cache-dir -e ".[vis]"

# FlashInfer: PyPI primary; CUDA/torch-specific wheels from FlashInfer index (must match image CUDA line).
RUN pip install --no-cache-dir flashinfer-python \
    --extra-index-url ${FLASHINFER_EXTRA_INDEX}

EXPOSE 8080

CMD ["bash"]
