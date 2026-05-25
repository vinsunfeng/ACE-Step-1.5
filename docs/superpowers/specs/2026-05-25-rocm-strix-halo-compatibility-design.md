# ROCm Strix Halo Compatibility Test Design

**Date**: 2026-05-25
**Target**: AMD Ryzen AI MAX+ 395 (Strix Halo), Radeon 8060S (gfx1151), 128GB unified memory, ROCm 7.1, Fedora 44
**Scope**: Basic inference verification — confirm ACE-Step DiT pipeline can load and generate audio on ROCm

## Hardware Profile

| Item | Value |
|------|-------|
| APU | AMD Ryzen AI MAX+ 395 w/ Radeon 8060S |
| GPU Arch | gfx1151 (RDNA 3.5) |
| Memory | 128GB shared (CPU/GPU unified) |
| ROCm | 7.1 (ROCk module loaded) |
| OS | Fedora 44, kernel 7.0.8-200.fc44.x86_64 |
| System Python | 3.14.4 (incompatible, project needs 3.11-3.12) |
| uv | 0.9.28 |

## Key Constraints

1. **pyproject.toml pins CUDA-only torch** — `torch==2.10.0+cu128` for Linux x86_64. `uv sync` will always install CUDA wheels. Must use pip with `requirements-rocm-linux.txt` instead.
2. **gfx1151 not in PyTorch ROCm 6.3 stable wheels** — Need TheRock community wheels or PyTorch nightly for native gfx1151 support. `HSA_OVERRIDE_GFX_VERSION=11.0.0` maps to gfx1100 (RDNA 3.0) and risks silent compute errors on RDNA 3.5.
3. **flash_attn unavailable on ROCm** — `ACESTEP_LM_BACKEND=pt` bypasses nano-vllm. SDPA fallback is used for attention.
4. **torch.compile/Triton unreliable on ROCm** — Must be prepared to disable with `TORCH_COMPILE_BACKEND=eager` or `compile_model=False`.
5. **bfloat16 defaults to float32 on Strix Halo** — Project has `_resolve_rocm_dtype()` that detects ROCm iGPU and uses float32 to avoid segfaults. 128GB unified memory can absorb the doubled memory footprint.
6. **torchao, torchcodec excluded from ROCm** — `requirements-rocm-linux.txt` already handles this. soundfile fallback for audio, quantization disabled.
7. **MIOPEN_FIND_MODE=FAST required** — Without this, first VAE decode convolution hangs for minutes while MIOpen benchmarks kernel configurations.

## Approach

After critical review, the chosen approach is:

- **Dockerfile base**: `rocm/dev-ubuntu-22.04:7.1` — matches host ROCm version, provides full userspace stack
- **Python**: 3.12 (TheRock community wheels best support cp312)
- **PyTorch source**: TheRock community wheels (gfx1151 native) > PyTorch nightly ROCm > build from source
- **Dependency install**: `pip install -r requirements-rocm-linux.txt`, NOT `uv sync`
- **GPU passthrough**: `/dev/kfd` + `/dev/dri` device mapping with `video`/`render` group access

## Phase 1: Host-Level Smoke Test

Verify ROCm 7.1 + gfx1151 PyTorch basics before investing in Docker setup. ~5 minutes.

| ID | Test | What It Validates | Pass Criteria |
|----|------|-------------------|---------------|
| T1.1 | `rocminfo` | GPU visible to ROCm | gfx1151 appears in Agent list |
| T1.2 | `torch.cuda.is_available()` | PyTorch sees GPU | Returns True |
| T1.3 | `torch.cuda.get_arch_list()` | Native arch support | Contains gfx1151 |
| T1.4 | bfloat16 matmul | Compute correctness | No crash, correct shape |
| T1.5 | `F.scaled_dot_product_attention` | SDPA kernel works | Returns output tensor |
| T1.6 | `torch.cuda.get_device_properties(0).total_memory` | VRAM detection | Reflects shared memory allocation |

**Failure policy**: Record error and stop. If PyTorch ROCm fails at this level, Docker cannot fix it.

## Phase 2: Docker ROCm Image Build

### Dockerfile (Dockerfile.rocm)

```dockerfile
FROM rocm/dev-ubuntu-22.04:7.1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common build-essential git curl wget \
    libsndfile1 libsndfile1-dev ffmpeg libffi-dev libssl-dev \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y python3.12 python3.12-dev python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

# Python env
RUN python3.12 -m venv /opt/acestep
ENV PATH="/opt/acestep/bin:${PATH}"

# PyTorch ROCm (TheRock or nightly)
RUN pip install --no-cache-dir torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/rocm6.3

# ACE-Step dependencies (skip torch, use ROCm requirements)
WORKDIR /app
COPY requirements-rocm-linux.txt .
RUN pip install --no-cache-dir -r requirements-rocm-linux.txt

# nano-vllm (local package)
COPY acestep/third_parts/nano-vllm /tmp/nano-vllm
RUN pip install --no-cache-dir -e /tmp/nano-vllm && rm -rf /tmp/nano-vllm

# Project source
COPY . .
RUN pip install --no-cache-dir -e .

# Runtime
RUN mkdir -p /app/checkpoints /app/gradio_outputs /app/output
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV ACESTEP_API_HOST=0.0.0.0
ENV ACESTEP_MODE=gradio
ENV ACESTEP_INIT_SERVICE=true
ENV ACESTEP_CONFIG_PATH=acestep-v15-turbo
ENV ACESTEP_LM_BACKEND=pt
ENV MIOPEN_FIND_MODE=FAST
ENV TOKENIZERS_PARALLELISM=false
EXPOSE 7860 8001

COPY <<'EOF' /app/docker-entrypoint.sh
#!/bin/bash
set -e
echo "=== ACE-Step 1.5 ROCm ==="
python -c "import torch; print(f'torch={torch.__version__}'); print(f'HIP={torch.version.hip}'); print(f'GPU={torch.cuda.get_device_name(0)}'); print(f'VRAM={torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB')"
exec python -m acestep.acestep_v15_pipeline \
    --server-name 0.0.0.0 --port 7860 \
    --backend pt --init_service true \
    --config_path acestep-v15-turbo
EOF
RUN chmod +x /app/docker-entrypoint.sh
ENTRYPOINT ["/app/docker-entrypoint.sh"]
```

### docker-compose.rocm.yml

```yaml
services:
  acestep:
    build:
      context: .
      dockerfile: Dockerfile.rocm
    container_name: acestep-rocm
    devices:
      - /dev/kfd:/dev/kfd
      - /dev/dri:/dev/dri
    group_add:
      - video
      - render
    shm_size: "4gb"
    env_file:
      - path: .env
        required: false
    environment:
      - ACESTEP_LM_BACKEND=pt
      - ACESTEP_INIT_SERVICE=true
      - ACESTEP_CONFIG_PATH=acestep-v15-turbo
      - MIOPEN_FIND_MODE=FAST
      - TOKENIZERS_PARALLELISM=false
    ports:
      - "7860:7860"
    volumes:
      - ./checkpoints:/app/checkpoints
      - hf_cache:/root/.cache/huggingface
      - ./gradio_outputs:/app/gradio_outputs
    restart: unless-stopped

volumes:
  hf_cache:
```

## Phase 3: Docker In-Container Verification

| ID | Test | What It Validates | Pass Criteria |
|----|------|-------------------|---------------|
| T3.1 | Container torch.cuda.is_available() | GPU passthrough works | True |
| T3.2 | Model download | acestep-download completes | Checkpoint files present |
| T3.3 | DiT model load | Model loads to GPU | No OOM, tensors on cuda |
| T3.4 | 15s audio generation | Short inference | .wav output, playable |
| T3.5 | 30s+ full song | Complete generation | .wav output, acceptable quality |

**Failure policy**: Record error (ROCm/HIP segfault, "no kernel image", OOM), stop immediately.

## Out of Scope

- LM CoT / thinking mode (requires flash_attn)
- Training / LoRA fine-tuning
- Repaint / remix / audio continuation
- REST API server
- Performance benchmarking vs CUDA
- torch.compile / Triton optimizations

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| PyTorch nightly lacks gfx1151 kernels | Medium | Blocker | Try TheRock wheels; fall back to HSA_OVERRIDE_GFX_VERSION=11.0.0 with careful validation |
| bfloat16 segfaults on gfx1151 | Medium | Performance (2x memory) | Use float32 default; 128GB can absorb it |
| MIOpen kernel compilation hangs | High | UX (long wait) | MIOPEN_FIND_MODE=FAST already set |
| torch.compile crashes on ROCm | High | Performance | Disable compile, use eager backend |
| Container permission denied on /dev/kfd | Low | Blocker | group_add video+render; verify host permissions |
