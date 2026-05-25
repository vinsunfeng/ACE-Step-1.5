# ROCm Strix Halo Compatibility Test Design

**Date**: 2026-05-25 (revised after audit)
**Target**: AMD Ryzen AI MAX+ 395 (Strix Halo), Radeon 8060S (gfx1151), 128GB unified memory, ROCm 7.1, Fedora 44
**Scope**: Basic inference verification — confirm ACE-Step DiT pipeline can load and generate audio on ROCm

## Hardware Profile

| Item | Value |
|------|-------|
| APU | AMD Ryzen AI MAX+ 395 w/ Radeon 8060S |
| GPU Arch | gfx1151 (RDNA 3.5) |
| Memory | 128GB shared (CPU/GPU unified) |
| BIOS iGPU VRAM | 512MB (dedicated frame buffer) |
| GTT Allocation | 126GB (`amdgpu.gttsize=126976`, `ttm.pages_limit=32505856`) |
| ROCm | 7.1 (ROCk module loaded) |
| OS | Fedora 44, kernel 7.0.8-200.fc44.x86_64 |
| System Python | 3.14.4 (incompatible, project needs 3.11-3.12) |
| uv | 0.9.28 |
| Docker | 29.5.1, buildx 0.34.0 |

## Key Constraints

1. **pyproject.toml pins CUDA-only torch** — `torch==2.10.0+cu128` for Linux x86_64. `uv sync` will always install CUDA wheels. Must use `pip install --no-deps` with project, install deps separately.
2. **gfx1151 not in PyTorch ROCm 6.3 stable wheels** — Need TheRock community wheels or PyTorch nightly. `HSA_OVERRIDE_GFX_VERSION=11.0.0` maps to gfx1100 (RDNA 3.0) and risks silent compute errors on RDNA 3.5 — use only as last resort.
3. **flash_attn unavailable on ROCm** — `ACESTEP_LM_BACKEND=pt` bypasses nano-vllm. SDPA fallback for attention.
4. **torch.compile/Triton unreliable on ROCm** — Set `TORCH_COMPILE_BACKEND=eager` to prevent crashes.
5. **bfloat16 defaults to float32 on Strix Halo** — `_resolve_rocm_dtype()` detects ROCm iGPU and uses float32 to avoid segfaults. Controllable via `ACESTEP_ROCM_DTYPE` env var.
6. **torchao in requirements-rocm-linux.txt depends on Triton** — Will fail to install on ROCm. Must install separately with `--no-deps` or comment out.
7. **MIOpen on gfx1151 has convolution correctness issues** — Not just slow, but actual errors reported (TheRock #2488). `MIOPEN_FIND_MODE=FAST` only helps with speed, not correctness.
8. **VRAM detection critical for APUs** — `rocm-smi` reports 512MB VRAM (BIOS frame buffer) despite 126GB GTT allocation. `torch.cuda.get_device_properties(0).total_memory` may report incorrectly, putting system in Tier 1 (worst). Must verify in Phase 1.
9. **nano-vllm triton dependency gated on Python 3.11** — Using Python 3.12 avoids triton install. Do NOT use Python 3.11 with nano-vllm on ROCm.
10. **ROCm container vs host version** — Available Docker image `rocm/pytorch:rocm7.2.2` has newer userspace (7.2.2) than host driver (7.1). ROCm requires host kernel driver >= container userspace. May need to verify forward compatibility.

## Prerequisites

Before starting, verify:
- [ ] Docker Engine installed (29.5.1 confirmed)
- [ ] BuildKit enabled for heredoc COPY syntax (buildx 0.34.0 confirmed)
- [ ] User in `docker` group or has sudo
- [ ] `checkpoints/` directory exists for model downloads
- [ ] GTT memory configured: `amdgpu.gttsize=126976` (confirmed in kernel cmdline)
- [ ] Host ROCm kernel driver version compatible with container userspace

## Approach: Host-First, Docker Later

After audit review, the strategy is: **validate on host first, Docker second**.

Rationale:
- Host already has ROCm 7.1 + ROCk loaded — no device passthrough complexity
- Faster iteration: no container rebuilds for env var changes
- Easier debugging: direct access to logs, rocm-smi, dmesg
- Docker is the right approach for deployment/reproducibility, not initial compatibility testing
- Project provides `requirements-rocm-linux.txt` specifically for this scenario

### Phase 1: Host Venv Smoke Test (~10 minutes)

Create a Python 3.12 venv on the host, install ROCm PyTorch, verify basic GPU operations.

**Setup:**
```bash
uv venv --python 3.12
source .venv/bin/activate

# Install ROCm PyTorch (priority: TheRock gfx1151 wheels > nightly > stable with override)
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/rocm6.3

# If nightly doesn't include gfx1151, try with override:
# HSA_OVERRIDE_GFX_VERSION=11.0.0 pip install ...
```

**Tests:**

| ID | Test | What It Validates | Pass Criteria |
|----|------|-------------------|---------------|
| T1.1 | `rocminfo \| grep gfx1151` | GPU visible to ROCm | gfx1151 in Agent list (pre-verified) |
| T1.2 | `torch.cuda.is_available()` | PyTorch sees GPU | Returns True |
| T1.3 | `torch.cuda.get_arch_list()` | Native arch support | Contains gfx1151 or compatible |
| T1.4 | bfloat16 matmul (100x100) | Basic compute | No crash, correct shape |
| T1.5 | `F.scaled_dot_product_attention` | SDPA kernel | Returns output tensor |
| T1.6 | `torch.cuda.get_device_properties(0).total_memory` | VRAM detection | **Critical**: must report >4GB usable, not just 512MB BIOS buffer |
| T1.7 | Conv1d forward pass | MIOpen convolution path | No error (detects TheRock #2488 issue) |

**Failure policy**: Record exact error and stop. If T1.2-T1.3 fail, the PyTorch build doesn't support gfx1151. If T1.6 reports <4GB, need to investigate APU VRAM reporting. If T1.7 fails, MIOpen conv issue needs workaround before proceeding.

### Phase 2: Host Venv Full Dependencies + DiT Inference (~20 minutes)

Install ACE-Step dependencies and run actual inference on host.

**Setup:**
```bash
source .venv/bin/activate

# Install ACE-Step ROCm requirements (skip torchao to avoid Triton)
# Create a modified requirements file without torchao
grep -v torchao requirements-rocm-linux.txt > /tmp/req-rocm-no-torchao.txt
pip install -r /tmp/req-rocm-no-torchao.txt

# Install nano-vllm (Python 3.12 avoids triton dep)
pip install --no-deps -e acestep/third_parts/nano-vllm
pip install xxhash

# Install project itself (skip deps since torch conflicts with pyproject.toml pin)
pip install --no-deps -e .

# Additional deps that requirements-rocm-linux.txt may miss
pip install typer-slim pytorch-wavelets pywavelets
```

**Environment:**
```bash
export ACESTEP_LM_BACKEND=pt
export ACESTEP_DEVICE=auto
export ACESTEP_ROCM_DTYPE=float32
export TORCH_COMPILE_BACKEND=eager
export MIOPEN_FIND_MODE=FAST
export TOKENIZERS_PARALLELISM=false
# If gfx1151 not in arch list:
# export HSA_OVERRIDE_GFX_VERSION=11.0.0
```

**Tests:**

| ID | Test | What It Validates | Pass Criteria |
|----|------|-------------------|---------------|
| T2.1 | `acestep-download` | Model download | Checkpoint files present |
| T2.2 | DiT model `.to("cuda")` | Model loads to GPU | No OOM, weights on device |
| T2.3 | VAE decode random latent | Conv/VAE path | No MIOpen error, output tensor produced |
| T2.4 | 15s audio generation | Short inference | .wav output, playable |
| T2.5 | 30s+ full song | Complete generation | .wav output, acceptable quality |

**Failure policy**: Record exact error (segfault, "no kernel image", MIOpen error, OOM) and stop.

### Phase 3: Docker Build and Verification (only after Phase 2 passes)

Only invest in Docker setup after host verification confirms ROCm compatibility.

**Dockerfile.rocm:**
```dockerfile
# Verify tag exists: docker pull rocm/pytorch:rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.7.1
# RISK: container ROCm 7.2.2 > host ROCm 7.1, may need forward-compat verification
FROM rocm/pytorch:rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.7.1

WORKDIR /app

# System deps not in base image
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 ffmpeg git curl \
    && rm -rf /var/lib/apt/lists/*

# Copy ROCm requirements (without torchao)
COPY requirements-rocm-linux.txt /tmp/req-original.txt
RUN grep -v torchao /tmp/req-original.txt > /tmp/req-rocm.txt \
    && pip install --no-cache-dir -r /tmp/req-rocm.txt \
    && rm /tmp/req-*.txt

# nano-vllm (Python 3.12, --no-deps to skip triton/flash-attn)
COPY acestep/third_parts/nano-vllm /tmp/nano-vllm
RUN pip install --no-cache-dir --no-deps -e /tmp/nano-vllm \
    && pip install xxhash \
    && rm -rf /tmp/nano-vllm

# Project source (skip deps, torch already installed, avoids CUDA pin conflict)
COPY . .
RUN pip install --no-cache-dir --no-deps -e .

# Additional deps
RUN pip install --no-cache-dir typer-slim pytorch-wavelets pywavelets

# Runtime
RUN mkdir -p /app/checkpoints /app/gradio_outputs /app/output
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV ACESTEP_LM_BACKEND=pt
ENV ACESTEP_ROCM_DTYPE=float32
ENV TORCH_COMPILE_BACKEND=eager
ENV MIOPEN_FIND_MODE=FAST
ENV TOKENIZERS_PARALLELISM=false
ENV ACESTEP_CONFIG_PATH=acestep-v15-turbo
EXPOSE 7860 8001

COPY <<'EOF' /app/docker-entrypoint.sh
#!/bin/bash
set -e
echo "=== ACE-Step 1.5 ROCm ==="
python -c "import torch; print(f'torch={torch.__version__}'); print(f'HIP={torch.version.hip}'); print(f'GPU={torch.cuda.get_device_name(0)}'); print(f'VRAM={torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB')" || true
exec python -m acestep.acestep_v15_pipeline \
    --server-name 0.0.0.0 --port 7860 \
    --backend pt --init_service true \
    --config_path acestep-v15-turbo
EOF
RUN chmod +x /app/docker-entrypoint.sh
ENTRYPOINT ["/app/docker-entrypoint.sh"]
```

**docker-compose.rocm.yml:**
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
    shm_size: "8gb"
    env_file:
      - path: .env
        required: false
    environment:
      - ACESTEP_LM_BACKEND=pt
      - ACESTEP_ROCM_DTYPE=float32
      - TORCH_COMPILE_BACKEND=eager
      - MIOPEN_FIND_MODE=FAST
      - TOKENIZERS_PARALLELISM=false
      - ACESTEP_CONFIG_PATH=acestep-v15-turbo
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

**Docker tests:**

| ID | Test | What It Validates | Pass Criteria |
|----|------|-------------------|---------------|
| T3.1 | Container `torch.cuda.is_available()` | GPU passthrough | True |
| T3.2 | Model download | Checkpoints volume | Files present |
| T3.3 | DiT load + VAE decode | Full pipeline | No MIOpen/conv error |
| T3.4 | 15s audio generation | End-to-end inference | .wav output, playable |

## Out of Scope

- LM CoT / thinking mode (requires flash_attn/vllm)
- Training / LoRA fine-tuning
- Repaint / remix / audio continuation
- REST API server
- Performance benchmarking vs CUDA
- torch.compile / Triton optimizations

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| PyTorch nightly rocm6.3 lacks gfx1151 kernels | High | Blocker | Try TheRock community wheels; HSA_OVERRIDE_GFX_VERSION=11.0.0 as last resort with validation |
| VRAM reported as 512MB (BIOS buffer only) | Medium | Tier 1 limits | Verify with torch.cuda; may need MAX_CUDA_VRAM override or gpu_config patch |
| MIOpen conv errors on gfx1151 | High | VAE crash | Disable MIOpen via env vars; test Conv1d in T1.7 before proceeding |
| bfloat16 segfaults on gfx1151 | Medium | 2x memory | float32 default; 128GB GTT can absorb |
| torchao install fails (Triton dep) | High | Build abort | Excluded from requirements via grep filter |
| Docker ROCm 7.2.2 > host 7.1 | Medium | Runtime ABI error | Use host venv (Phase 2) to validate first; Docker only after confirmation |
| Container /dev/kfd permission denied | Low | Blocker | group_add video+render; verify host groups |
