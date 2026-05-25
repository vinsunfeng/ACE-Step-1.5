# ROCm Strix Halo Compatibility Test Design

**Date**: 2026-05-25 (v3 — vllm toolbox approach)
**Target**: AMD Ryzen AI MAX+ 395 (Strix Halo), Radeon 8060S (gfx1151), 128GB unified memory, Fedora 44
**Scope**: Basic inference verification — confirm ACE-Step DiT pipeline can load and generate audio on ROCm

## Hardware Profile

| Item | Value |
|------|-------|
| APU | AMD Ryzen AI MAX+ 395 w/ Radeon 8060S |
| GPU Arch | gfx1151 (RDNA 3.5) |
| Memory | 128GB shared (CPU/GPU unified) |
| BIOS iGPU VRAM | 512MB (dedicated frame buffer) |
| GTT Allocation | 126GB (`amdgpu.gttsize=126976`, `ttm.pages_limit=32505856`) |
| Host ROCm | 7.1 (ROCk module loaded, Fedora 44 dnf packages) |
| OS | Fedora 44, kernel 7.0.8-200.fc44.x86_64 |
| Kernel params | `iommu=pt` (consider `amd_iommu=off` for 5-12% perf gain per Strix Halo benchmarks) |
| Docker/Podman | 29.5.1, buildx 0.34.0 |
| uv | 0.9.28 |

## Approach: vllm Toolbox

Use `amd-strix-halo-vllm-toolboxes` as the execution environment. This toolbox already provides:
- **Python 3.12 venv** at `/opt/venv`
- **PyTorch TheRock nightly** built specifically for gfx1151 (`rocm.nightlies.amd.com/v2-staging/gfx1151/`)
- **ROCm flash-attention** compiled for gfx1151 (ROCm fork, not standard flash-attn)
- **Full ROCm SDK** (TheRock tarball method, not minimal runtime)
- **bitsandbytes** (ROCm build), **aiter**, **RCCL** (gfx1151 custom build)

This eliminates the top 3 blockers from previous designs:
1. ~~PyTorch lacks gfx1151 kernels~~ → TheRock nightly has native gfx1151 support
2. ~~flash_attn unavailable on ROCm~~ → ROCm flash-attention already built
3. ~~Need to build Python env from scratch~~ → Python 3.12 + venv ready

### Why vllm toolbox over llama.cpp toolbox

| Need | llama.cpp toolbox | vllm toolbox |
|------|------------------|-------------|
| Python | None (C++ binary) | Python 3.12 + venv |
| PyTorch | None | TheRock nightly gfx1151 |
| ROCm SDK | Runtime only (7.2.3) | Full SDK (TheRock nightly) |
| flash-attention | None | Built for gfx1151 |
| Build tools | None | cmake, ninja, gcc, clang |

## Key Constraints (resolved and remaining)

1. ~~PyTorch lacks gfx1151~~ → **RESOLVED** by TheRock nightly
2. ~~flash_attn unavailable~~ → **RESOLVED** by ROCm flash-attention
3. ~~Python env missing~~ → **RESOLVED** by venv in toolbox
4. **torchao depends on Triton** — Skip torchao from requirements, quantization unavailable
5. **MIOpen conv issues on gfx1151** — Set `MIOPEN_DEBUG_CONV_DIRECT=0` as preventive measure
6. **VRAM detection for APUs** — rocm-smi reports 512MB, `torch.cuda` may report similarly. Must verify.
7. **nano-vllm triton dep gated on Python 3.11** — Python 3.12 in toolbox avoids this
8. **vLLM deps may conflict with ACE-Step** — Use `--no-deps` for ACE-Step packages where possible
9. **pyproject.toml CUDA torch pin** — Use `pip install --no-deps -e .` to bypass
10. **bfloat16 on Strix Halo** — `_resolve_rocm_dtype()` defaults to float32; try bfloat16 since ROCm 7.2 has better support

## Prerequisites

- [x] Docker/Podman installed (29.5.1)
- [x] GTT memory configured in kernel cmdline
- [x] `~/amd-strix-halo-vllm-toolboxes` cloned
- [ ] Create toolbox container (one-time setup)
- [ ] `checkpoints/` directory for model downloads

## Phase 1: Toolbox Setup + Smoke Test (~15 min)

### 1.1 Create toolbox

```bash
toolbox create acestep-rocm \
  --image docker.io/kyuz0/vllm-therock-gfx1151:stable \
  -- --device /dev/dri --device /dev/kfd \
  --group-add video --group-add render --group-add sudo \
  --security-opt seccomp=unconfined

toolbox enter acestep-rocm
```

### 1.2 Verify toolbox environment

```bash
# Inside toolbox:
source /opt/venv/bin/activate  # should auto-activate via /etc/profile.d/venv.sh
python -c "import torch; print(f'torch={torch.__version__}'); print(f'HIP={torch.version.hip}'); print(f'GPU={torch.cuda.get_device_name(0)}'); print(f'VRAM={torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB')"
```

### 1.3 Smoke tests

| ID | Test | What It Validates | Pass Criteria |
|----|------|-------------------|---------------|
| T1.1 | `rocminfo \| grep gfx1151` | GPU visible in toolbox | gfx1151 in Agent list |
| T1.2 | `torch.cuda.is_available()` | PyTorch sees GPU | True |
| T1.3 | `torch.cuda.get_arch_list()` | Native gfx1151 support | Contains gfx1151 |
| T1.4 | bfloat16 matmul | Compute correctness | No crash |
| T1.5 | `F.scaled_dot_product_attention` | SDPA kernel works | Output tensor returned |
| T1.6 | `torch.cuda.get_device_properties(0).total_memory` | VRAM detection | Reports >4GB (not just 512MB BIOS buffer) |
| T1.7 | Conv1d forward pass | MIOpen conv path | No error |

**Failure policy**: Record error, stop. Toolbox-level failures mean the container ROCm stack doesn't support gfx1151 properly.

## Phase 2: ACE-Step Install + Inference (~20 min)

### 2.1 Install ACE-Step dependencies

```bash
# Inside toolbox, venv already active
cd ~/project/ACE-Step-1.5  # $HOME is mounted in toolbox

# ROCm requirements without torchao (Triton dep)
grep -v torchao requirements-rocm-linux.txt > /tmp/req-rocm.txt
pip install -r /tmp/req-rocm.txt

# nano-vllm (Python 3.12 avoids triton, --no-deps to be safe)
pip install --no-deps -e acestep/third_parts/nano-vllm
pip install xxhash

# Project itself (skip deps to avoid CUDA torch pin conflict)
pip install --no-deps -e .

# Additional deps
pip install typer-slim pytorch-wavelets pywavelets
```

### 2.2 Set environment

```bash
export ACESTEP_LM_BACKEND=pt
export ACESTEP_DEVICE=auto
export ACESTEP_ROCM_DTYPE=float32
export TORCH_COMPILE_BACKEND=eager
export MIOPEN_FIND_MODE=FAST
export MIOPEN_DEBUG_CONV_DIRECT=0
export TOKENIZERS_PARALLELISM=false
```

### 2.3 Inference tests

| ID | Test | What It Validates | Pass Criteria |
|----|------|-------------------|---------------|
| T2.1 | `acestep-download` | Model download | Checkpoint files present |
| T2.2 | DiT model `.to("cuda")` | Model loads to GPU | No OOM |
| T2.3 | VAE decode random latent | Conv/VAE path isolated | No MIOpen error |
| T2.4 | 15s audio generation | Short inference | .wav output, playable |
| T2.5 | 30s+ full song | Complete generation | .wav output, acceptable quality |

**Failure policy**: Record exact error, stop.

## Phase 3: Service Deployment (after Phase 2 passes)

Options for running ACE-Step as a persistent service:

**Option A: podman run (simplest)**
```bash
podman run -d --name acestep-service \
  --device /dev/dri --device /dev/kfd \
  --group-add video --group-add render \
  -p 7860:7860 \
  -v ~/checkpoints:/root/.cache/huggingface \
  kyuz0/vllm-therock-gfx1151:stable \
  bash -c "source /opt/venv/bin/activate && python -m acestep.acestep_v15_pipeline --server-name 0.0.0.0"
```

**Option B: systemd user service**
```ini
[Service]
ExecStart=podman run --rm --name acestep ...
Restart=on-failure
```

**Option C: Custom Dockerfile** (if vLLM deps conflict with ACE-Step)
Base on vllm toolbox Dockerfile steps 1-5 (system + ROCm + Python + PyTorch), replace vLLM with ACE-Step deps.

## Out of Scope

- LM CoT / thinking mode (requires vllm/flash_attn integration — may work now since flash-attention is available)
- Training / LoRA fine-tuning
- Repaint / remix / audio continuation
- Performance benchmarking vs CUDA
- torch.compile / Triton optimizations

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| vLLM deps conflict with ACE-Step | Medium | Install failure | Use `--no-deps`; fallback: custom Dockerfile from vllm toolbox base |
| VRAM reported as 512MB | Medium | Tier 1 limits | Verify T1.6; may need `MAX_CUDA_VRAM` override |
| MIOpen conv errors on gfx1151 | High | VAE crash | `MIOPEN_DEBUG_CONV_DIRECT=0` set preemptively; T1.7 catches it |
| bfloat16 segfaults | Medium | 2x memory | float32 default; 128GB can absorb |
| torchao fails to install | High | No quantization | Already excluded from requirements |
| Kernel `iommu=pt` vs `amd_iommu=off` | Low | 5-12% perf loss | Can change later; not a blocker |
