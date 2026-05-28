# ROCm Container Setup for ACE-Step 1.5

Detailed guide for running ACE-Step in a container on AMD Strix Halo (gfx1151).

## Prerequisites

- AMD Ryzen AI MAX+ 395 (Strix Halo) or compatible AMD GPU
- Fedora 42/43 (Silverblue or Workstation) with Podman
- At least 64 GB system RAM
- Kernel parameters configured (see README-ROCm.md)

## Container Image

The recommended container image is [kyuz0/vllm-therock-gfx1151](https://github.com/kyuz0/amd-strix-halo-vllm-toolboxes):
- Fedora 43 base
- TheRock ROCm nightly SDK for gfx1151
- PyTorch nightly with ROCm support
- vLLM patched for gfx1151
- flash-attention built with Triton backend

## Option 1: Fedora Toolbox (Recommended)

### Create the toolbox

```bash
toolbox create acestep-rocm \
  --image docker.io/kyuz0/vllm-therock-gfx1151:latest \
  -- --device /dev/dri --device /dev/kfd \
  --group-add video --group-add render --security-opt seccomp=unconfined
```

The `--device` flags pass through the AMD GPU device nodes:
- `/dev/dri` — DRI render nodes for GPU compute
- `/dev/kfd` — Kernel Fusion Driver for HSA/ROCm

The `--group-add video --group-add render` flags grant device access permissions.

### Enter the toolbox

```bash
toolbox enter acestep-rocm
```

### Install ACE-Step

```bash
git clone https://github.com/ace-step/ACE-Step-1.5.git
cd ACE-Step-1.5
pip install -e .
```

### Launch API server

```bash
python -m acestep.api_server
```

The server starts on port 8010 by default.

## Option 2: Distrobox (Ubuntu)

```bash
distrobox create -n acestep-rocm \
  --image docker.io/kyuz0/vllm-therock-gfx1151:latest \
  --additional-flags "--device /dev/kfd --device /dev/dri --group-add video --group-add render --security-opt seccomp=unconfined"

distrobox enter acestep-rocm
```

## Option 3: Podman/Docker (standalone)

```bash
podman run -it \
  --device /dev/dri --device /dev/kfd \
  --group-add video --group-add render \
  --security-opt seccomp=unconfined \
  -v ~/acestep-models:/root/.cache/huggingface \
  -p 8010:8010 \
  docker.io/kyuz0/vllm-therock-gfx1151:latest \
  bash
```

## Pre-configured Environment Variables

The container sets these via `/etc/profile.d/01-rocm-env-for-triton.sh`:

| Variable | Value | Purpose |
|----------|-------|---------|
| `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL` | `1` | Enable experimental aOTriton kernels |
| `FLASH_ATTENTION_TRITON_AMD_ENABLE` | `TRUE` | Use Triton backend for flash-attention |
| `VLLM_TARGET_DEVICE` | `rocm` | Tell vLLM to use ROCm backend |
| `VLLM_USE_MMAP` | `0` | Disable mmap (workaround for gfx1151) |
| `VLLM_ROCM_USE_AITER` | `0` | Disable aiter (unstable on gfx1151) |
| `VLLM_ROCM_USE_AITER_MOE` | `0` | Disable aiter MoE |
| `VLLM_USE_TRITON_AWQ` | `1` | Enable Triton AWQ quantization |

These are automatically sourced when entering the container.

## Model Storage

- **HuggingFace cache**: `~/.cache/huggingface/` (shared with host in toolbox mode)
- **ACE-Step checkpoints**: `~/.cache/acestep/`
- **vLLM compiled kernels**: `~/.cache/vllm/`

In toolbox mode, the host HOME directory is mounted, so models persist across container restarts.

## Cache Permission Issues

Containers may run with a different UID than the host. If you see `sqlite3.OperationalError: readonly database`:

```bash
rm -rf ~/.cache/acestep/local_redis/*.db*
rm -rf ~/.cache/miopen/*.db*
rm -rf ~/.cache/vllm/*.db*
```

## RDMA Clustering

For multi-node setups with InfiniBand/RoCE, see the [RDMA Cluster Setup Guide](https://github.com/kyuz0/amd-strix-halo-vllm-toolboxes/blob/master/rdma_cluster/setup_guide.md) in the toolbox repository.

## Troubleshooting

### rocm-smi not found

The container includes ROCm tools. If `rocm-smi` fails, verify the container was created with device pass-through:
```bash
ls -la /dev/dri /dev/kfd
```

### Out of VRAM

Check available memory:
```bash
python -c "import torch; props = torch.cuda.get_device_properties(0); print(f'{props.total_memory / 1e9:.1f} GB')"
```

Enable CPU offload:
```bash
ACESTEP_OFFLOAD_TO_CPU=true python -m acestep.api_server
```

### flash_attn import error

The container builds flash-attention from ROCm's fork. If it fails:
```bash
pip uninstall flash-attn
cd /opt && git clone https://github.com/ROCm/flash-attention.git
cd flash-attention && git checkout main_perf && python setup.py install
```
