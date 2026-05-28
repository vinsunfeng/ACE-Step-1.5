# ACE-Step 1.5 - AMD ROCm Setup Guide

Quick start guide for running ACE-Step on AMD GPUs with ROCm.

## Supported Hardware

| GPU | gfx | bf16 | HSA_OVERRIDE_GFX_VERSION | Notes |
|-----|-----|------|--------------------------|-------|
| Ryzen AI MAX+ 395 (Strix Halo) | gfx1151 | Unreliable | `11.5.1` | Use toolbox container |
| RX 7900 XTX | gfx1100 | Partial | `11.0.0` | ROCm 6.3+ |
| RX 7800 XT | gfx1101 | Partial | `11.0.1` | ROCm 6.3+ |
| RX 7600 | gfx1102 | Partial | `11.0.2` | ROCm 6.3+ |
| MI250X | gfx90a | Native | Not needed | Data center |
| MI300X | gfx942 | Native | Not needed | Data center |

## Quick Start

### Option 1: Container (Recommended for Strix Halo)

Using the [kyuz0/vllm-therock-gfx1151](https://github.com/kyuz0/amd-strix-halo-vllm-toolboxes) toolbox image:

```bash
# Create the toolbox
toolbox create acestep-rocm \
  --image docker.io/kyuz0/vllm-therock-gfx1151:latest \
  -- --device /dev/dri --device /dev/kfd \
  --group-add video --group-add render --security-opt seccomp=unconfined

# Enter the toolbox
toolbox enter acestep-rocm

# Install ACE-Step
pip install -e .

# Launch API server
python -m acestep.api_server
```

The container pre-configures these environment variables:
- `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`
- `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`
- `VLLM_ROCM_USE_AITER=0`

See [docs/rocm-container-setup.md](docs/rocm-container-setup.md) for detailed container setup instructions.

### Option 2: Bare Metal (Data Center GPUs)

```bash
# Install ROCm PyTorch
pip install torch --index-url https://download.pytorch.org/whl/rocm6.3

# Install ACE-Step
pip install -e .

# Launch with PyTorch LM backend (vLLM requires special ROCm build)
HSA_OVERRIDE_GFX_VERSION=11.5.1 ACESTEP_LM_BACKEND=pt python -m acestep.api_server
```

## Host Kernel Parameters (Strix Halo)

Add to kernel boot parameters to enable full unified memory:

```
iommu=pt amdgpu.gttsize=126976 ttm.pages_limit=32505856
```

This allocates up to 124 GiB for the iGPU. Apply via GRUB:

```bash
# Edit /etc/default/grub, add to GRUB_CMDLINE_LINUX
sudo grub2-mkconfig -o /boot/grub2/grub.cfg
sudo reboot
```

## Environment Variables

| Variable | Purpose | Default | Required |
|----------|---------|---------|----------|
| `HSA_OVERRIDE_GFX_VERSION` | GPU architecture override | Auto | Consumer RDNA only |
| `ACESTEP_ROCM_DTYPE` | Override dtype (DiT + LM) | fp32 (RDNA), bf16 (CDNA) | No |
| `ACESTEP_LM_BACKEND` | Force LM backend | pt | Recommended on bare metal |
| `MIOPEN_USER_DB_PATH` | MIOpen cache location | Auto | Only if cache errors |

## Known Limitations

- **flash_attn**: Uses Triton backend on ROCm (not CUDA kernel). Set `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`.
- **aiter**: Disabled (`VLLM_ROCM_USE_AITER=0`) on gfx1151.
- **bfloat16**: May produce NaN/inf on consumer RDNA GPUs (gfx11xx). ACE-Step defaults to float32 on these GPUs.
- **vLLM**: Requires patched build for gfx1151 (not yet upstream). Use PyTorch backend on bare metal.
- **torch.compile**: Disabled on consumer RDNA by default due to Triton kernel compilation issues.

## Troubleshooting

### "GPU NOT DETECTED" on ROCm

1. Verify ROCm sees the GPU: `rocm-smi`
2. Set HSA_OVERRIDE_GFX_VERSION for your GPU (see table above)
3. Verify PyTorch: `python -c "import torch; print(torch.version.hip)"`

### "torch.cuda.is_available() returns False"

Reinstall PyTorch with ROCm:
```bash
pip install torch --index-url https://download.pytorch.org/whl/rocm6.3
```

### Out of memory on Strix Halo

1. Check kernel parameters are applied: `cat /proc/cmdline`
2. Verify VRAM: `python -c "import torch; print(torch.cuda.get_device_properties(0).total_memory / 1e9, 'GB')"`
3. Enable CPU offload: `ACESTEP_OFFLOAD_TO_CPU=true`

### Cache permission errors in container

```bash
rm -rf ~/.cache/acestep/local_redis/*.db*
rm -rf ~/.cache/miopen/*.db*
```

## References

- [Strix Halo vLLM Toolbox](https://github.com/kyuz0/amd-strix-halo-vllm-toolboxes)
- [ROCm bf16 bugs on gfx1151](https://github.com/ROCm/ROCm/issues/6034)
- [vLLM gfx1151 support](https://github.com/vllm-project/vllm/issues/32180)
- [Main README](README.md)
