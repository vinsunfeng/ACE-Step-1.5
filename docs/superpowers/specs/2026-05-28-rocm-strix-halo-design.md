# AMD Strix Halo (ROCm) Support — Design Spec

**Date:** 2026-05-28
**Status:** Draft
**Approach:** Platform Parity with XPU (B v2, revised after subagent debate)

## Goal

Make ACE-Step 1.5 work correctly out-of-the-box on AMD ROCm GPUs, especially Strix Halo (gfx1151), by adding runtime platform overrides, fixing dtype inconsistencies, and providing setup documentation.

## Scope

- Runtime code fixes in `gpu_config.py` and `llm_inference.py`
- Platform-specific documentation (`README-ROCm.md`)
- Container setup guide referencing kyuz0/vllm-therock-gfx1151
- Unit tests for ROCm detection and overrides

Out of scope: refactoring existing MPS overrides, creating a new Dockerfile, adding gfx1151 support to vLLM upstream.

## Debate Outcomes

Three subagents (Challenger, Architect, ROCm Expert) debated the initial design. Key revisions:

1. **Cannot blanket-force float32 on all ROCm** — MI250/MI300 have native bf16. Must distinguish consumer RDNA vs data center CDNA.
2. **`torch.compile` is not universally broken** — works on CDNA, may fail on consumer RDNA. Soften to arch-aware default.
3. **Consolidate env vars** — single `ACESTEP_ROCM_DTYPE` for both DiT and LM, not two separate vars.
4. **Use post-construction override pattern** — `_apply_rocm_overrides(config)` instead of inline ternaries in GPUConfig constructor.
5. **`get_gpu_config_for_tier()` also needs overrides** — manual tier selection currently loses platform overrides.
6. **Replace inline ROCm checks** — use `gpu_config.is_rocm_available()` everywhere.

---

## 1. gpu_config.py Changes

### 1.1 New helper functions

```python
def get_rocm_gfx_version() -> Optional[str]:
    """Return ROCm gfx architecture string (e.g. 'gfx1151') or None."""

def is_rocm_consumer_gpu() -> bool:
    """True for RDNA consumer GPUs (gfx11xx) where bf16/compile may be unreliable."""
```

`get_rocm_gfx_version()` tries three sources in order:
1. `torch.cuda.get_device_properties(0).gcnArchName` — returns the gfx string directly (e.g. `"gfx1151"`) when available
2. Parse `HSA_OVERRIDE_GFX_VERSION` env var (format: `"major.minor.patch"`) and convert to gfx notation
3. Read `/sys/class/kfd/kfd/topology/nodes/0/properties` and extract the `gfx_version` line

`is_rocm_consumer_gpu()` returns `True` when the gfx version matches the RDNA consumer pattern: `gfx11*` (gfx1100, gfx1101, gfx1102, gfx1151).

### 1.2 Post-construction override function

```python
def _apply_rocm_overrides(config: GPUConfig) -> GPUConfig:
    """Apply ROCm-specific post-construction overrides."""
    if not is_rocm_available():
        return config

    gfx = get_rocm_gfx_version()
    is_consumer = gfx is not None and gfx.startswith("gfx11")

    if is_consumer:
        config.compile_model_default = False

    config.recommended_backend = "pt"

    return config
```

### 1.3 Integration points

Call `_apply_rocm_overrides()` in two places:

- `get_gpu_config()` (line ~891): after MPS overrides, before `_apply_lm_backend_compatibility_overrides()`
- `get_gpu_config_for_tier()` (line ~1555): same position

### 1.4 Diagnostic info enhancement

Add gfx1151 entry to `_log_gpu_diagnostic_info()` HSA_OVERRIDE_GFX_VERSION recommendations:

```python
logger.warning("       - Strix Halo (Ryzen AI MAX+ 395): set HSA_OVERRIDE_GFX_VERSION=11.5.1")
```

### 1.5 URL update

Change `PYTORCH_ROCM_INSTALL_URL` from `rocm6.0` to `rocm6.3`.

---

## 2. llm_inference.py LM dtype fix

### 2.1 Problem

Line 572: `if device in ["cuda", "xpu"]: self.dtype = torch.bfloat16`

On ROCm, this sets bfloat16 for ALL AMD GPUs. But consumer RDNA GPUs (gfx11xx) have unreliable bf16 support (ROCm issue #6034 documents 5 critical bf16 bugs on gfx1151). The DiT already forces float32 for ROCm via `_resolve_rocm_dtype()`, creating an inconsistency.

### 2.2 Solution

```python
if dtype is None:
    if device in ["cuda", "xpu"]:
        _is_rocm = is_rocm_available()
        if _is_rocm and not cuda_supports_bfloat16():
            self.dtype = torch.float32
        elif _is_rocm:
            rocm_dtype = os.getenv("ACESTEP_ROCM_DTYPE", "").lower()
            if rocm_dtype in ("float32", "fp32"):
                self.dtype = torch.float32
            else:
                self.dtype = torch.bfloat16
        else:
            self.dtype = torch.bfloat16
    else:
        self.dtype = torch.float32
```

Logic:
- Consumer RDNA (no native bf16): force float32
- Data center CDNA (native bf16): use bfloat16, but respect `ACESTEP_ROCM_DTYPE` env var
- Non-ROCm: existing bfloat16 behavior

### 2.3 Import

Add `from acestep.gpu_config import is_rocm_available, cuda_supports_bfloat16` at the top of `llm_inference.py`.

Replace existing inline `hasattr(torch.version, "hip") and torch.version.hip is not None` checks with `is_rocm_available()`.

---

## 3. README-ROCm.md

Following the `README-XPU.md` pattern.

### 3.1 Hardware support table

| GPU | gfx | bf16 | HSA_OVERRIDE_GFX_VERSION | Notes |
|-----|-----|------|--------------------------|-------|
| Ryzen AI MAX+ 395 (Strix Halo) | gfx1151 | Unreliable | `11.5.1` | Use toolbox container |
| RX 7900 XTX | gfx1100 | Partial | `11.0.0` | ROCm 6.3+ |
| RX 7800 XT | gfx1101 | Partial | `11.0.1` | ROCm 6.3+ |
| RX 7600 | gfx1102 | Partial | `11.0.2` | ROCm 6.3+ |
| MI250X | gfx90a | Native | Not needed | Data center |
| MI300X | gfx942 | Native | Not needed | Data center |

### 3.2 Setup options

**Option 1: Container (recommended for Strix Halo)**

Using kyuz0/vllm-therock-gfx1151 toolbox:
```bash
toolbox create acestep-rocm \
  --image docker.io/kyuz0/vllm-therock-gfx1151:latest \
  -- --device /dev/dri --device /dev/kfd \
  --group-add video --group-add render --security-opt seccomp=unconfined

toolbox enter acestep-rocm
pip install -e .
python -m acestep.api_server
```

Key container environment variables (pre-set by toolbox):
- `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`
- `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`
- `VLLM_ROCM_USE_AITER=0`

**Option 2: Bare metal (data center GPUs)**

```bash
pip install torch --index-url https://download.pytorch.org/whl/rocm6.3
pip install -e .
ACESTEP_LM_BACKEND=pt python -m acestep.api_server
```

### 3.3 Environment variables

| Variable | Purpose | Default | Required |
|----------|---------|---------|----------|
| `HSA_OVERRIDE_GFX_VERSION` | GPU architecture override | Auto | Only for consumer RDNA |
| `ACESTEP_ROCM_DTYPE` | Override dtype for DiT and LM | float32 (RDNA), bfloat16 (CDNA) | No |
| `ACESTEP_LM_BACKEND` | Force LM backend | pt | Recommended for bare metal |
| `MIOPEN_USER_DB_PATH` | MIOpen cache location | Auto | Only if cache errors occur |

### 3.4 Host kernel parameters (Strix Halo)

```
iommu=pt amdgpu.gttsize=126976 ttm.pages_limit=32505856
```

Source: kyuz0/amd-strix-halo-vllm-toolboxes README, verified on Framework Desktop.

### 3.5 Known limitations

- flash_attn uses Triton backend on ROCm (not CUDA kernel)
- aiter is disabled (`VLLM_ROCM_USE_AITER=0`)
- bfloat16 may produce NaN/inf on consumer RDNA GPUs
- vLLM requires patched build for gfx1151 (not yet upstream)

---

## 4. tests/test_rocm_compat.py

10 test cases:

1. `test_is_rocm_available_true` — mock torch.version.hip, verify True
2. `test_is_rocm_available_false` — mock no hip, verify False
3. `test_get_rocm_gfx_version_gfx1151` — mock device name, verify "gfx1151"
4. `test_is_rocm_consumer_gpu_gfx1151` — verify True for gfx1151
5. `test_is_rocm_consumer_gpu_gfx90a` — verify False for gfx90a (data center)
6. `test_get_gpu_config_rocm_consumer_overrides` — verify compile=False on gfx1151
7. `test_get_gpu_config_rocm_datacenter_keeps_compile` — verify compile=True on gfx90a
8. `test_get_gpu_config_rocm_recommended_backend_pt` — verify pt backend
9. `test_get_gpu_config_for_tier_rocm_overrides` — verify overrides survive manual tier selection
10. `test_lm_dtype_rocm_consumer_float32` — mock ROCm consumer GPU, verify float32 default

All tests mock `torch.cuda` and `torch.version.hip` — no AMD hardware required.

---

## 5. docs/rocm-container-setup.md

Detailed guide for the kyuz0/vllm-therock-gfx1151 container:

- Container creation (toolbox and distrobox)
- Device pass-through requirements (`/dev/dri`, `/dev/kfd`)
- Environment variables (pre-set by container, documented for reference)
- Model storage (shared with host via HOME mount)
- Cache permission handling (UID mismatch in containers)
- RDMA clustering (pointer to kyuz0 repo)
- Troubleshooting common issues

---

## Files Changed

| File | Action | Description |
|------|--------|-------------|
| `acestep/gpu_config.py` | Modify | Add `get_rocm_gfx_version()`, `is_rocm_consumer_gpu()`, `_apply_rocm_overrides()`, update URLs, enhance diagnostics |
| `acestep/llm_inference.py` | Modify | Fix LM dtype for ROCm, replace inline checks with `is_rocm_available()` |
| `README-ROCm.md` | Create | Platform setup guide following XPU pattern |
| `tests/test_rocm_compat.py` | Create | 10 unit tests for ROCm detection and overrides |
| `docs/rocm-container-setup.md` | Create | Detailed container setup guide |

Total: 2 modified files, 3 new files.

## References

- kyuz0/amd-strix-halo-vllm-toolboxes: https://github.com/kyuz0/amd-strix-halo-vllm-toolboxes
- ROCm bf16 bugs on gfx1151: https://github.com/ROCm/ROCm/issues/6034
- vLLM gfx1151 instability: https://github.com/vllm-project/vllm/issues/32180
- Framework vLLM on Strix Halo HOW-TO: https://community.frame.work/t/how-to-compiling-vllm-from-source-on-strix-halo/77241
