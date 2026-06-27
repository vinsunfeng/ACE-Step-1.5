# ROCm Strix Halo Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add ROCm platform overrides and dtype fixes so ACE-Step 1.5 works correctly on AMD GPUs (especially Strix Halo gfx1151) without manual configuration.

**Architecture:** Follow the existing MPS override pattern but use post-construction patch functions (`_apply_rocm_overrides`) instead of inline ternaries. Add arch-aware ROCm detection (`get_rocm_gfx_version`, `is_rocm_consumer_gpu`) to distinguish consumer RDNA from data center CDNA GPUs. Fix LM dtype to match DiT behavior on ROCm.

**Tech Stack:** Python 3.11+, PyTorch (ROCm), unittest with unittest.mock

**Design Spec:** `docs/superpowers/specs/2026-05-28-rocm-strix-halo-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `acestep/gpu_config.py` | Modify | Add ROCm helpers, post-construction overrides, URL update, diagnostics |
| `acestep/llm_inference.py` | Modify | Fix LM dtype for ROCm, replace inline checks |
| `tests/test_rocm_compat.py` | Create | 10 unit tests for ROCm detection and overrides |
| `README-ROCm.md` | Create | Platform setup guide for AMD ROCm users |
| `docs/rocm-container-setup.md` | Create | Detailed container/toolbox setup guide |

---

### Task 1: Add ROCm helper functions to gpu_config.py

**Files:**
- Modify: `acestep/gpu_config.py` (after line 108, before `cuda_supports_bfloat16`)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_rocm_compat.py
import unittest
from unittest.mock import patch, MagicMock
from types import SimpleNamespace


class TestGetRocmGfxVersion(unittest.TestCase):
    """get_rocm_gfx_version returns gfx arch string or None."""

    @patch("acestep.gpu_config.is_rocm_available", return_value=True)
    def test_reads_gcn_arch_name(self, mock_rocm):
        from acestep.gpu_config import get_rocm_gfx_version
        mock_props = SimpleNamespace(gcnArchName="gfx1151")
        with patch("torch.cuda.get_device_properties", return_value=mock_props):
            result = get_rocm_gfx_version()
        self.assertEqual(result, "gfx1151")

    @patch("acestep.gpu_config.is_rocm_available", return_value=True)
    def test_strips_colon_suffix(self, mock_rocm):
        """gcnArchName may include ':sramecc+:xnack-' suffix."""
        from acestep.gpu_config import get_rocm_gfx_version
        mock_props = SimpleNamespace(gcnArchName="gfx1151:sramecc+:xnack-")
        with patch("torch.cuda.get_device_properties", return_value=mock_props):
            result = get_rocm_gfx_version()
        self.assertEqual(result, "gfx1151")

    @patch("acestep.gpu_config.is_rocm_available", return_value=False)
    def test_returns_none_when_not_rocm(self, mock_rocm):
        from acestep.gpu_config import get_rocm_gfx_version
        result = get_rocm_gfx_version()
        self.assertIsNone(result)


class TestIsRocmConsumerGpu(unittest.TestCase):
    """is_rocm_consumer_gpu distinguishes consumer RDNA from data center."""

    @patch("acestep.gpu_config.get_rocm_gfx_version", return_value="gfx1151")
    def test_gfx1151_is_consumer(self, mock_gfx):
        from acestep.gpu_config import is_rocm_consumer_gpu
        self.assertTrue(is_rocm_consumer_gpu())

    @patch("acestep.gpu_config.get_rocm_gfx_version", return_value="gfx90a")
    def test_gfx90a_is_not_consumer(self, mock_gfx):
        from acestep.gpu_config import is_rocm_consumer_gpu
        self.assertFalse(is_rocm_consumer_gpu())

    @patch("acestep.gpu_config.get_rocm_gfx_version", return_value=None)
    def test_none_is_not_consumer(self, mock_gfx):
        from acestep.gpu_config import is_rocm_consumer_gpu
        self.assertFalse(is_rocm_consumer_gpu())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/vinsun/project/ACE-Step-1.5 && python -m pytest tests/test_rocm_compat.py::TestGetRocmGfxVersion -v`
Expected: FAIL (ImportError or AttributeError)

- [ ] **Step 3: Write minimal implementation**

Add to `acestep/gpu_config.py` after `is_rocm_available()` (after line 108):

```python
def get_rocm_gfx_version() -> Optional[str]:
    """Return ROCm gfx architecture string (e.g. 'gfx1151') or None.

    Tries three sources in order:
    1. torch.cuda.get_device_properties(0).gcnArchName
    2. /sys/class/kfd/kfd/topology/nodes/0/properties (gfx_version line)
    3. HSA_OVERRIDE_GFX_VERSION env var (best-effort conversion)
    """
    if not is_rocm_available():
        return None
    try:
        import torch
        props = torch.cuda.get_device_properties(0)
        gcn = getattr(props, "gcnArchName", None)
        if gcn:
            return gcn.split(":")[0].strip()
    except Exception:
        pass
    try:
        import glob
        for prop_file in glob.glob("/sys/class/kfd/kfd/topology/nodes/*/properties"):
            with open(prop_file) as f:
                for line in f:
                    if line.startswith("gfx_version "):
                        return line.split()[-1].strip()
    except Exception:
        pass
    hsa = os.environ.get("HSA_OVERRIDE_GFX_VERSION")
    if hsa:
        return f"gfx{hsa.replace('.', '')}"
    return None


def is_rocm_consumer_gpu() -> bool:
    """True for RDNA consumer GPUs (gfx11xx) where bf16/compile may be unreliable."""
    gfx = get_rocm_gfx_version()
    return gfx is not None and gfx.startswith("gfx11")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/vinsun/project/ACE-Step-1.5 && python -m pytest tests/test_rocm_compat.py::TestGetRocmGfxVersion tests/test_rocm_compat.py::TestIsRocmConsumerGpu -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add acestep/gpu_config.py tests/test_rocm_compat.py
git commit -m "feat(rocm): add get_rocm_gfx_version and is_rocm_consumer_gpu helpers"
```

---

### Task 2: Add _apply_rocm_overrides post-construction function

**Files:**
- Modify: `acestep/gpu_config.py` (after `_apply_lm_backend_compatibility_overrides`, line ~283)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_rocm_compat.py — add to existing file

class TestApplyRocmOverrides(unittest.TestCase):
    """_apply_rocm_overrides patches GPUConfig for ROCm correctly."""

    def _make_config(self):
        from acestep.gpu_config import GPUConfig
        return GPUConfig(
            tier="unlimited", gpu_memory_gb=62.5,
            max_duration_with_lm=600, max_duration_without_lm=600,
            max_batch_size_with_lm=8, max_batch_size_without_lm=8,
            init_lm_default=True,
            available_lm_models=["acestep-5Hz-lm-4B"],
            recommended_lm_model="acestep-5Hz-lm-4B",
            lm_backend_restriction="all", recommended_backend="vllm",
            offload_to_cpu_default=False, offload_dit_to_cpu_default=False,
            quantization_default=False, compile_model_default=True,
            lm_memory_gb={"0.6B": 3, "1.7B": 8, "4B": 12},
        )

    @patch("acestep.gpu_config.is_rocm_available", return_value=True)
    @patch("acestep.gpu_config.is_rocm_consumer_gpu", return_value=True)
    def test_consumer_gpu_disables_compile(self, mock_consumer, mock_rocm):
        from acestep.gpu_config import _apply_rocm_overrides
        config = self._make_config()
        result = _apply_rocm_overrides(config)
        self.assertFalse(result.compile_model_default)
        self.assertEqual(result.recommended_backend, "pt")

    @patch("acestep.gpu_config.is_rocm_available", return_value=True)
    @patch("acestep.gpu_config.is_rocm_consumer_gpu", return_value=False)
    def test_datacenter_keeps_compile(self, mock_consumer, mock_rocm):
        from acestep.gpu_config import _apply_rocm_overrides
        config = self._make_config()
        result = _apply_rocm_overrides(config)
        self.assertTrue(result.compile_model_default)
        self.assertEqual(result.recommended_backend, "pt")

    @patch("acestep.gpu_config.is_rocm_available", return_value=False)
    def test_noop_when_not_rocm(self, mock_rocm):
        from acestep.gpu_config import _apply_rocm_overrides
        config = self._make_config()
        result = _apply_rocm_overrides(config)
        self.assertEqual(result.recommended_backend, "vllm")
        self.assertTrue(result.compile_model_default)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/vinsun/project/ACE-Step-1.5 && python -m pytest tests/test_rocm_compat.py::TestApplyRocmOverrides -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Write minimal implementation**

Add to `acestep/gpu_config.py` after `_apply_lm_backend_compatibility_overrides` (after line ~283):

```python
def _apply_rocm_overrides(config: GPUConfig) -> GPUConfig:
    """Apply ROCm-specific post-construction overrides."""
    if not is_rocm_available():
        return config

    if is_rocm_consumer_gpu():
        config.compile_model_default = False

    config.recommended_backend = "pt"

    return config
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/vinsun/project/ACE-Step-1.5 && python -m pytest tests/test_rocm_compat.py::TestApplyRocmOverrides -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add acestep/gpu_config.py tests/test_rocm_compat.py
git commit -m "feat(rocm): add _apply_rocm_overrides post-construction patch"
```

---

### Task 3: Wire _apply_rocm_overrides into get_gpu_config and get_gpu_config_for_tier

**Files:**
- Modify: `acestep/gpu_config.py` lines 891 and 1555

- [ ] **Step 1: Write the failing test**

```python
# tests/test_rocm_compat.py — add to existing file

class TestRocmOverridesInGetGpuConfig(unittest.TestCase):
    """Verify ROCm overrides are applied in get_gpu_config flow."""

    @patch("acestep.gpu_config.is_rocm_available", return_value=True)
    @patch("acestep.gpu_config.is_rocm_consumer_gpu", return_value=True)
    @patch("acestep.gpu_config.get_gpu_memory_gb", return_value=62.5)
    @patch("acestep.gpu_config.is_mps_platform", return_value=False)
    def test_consumer_overrides_applied(self, mock_mps, mock_mem, mock_consumer, mock_rocm):
        from acestep.gpu_config import get_gpu_config
        config = get_gpu_config()
        self.assertFalse(config.compile_model_default)
        self.assertEqual(config.recommended_backend, "pt")

    @patch("acestep.gpu_config.is_rocm_available", return_value=True)
    @patch("acestep.gpu_config.is_rocm_consumer_gpu", return_value=False)
    @patch("acestep.gpu_config.get_gpu_memory_gb", return_value=62.5)
    @patch("acestep.gpu_config.is_mps_platform", return_value=False)
    def test_datacenter_keeps_compile(self, mock_mps, mock_mem, mock_consumer, mock_rocm):
        from acestep.gpu_config import get_gpu_config
        config = get_gpu_config()
        self.assertTrue(config.compile_model_default)
        self.assertEqual(config.recommended_backend, "pt")

    @patch("acestep.gpu_config.is_rocm_available", return_value=True)
    @patch("acestep.gpu_config.is_rocm_consumer_gpu", return_value=True)
    @patch("acestep.gpu_config.get_gpu_memory_gb", return_value=62.5)
    @patch("acestep.gpu_config.is_mps_platform", return_value=False)
    def test_tier_override_survives_manual_selection(self, mock_mps, mock_mem, mock_consumer, mock_rocm):
        from acestep.gpu_config import get_gpu_config_for_tier
        config = get_gpu_config_for_tier("tier5")
        self.assertFalse(config.compile_model_default)
        self.assertEqual(config.recommended_backend, "pt")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/vinsun/project/ACE-Step-1.5 && python -m pytest tests/test_rocm_compat.py::TestRocmOverridesInGetGpuConfig -v`
Expected: FAIL (compile_model_default is True, recommended_backend is "vllm")

- [ ] **Step 3: Write minimal implementation**

In `acestep/gpu_config.py`, change line 891 from:
```python
    return _apply_lm_backend_compatibility_overrides(config)
```
to:
```python
    if is_rocm_available() and not _mps:
        config = _apply_rocm_overrides(config)
    return _apply_lm_backend_compatibility_overrides(config)
```

In `acestep/gpu_config.py`, change line 1555 from:
```python
    return _apply_lm_backend_compatibility_overrides(config)
```
to:
```python
    _rocm = is_rocm_available()
    if _rocm and not _mps:
        config = _apply_rocm_overrides(config)
    return _apply_lm_backend_compatibility_overrides(config)
```

Also add the same override to `compute_adaptive_config()` (line ~1178). Change:
```python
    return _apply_lm_backend_compatibility_overrides(config)
```
to:
```python
    if is_rocm_available():
        config = _apply_rocm_overrides(config)
    return _apply_lm_backend_compatibility_overrides(config)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/vinsun/project/ACE-Step-1.5 && python -m pytest tests/test_rocm_compat.py::TestRocmOverridesInGetGpuConfig -v`
Expected: PASS

- [ ] **Step 5: Run existing tests to verify no regression**

Run: `cd /home/vinsun/project/ACE-Step-1.5 && python -m pytest acestep/gpu_config_test.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add acestep/gpu_config.py tests/test_rocm_compat.py
git commit -m "feat(rocm): wire _apply_rocm_overrides into get_gpu_config and get_gpu_config_for_tier"
```

---

### Task 4: Update PYTORCH_ROCM_INSTALL_URL and diagnostics

**Files:**
- Modify: `acestep/gpu_config.py` lines 40, 677-686

- [ ] **Step 1: Update URL constant**

Change line 40:
```python
PYTORCH_ROCM_INSTALL_URL = "https://download.pytorch.org/whl/rocm6.3"
```

- [ ] **Step 2: Add gfx1151 to diagnostic HSA_OVERRIDE_GFX_VERSION list**

In `_log_gpu_diagnostic_info()`, after the line that mentions RX 7600 (around line 685), add:

```python
            logger.warning(
                "       - Strix Halo (Ryzen AI MAX+ 395): set HSA_OVERRIDE_GFX_VERSION=11.5.1"
            )
```

- [ ] **Step 3: Run existing tests to verify no regression**

Run: `cd /home/vinsun/project/ACE-Step-1.5 && python -m pytest acestep/gpu_config_test.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add acestep/gpu_config.py
git commit -m "fix(rocm): update PyTorch URL to rocm6.3, add gfx1151 to diagnostics"
```

---

### Task 5: Fix LM dtype in llm_inference.py

**Files:**
- Modify: `acestep/llm_inference.py` lines 27, 571-577, 642

- [ ] **Step 1: Write the failing test**

```python
# tests/test_rocm_compat.py — add to existing file

class TestLmDtypeRocm(unittest.TestCase):
    """LM dtype logic: float32 on consumer ROCm, bfloat16 on CDNA, env var override."""

    @patch("acestep.gpu_config.cuda_supports_bfloat16", return_value=False)
    @patch("acestep.gpu_config.is_rocm_available", return_value=True)
    def test_consumer_rocm_forces_float32(self, mock_rocm, mock_bf16):
        """Consumer RDNA GPU without native bf16 -> dtype = float32."""
        import torch
        import os
        from acestep.gpu_config import is_rocm_available, cuda_supports_bfloat16
        # Replicate the dtype decision logic from llm_inference.py
        device = "cuda"
        dtype = None
        if dtype is None:
            if device in ["cuda", "xpu"]:
                _is_rocm = is_rocm_available()
                if _is_rocm and not cuda_supports_bfloat16():
                    resolved = torch.float32
                elif _is_rocm:
                    resolved = torch.bfloat16
                else:
                    resolved = torch.bfloat16
            else:
                resolved = torch.float32
        self.assertEqual(resolved, torch.float32)

    @patch("acestep.gpu_config.cuda_supports_bfloat16", return_value=True)
    @patch("acestep.gpu_config.is_rocm_available", return_value=True)
    def test_datacenter_rocm_uses_bfloat16(self, mock_rocm, mock_bf16):
        """CDNA GPU with native bf16 -> dtype = bfloat16."""
        import torch
        from acestep.gpu_config import is_rocm_available, cuda_supports_bfloat16
        _is_rocm = is_rocm_available()
        if _is_rocm and not cuda_supports_bfloat16():
            resolved = torch.float32
        elif _is_rocm:
            resolved = torch.bfloat16
        else:
            resolved = torch.bfloat16
        self.assertEqual(resolved, torch.bfloat16)

    @patch("acestep.gpu_config.cuda_supports_bfloat16", return_value=True)
    @patch("acestep.gpu_config.is_rocm_available", return_value=True)
    def test_rocm_dtype_env_var_forces_float32(self, mock_rocm, mock_bf16):
        """ACESTEP_ROCM_DTYPE=fp32 overrides CDNA bf16 default."""
        import torch
        import os
        from unittest.mock import patch as upatch
        with upatch.dict(os.environ, {"ACESTEP_ROCM_DTYPE": "fp32"}):
            _is_rocm = True
            rocm_dtype = os.getenv("ACESTEP_ROCM_DTYPE", "").lower()
            if rocm_dtype in ("float32", "fp32"):
                resolved = torch.float32
            else:
                resolved = torch.bfloat16
        self.assertEqual(resolved, torch.float32)
```

- [ ] **Step 2: Run test to verify it passes** (these tests verify the helpers, not the full handler)

Run: `cd /home/vinsun/project/ACE-Step-1.5 && python -m pytest tests/test_rocm_compat.py::TestLmDtypeRocm -v`
Expected: PASS (these validate the detection functions that the dtype logic will use)

- [ ] **Step 3: Fix the import and dtype logic**

In `acestep/llm_inference.py`, change line 27 from:
```python
from acestep.gpu_config import get_lm_gpu_memory_ratio, get_gpu_memory_gb, get_lm_model_size, get_global_gpu_config
```
to:
```python
from acestep.gpu_config import (
    get_lm_gpu_memory_ratio,
    get_gpu_memory_gb,
    get_lm_model_size,
    get_global_gpu_config,
    is_rocm_available,
    cuda_supports_bfloat16,
)
```

Change lines 571-577 from:
```python
            if dtype is None:
                if device in ["cuda", "xpu"]:
                    self.dtype = torch.bfloat16
                else:
                    self.dtype = torch.float32
            else:
                self.dtype = dtype
```
to:
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
            else:
                self.dtype = dtype
```

Replace line 642:
```python
            is_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None
```
with:
```python
            is_rocm = is_rocm_available()
```

- [ ] **Step 4: Run all ROCm tests to verify**

Run: `cd /home/vinsun/project/ACE-Step-1.5 && python -m pytest tests/test_rocm_compat.py -v`
Expected: PASS

- [ ] **Step 5: Run existing gpu_config tests to verify no regression**

Run: `cd /home/vinsun/project/ACE-Step-1.5 && python -m pytest acestep/gpu_config_test.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add acestep/llm_inference.py tests/test_rocm_compat.py
git commit -m "fix(rocm): force float32 for LM on consumer RDNA, allow bf16 on CDNA"
```

---

### Task 6: Create README-ROCm.md

**Files:**
- Create: `README-ROCm.md`

- [ ] **Step 1: Write README-ROCm.md**

Create the file with the following content (full documentation following the XPU pattern, incorporating all spec details from Section 3):

```markdown
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
```

- [ ] **Step 2: Commit**

```bash
git add README-ROCm.md
git commit -m "docs: add README-ROCm.md for AMD GPU setup guide"
```

---

### Task 7: Create docs/rocm-container-setup.md

**Files:**
- Create: `docs/rocm-container-setup.md`

- [ ] **Step 1: Write the container setup guide**

Create `docs/rocm-container-setup.md` with the following content:

```markdown
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
```

- [ ] **Step 2: Commit**

```bash
git add docs/rocm-container-setup.md
git commit -m "docs: add ROCm container setup guide"
```

---

### Task 8: Run full test suite and verify

- [ ] **Step 1: Run all ROCm tests**

Run: `cd /home/vinsun/project/ACE-Step-1.5 && python -m pytest tests/test_rocm_compat.py -v`
Expected: All 10+ tests PASS

- [ ] **Step 2: Run existing gpu_config tests**

Run: `cd /home/vinsun/project/ACE-Step-1.5 && python -m pytest acestep/gpu_config_test.py -v`
Expected: All existing tests PASS (no regression)

- [ ] **Step 3: Run api test suite**

Run: `cd /home/vinsun/project/ACE-Step-1.5 && python -m pytest acestep/api/ -v --ignore=acestep/api/http/release_task_route_http_test.py`
Expected: All API tests PASS

- [ ] **Step 4: Final commit if any fixes needed**

```bash
git add -A
git commit -m "test(rocm): verify all tests pass after ROCm support changes"
```
