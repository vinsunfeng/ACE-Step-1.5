# ROCm Strix Halo Compatibility Verification Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Verify ACE-Step music generation runs on AMD Strix Halo (gfx1151) using a vllm toolbox container with ROCm 7.2 + TheRock PyTorch.

**Architecture:** Use Fedora toolbox (`kyuz0/vllm-therock-gfx1151:stable`) which provides Python 3.12 + TheRock PyTorch nightly + ROCm flash-attention. Install ACE-Step deps inside the toolbox, run smoke tests, then generate audio.

**Tech Stack:** Fedora toolbox, podman, TheRock ROCm SDK, PyTorch nightly (gfx1151), ACE-Step 1.5

**Spec:** `docs/superpowers/specs/2026-05-25-rocm-strix-halo-compatibility-design.md`

---

## Task 1: Create ACE-Step Toolbox Container

**Files:**
- Reference: `~/amd-strix-halo-vllm-toolboxes/README.md` (quickstart section)

- [ ] **Step 1: Pull the vllm toolbox image**

```bash
podman pull docker.io/kyuz0/vllm-therock-gfx1151:stable
```

Expected: Image pulled successfully. This may take several minutes depending on network speed.

- [ ] **Step 2: Create the toolbox container**

```bash
toolbox create acestep-rocm \
  --image docker.io/kyuz0/vllm-therock-gfx1151:stable \
  -- --device /dev/dri --device /dev/kfd \
  --group-add video --group-add render \
  --security-opt seccomp=unconfined
```

Expected: `Image docker.io/kyuz0/vllm-therock-gfx1151:stable resolved to ... Created container: acestep-rocm`

- [ ] **Step 3: Enter toolbox and verify basic shell**

```bash
toolbox enter acestep-rocm
```

Expected: Shell prompt changes to indicate toolbox environment. You should see a banner from `/etc/profile.d/99-toolbox-banner.sh`.

- [ ] **Step 4: Verify Python and venv activation**

Inside toolbox:
```bash
python --version
which python
echo $VIRTUAL_ENV
```

Expected: `Python 3.12.x`, `/opt/venv/bin/python`, `/opt/venv`

- [ ] **Step 5: Commit (leave toolbox first)**

```bash
exit
```

---

## Task 2: GPU and PyTorch Smoke Tests (Phase 1)

**Files:**
- Reference: `~/project/ACE-Step-1.5/acestep/gpu_config.py` (for understanding VRAM detection)

- [ ] **Step 1: Enter toolbox**

```bash
toolbox enter acestep-rocm
```

- [ ] **Step 2: Verify GPU visibility (T1.1)**

Inside toolbox:
```bash
rocminfo | grep -A2 gfx1151
```

Expected: Output showing `Name: gfx1151` and `Marketing Name: AMD Radeon 8060S Graphics`

- [ ] **Step 3: Verify PyTorch CUDA/HIP availability (T1.2 + T1.3)**

Inside toolbox:
```bash
python -c "
import torch
print(f'torch={torch.__version__}')
print(f'HIP={torch.version.hip}')
print(f'CUDA available={torch.cuda.is_available()}')
print(f'Arch list={torch.cuda.get_arch_list()}')
print(f'GPU={torch.cuda.get_device_name(0)}')
"
```

Expected: `CUDA available=True`, Arch list contains `gfx1151`, GPU name shows `AMD Radeon 8060S Graphics` or similar.

**If this fails:** The TheRock PyTorch build doesn't support gfx1151. Record the error. Stop.

- [ ] **Step 4: Test bfloat16 compute (T1.4)**

Inside toolbox:
```bash
python -c "
import torch
x = torch.randn(100, 100, device='cuda', dtype=torch.bfloat16)
y = x @ x
print(f'bfloat16 matmul: OK, shape={y.shape}')
"
```

Expected: `bfloat16 matmul: OK, shape=torch.Size([100, 100])`

**If this segfaults:** bfloat16 kernels incomplete for gfx1151. ACE-Step will use float32 (already the default). Note this and continue.

- [ ] **Step 5: Test SDPA attention (T1.5)**

Inside toolbox:
```bash
python -c "
import torch
import torch.nn.functional as F
q = torch.randn(1, 8, 10, 64, device='cuda', dtype=torch.bfloat16)
k = torch.randn(1, 8, 10, 64, device='cuda', dtype=torch.bfloat16)
v = torch.randn(1, 8, 10, 64, device='cuda', dtype=torch.bfloat16)
out = F.scaled_dot_product_attention(q, k, v)
print(f'SDPA: OK, shape={out.shape}')
"
```

Expected: `SDPA: OK, shape=torch.Size([1, 8, 10, 64])`

- [ ] **Step 6: Test VRAM detection (T1.6) — CRITICAL for APU**

Inside toolbox:
```bash
python -c "
import torch
props = torch.cuda.get_device_properties(0)
print(f'Total VRAM: {props.total_memory / 1024**3:.1f} GB')
print(f'GPU name: {props.name}')
"
```

Expected: Reports >4GB. If it reports only 512MB (BIOS buffer), ACE-Step will be stuck in Tier 1 (worst). We'll need to investigate `MAX_CUDA_VRAM` override.

**If <4GB:** Record exact value. Try: `MAX_CUDA_VRAM=126 python -c "..."` to see if override works.

- [ ] **Step 7: Test Conv1d / MIOpen path (T1.7)**

Inside toolbox:
```bash
MIOPEN_DEBUG_CONV_DIRECT=0 python -c "
import torch
conv = torch.nn.Conv1d(64, 128, kernel_size=3, padding=1).cuda()
x = torch.randn(1, 64, 100, device='cuda')
y = conv(x)
print(f'Conv1d: OK, shape={y.shape}')
"
```

Expected: `Conv1d: OK, shape=torch.Size([1, 128, 100])`

**If MIOpen error:** Record the exact error. This indicates VAE decode will fail. Try with `MIOPEN_DEBUG_CONV_DIRECT_NAIVE_CONV_FWD=0` instead.

- [ ] **Step 8: Test flash_attn import (T1.8)**

Inside toolbox:
```bash
python -c "
import flash_attn
print(f'flash_attn version: {flash_attn.__version__}')
"
```

Expected: Prints version number without ImportError.

**If ImportError:** flash_attn not compatible with this Python/nano-vllm. Use `ACESTEP_LM_BACKEND=pt` (SDPA fallback). This is expected and acceptable.

**If import succeeds:** Note this — we can try `ACESTEP_LM_BACKEND=vllm` later to enable LM CoT features.

- [ ] **Step 9: Exit toolbox and record results**

```bash
exit
```

Record pass/fail for T1.1-T1.8. **All tests must pass before proceeding to Task 3.** If any fails, document the error and stop.

---

## Task 3: Install ACE-Step Dependencies (Phase 2 setup)

**Files:**
- Read: `~/project/ACE-Step-1.5/requirements-rocm-linux.txt`
- Read: `~/project/ACE-Step-1.5/pyproject.toml`
- Read: `~/project/ACE-Step-1.5/acestep/third_parts/nano-vllm/pyproject.toml`

- [ ] **Step 1: Enter toolbox**

```bash
toolbox enter acestep-rocm
```

- [ ] **Step 2: Navigate to project and prepare requirements**

Inside toolbox:
```bash
cd ~/project/ACE-Step-1.5

# Remove torchao (Triton dep, fails on ROCm) from requirements
grep -v torchao requirements-rocm-linux.txt > /tmp/req-rocm.txt

# Verify the filtered file
cat /tmp/req-rocm.txt
```

Expected: File lists all deps except torchao. Lines like `transformers>=4.51.0,<4.58.0`, `diffusers`, `gradio==6.2.0`, etc.

- [ ] **Step 3: Install ROCm requirements**

Inside toolbox:
```bash
pip install -r /tmp/req-rocm.txt
```

Expected: All packages install successfully. This may take 5-10 minutes. `gradio==6.2.0` may pull many deps.

**If conflict with vLLM's existing deps:** pip may show warnings about version overrides. This is expected and usually harmless. If a hard error occurs, record the conflicting packages.

- [ ] **Step 4: Verify vLLM still importable (health check)**

Inside toolbox:
```bash
python -c "import vllm; print(f'vllm={vllm.__version__}')" || echo "WARNING: vLLM broken (not needed for ACE-Step, but noted)"
```

Expected: Either prints vllm version or shows the WARNING. Not a blocker since ACE-Step uses `pt` backend.

- [ ] **Step 5: Install nano-vllm (skip deps to avoid triton/flash-attn)**

Inside toolbox:
```bash
pip install --no-deps -e acestep/third_parts/nano-vllm
```

Expected: `Successfully installed nano-vllm-0.2.0`

- [ ] **Step 6: Install ACE-Step project (skip deps to avoid CUDA torch pin)**

Inside toolbox:
```bash
pip install --no-deps -e .
```

Expected: `Successfully installed ace-step-1.5.0`

- [ ] **Step 7: Install additional deps not in requirements-rocm-linux.txt**

Inside toolbox:
```bash
pip install typer-slim pytorch-wavelets pywavelets
```

Expected: All three install successfully.

- [ ] **Step 8: Verify ACE-Step CLI is accessible**

Inside toolbox:
```bash
python -m acestep.acestep_v15_pipeline --help | head -20
```

Expected: Prints argparse help showing `--server-name`, `--port`, `--backend`, `--init_service`, `--config_path`, `--init_llm`, etc.

- [ ] **Step 9: Exit toolbox**

```bash
exit
```

---

## Task 4: Download Models and Run Inference (Phase 2 tests)

**Files:**
- Write output to: `~/project/ACE-Step-1.5/checkpoints/`
- Write output to: `~/project/ACE-Step-1.5/gradio_outputs/`

- [ ] **Step 1: Enter toolbox**

```bash
toolbox enter acestep-rocm
```

- [ ] **Step 2: Set environment variables**

Inside toolbox:
```bash
cd ~/project/ACE-Step-1.5

export ACESTEP_LM_BACKEND=pt
export ACESTEP_DEVICE=auto
export ACESTEP_ROCM_DTYPE=float32
export TORCH_COMPILE_BACKEND=eager
export MIOPEN_DEBUG_CONV_DIRECT=0
export TOKENIZERS_PARALLELISM=false
```

- [ ] **Step 3: Download DiT model (T2.1)**

Inside toolbox:
```bash
python -m acestep.model_downloader
```

Expected: Downloads model files to `./checkpoints/`. May take several minutes depending on network. Look for files like `acestep-v15-turbo/` directory under `checkpoints/`.

- [ ] **Step 4: Verify checkpoint files exist**

Inside toolbox:
```bash
ls -la checkpoints/
```

Expected: Shows model directories/files.

- [ ] **Step 5: Test DiT model load to GPU (T2.2)**

Inside toolbox:
```bash
python -c "
import torch
from acestep.model_downloader import get_model_path
print(f'torch: {torch.cuda.is_available()}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB')
print('Model path check...')
import os
ckpt_dir = 'checkpoints'
if os.path.exists(ckpt_dir):
    print(f'Checkpoint contents: {os.listdir(ckpt_dir)}')
else:
    print('WARNING: checkpoints/ not found')
"
```

Expected: torch CUDA available, VRAM reported, checkpoint directory contents listed.

- [ ] **Step 6: Launch Gradio UI with init_service (T2.4 - full test)**

Inside toolbox:
```bash
python -m acestep.acestep_v15_pipeline \
  --server-name 127.0.0.1 \
  --port 7860 \
  --backend pt \
  --init_service true \
  --config_path acestep-v15-turbo \
  --init_llm false
```

Expected:
- Console shows model loading progress
- DiT model loads successfully to GPU
- Gradio UI starts on `http://127.0.0.1:7860`
- If VAE/MIOpen errors occur during init, they will appear here

**Watch for:**
- `No kernel image found for executing on device` → PyTorch doesn't support gfx1151
- MIOpen errors → Conv path broken
- OOM → VRAM detection problem
- If it starts successfully → proceed to Step 7

**If startup fails:** Record the full error traceback. Stop.

- [ ] **Step 7: Generate audio via Gradio UI (T2.4)**

Open browser to `http://127.0.0.1:7860`:
1. Enter a simple prompt like "A calm piano melody in C major"
2. Set duration to 15 seconds
3. Click Generate
4. Wait for generation to complete

Expected: Audio file generated and playable in the browser.

**If generation fails:** Record the error from the console output. Note whether it failed during:
- Text encoding (likely not GPU-related)
- DiT diffusion (GPU compute issue)
- VAE decode (MIOpen/conv issue — most likely failure point)

- [ ] **Step 8: Generate longer audio (T2.5)**

If Step 7 succeeded, try a 30+ second generation with more complex prompt.

Expected: Longer audio generated successfully.

- [ ] **Step 9: Stop the server and exit**

```bash
# Ctrl+C to stop the server
exit
```

---

## Task 5: Record Results and Next Steps

- [ ] **Step 1: Document test results**

Create a summary of all test results:

| Test | Result | Notes |
|------|--------|-------|
| T1.1 GPU visible | PASS/FAIL | |
| T1.2 torch.cuda | PASS/FAIL | |
| T1.3 arch list | PASS/FAIL | |
| T1.4 bfloat16 | PASS/FAIL | |
| T1.5 SDPA | PASS/FAIL | |
| T1.6 VRAM | PASS/FAIL | reported value: ___ GB |
| T1.7 Conv1d | PASS/FAIL | |
| T1.8 flash_attn | PASS/FAIL | |
| T2.1 Model download | PASS/FAIL | |
| T2.2 Model load | PASS/FAIL | |
| T2.4 15s generation | PASS/FAIL | |
| T2.5 30s+ generation | PASS/FAIL | |

- [ ] **Step 2: Based on results, decide next steps**

**If all PASS:**
- ACE-Step is verified on ROCm Strix Halo
- Consider Phase 3: set up as persistent service (see spec)
- If T1.8 flash_attn passed, try `ACESTEP_LM_BACKEND=vllm` to enable LM CoT features
- Consider `amd_iommu=off` kernel param for 5-12% performance gain

**If Conv1d (T1.7) or VAE fails:**
- Try different MIOpen env vars
- Report as ROCm gfx1151 issue

**If VRAM (T1.6) reports 512MB:**
- Try `MAX_CUDA_VRAM=126` env var
- May need patch in `acestep/gpu_config.py` for APU detection

**If bfloat16 (T1.4) fails:**
- ACE-Step already defaults to float32 on ROCm — should be fine
- 128GB GTT can absorb the doubled memory

- [ ] **Step 3: Commit any project changes**

If any fixes were needed (env vars, config tweaks):
```bash
cd ~/project/ACE-Step-1.5
git add -A
git status
# Review changes before committing
git commit -m "feat: add ROCm Strix Halo compatibility verification results"
```
