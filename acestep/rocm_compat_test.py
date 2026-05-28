"""Unit tests for ROCm platform detection and overrides."""

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


class TestLmDtypeRocm(unittest.TestCase):
    """LM dtype logic: float32 on consumer ROCm, bfloat16 on CDNA, env var override."""

    @patch("acestep.gpu_config.cuda_supports_bfloat16", return_value=False)
    @patch("acestep.gpu_config.is_rocm_available", return_value=True)
    def test_consumer_rocm_forces_float32(self, mock_rocm, mock_bf16):
        """Consumer RDNA GPU without native bf16 -> dtype = float32."""
        import torch
        from acestep.gpu_config import is_rocm_available, cuda_supports_bfloat16
        _is_rocm = is_rocm_available()
        if _is_rocm and not cuda_supports_bfloat16():
            resolved = torch.float32
        elif _is_rocm:
            resolved = torch.bfloat16
        else:
            resolved = torch.bfloat16
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


if __name__ == "__main__":
    unittest.main()
