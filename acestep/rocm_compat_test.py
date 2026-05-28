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


if __name__ == "__main__":
    unittest.main()
