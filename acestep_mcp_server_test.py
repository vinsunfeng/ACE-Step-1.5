"""Tests for mcp/acestep_mcp_server.py.

Run from repo root:  python -m unittest acestep_mcp_server_test -v

The server is a script in mcp/ (shadowed by the PyPI mcp SDK), so we load it
by file path under a distinct module name; the SDK stays importable.
"""
import base64
import importlib.util
import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "acestep_mcp_srv", os.path.join(_HERE, "mcp", "acestep_mcp_server.py")
)
server = importlib.util.module_from_spec(_spec)
sys.modules["acestep_mcp_srv"] = server
_spec.loader.exec_module(server)


class TestImport(unittest.TestCase):
    def test_module_loaded_with_all_tools(self):
        self.assertTrue(hasattr(server, "generate_music"))
        for name in ("generate_music", "list_models", "enhance_prompt", "check_health", "get_examples"):
            self.assertTrue(hasattr(server, name), f"missing tool {name}")


class TestResolveModel(unittest.TestCase):
    def setUp(self):
        server._clear_model_cache()

    def test_resolves_first_model_id(self):
        fake = {"object": "list", "data": [{"id": "acestep/acestep-v15-turbo"}]}
        with patch.object(server, "_request", return_value=fake):
            self.assertEqual(server._resolve_model(), "acestep/acestep-v15-turbo")

    def test_caches_across_calls(self):
        fake = {"object": "list", "data": [{"id": "acestep/acestep-v15-turbo"}]}
        with patch.object(server, "_request", return_value=fake) as mock_req:
            server._resolve_model()
            server._resolve_model()
            self.assertEqual(mock_req.call_count, 1)

    def test_raises_when_no_models(self):
        with patch.object(server, "_request", return_value={"object": "list", "data": []}):
            with self.assertRaises(RuntimeError):
                server._resolve_model()

    def test_raises_on_request_error(self):
        with patch.object(server, "_request", return_value={"error": "boom"}):
            with self.assertRaises(RuntimeError):
                server._resolve_model()

    def test_force_clears_cache(self):
        fake = {"object": "list", "data": [{"id": "acestep/acestep-v15-turbo"}]}
        with patch.object(server, "_request", return_value=fake) as mock_req:
            server._resolve_model()
            server._resolve_model(force=True)
            self.assertEqual(mock_req.call_count, 2)


if __name__ == "__main__":
    unittest.main()
