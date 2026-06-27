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


if __name__ == "__main__":
    unittest.main()
