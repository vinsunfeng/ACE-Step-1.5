"""Tests for agent_discovery_route — GET /.well-known/agent."""

import os
import re
import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from acestep.api.agent_discovery_route import register_agent_discovery_route


def _make_app() -> FastAPI:
    app = FastAPI()
    register_agent_discovery_route(app)
    return app


class TestAgentDiscoveryEndpoint(unittest.TestCase):
    """GET /.well-known/agent returns a machine-readable capability document."""

    def setUp(self):
        self.client = TestClient(_make_app())

    def test_returns_200(self):
        resp = self.client.get("/.well-known/agent")
        self.assertEqual(resp.status_code, 200)

    def test_returns_json(self):
        resp = self.client.get("/.well-known/agent")
        data = resp.json()
        self.assertIsInstance(data, dict)

    def test_has_required_fields(self):
        resp = self.client.get("/.well-known/agent")
        data = resp.json()
        for field in ("name", "version", "capabilities", "primary_endpoints"):
            self.assertIn(field, data, f"Missing field: {field}")

    def test_capabilities_is_list(self):
        resp = self.client.get("/.well-known/agent")
        data = resp.json()
        self.assertIsInstance(data["capabilities"], list)
        self.assertIn("text-to-music", data["capabilities"])

    def test_primary_endpoints_has_generate(self):
        resp = self.client.get("/.well-known/agent")
        data = resp.json()
        self.assertIn("generate", data["primary_endpoints"])
        self.assertEqual(data["primary_endpoints"]["generate"], "/v1/chat/completions")

    def test_has_schema_url(self):
        resp = self.client.get("/.well-known/agent")
        data = resp.json()
        self.assertEqual(data["schema_url"], "/openapi.json")

    def test_has_docs_urls(self):
        resp = self.client.get("/.well-known/agent")
        data = resp.json()
        self.assertIn("llms_txt_url", data)
        self.assertIn("llms_full_txt_url", data)

    def test_discovery_tools_match_mcp_file(self):
        """The discovery route's tools list must stay in sync with the
        @mcp.tool() functions defined in the MCP server file. Parsed
        generically, so it catches both renames and count drift."""
        project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        with open(os.path.join(project_root, "mcp", "acestep_mcp_server.py")) as f:
            srv_src = f.read()
        with open(
            os.path.join(project_root, "acestep", "api", "agent_discovery_route.py")
        ) as f:
            route_src = f.read()
        defined = set(re.findall(r"@mcp\.tool\(\)\s*\ndef\s+(\w+)", srv_src))
        m = re.search(r'"tools":\s*\[(.*?)\]', route_src, re.DOTALL)
        self.assertIsNotNone(m, "tools list not found in discovery route")
        listed = set(re.findall(r'"([^"]+)"', m.group(1)))
        self.assertEqual(
            defined,
            listed,
            f"discovery tools {listed} != MCP tools {defined}",
        )


if __name__ == "__main__":
    unittest.main()
