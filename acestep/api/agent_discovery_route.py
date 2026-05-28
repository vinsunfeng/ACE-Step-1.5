"""Agent discovery endpoints — /.well-known/agent, /llms.txt, /llms-full.txt.

Returns machine-readable documents so AI agents can understand what the
service offers and how to integrate with it.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def register_agent_discovery_route(app: FastAPI) -> None:
    """Register agent discovery and llms.txt routes on the FastAPI app."""

    @app.get(
        "/.well-known/agent",
        tags=["Agent Discovery"],
        summary="Agent Capability Discovery",
        description="Returns a machine-readable document describing API capabilities, supported task types, and key endpoint URLs.",
    )
    async def agent_discovery() -> Dict[str, Any]:
        return {
            "name": "ACE-Step",
            "version": "1.5",
            "description": "AI music generation — text to music, cover, repaint, and more",
            "capabilities": [
                "text-to-music",
                "cover",
                "repaint",
                "complete",
                "lego",
                "extract",
            ],
            "input_modalities": ["text", "audio"],
            "output_modalities": ["audio", "text"],
            "primary_endpoints": {
                "generate": "/v1/chat/completions",
                "generate_simple": "/release_task",
                "query_result": "/query_result",
                "format_input": "/format_input",
                "create_sample": "/v1/create_sample",
                "random_sample": "/create_random_sample",
                "list_models": "/v1/models",
            },
            "schema_url": "/openapi.json",
            "docs_url": "/docs",
            "llms_txt_url": "/llms.txt",
            "llms_full_txt_url": "/llms-full.txt",
            "mcp_server": {
                "description": "MCP server wrapping this API for agent tool use",
                "source": "mcp/acestep_mcp_server.py",
                "tools": [
                    "generate_music",
                    "list_models",
                    "enhance_prompt",
                    "check_health",
                ],
                "config": {
                    "ACESTEP_API_URL": "http://localhost:8010",
                    "ACESTEP_API_KEY": "optional API key",
                },
            },
            "agent_skills": {
                "hermes": "mcp/hermes-skill/SKILL.md",
                "claude_code": "CLAUDE.md",
                "codex": "AGENTS.md",
            },
        }

    @app.get(
        "/llms.txt",
        tags=["Agent Discovery"],
        summary="Concise API overview for agents",
        response_class=PlainTextResponse,
    )
    async def llms_txt() -> str:
        path = _PROJECT_ROOT / "llms.txt"
        if path.exists():
            return path.read_text(encoding="utf-8")
        return "# llms.txt not found\n"

    @app.get(
        "/llms-full.txt",
        tags=["Agent Discovery"],
        summary="Full API reference for agents",
        response_class=PlainTextResponse,
    )
    async def llms_full_txt() -> str:
        path = _PROJECT_ROOT / "llms-full.txt"
        if path.exists():
            return path.read_text(encoding="utf-8")
        return "# llms-full.txt not found\n"
