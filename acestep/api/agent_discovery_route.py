"""Agent discovery endpoint — GET /.well-known/agent.

Returns a machine-readable capability document so AI agents can
understand what the service offers without reading source code.
"""

from __future__ import annotations

from typing import Any, Dict

from fastapi import FastAPI


def register_agent_discovery_route(app: FastAPI) -> None:
    """Register the agent discovery endpoint on the FastAPI app."""

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
        }
