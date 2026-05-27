"""Lightweight Pydantic request models for OpenAPI spec documentation.

These models describe the request shapes for endpoints that use
``request: Request`` with manual body parsing. They are used exclusively
for OpenAPI schema generation — no runtime validation change.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class QueryResultRequest(BaseModel):
    """Request body for POST /query_result."""

    task_id_list: List[str] = Field(
        default_factory=list,
        description="List of task IDs to query results for",
    )


class FormatInputRequest(BaseModel):
    """Request body for POST /format_input — enhance caption/lyrics via LLM."""

    prompt: str = Field(default="", description="Text description of the desired music")
    lyrics: str = Field(default="", description="Lyrics with section tags [Verse], [Chorus], etc.")
    temperature: float = Field(default=0.85, ge=0.0, le=2.0, description="LLM sampling temperature")


class CreateRandomSampleRequest(BaseModel):
    """Request body for POST /create_random_sample."""

    sample_type: str = Field(
        default="simple_mode",
        description="Example data mode: 'simple_mode' or 'custom_mode'",
    )


class CreateSampleRequest(BaseModel):
    """Request body for POST /v1/create_sample — auto-generate music parameters."""

    query: str = Field(default="", description="Natural language description for sample generation")
    vocal_language: str = Field(default="en", description="Language code: en, zh, ja, ko, fr, etc.")
    temperature: float = Field(default=0.85, ge=0.0, le=2.0, description="LLM sampling temperature")


class LoadTensorInfoRequest(BaseModel):
    """Request body for POST /v1/training/load_tensor_info."""

    tensor_dir: str = Field(..., description="Path to directory containing tensor files")
