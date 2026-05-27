"""OpenAPI schema patch — inject Pydantic model schemas into the generated spec.

Routes that use ``request: Request`` produce empty OpenAPI schemas. This module
overrides ``FastAPI.openapi()`` to merge existing Pydantic model definitions
into the spec without changing any route handler code.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List

from fastapi import FastAPI
from pydantic import BaseModel

from acestep.api.http.release_task_models import GenerateMusicRequest
from acestep.api.openapi_models import (
    CreateRandomSampleRequest,
    CreateSampleRequest,
    FormatInputRequest,
    LoadTensorInfoRequest,
    QueryResultRequest,
)
from acestep.openrouter_models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ModelsResponse,
)


OPENAPI_TAGS: List[Dict[str, str]] = [
    {"name": "Music Generation", "description": "Core music generation endpoints"},
    {"name": "Task Management", "description": "Query and manage generation tasks"},
    {"name": "Audio", "description": "Audio file access"},
    {"name": "Model Management", "description": "Model loading, initialization, and status"},
    {"name": "LoRA", "description": "LoRA adapter management"},
    {"name": "Training", "description": "LoRA/LoKR training operations"},
    {"name": "Dataset", "description": "Training dataset operations"},
    {"name": "Agent Discovery", "description": "Endpoints for AI agent self-discovery"},
]

_PATH_TAG_MAP: Dict[str, List[str]] = {
    "/release_task": ["Music Generation"],
    "/v1/chat/completions": ["Music Generation"],
    "/create_random_sample": ["Music Generation"],
    "/format_input": ["Music Generation"],
    "/v1/create_sample": ["Music Generation"],
    "/query_result": ["Task Management"],
    "/v1/audio": ["Audio"],
    "/v1/models": ["Model Management"],
    "/v1/model_inventory": ["Model Management"],
    "/v1/init": ["Model Management"],
    "/v1/reinitialize": ["Model Management"],
    "/v1/stats": ["Model Management"],
    "/health": ["Model Management"],
    "/.well-known/agent": ["Agent Discovery"],
    "/v1/lora": ["LoRA"],
    "/v1/training": ["Training"],
    "/v1/dataset": ["Dataset"],
}

_PATH_REQUEST_BODY_MAP: Dict[str, Dict[str, Any]] = {
    "/release_task": {
        "content_types": ["application/json", "multipart/form-data"],
        "model": GenerateMusicRequest,
    },
    "/v1/chat/completions": {
        "content_types": ["application/json"],
        "model": ChatCompletionRequest,
    },
    "/query_result": {
        "content_types": ["application/json"],
        "model": QueryResultRequest,
    },
    "/format_input": {
        "content_types": ["application/json"],
        "model": FormatInputRequest,
    },
    "/create_random_sample": {
        "content_types": ["application/json"],
        "model": CreateRandomSampleRequest,
    },
    "/v1/create_sample": {
        "content_types": ["application/json"],
        "model": CreateSampleRequest,
    },
}


def _model_to_schema(model: type[BaseModel]) -> Dict[str, Any]:
    """Convert a Pydantic model to an OpenAPI-compatible JSON Schema dict."""
    return model.model_json_schema(ref_template="#/components/schemas/{model}")


def _collect_schemas() -> Dict[str, Any]:
    """Collect all model schemas into a components.schemas dict."""
    models: List[type[BaseModel]] = [
        GenerateMusicRequest,
        ChatCompletionRequest,
        ChatCompletionResponse,
        ModelsResponse,
        QueryResultRequest,
        FormatInputRequest,
        CreateRandomSampleRequest,
        CreateSampleRequest,
        LoadTensorInfoRequest,
    ]
    schemas: Dict[str, Any] = {}
    for model in models:
        model_schema = _model_to_schema(model)
        defs = model_schema.pop("$defs", {})
        schemas.update(defs)
        schemas[model.__name__] = model_schema
    return schemas


def _inject_request_bodies(paths: Dict[str, Any]) -> None:
    """Inject requestBody into path items that lack them."""
    for path, path_item in paths.items():
        for method in ("post", "put", "patch"):
            if method not in path_item:
                continue
            operation = path_item[method]
            if "requestBody" in operation:
                existing_json_schema = operation["requestBody"].get("content", {}).get("application/json", {}).get("schema", {})
                if "$ref" in existing_json_schema:
                    continue

            config = _PATH_REQUEST_BODY_MAP.get(path)
            if config is None:
                continue

            model_name = config["model"].__name__
            content: Dict[str, Any] = {}
            for ct in config["content_types"]:
                content[ct] = {
                    "schema": {"$ref": f"#/components/schemas/{model_name}"}
                }
            operation["requestBody"] = {
                "required": True,
                "content": content,
            }


def _inject_tags(paths: Dict[str, Any]) -> None:
    """Add tags to path operations based on path prefix."""
    for path, path_item in paths.items():
        for method in ("get", "post", "put", "patch", "delete"):
            if method not in path_item:
                continue
            operation = path_item[method]
            existing_tags = operation.get("tags", [])
            for prefix, tags in _PATH_TAG_MAP.items():
                if path.startswith(prefix) or path == prefix:
                    for tag in tags:
                        if tag not in existing_tags:
                            existing_tags.append(tag)
            if existing_tags:
                operation["tags"] = existing_tags


def patch_openapi(app: FastAPI) -> None:
    """Override app.openapi() to inject Pydantic model schemas.

    Call this after all routes are registered on the app.
    """
    original_openapi: Callable[[], Dict[str, Any]] = app.openapi
    _extra_schemas = _collect_schemas()

    def _patched_openapi() -> Dict[str, Any]:
        spec = original_openapi()

        components = spec.setdefault("components", {})
        existing_schemas = components.setdefault("schemas", {})
        for name, schema in _extra_schemas.items():
            if name not in existing_schemas:
                existing_schemas[name] = schema

        _inject_request_bodies(spec.get("paths", {}))
        _inject_tags(spec.get("paths", {}))

        existing_tag_names = {t["name"] for t in spec.get("tags", [])}
        merged_tags = list(spec.get("tags", []))
        for tag in OPENAPI_TAGS:
            if tag["name"] not in existing_tag_names:
                merged_tags.append(tag)
        spec["tags"] = merged_tags

        return spec

    app.openapi = _patched_openapi
