# OpenAPI Agent-Friendly Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Inject Pydantic model schemas into the OpenAPI spec so external AI agents can discover API parameters without reading source code.

**Architecture:** Override `FastAPI.openapi()` to merge existing Pydantic model schemas into the generated spec. Add a `/.well-known/agent` discovery endpoint. No route handler code changes.

**Tech Stack:** Python 3.12, FastAPI, Pydantic v2, unittest

**Spec:** `docs/superpowers/specs/2026-05-28-openapi-agent-friendly-design.md`

---

## File Structure

| Action | File | Responsibility |
|--------|------|----------------|
| Create | `acestep/api/openapi_models.py` | Lightweight request models for spec-only endpoints |
| Create | `acestep/api/openapi_schema_patch.py` | `patch_openapi()` — merges models into OpenAPI spec |
| Create | `acestep/api/openapi_models_test.py` | Tests for new models |
| Create | `acestep/api/openapi_schema_patch_test.py` | Tests for spec patching |
| Create | `acestep/api/agent_discovery_route.py` | `GET /.well-known/agent` endpoint |
| Create | `acestep/api/agent_discovery_route_test.py` | Tests for discovery endpoint |
| Modify | `acestep/api_server.py` | Wire in `patch_openapi()` after route setup |
| Modify | `acestep/api/route_setup.py` | Register agent discovery route |

---

### Task 1: Create lightweight request models

**Files:**
- Create: `acestep/api/openapi_models.py`
- Create: `acestep/api/openapi_models_test.py`

- [ ] **Step 1: Write the failing test**

Create `acestep/api/openapi_models_test.py`:

```python
"""Tests for openapi_models — lightweight request models for spec documentation."""

import unittest

from pydantic import BaseModel


class TestQueryResultRequest(unittest.TestCase):
    """QueryResultRequest model for /query_result spec."""

    def test_accepts_task_id_list(self):
        from acestep.api.openapi_models import QueryResultRequest

        req = QueryResultRequest(task_id_list=["id-1", "id-2"])
        self.assertEqual(req.task_id_list, ["id-1", "id-2"])

    def test_default_empty_list(self):
        from acestep.api.openapi_models import QueryResultRequest

        req = QueryResultRequest()
        self.assertEqual(req.task_id_list, [])

    def test_json_schema_has_description(self):
        from acestep.api.openapi_models import QueryResultRequest

        schema = QueryResultRequest.model_json_schema()
        props = schema["properties"]
        self.assertIn("task_id_list", props)
        self.assertIn("description", props["task_id_list"])


class TestFormatInputRequest(unittest.TestCase):
    """FormatInputRequest model for /format_input spec."""

    def test_accepts_prompt_and_lyrics(self):
        from acestep.api.openapi_models import FormatInputRequest

        req = FormatInputRequest(prompt="pop song", lyrics="[Verse]\nHello")
        self.assertEqual(req.prompt, "pop song")
        self.assertEqual(req.lyrics, "[Verse]\nHello")

    def test_defaults(self):
        from acestep.api.openapi_models import FormatInputRequest

        req = FormatInputRequest()
        self.assertEqual(req.prompt, "")
        self.assertEqual(req.lyrics, "")
        self.assertEqual(req.temperature, 0.85)


class TestCreateRandomSampleRequest(unittest.TestCase):
    """CreateRandomSampleRequest model for /create_random_sample spec."""

    def test_default_sample_type(self):
        from acestep.api.openapi_models import CreateRandomSampleRequest

        req = CreateRandomSampleRequest()
        self.assertEqual(req.sample_type, "simple_mode")


class TestCreateSampleRequest(unittest.TestCase):
    """CreateSampleRequest model for /v1/create_sample spec."""

    def test_defaults(self):
        from acestep.api.openapi_models import CreateSampleRequest

        req = CreateSampleRequest()
        self.assertEqual(req.query, "")
        self.assertEqual(req.vocal_language, "en")
        self.assertEqual(req.temperature, 0.85)


class TestLoadTensorInfoRequest(unittest.TestCase):
    """LoadTensorInfoRequest — fixes the broken request: dict in training route."""

    def test_requires_tensor_dir(self):
        from acestep.api.openapi_models import LoadTensorInfoRequest

        req = LoadTensorInfoRequest(tensor_dir="/path/to/tensors")
        self.assertEqual(req.tensor_dir, "/path/to/tensors")

    def test_validation_error_on_missing(self):
        from pydantic import ValidationError
        from acestep.api.openapi_models import LoadTensorInfoRequest

        with self.assertRaises(ValidationError):
            LoadTensorInfoRequest()


class TestModelJsonSchemaExport(unittest.TestCase):
    """All new models produce valid JSON Schema for OpenAPI injection."""

    def test_all_models_export_schema(self):
        from acestep.api import openapi_models

        model_classes = [
            openapi_models.QueryResultRequest,
            openapi_models.FormatInputRequest,
            openapi_models.CreateRandomSampleRequest,
            openapi_models.CreateSampleRequest,
            openapi_models.LoadTensorInfoRequest,
        ]
        for cls in model_classes:
            schema = cls.model_json_schema()
            self.assertIn("properties", schema)
            self.assertIn("type", schema)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/vinsun/project/ACE-Step-1.5 && uv run python -m unittest acestep/api/openapi_models_test.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'acestep.api.openapi_models'`

- [ ] **Step 3: Write implementation**

Create `acestep/api/openapi_models.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/vinsun/project/ACE-Step-1.5 && uv run python -m unittest acestep/api/openapi_models_test.py -v`
Expected: All 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add acestep/api/openapi_models.py acestep/api/openapi_models_test.py
git commit -m "feat(api): add lightweight Pydantic request models for OpenAPI spec"
```

---

### Task 2: Create OpenAPI schema patch module

**Files:**
- Create: `acestep/api/openapi_schema_patch.py`
- Create: `acestep/api/openapi_schema_patch_test.py`

- [ ] **Step 1: Write the failing test**

Create `acestep/api/openapi_schema_patch_test.py`:

```python
"""Tests for openapi_schema_patch — injects Pydantic schemas into OpenAPI spec."""

import unittest

from fastapi import FastAPI

from acestep.api.openapi_schema_patch import patch_openapi


def _make_app_with_untyped_route() -> FastAPI:
    """Create a minimal FastAPI app with an untyped route like the real server."""

    app = FastAPI(title="Test API")

    @app.post("/release_task")
    async def release_task(request: dict):
        return {"data": {}}

    @app.post("/query_result")
    async def query_result(request: dict):
        return {"data": {}}

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return app


class TestPatchOpenAPI(unittest.TestCase):
    """patch_openapi injects schemas into the generated OpenAPI spec."""

    def test_patch_does_not_crash(self):
        app = _make_app_with_untyped_route()
        patch_openapi(app)
        spec = app.openapi()
        self.assertIsInstance(spec, dict)

    def test_components_schemas_populated(self):
        app = _make_app_with_untyped_route()
        patch_openapi(app)
        spec = app.openapi()
        schemas = spec.get("components", {}).get("schemas", {})
        self.assertIn("GenerateMusicRequest", schemas)
        self.assertIn("ChatCompletionRequest", schemas)

    def test_release_task_has_request_body(self):
        app = _make_app_with_untyped_route()
        patch_openapi(app)
        spec = app.openapi()
        post = spec["paths"]["/release_task"]["post"]
        self.assertIn("requestBody", post)
        content = post["requestBody"]["content"]
        self.assertIn("application/json", content)
        json_schema = content["application/json"]["schema"]
        self.assertIn("GenerateMusicRequest", str(json_schema.get("$ref", "")))

    def test_release_task_accepts_multipart(self):
        app = _make_app_with_untyped_route()
        patch_openapi(app)
        spec = app.openapi()
        post = spec["paths"]["/release_task"]["post"]
        content = post["requestBody"]["content"]
        self.assertIn("multipart/form-data", content)

    def test_query_result_has_request_body(self):
        app = _make_app_with_untyped_route()
        patch_openapi(app)
        spec = app.openapi()
        post = spec["paths"]["/query_result"]["post"]
        self.assertIn("requestBody", post)

    def test_tags_added(self):
        app = _make_app_with_untyped_route()
        patch_openapi(app)
        spec = app.openapi()
        self.assertIn("tags", spec)
        tag_names = [t["name"] for t in spec["tags"]]
        self.assertIn("Music Generation", tag_names)

    def test_path_tags_assigned(self):
        app = _make_app_with_untyped_route()
        patch_openapi(app)
        spec = app.openapi()
        release_tags = spec["paths"]["/release_task"]["post"].get("tags", [])
        self.assertIn("Music Generation", release_tags)

    def test_health_not_modified(self):
        app = _make_app_with_untyped_route()
        patch_openapi(app)
        spec = app.openapi()
        get = spec["paths"]["/health"]["get"]
        self.assertNotIn("requestBody", get)


class TestSchemaContent(unittest.TestCase):
    """Injected schemas contain the expected field definitions."""

    def test_generate_music_request_has_prompt(self):
        app = _make_app_with_untyped_route()
        patch_openapi(app)
        spec = app.openapi()
        schema = spec["components"]["schemas"]["GenerateMusicRequest"]
        props = schema.get("properties", {})
        self.assertIn("prompt", props)
        self.assertIn("lyrics", props)
        self.assertIn("audio_duration", props)
        self.assertIn("task_type", props)

    def test_chat_completion_request_has_messages(self):
        app = _make_app_with_untyped_route()
        patch_openapi(app)
        spec = app.openapi()
        schema = spec["components"]["schemas"]["ChatCompletionRequest"]
        props = schema.get("properties", {})
        self.assertIn("messages", props)
        self.assertIn("model", props)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/vinsun/project/ACE-Step-1.5 && uv run python -m unittest acestep/api/openapi_schema_patch_test.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'acestep.api.openapi_schema_patch'`

- [ ] **Step 3: Write implementation**

Create `acestep/api/openapi_schema_patch.py`:

```python
"""OpenAPI schema patch — inject Pydantic model schemas into the generated spec.

Routes that use ``request: Request`` produce empty OpenAPI schemas. This module
overrides ``FastAPI.openapi()`` to merge existing Pydantic model definitions
into the spec without changing any route handler code.
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Dict, List, Optional, Set

from fastapi import FastAPI
from loguru import logger
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


# =============================================================================
# Tag definitions
# =============================================================================

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

# Map: path prefix -> tag name(s)
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

# Map: path -> request body schema config
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
        # Flatten $defs into top-level schemas
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
            if "requestBody" in operation and operation["requestBody"].get("content", {}).get("application/json", {}).get("schema", {}):
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

        # Merge schemas into components
        components = spec.setdefault("components", {})
        existing_schemas = components.setdefault("schemas", {})
        for name, schema in _extra_schemas.items():
            if name not in existing_schemas:
                existing_schemas[name] = schema

        # Inject request bodies
        _inject_request_bodies(spec.get("paths", {}))

        # Inject tags
        _inject_tags(spec.get("paths", {}))

        # Set top-level tags
        existing_tag_names = {t["name"] for t in spec.get("tags", [])}
        merged_tags = list(spec.get("tags", []))
        for tag in OPENAPI_TAGS:
            if tag["name"] not in existing_tag_names:
                merged_tags.append(tag)
        spec["tags"] = merged_tags

        return spec

    app.openapi = _patched_openapi
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/vinsun/project/ACE-Step-1.5 && uv run python -m unittest acestep/api/openapi_schema_patch_test.py -v`
Expected: All 10 tests PASS

- [ ] **Step 5: Commit**

```bash
git add acestep/api/openapi_schema_patch.py acestep/api/openapi_schema_patch_test.py
git commit -m "feat(api): add OpenAPI schema patch module for agent discovery"
```

---

### Task 3: Create agent discovery endpoint

**Files:**
- Create: `acestep/api/agent_discovery_route.py`
- Create: `acestep/api/agent_discovery_route_test.py`

- [ ] **Step 1: Write the failing test**

Create `acestep/api/agent_discovery_route_test.py`:

```python
"""Tests for agent_discovery_route — GET /.well-known/agent."""

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


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/vinsun/project/ACE-Step-1.5 && uv run python -m unittest acestep/api/agent_discovery_route_test.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'acestep.api.agent_discovery_route'`

- [ ] **Step 3: Write implementation**

Create `acestep/api/agent_discovery_route.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/vinsun/project/ACE-Step-1.5 && uv run python -m unittest acestep/api/agent_discovery_route_test.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add acestep/api/agent_discovery_route.py acestep/api/agent_discovery_route_test.py
git commit -m "feat(api): add /.well-known/agent discovery endpoint"
```

---

### Task 4: Wire everything into the app

**Files:**
- Modify: `acestep/api_server.py`
- Modify: `acestep/api/route_setup.py`

- [ ] **Step 1: Register agent discovery route in route_setup.py**

In `acestep/api/route_setup.py`, add import and call inside `configure_api_routes`:

At the top, add to imports:
```python
from acestep.api.agent_discovery_route import register_agent_discovery_route
```

At the end of `configure_api_routes` function body (after `register_audio_route`), add:
```python
register_agent_discovery_route(app=app)
```

- [ ] **Step 2: Wire patch_openapi in api_server.py**

In `acestep/api_server.py`, add import at the top (after existing local imports):
```python
from acestep.api.openapi_schema_patch import patch_openapi
```

Inside `create_app()`, after the `configure_api_routes(...)` call (after line ~353), add:
```python
patch_openapi(app)
```

- [ ] **Step 3: Verify with running server**

Run: `cd /home/vinsun/project/ACE-Step-1.5 && uv run python -m unittest acestep/api/openapi_schema_patch_test.py acestep/api/openapi_models_test.py acestep/api/agent_discovery_route_test.py -v`
Expected: All 25 tests PASS

- [ ] **Step 4: Run all existing API tests to check no regressions**

Run: `cd /home/vinsun/project/ACE-Step-1.5 && uv run python -m unittest discover -s acestep/api -p "*_test.py" -v`
Expected: All existing tests PASS (no regression)

- [ ] **Step 5: Commit**

```bash
git add acestep/api_server.py acestep/api/route_setup.py
git commit -m "feat(api): wire OpenAPI schema patch and agent discovery into app"
```

---

### Task 5: Integration validation with live server

**Files:** None (validation only)

- [ ] **Step 1: Restart the API server**

```bash
# Kill existing server
pkill -f "acestep.api_server"
# Start fresh
toolbox run --container acestep-rocm python -m acestep.api_server --port 8010 &
```

- [ ] **Step 2: Verify /.well-known/agent**

Run: `curl -s http://127.0.0.1:8010/.well-known/agent | python3 -m json.tool`

Expected: JSON with `name`, `capabilities`, `primary_endpoints` fields.

- [ ] **Step 3: Verify OpenAPI spec has schemas**

Run: `curl -s http://127.0.0.1:8010/openapi.json | python3 -c "import sys,json; d=json.load(sys.stdin); print('Components:', list(d.get('components',{}).get('schemas',{}).keys())[:10]); print('/release_task requestBody:', 'requestBody' in d['paths']['/release_task']['post']); print('/v1/chat/completions requestBody:', 'requestBody' in d['paths']['/v1/chat/completions']['post'])"`

Expected: `GenerateMusicRequest` and `ChatCompletionRequest` in components, both endpoints have `requestBody: True`.

- [ ] **Step 4: Verify Swagger UI**

Run: `curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8010/docs`
Expected: 200

- [ ] **Step 5: Verify existing /release_task still works**

Run: `curl -s -X POST http://127.0.0.1:8010/release_task -H "Content-Type: application/json" -d '{"prompt":"test","lyrics":"[inst]","audio_duration":10}' | python3 -c "import sys,json; d=json.load(sys.stdin); print('status:', d['code'])"`

Expected: `status: 200`

- [ ] **Step 6: Commit all remaining changes**

```bash
git add -A
git status
git diff --cached --stat
```

Review and commit any remaining changes.
