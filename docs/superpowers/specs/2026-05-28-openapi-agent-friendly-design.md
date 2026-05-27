# OpenAPI Agent-Friendly Spec Design

**Date**: 2026-05-28
**Status**: Approved
**Scope**: Make all 37 FastAPI endpoints AI-agent-discoverable via OpenAPI spec

## Problem

All 37 API endpoints use `request: Request` for manual body parsing. FastAPI cannot
auto-generate OpenAPI schemas from raw `Request` objects, so external AI agents
(OpenAI SDK, Claude tool-use, MCP clients) see an empty spec with zero parameter
descriptions. Agent score: 4.7/10.

## Root Cause Analysis

- 15 endpoints already use Pydantic models as route parameters (LoRA, training,
  dataset routes). These already produce correct OpenAPI schemas.
- 7 endpoints use `request: Request` with manual parsing. These produce empty schemas.
- The `/release_task` endpoint accepts JSON, form-data, AND multipart file uploads
  in a single handler. Pydantic model binding cannot cover all three, which is why
  the codebase intentionally uses `request: Request`.
- The `/v1/chat/completions` endpoint already has `ChatCompletionRequest` but parses
  manually because `extra = "allow"` on the model would lose passthrough fields under
  FastAPI's automatic binding.

FastAPI does **not** allow two routes at the same path+method, so a dual-track
approach (original + typed) is blocked.

## Design: Hybrid A+ (openapi_extra + Global Spec Override)

**Core principle**: Inject existing Pydantic model schemas into the OpenAPI spec
without changing any route handler code or runtime behavior.

### Phase 1: OpenAPI Schema Patch Module

Create `acestep/api/openapi_schema_patch.py` (~120 lines).

This module provides `patch_openapi(app: FastAPI)` which:

1. Saves the original `app.openapi()` method
2. Replaces it with a wrapper that:
   a. Calls the original to get the base spec
   b. Merges Pydantic model JSON schemas into `components.schemas`
   c. Injects `requestBody` into the 7 untyped path items
   d. Injects response descriptions
   e. Adds OpenAPI tags for endpoint grouping
   f. Returns the patched spec

Models to register (already exist):
- `GenerateMusicRequest` from `acestep/api/http/release_task_models.py`
- `ChatCompletionRequest` from `acestep/openrouter_models.py`
- `ChatCompletionResponse` from `acestep/openrouter_models.py`
- `ModelsResponse` from `acestep/openrouter_models.py`

Models to create (lightweight, for spec only):
- `QueryResultRequest` — `task_id_list: list[str]`
- `FormatInputRequest` — `prompt: str`, `lyrics: str`, `temperature: float`
- `CreateRandomSampleRequest` — `sample_type: str`
- `CreateSampleRequest` — `query: str`, `vocal_language: str`, `temperature: float`
- `LoadTensorInfoRequest` — `tensor_dir: str` (fixes existing bug)
- `ApiResponseEnvelope` — generic wrapper model for documentation

New models go in `acestep/api/openapi_models.py` (~80 lines).

### Phase 2: Wire Patch into App

In `acestep/api_server.py`, after `app = create_app()` or inside `create_app()`
after route registration:

```python
from acestep.api.openapi_schema_patch import patch_openapi
patch_openapi(app)
```

This is a single import + function call. No other changes to `api_server.py`.

### Phase 3: Add Agent Discovery Endpoint

Add `GET /.well-known/agent` to `acestep/api/route_setup.py` (or as a separate
tiny route module). Returns:

```json
{
  "name": "ACE-Step",
  "version": "1.5",
  "capabilities": ["text-to-music", "cover", "repaint", "complete", "lego", "extract"],
  "input_modalities": ["text", "audio"],
  "output_modalities": ["audio", "text"],
  "models": [],
  "primary_endpoints": {
    "generate": "/v1/chat/completions",
    "generate_simple": "/release_task",
    "query_result": "/query_result",
    "format_input": "/format_input",
    "list_models": "/v1/models"
  },
  "schema_url": "/openapi.json",
  "docs_url": "/docs",
  "llms_txt_url": "/llms.txt",
  "llms_full_txt_url": "/llms-full.txt"
}
```

This endpoint has no auth requirement and returns even if models are not initialized.

### Phase 4: OpenAPI Tags and Descriptions

The `patch_openapi` function adds tags to all paths:

| Tag | Endpoints |
|-----|-----------|
| Music Generation | `/release_task`, `/v1/chat/completions`, `/create_random_sample`, `/format_input`, `/v1/create_sample` |
| Task Management | `/query_result` |
| Audio | `/v1/audio` |
| Model Management | `/v1/models`, `/v1/model_inventory`, `/v1/init`, `/v1/reinitialize`, `/v1/stats`, `/health` |
| LoRA | `/v1/lora/*` |
| Training | `/v1/training/*` |
| Dataset | `/v1/dataset/*` |

Tag descriptions are set in `openapi.tags` at the top level.

### Phase 5: Validation

After implementation, verify:

1. `GET /openapi.json` — every endpoint has non-empty request/response schemas
2. `GET /docs` — Swagger UI shows all fields for `/release_task` and `/v1/chat/completions`
3. OpenAI SDK test: `client.chat.completions.create(...)` still works
4. `/release_task` with multipart file upload still works (unchanged runtime)
5. `GET /.well-known/agent` returns capability document

## Files to Create

| File | Lines | Purpose |
|------|-------|---------|
| `acestep/api/openapi_schema_patch.py` | ~120 | Spec override logic |
| `acestep/api/openapi_models.py` | ~80 | Missing request models for spec |

## Files to Modify

| File | Change | Lines Changed |
|------|--------|---------------|
| `acestep/api_server.py` | Add `patch_openapi(app)` call after route setup | +3 |
| `acestep/api/route_setup.py` | Add `/.well-known/agent` route | +30 |

## Files NOT Changed

All route handlers remain untouched:
- `acestep/api/http/release_task_route.py`
- `acestep/api/http/query_result_route.py`
- `acestep/api/http/sample_format_routes.py`
- `acestep/openrouter_adapter.py`
- `acestep/ui/gradio/api/api_routes.py`

## Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Spec drift from runtime | Low | Models already exist and are actively maintained |
| Patch breaks OpenAPI generation | Low | Purely additive; original spec is preserved as base |
| Gradio mode unaffected | Certain | Patch only runs in API server mode |
| Existing clients affected | None | Zero runtime behavior change |

## Estimated Effort

~7 hours total across 5 phases.
