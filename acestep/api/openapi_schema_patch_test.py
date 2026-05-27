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
