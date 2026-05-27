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
