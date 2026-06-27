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


class TestResolveModel(unittest.TestCase):
    def setUp(self):
        server._clear_model_cache()

    def test_resolves_first_model_id(self):
        fake = {"object": "list", "data": [{"id": "acestep/acestep-v15-turbo"}]}
        with patch.object(server, "_request", return_value=fake):
            self.assertEqual(server._resolve_model(), "acestep/acestep-v15-turbo")

    def test_caches_across_calls(self):
        fake = {"object": "list", "data": [{"id": "acestep/acestep-v15-turbo"}]}
        with patch.object(server, "_request", return_value=fake) as mock_req:
            server._resolve_model()
            server._resolve_model()
            self.assertEqual(mock_req.call_count, 1)

    def test_raises_when_no_models(self):
        with patch.object(server, "_request", return_value={"object": "list", "data": []}):
            with self.assertRaises(RuntimeError):
                server._resolve_model()

    def test_raises_on_request_error(self):
        with patch.object(server, "_request", return_value={"error": "boom"}):
            with self.assertRaises(RuntimeError):
                server._resolve_model()

    def test_force_clears_cache(self):
        fake = {"object": "list", "data": [{"id": "acestep/acestep-v15-turbo"}]}
        with patch.object(server, "_request", return_value=fake) as mock_req:
            server._resolve_model()
            server._resolve_model(force=True)
            self.assertEqual(mock_req.call_count, 2)


class TestBuildMessages(unittest.TestCase):
    def test_text_only_is_string_content(self):
        self.assertEqual(
            server._build_messages("a pop song", None),
            [{"role": "user", "content": "a pop song"}],
        )

    def test_src_audio_builds_multimodal(self):
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(b"fake-audio-bytes")
            path = f.name
        try:
            content = server._build_messages("rock cover", path)[0]["content"]
            self.assertIsInstance(content, list)
            self.assertEqual(content[0], {"type": "text", "text": "rock cover"})
            self.assertEqual(content[1]["type"], "input_audio")
            self.assertEqual(content[1]["input_audio"]["format"], "mp3")
            self.assertEqual(base64.b64decode(content[1]["input_audio"]["data"]), b"fake-audio-bytes")
        finally:
            os.unlink(path)


class TestResolveTaskType(unittest.TestCase):
    def test_src_audio_auto_cover_from_default(self):
        self.assertEqual(server._resolve_task_type("text2music", "x.mp3"), "cover")

    def test_src_audio_keeps_explicit_src_type(self):
        self.assertEqual(server._resolve_task_type("repaint", "x.mp3"), "repaint")

    def test_src_audio_rejects_non_src_type(self):
        self.assertIsNone(server._resolve_task_type("made-up", "x.mp3"))

    def test_no_src_audio_keeps_type(self):
        self.assertEqual(server._resolve_task_type("cover", None), "cover")
        self.assertEqual(server._resolve_task_type("text2music", None), "text2music")


class TestSaveAudio(unittest.TestCase):
    def _b64url(self, payload: bytes, mime: str = "audio/mpeg") -> str:
        return "data:" + mime + ";base64," + base64.b64encode(payload).decode()

    def test_saves_file_and_returns_path(self):
        with tempfile.TemporaryDirectory() as d:
            paths = server._save_audio(
                [{"type": "audio_url", "audio_url": {"url": self._b64url(b"ABC")}}], d
            )
            self.assertEqual(len(paths), 1)
            self.assertTrue(os.path.exists(paths[0]))
            with open(paths[0], "rb") as f:
                self.assertEqual(f.read(), b"ABC")
            self.assertTrue(paths[0].endswith(".mp3"))

    def test_mime_drives_extension(self):
        with tempfile.TemporaryDirectory() as d:
            paths = server._save_audio(
                [{"type": "audio_url", "audio_url": {"url": self._b64url(b"X", "audio/mp4")}}], d
            )
            self.assertTrue(paths[0].endswith(".m4a"))

    def test_empty_or_none_returns_empty(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(server._save_audio(None, d), [])
            self.assertEqual(server._save_audio([], d), [])
            self.assertEqual(
                server._save_audio([{"type": "audio_url", "audio_url": {"url": ""}}], d), []
            )

    def test_creates_output_dir(self):
        nested = os.path.join(tempfile.mkdtemp(), "deep", "out")
        try:
            server._save_audio(
                [{"type": "audio_url", "audio_url": {"url": self._b64url(b"Y")}}], nested
            )
            self.assertTrue(os.path.isdir(nested))
        finally:
            shutil.rmtree(os.path.dirname(nested), ignore_errors=True)

    def test_skips_malformed_data_urls(self):
        with tempfile.TemporaryDirectory() as d:
            bad = [
                {"type": "audio_url", "audio_url": {"url": "data:audio/mpeg;base64"}},  # no comma
                {"type": "audio_url", "audio_url": {"url": "data:audio/mpeg;base64,!!!notb64!!!"}},  # bad b64
                {"type": "audio_url", "audio_url": {"url": self._b64url(b"OK")}},  # good
            ]
            paths = server._save_audio(bad, d)
            self.assertEqual(len(paths), 1)
            with open(paths[0], "rb") as f:
                self.assertEqual(f.read(), b"OK")


class TestParseMetadata(unittest.TestCase):
    def test_parses_all_fields(self):
        content = (
            "## Metadata\n**Caption:** dreamy pop\n**BPM:** 120\n"
            "**Duration:** 30.0s\n**Key:** C major\n**Time Signature:** 4/4\n\n"
            "## Lyrics\n[Verse]\nHello"
        )
        m = server._parse_metadata(content)
        self.assertEqual(m["caption"], "dreamy pop")
        self.assertEqual(m["bpm"], "120")
        self.assertEqual(m["duration"], "30.0")
        self.assertEqual(m["key"], "C major")
        self.assertEqual(m["time_signature"], "4/4")
        self.assertIn("Hello", m["lyrics"])

    def test_instrumental_has_no_lyrics(self):
        m = server._parse_metadata("## Metadata\n**Caption:** beat\n**BPM:** 90")
        self.assertEqual(m["caption"], "beat")
        self.assertNotIn("lyrics", m)

    def test_missing_fields_omitted(self):
        self.assertEqual(server._parse_metadata("## Metadata\n**Caption:** only"), {"caption": "only"})

    def test_empty_content(self):
        self.assertEqual(server._parse_metadata(""), {})
        self.assertEqual(server._parse_metadata("Music generated successfully."), {})

    def test_lyrics_stop_at_next_section(self):
        content = "## Metadata\n**Caption:** x\n\n## Lyrics\n[Verse]\nHello\n\n## Notes\nextra"
        m = server._parse_metadata(content)
        self.assertIn("Hello", m["lyrics"])
        self.assertNotIn("extra", m["lyrics"])


class TestGenerateMusic(unittest.TestCase):
    def setUp(self):
        server._clear_model_cache()
        self.tmp = tempfile.mkdtemp()
        self._orig_out = server.OUTPUT_DIR
        server.OUTPUT_DIR = self.tmp

    def tearDown(self):
        server.OUTPUT_DIR = self._orig_out
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _audio_url(self, payload=b"SONG", mime="audio/mpeg"):
        return "data:" + mime + ";base64," + base64.b64encode(payload).decode()

    def _gen_resp(self, content="## Metadata\n**Caption:** x", with_audio=True):
        audio = (
            [{"type": "audio_url", "audio_url": {"url": self._audio_url()}}] if with_audio else None
        )
        return {"choices": [{"message": {"content": content, "audio": audio}}]}

    _MODELS = {"object": "list", "data": [{"id": "acestep/acestep-v15-turbo"}]}

    def test_text2music_saves_and_returns_path_no_base64(self):
        with patch.object(server, "_request", side_effect=[self._MODELS, self._gen_resp()]) as mr:
            out = server.generate_music(prompt="pop", lyrics="[inst]")
        self.assertIn("Audio saved:", out)
        self.assertNotIn("base64,", out)
        self.assertEqual(mr.call_args_list[1].args[2]["model"], "acestep/acestep-v15-turbo")
        self.assertEqual(mr.call_args_list[1].args[2]["task_type"], "text2music")

    def test_invalid_format_returns_error(self):
        self.assertIn("Invalid format", server.generate_music(prompt="x", format="mp9"))

    def test_src_audio_auto_cover(self):
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(b"src"); src = f.name
        try:
            with patch.object(server, "_request", side_effect=[self._MODELS, self._gen_resp()]) as mr:
                server.generate_music(prompt="rock cover", src_audio=src)
            body = mr.call_args_list[1].args[2]
            self.assertEqual(body["task_type"], "cover")
            self.assertIsInstance(body["messages"][0]["content"], list)
        finally:
            os.unlink(src)

    def test_cover_without_src_audio_errors(self):
        self.assertIn("requires src_audio", server.generate_music(prompt="x", task_type="cover"))

    def test_empty_audio_returns_retry_message_no_file(self):
        with patch.object(server, "_request", side_effect=[self._MODELS, self._gen_resp(with_audio=False)]):
            out = server.generate_music(prompt="x")
        self.assertIn("No audio produced", out)
        self.assertEqual(os.listdir(self.tmp), [])

    def test_error_response_returns_failure(self):
        with patch.object(server, "_request", side_effect=[self._MODELS, {"error": "boom"}]):
            self.assertIn("Generation failed: boom", server.generate_music(prompt="x"))

    def test_retry_on_model_error_then_succeeds(self):
        models_v1 = {"object": "list", "data": [{"id": "acestep/old"}]}
        models_v2 = {"object": "list", "data": [{"id": "acestep/acestep-v15-turbo"}]}
        seq = [models_v1, {"error": "model acestep/old not found"}, models_v2, self._gen_resp()]
        with patch.object(server, "_request", side_effect=seq) as mr:
            out = server.generate_music(prompt="pop")
        self.assertIn("Audio saved:", out)
        self.assertEqual(mr.call_count, 4)
        self.assertEqual(mr.call_args_list[3].args[2]["model"], "acestep/acestep-v15-turbo")

    def test_missing_src_audio_file_returns_error(self):
        self.assertIn("src_audio file not found", server.generate_music(prompt="x", src_audio="/no/such/file.mp3"))


if __name__ == "__main__":
    unittest.main()
