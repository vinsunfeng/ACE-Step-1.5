# MCP Reliability + Doc Single-Source-of-Truth — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the MCP `generate_music` tool actually work end-to-end (correct model, working cover/repaint, audio saved to disk + metadata) and unify the agent docs (`llms.txt`, `llms-full.txt`, `AGENTS.md`, `hermes-skill`) into a single-source-of-truth structure.

**Architecture:** Add small pure/testable helper functions to `mcp/acestep_mcp_server.py` (model resolution, multimodal message building, audio save, metadata parse), then rewrite `generate_music` to orchestrate them. Docs are reconciled in place with `llms-full.txt` as canonical and the others delegating. A drift-guard test pins tool-list consistency.

**Tech Stack:** Python 3.14 (host), FastMCP (PyPI `mcp` SDK), unittest, stdlib only (no new deps).

**Spec:** `docs/superpowers/specs/2026-06-27-mcp-llms-design.md`

---

## CRITICAL IMPORT NOTE (read first)

`mcp/` is a **script directory, not a package** (no `__init__.py`). The PyPI `mcp` SDK (regular package in site-packages) shadows the local `mcp/` dir, so **`import mcp.acestep_mcp_server` raises `ModuleNotFoundError`** (verified). Therefore:
- The server stays a **script** (`python mcp/acestep_mcp_server.py`) — its `from mcp.server.fastmcp import FastMCP` resolves to the SDK because sys.path[0] is the `mcp/` dir itself.
- The **test file lives at repo ROOT** (`acestep_mcp_server_test.py`, not under `mcp/`) so it is importable as a top-level module.
- The test loads the server by **file path** via `importlib.util` under a distinct name (`acestep_mcp_srv`), which leaves the SDK `mcp` intact.
- Do NOT add `mcp/__init__.py` — it would make `mcp/` shadow the SDK and break the server's own import.

## File Structure

| File | Responsibility | Action |
|------|---------------|--------|
| `mcp/acestep_mcp_server.py` | MCP tools; add helpers + rewrite `generate_music` | Modify |
| `acestep_mcp_server_test.py` (repo root) | Unit tests; loads server by path | Create |
| `llms-full.txt` | Canonical API reference (fix bugs, split endpoint tables) | Modify |
| `llms.txt` | Thin summary delegating to llms-full.txt | Modify |
| `AGENTS.md` | MCP section: tool table (5), env vars, Claude Code+Codex, drift-checklist rule | Modify |
| `mcp/hermes-skill/SKILL.md` | Endpoint-scoped curl skill; fix model id (×3) | Modify |
| `acestep/api/agent_discovery_route.py` | `mcp_server.tools` 4→5 | Modify |
| `acestep/api/agent_discovery_route_test.py` | Add tools-sync assertion | Modify |

Helpers added to `mcp/acestep_mcp_server.py`: `_resolve_model`/`_clear_model_cache`, `_build_messages`, `_resolve_task_type`, `_save_audio`, `_parse_metadata`.

---

## Part A — MCP code (TDD)

### Task 1: Test scaffold (repo root) + module constants + _CAPTION_GUIDE rule comment

**Files:**
- Create: `acestep_mcp_server_test.py` (repo root)
- Modify: `mcp/acestep_mcp_server.py`

- [ ] **Step 1: Create the test file with the path-based loader + import test**

Create `acestep_mcp_server_test.py` at repo root:
```python
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


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test to verify it passes (proves the loader works)**

Run: `python -m unittest acestep_mcp_server_test -v`
Expected: PASS (1 test). If it fails with ModuleNotFoundError on `mcp.server.fastmcp`, the loader path is wrong — confirm `_HERE/mcp/acestep_mcp_server.py` exists.

- [ ] **Step 3: Add module constants to the server**

In `mcp/acestep_mcp_server.py`, immediately after the `API_KEY = os.getenv("ACESTEP_API_KEY", "")` line, insert:
```python
OUTPUT_DIR = os.getenv("ACESTEP_OUTPUT_DIR", os.path.join(os.getcwd(), "acestep_output"))
REQUEST_TIMEOUT = int(os.getenv("ACESTEP_REQUEST_TIMEOUT", "650"))

_SRC_TASK_TYPES = {"cover", "cover-nofsq", "repaint", "lego", "extract", "complete"}
_CHAT_FORMATS = {"mp3", "wav", "flac", "ogg", "m4a", "aac"}
_MIME_TO_EXT = {
    "audio/mpeg": "mp3",
    "audio/wav": "wav",
    "audio/flac": "flac",
    "audio/ogg": "ogg",
    "audio/mp4": "m4a",
    "audio/aac": "aac",
}
_model_cache = None
```

- [ ] **Step 4: Bump the request timeout + add stdlib imports**

In the same file:
- Add `import re` and `import uuid` to the stdlib import block at the top (after `from urllib.error import URLError, HTTPError`).
- In `_request`, change `urlopen(req, timeout=600)` to `urlopen(req, timeout=REQUEST_TIMEOUT)`.

- [ ] **Step 5: Record the _CAPTION_GUIDE rule as a code comment**

Immediately above the `_CAPTION_GUIDE = """\` line, insert:
```python
# CONSTRAINT: keep this guide to prose caption/lyrics writing guidance only.
# Do NOT add endpoint URLs, API field/param names, or format-option claims
# (those live in llms-full.txt). Metadata *ranges* (BPM/key/duration) as
# writing guidance are allowed.
```

- [ ] **Step 6: Re-run the test, confirm PASS**

Run: `python -m unittest acestep_mcp_server_test -v`
Expected: PASS (module still imports; constants added cleanly).

- [ ] **Step 7: Commit**
```bash
git add mcp/acestep_mcp_server.py acestep_mcp_server_test.py
git commit -m "test(mcp): add path-based test scaffold + module constants"
```

---

### Task 2: Model resolution helper (fix the wrong hardcoded model)

**Files:** Modify `mcp/acestep_mcp_server.py`; test in `acestep_mcp_server_test.py`

- [ ] **Step 1: Write the failing tests**

Append to `acestep_mcp_server_test.py` (before the `if __name__` block):
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m unittest acestep_mcp_server_test.TestResolveModel -v`
Expected: FAIL (`_resolve_model`/`_clear_model_cache` not defined).

- [ ] **Step 3: Implement the helpers**

In `mcp/acestep_mcp_server.py`, after the `_request` function, add:
```python
def _clear_model_cache() -> None:
    """Clear the cached resolved model id."""
    global _model_cache
    _model_cache = None


def _resolve_model(force: bool = False) -> str:
    """Resolve the primary model id from GET /v1/models (data[0].id), cached.

    The OpenRouter list emits the primary model first; there is no is_default
    marker, so data[0] is the documented default. ``force`` clears the cache
    first (retry path after a stale-model error).
    """
    global _model_cache
    if force:
        _clear_model_cache()
    if _model_cache is None:
        r = _request("GET", "/v1/models")
        if r.get("error"):
            raise RuntimeError(f"Could not resolve model: {r['error']}")
        data = r.get("data", [])
        if not (isinstance(data, list) and data):
            raise RuntimeError("No models available on /v1/models")
        first = data[0]
        _model_cache = first.get("id") if isinstance(first, dict) else str(first)
    return _model_cache
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m unittest acestep_mcp_server_test.TestResolveModel -v`
Expected: PASS (5).

- [ ] **Step 5: Commit**
```bash
git add mcp/acestep_mcp_server.py acestep_mcp_server_test.py
git commit -m "feat(mcp): add cached model resolution from /v1/models"
```

---

### Task 3: Multimodal message builder + src_audio/task_type rules

**Files:** Modify `mcp/acestep_mcp_server.py`; test in `acestep_mcp_server_test.py`

- [ ] **Step 1: Write the failing tests**

Append:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m unittest acestep_mcp_server_test.TestBuildMessages acestep_mcp_server_test.TestResolveTaskType -v`
Expected: FAIL (helpers not defined).

- [ ] **Step 3: Implement the helpers**

After `_resolve_model` in `mcp/acestep_mcp_server.py`, add:
```python
def _build_messages(prompt: str, src_audio: str | None) -> list:
    """Build the messages list. With src_audio -> multimodal input_audio block."""
    if src_audio:
        with open(src_audio, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        ext = os.path.splitext(src_audio)[1].lstrip(".").lower() or "mp3"
        content = [
            {"type": "text", "text": prompt},
            {"type": "input_audio", "input_audio": {"data": b64, "format": ext}},
        ]
    else:
        content = prompt
    return [{"role": "user", "content": content}]


def _resolve_task_type(task_type: str, src_audio: str | None) -> str | None:
    """Apply src_audio <-> task_type rules. Returns the task_type to use, or
    None if the combination is invalid (caller returns an error string)."""
    if src_audio:
        if task_type == "text2music":
            return "cover"  # auto-cover
        return task_type if task_type in _SRC_TASK_TYPES else None
    return task_type
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m unittest acestep_mcp_server_test.TestBuildMessages acestep_mcp_server_test.TestResolveTaskType -v`
Expected: PASS (6).

- [ ] **Step 5: Commit**
```bash
git add mcp/acestep_mcp_server.py acestep_mcp_server_test.py
git commit -m "feat(mcp): add multimodal message builder + src_audio/task_type rules"
```

---

### Task 4: Audio save-to-disk + empty-audio handling + mime extension

**Files:** Modify `mcp/acestep_mcp_server.py`; test in `acestep_mcp_server_test.py`

- [ ] **Step 1: Write the failing tests**

Append:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m unittest acestep_mcp_server_test.TestSaveAudio -v`
Expected: FAIL (`_save_audio` not defined).

- [ ] **Step 3: Implement the helper**

After `_resolve_task_type` in `mcp/acestep_mcp_server.py`, add:
```python
def _save_audio(audio_list, out_dir: str) -> list:
    """Decode base64 data URLs into files in out_dir. Returns saved paths.

    Returns [] if no decodable audio (caller treats empty as 'no audio').
    Extension is derived from the response mime via _MIME_TO_EXT (fallback mp3).
    """
    os.makedirs(out_dir, exist_ok=True)
    saved = []
    for a in audio_list or []:
        url = a.get("audio_url", {}).get("url", "") if isinstance(a, dict) else ""
        if not url.startswith("data:"):
            continue
        header, b64data = url.split(",", 1)
        mime = header.split(":")[1].split(";")[0]
        ext = _MIME_TO_EXT.get(mime, "mp3")
        path = os.path.join(out_dir, f"{uuid.uuid4().hex}.{ext}")
        with open(path, "wb") as f:
            f.write(base64.b64decode(b64data))
        saved.append(path)
    return saved
```
(`import uuid` was added in Task 1 Step 4.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m unittest acestep_mcp_server_test.TestSaveAudio -v`
Expected: PASS (4).

- [ ] **Step 5: Commit**
```bash
git add mcp/acestep_mcp_server.py acestep_mcp_server_test.py
git commit -m "feat(mcp): add audio save-to-disk with mime-derived extension"
```

---

### Task 5: Metadata parsing

**Files:** Modify `mcp/acestep_mcp_server.py`; test in `acestep_mcp_server_test.py`

- [ ] **Step 1: Write the failing tests**

Append:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m unittest acestep_mcp_server_test.TestParseMetadata -v`
Expected: FAIL (`_parse_metadata` not defined).

- [ ] **Step 3: Implement the helper**

After `_save_audio` in `mcp/acestep_mcp_server.py`, add:
```python
_META_PATTERNS = {
    "caption": r"\*\*Caption:\*\*\s*(.+)",
    "bpm": r"\*\*BPM:\*\*\s*(.+)",
    "duration": r"\*\*Duration:\*\*\s*(.+?)s",
    "key": r"\*\*Key:\*\*\s*(.+)",
    "time_signature": r"\*\*Time Signature:\*\*\s*(.+)",
}


def _parse_metadata(content: str) -> dict:
    """Best-effort parse of the ## Metadata / ## Lyrics markdown block.

    Each field is optional (server omits N/A lines). The ## Lyrics block is
    absent for instrumental tracks.
    """
    meta = {}
    if not content:
        return meta
    for key, pat in _META_PATTERNS.items():
        m = re.search(pat, content)
        if m:
            meta[key] = m.group(1).strip()
    m = re.search(r"## Lyrics\s*\n(.+)", content, re.DOTALL)
    if m:
        meta["lyrics"] = m.group(1).strip()
    return meta
```
(`import re` was added in Task 1 Step 4.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m unittest acestep_mcp_server_test.TestParseMetadata -v`
Expected: PASS (4).

- [ ] **Step 5: Commit**
```bash
git add mcp/acestep_mcp_server.py acestep_mcp_server_test.py
git commit -m "feat(mcp): add metadata markdown parser for generate_music returns"
```

---

### Task 6: Rewrite `generate_music` to orchestrate helpers (integration + retry)

**Files:** Modify `mcp/acestep_mcp_server.py`; test in `acestep_mcp_server_test.py`

- [ ] **Step 1: Write the failing integration tests (including the cache-retry path)**

Append:
```python
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
        # seq: resolve(v1) -> POST model-error -> force-resolve(v2) -> POST success
        models_v1 = {"object": "list", "data": [{"id": "acestep/old"}]}
        models_v2 = {"object": "list", "data": [{"id": "acestep/acestep-v15-turbo"}]}
        seq = [models_v1, {"error": "model acestep/old not found"}, models_v2, self._gen_resp()]
        with patch.object(server, "_request", side_effect=seq) as mr:
            out = server.generate_music(prompt="pop")
        self.assertIn("Audio saved:", out)
        self.assertEqual(mr.call_count, 4)
        self.assertEqual(mr.call_args_list[3].args[2]["model"], "acestep/acestep-v15-turbo")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m unittest acestep_mcp_server_test.TestGenerateMusic -v`
Expected: FAIL (old `generate_music` doesn't save/resolve/retry).

- [ ] **Step 3: Rewrite `generate_music`**

In `mcp/acestep_mcp_server.py`, replace the ENTIRE existing `generate_music` (signature through its return) with:
```python
@mcp.tool()
def generate_music(
    prompt: str,
    lyrics: str = "",
    duration: float | None = None,
    format: str = "mp3",
    bpm: int | None = None,
    key_scale: str | None = None,
    time_signature: str | None = None,
    vocal_language: str = "en",
    instrumental: bool = False,
    seed: int | None = None,
    task_type: str = "text2music",
    src_audio: str | None = None,
    repaint_start: float | None = None,
    repaint_end: float | None = None,
    cover_strength: float | None = None,
    guidance_scale: float | None = None,
    inference_steps: int | None = None,
    model: str | None = None,
) -> str:
    """Generate music from a text description and optional lyrics/source audio.

    Saves generated audio to disk and returns the file path plus parsed
    metadata. For cover/repaint pass src_audio (local file path) — task_type
    auto-defaults to 'cover'. task_type cover/repaint/complete require src_audio.
    """
    fmt = (format or "mp3").lower()
    if fmt not in _CHAT_FORMATS:
        return f"Invalid format '{format}'. Valid: {sorted(_CHAT_FORMATS)}"

    if instrumental and not lyrics:
        lyrics = "[inst]"

    resolved_task = _resolve_task_type(task_type, src_audio)
    if resolved_task is None:
        return f"src_audio requires a source-audio task type, got '{task_type}'."
    if src_audio and not prompt:
        return "src_audio requires a non-empty prompt."
    if not src_audio and resolved_task in {"cover", "repaint", "complete"}:
        return f"task_type '{resolved_task}' requires src_audio."

    # Capture before model is overwritten, so the retry only fires when we resolved it.
    model_resolved_here = model is None
    if model is None:
        model = _resolve_model()

    audio_config = {"format": fmt, "vocal_language": vocal_language}
    if duration is not None:
        audio_config["duration"] = duration
    if bpm is not None:
        audio_config["bpm"] = bpm
    if key_scale is not None:
        audio_config["key_scale"] = key_scale
    if time_signature is not None:
        audio_config["time_signature"] = time_signature
    if instrumental:
        audio_config["instrumental"] = True

    body = {
        "model": model,
        "messages": _build_messages(prompt, src_audio),
        "stream": False,
        "audio_config": audio_config,
        "task_type": resolved_task,
    }
    if lyrics:
        body["lyrics"] = lyrics
    if seed is not None:
        body["seed"] = seed
    if guidance_scale is not None:
        body["guidance_scale"] = guidance_scale
    if inference_steps is not None:
        body["inference_steps"] = inference_steps
    if repaint_start is not None:
        body["repainting_start"] = repaint_start
    if repaint_end is not None:
        body["repainting_end"] = repaint_end
    if cover_strength is not None:
        body["audio_cover_strength"] = cover_strength

    r = _request("POST", "/v1/chat/completions", body)

    # Retry once on a model error when we auto-resolved (cache may be stale).
    if r.get("error") and model_resolved_here and "model" in r["error"].lower():
        model = _resolve_model(force=True)
        body["model"] = model
        r = _request("POST", "/v1/chat/completions", body)

    if r.get("error"):
        return f"Generation failed: {r['error']}"

    choices = r.get("choices", [])
    if not choices:
        return f"No output. Full response: {json.dumps(r, indent=2)[:1000]}"

    msg = choices[0].get("message", {})
    content = msg.get("content", "")
    audio_list = msg.get("audio", [])

    saved = _save_audio(audio_list, OUTPUT_DIR)
    if not saved:
        return "No audio produced (the model may still be loading). Retry in a few seconds."

    meta = _parse_metadata(content)
    lines = [f"Audio saved: {p}" for p in saved]
    for key, label in (
        ("caption", "Caption"), ("bpm", "BPM"), ("duration", "Duration"),
        ("key", "Key"), ("time_signature", "Time signature"), ("lyrics", "Lyrics"),
    ):
        if meta.get(key):
            lines.append(f"{label}: {meta[key]}")
    if seed is not None:
        lines.append(f"Seed: {seed}")
    return "\n".join(lines)
```

- [ ] **Step 4: Run the full MCP test suite**

Run: `python -m unittest acestep_mcp_server_test -v`
Expected: PASS (import 1 + resolve 5 + build/task 6 + save 4 + parse 4 + generate 7 = 27).

- [ ] **Step 5: Commit**
```bash
git add mcp/acestep_mcp_server.py acestep_mcp_server_test.py
git commit -m "feat(mcp): rewrite generate_music — save audio, model resolution, cover/repaint, retry"
```

---

## Part B — Docs (single-source-of-truth)

### Task 7: llms-full.txt canonical fixes

**Files:** Modify `llms-full.txt`

- [ ] **Step 1: Fix the `/v1/models` response example (C10)**

In the `### List Models` section, make `data` a flat list with no `default_model`/nested `models`:
```
GET /v1/models
→ {"object":"list","data":[{"id":"acestep/acestep-v15-turbo","name":"ACE-Step ...","created":...,"input_modalities":["text","audio"],"output_modalities":["audio","text"],"pricing":{...}}]}
```
Add: "Note: `data` is a flat list; there is no `default_model` field. The primary model is first."

- [ ] **Step 2: Split the field reference into two endpoint-specific tables**

Replace the merged field list with two labeled tables:

**Table A — `/release_task` (native):** `audio_duration` (float), `audio_format` (default `flac`; accepts `mp3/flac/wav/opus/aac/wav32`, NOT ogg/m4a), `bpm`, `key_scale`, `time_signature`, `vocal_language`, `src_audio_path`, `task_type`, `repainting_start`/`repainting_end`, `audio_cover_strength`, `batch_size` (default `2`), `lm_cfg_scale` (runtime default `2.0`).

**Table B — `/v1/chat/completions` (OpenRouter):** `audio_config.duration`, `audio_config.format` (default `mp3`; accepts `mp3/wav/flac/ogg/m4a/aac`, NOT opus/wav32), `audio_config.{bpm,key_scale,time_signature,vocal_language,instrumental}`, top-level `lyrics`, `task_type`, `repainting_start`/`repainting_end`, `audio_cover_strength`, `seed`, `guidance_scale`, `inference_steps`.

- [ ] **Step 3: Fix lm_cfg_scale default**

Wherever `lm_cfg_scale` appears, state runtime default `2.0` (not `2.5`).

- [ ] **Step 4: Add a cover/repaint worked example**

After the chat-completions section:
```
### Cover / Repaint (chat-completions, source audio inline)

POST /v1/chat/completions
{
  "model": "acestep/acestep-v15-turbo",
  "messages": [{"role":"user","content":[
    {"type":"text","text":"jazz cover"},
    {"type":"input_audio","input_audio":{"data":"<base64>","format":"mp3"}}
  ]}],
  "task_type": "cover",
  "audio_cover_strength": 0.8,
  "audio_config": {"duration": 120}
}
Repaint a region: set "task_type":"repaint", "repainting_start":30, "repainting_end":90.
```
For `/release_task`, source audio is passed via `src_audio_path` (or multipart `src_audio`).

- [ ] **Step 5: Add Integration Patterns section at the very top (after the title)**

```
## Integration Patterns

- **MCP-first (preferred for MCP-capable agents):** run `mcp/acestep_mcp_server.py`.
  Tools: generate_music, enhance_prompt, list_models, check_health, get_examples.
  See AGENTS.md for config. generate_music saves audio to disk and returns the path.
- **HTTP fallback:** POST /v1/chat/completions (sync, audio as base64) — primary.
  For native/long jobs: /release_task + /query_result (async, files saved server-side).
```

- [ ] **Step 6: Verify it still serves**

Run: `curl -s http://localhost:8010/llms-full.txt | head -8`
Expected: title + new `## Integration Patterns` heading.

- [ ] **Step 7: Commit**
```bash
git add llms-full.txt
git commit -m "docs(llms-full): fix /v1/models shape, split endpoint tables, cover example, integration patterns"
```

---

### Task 8: llms.txt thin summary (delegate, no normative tables)

**Files:** Modify `llms.txt`

- [ ] **Step 1: Rewrite llms.txt as a thin summary**

Replace the body with this skeleton (keep title + one-line description):
```
# ACE-Step 1.5 — AI Music Generator

> Text + optional lyrics/source audio → full songs (vocals + instruments).

## Integration
- **MCP (preferred for MCP-capable agents):** `mcp/acestep_mcp_server.py` —
  tools generate_music / enhance_prompt / list_models / check_health / get_examples.
  See AGENTS.md. generate_music saves audio to disk and returns the path.
- **HTTP:** POST /v1/chat/completions (sync, base64 audio) — primary HTTP path.

## Quick examples
GET /health            → {"data":{"status":"ok"},...}
POST /v1/chat/completions  {"model":"acestep/acestep-v15-turbo",
        "messages":[{"role":"user","content":"chill lo-fi beat"}],"stream":false}

## Model
`acestep/acestep-v15-turbo` (namespaced for chat-completions; bare for /release_task).
GET /v1/models returns a flat data[] list (no default_model).

## Full reference
All parameters, defaults, per-endpoint differences, and task types:
see llms-full.txt (served at /llms-full.txt).
```
Keep the `/v1/models` shape consistent with llms-full.txt (flat `data[]`).

- [ ] **Step 2: Verify it serves + is shorter**

Run: `curl -s http://localhost:8010/llms.txt | wc -l`
Expected: notably fewer lines than before (was ~110).

- [ ] **Step 3: Commit**
```bash
git add llms.txt
git commit -m "docs(llms): trim to thin summary delegating field reference to llms-full.txt"
```

---

### Task 9: AGENTS.md MCP section + env vars + drift-checklist rule

**Files:** Modify `AGENTS.md`

- [ ] **Step 1: Fix the MCP tools table (4 → 5 tools)**

In `### MCP Tools`, add a row:
```
| `get_examples` | Get example music generation parameters (simple or full) from the sample pool |
```

- [ ] **Step 2: Document the new env vars**

In the MCP config `env` block area, add (or note alongside):
```
Additional MCP env vars:
- `ACESTEP_OUTPUT_DIR` — where generated audio is saved (default `./acestep_output`, relative to the MCP process CWD).
- `ACESTEP_REQUEST_TIMEOUT` — per-request timeout in seconds (default `650`, kept above the server's 600s).
```

- [ ] **Step 3: Mention Claude Code + Codex; point to llms-full.txt**

Rename `### Codex MCP Configuration` → `### MCP Configuration (Claude Code / Codex)`; add that Claude Code uses the same JSON in `.claude/settings.json` `mcpServers` (or `claude mcp add`). After the tools table add: "Full parameter reference and per-endpoint differences: see `llms-full.txt` (served at `/llms-full.txt`)."

- [ ] **Step 4: Add the drift-checklist rule**

In `## PR Readiness Checklist`, add:
```
- [ ] If you added/removed/renamed an `@mcp.tool()` in `mcp/acestep_mcp_server.py`, the AGENTS.md MCP Tools table and `acestep/api/agent_discovery_route.py` `mcp_server.tools` were updated in the same commit.
```

- [ ] **Step 5: Add a cover example line**

After the existing usage example, add:
```
generate_music(prompt="jazz cover", src_audio="song.mp3")   # task_type auto=cover
```

- [ ] **Step 6: Commit**
```bash
git add AGENTS.md
git commit -m "docs(agents): 5-tool MCP table, env vars, Claude Code config, drift-checklist rule"
```

---

### Task 10: hermes-skill SKILL.md — endpoint-scoped + fix model id

**Files:** Modify `mcp/hermes-skill/SKILL.md`

- [ ] **Step 1: Add endpoint-scope header note**

After the `# ACE-Step Music Generation` title, add:
```
> This skill targets `/v1/chat/completions` only. Field shapes (`audio_config.*`)
> and format options differ from `/release_task` — see `llms-full.txt`.
```

- [ ] **Step 2: Replace the three wrong model ids**

Replace all three occurrences of `acestep/acestep-v15-chinese-lyric` (in the three curl `-d` bodies) with `acestep/acestep-v15-turbo`.

- [ ] **Step 3: Align the Parameters Reference table to chat-completions ground truth**

Set the `format` row to: `mp3, wav, flac, ogg, m4a, aac (chat-completions)`. Add rows for `time_signature` and `task_type` (text2music/cover/repaint/complete); note source audio is passed as a multimodal `input_audio` content part, not a body field.

- [ ] **Step 4: Add a cover example block**

Under `## Task Types`:
```
### Cover (source audio inline)
curl -X POST {{API_URL}}/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"acestep/acestep-v15-turbo","messages":[{"role":"user","content":[
    {"type":"text","text":"jazz cover"},
    {"type":"input_audio","input_audio":{"data":"<base64-of-src.mp3>","format":"mp3"}}]}],
    "task_type":"cover","audio_cover_strength":0.8}'
```

- [ ] **Step 5: Commit**
```bash
git add mcp/hermes-skill/SKILL.md
git commit -m "docs(hermes): endpoint-scope note, fix model id, cover example, aligned param table"
```

---

### Task 11: Discovery route tools list 4→5 + restart acestep-rocm

**Files:** Modify `acestep/api/agent_discovery_route.py`

- [ ] **Step 1: Update the hardcoded tools list**

In the `mcp_server` dict, make `tools` read:
```python
"tools": [
    "generate_music",
    "list_models",
    "enhance_prompt",
    "check_health",
    "get_examples",
],
```

- [ ] **Step 2: Restart the API server in the acestep-rocm container**

The 8010 server is `python -m acestep.api_server --host 0.0.0.0 --port 8010 --api-key acestep-rocm` running inside `acestep-rocm` (a manually-launched long process, NOT the container entrypoint). `/home/vinsun` is bind-mounted in, so the edit is visible. First confirm the live process, then restart:
```bash
# Confirm current launch (sanity):
podman exec acestep-rocm bash -lc 'ps -eo pid,args | grep "acestep.api_server" | grep -v grep'
# Restart it:
podman exec -d acestep-rocm bash -lc 'pkill -f "acestep.api_server"; sleep 3; cd /home/vinsun/project/ACE-Step-1.5 && nohup python -m acestep.api_server --host 0.0.0.0 --port 8010 --api-key acestep-rocm >/tmp/acestep_api.log 2>&1 &'
```
**Note:** the executor must confirm this matches how the server was originally launched (if it was started differently, match that). The user has confirmed restart is safe (testing, no users).

- [ ] **Step 3: Poll until re-initialized, then verify the tools list**

```bash
for i in $(seq 1 60); do
  curl -s http://localhost:8010/health | grep -q '"models_initialized":true' && break
  sleep 2
done
curl -s http://localhost:8010/health   # expect models_initialized:true, loaded_model:acestep-v15-turbo
curl -s http://localhost:8010/.well-known/agent | python3 -c 'import sys,json;print(json.load(sys.stdin)["mcp_server"]["tools"])'
```
Expected: `['generate_music', 'list_models', 'enhance_prompt', 'check_health', 'get_examples']`.

- [ ] **Step 4: Commit**
```bash
git add acestep/api/agent_discovery_route.py
git commit -m "fix(api): add get_examples to agent discovery mcp_server.tools"
```

---

### Task 12: Discovery-sync test (generic drift guard)

**Files:** Modify `acestep/api/agent_discovery_route_test.py`

- [ ] **Step 1: Read the existing test file to match its style**

Run: `sed -n '1,70p' acestep/api/agent_discovery_route_test.py` — match existing imports/test-class style.

- [ ] **Step 2: Add a test that parses tool names generically (no hardcoded names)**

Add a test (adapting to the file's style) that compares the `@mcp.tool()` function names in `mcp/acestep_mcp_server.py` against the `tools` list literal in `agent_discovery_route.py`, both parsed without pre-baking names:
```python
def test_discovery_tools_match_mcp_file(self):
    import re
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    srv_src = open(os.path.join(project_root, "mcp", "acestep_mcp_server.py")).read()
    route_src = open(os.path.join(project_root, "acestep", "api", "agent_discovery_route.py")).read()
    defined = set(re.findall(r"@mcp\.tool\(\)\s*\ndef\s+(\w+)", srv_src))
    m = re.search(r'"tools":\s*\[(.*?)\]', route_src, re.DOTALL)
    self.assertIsNotNone(m, "tools list not found in discovery route")
    listed = set(re.findall(r'"([^"]+)"', m.group(1)))
    self.assertEqual(defined, listed, f"discovery tools {listed} != MCP tools {defined}")
```
This fails on both renames AND count drift (a new tool added to the MCP file but not the route → `defined ≠ listed`).

- [ ] **Step 3: Run the discovery test suite**

Run: `python -m unittest acestep.api.agent_discovery_route_test -v`
Expected: PASS (existing tests + new sync test). Note: the existing tests run against the route module directly (no live server needed); the new test only reads source files.

- [ ] **Step 4: Commit**
```bash
git add acestep/api/agent_discovery_route_test.py
git commit -m "test(api): assert discovery tools list stays in sync with MCP @mcp.tool() functions"
```

---

## Part C — Verification

### Task 13: End-to-end live verification + final doc consistency check

**Files:** none (verification only)

- [ ] **Step 1: Full MCP test suite green**

Run: `python -m unittest acestep_mcp_server_test -v`
Expected: all PASS (27 tests).

- [ ] **Step 2: Live text2music through the actual MCP tool**

Confirm server up: `curl -s http://localhost:8010/health`. Then:
```bash
python3 -c "
import importlib.util, os, sys
spec = importlib.util.spec_from_file_location('s', os.path.join('mcp','acestep_mcp_server.py'))
s = importlib.util.module_from_spec(spec); sys.modules['s']=s; spec.loader.exec_module(s)
s.OUTPUT_DIR = '/tmp/acestep_e2e'
print(s.generate_music(prompt='chill lo-fi hip hop beat', lyrics='[inst]', duration=10))
print('files:', os.listdir('/tmp/acestep_e2e'))
"
```
Expected: `Audio saved: /tmp/acestep_e2e/<hex>.mp3`; dir lists a `.mp3`; no `base64,` in output; any model referenced is `acestep-v15-turbo`.

- [ ] **Step 3: Live cover via src_audio**

```bash
python3 -c "
import importlib.util, os, sys, glob
spec = importlib.util.spec_from_file_location('s', os.path.join('mcp','acestep_mcp_server.py'))
s = importlib.util.module_from_spec(spec); sys.modules['s']=s; spec.loader.exec_module(s)
s.OUTPUT_DIR = '/tmp/acestep_e2e'
src = glob.glob('acestep_output/*.mp3')[0]
print(s.generate_music(prompt='rock cover', src_audio=src, duration=15))
"
```
Expected: `Audio saved:` + a file; task auto-set to cover (no error).

- [ ] **Step 4: Doc consistency cross-check**

```bash
echo "--- stale model id in MCP/docs (expect none) ---"
grep -rn "chinese-lyric" mcp/ llms.txt AGENTS.md || echo "none"
echo "--- discovery tool count (expect 5) ---"
curl -s http://localhost:8010/.well-known/agent | python3 -c 'import sys,json;print(len(json.load(sys.stdin)["mcp_server"]["tools"]))'
```
Expected: no `chinese-lyric` in MCP/docs; discovery reports 5 tools.

- [ ] **Step 5: Final commit (if verification surfaced fixes)**
```bash
git status   # clean → nothing to commit; else commit the fix
```

---

## Self-Review (completed during authoring, after reviewer pass)

- **Spec coverage:** Part 1 → Tasks 1-6 (incl. cache-retry test P1-1). Part 2 → Tasks 7-10 (+ `_CAPTION_GUIDE` rule comment in Task 1 Step 5 = P1-3; env vars in Task 9 = P1-4). Part 3 → Tasks 11-12 (generic drift guard = P1-2). Part 4 tests → Tasks 1-6 + 12. Success criteria → Task 13.
- **Reviewer P0s fixed:** P0-1 (test at repo root + importlib loader, NOT under mcp/); P0-2 (pinned acestep-rocm restart + poll). P1-1 (retry test), P1-2 (generic parse), P1-3 (caption-guide rule), P1-4 (env docs), P1-5 (explicit import edit) all addressed.
- **Placeholder scan:** all steps have concrete code / exact edits / exact commands.
- **Type/name consistency:** helper + constant names are identical across tasks; `_request(method, path, body)` call-arg indexing (`call_args_list[n].args[2]` = body) verified against the side_effect sequences.
