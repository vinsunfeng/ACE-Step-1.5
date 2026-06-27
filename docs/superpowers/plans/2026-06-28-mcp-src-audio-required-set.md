# MCP src_audio-required set — Implementation Plan

> **For agentic workers:** Small follow-up; inline TDD execution (one logical change). Steps use `- [ ]`.

**Goal:** MCP `generate_music` requires `src_audio` for all six source-routing task types (currently only cover/repaint/complete — missing cover-nofsq/lego/extract; complete correctly stays).

**Spec:** `docs/superpowers/specs/2026-06-28-mcp-src-audio-required-set-design.md`

**Key insight (from debate):** the MCP has no `audio_code_string` surface, so `src_audio` is the only source path → require it for every task type the MCP routes source audio to = `_SRC_TASK_TYPES`. Reuse that constant; no new constant.

---

### Task 1: Red — add the looping test + accept-set assertion

In `acestep_mcp_server_test.py`, add to `TestGenerateMusic` (uses the class's existing `_MODELS` / `_gen_resp` helpers):
```python
def test_all_src_task_types_require_src_audio(self):
    def fake(method, path, body=None):
        return self._MODELS if path == "/v1/models" else self._gen_resp()
    for t in sorted(server._SRC_TASK_TYPES):
        with self.subTest(task_type=t):
            with patch.object(server, "_request", side_effect=fake):
                out = server.generate_music(prompt="x", task_type=t)
            self.assertIn("requires src_audio", out, f"{t} should require src_audio")
```
And in `TestResolveTaskType`, add (pins the accept-set):
```python
def test_complete_with_src_audio_is_accepted(self):
    self.assertEqual(server._resolve_task_type("complete", "x.mp3"), "complete")
```
Run: `python -m unittest acestep_mcp_server_test -v` → expect **3 failures** (cover-nofsq, lego, extract proceed instead of erroring); the other 3 + the new accept test pass.

- [ ] add tests, run, confirm 3 red.

### Task 2: Green — fix the guard + docstring

In `mcp/acestep_mcp_server.py`:
- Line ~320: `if not src_audio and resolved_task in {"cover", "repaint", "complete"}:` → `if not src_audio and resolved_task in _SRC_TASK_TYPES:`
- Docstring (~306): `task_type cover/repaint/complete require src_audio.` → `task_type cover/cover-nofsq/repaint/lego/extract/complete require src_audio (cover-nofsq/lego/extract are server-side types surfaced for completeness).`

Run: `python -m unittest acestep_mcp_server_test -v` → **all pass** (32).

- [ ] apply fix, run, confirm green.

### Task 3: Doc sync (single-source-of-truth)

- `llms-full.txt` (~96): "Source audio file path (cover/repaint/complete/lego)" → add `cover-nofsq`, `extract` → full six.
- `mcp/hermes-skill/SKILL.md`: task_type table (~110, ~148-156) → add `cover-nofsq`/`lego`/`extract`.

- [ ] sync both docs to all six.

### Task 4: Verify + commit

- [ ] `python -m unittest acestep_mcp_server_test -v` all pass.
- [ ] `grep -rn "cover-nofsq" llms-full.txt mcp/hermes-skill/SKILL.md mcp/acestep_mcp_server.py` → present in all three.
- [ ] One commit: `fix(mcp): require src_audio for all source-routing task types (cover-nofsq/lego/extract)`
