# Design: Align MCP src_audio-required set with full routing set

- **Date:** 2026-06-28
- **Status:** Approved, pending implementation
- **Scope:** `mcp/acestep_mcp_server.py`, `acestep_mcp_server_test.py`, `llms-full.txt`, `mcp/hermes-skill/SKILL.md`

## 1. Background

Follow-up to the 2026-06-27 MCP reliability work. The final review flagged an asymmetry: the MCP `generate_music` guard (line ~320) rejects no-src_audio calls for an inline set `{"cover","repaint","complete"}`, but the server routes source audio for **six** task types (`_SRC_AUDIO_TASK_TYPES = {cover, cover-nofsq, repaint, lego, extract, complete}`). So `cover-nofsq`/`lego`/`extract` slip past the MCP guard without source audio and only fail (or misbehave) at the server.

An adversarial debate refined the fix direction and **overturned an initial proposal**:

- The server's `_src_audio_required_tasks = {cover, cover-nofsq, repaint, lego, extract}` (`generate_music_request.py:130`) is a **file-only** gate — checked only when `src_audio is None`. For those five, the server also accepts `audio_code_string` as a substitute source.
- **The MCP does NOT expose `audio_code_string`** (grep: zero matches). So via the MCP surface, `src_audio` is the **only** way to supply source content.
- `complete` is NOT in the server's 5-set, so a source-less `complete` does **not** error — but it runs a "complete the input track" generation against a **silent** target (`padding_utils.py:73-88`), producing a meaningless "completion of silence" that is still saved to disk. The MCP's current inclusion of `complete` in its guard therefore **correctly blocks** this garbage path.

**Conclusion:** for the MCP surface, **all six** src-routing task types should require `src_audio` (there is no codes escape hatch). The required set equals the routing set. The bug is only the three missing types; `complete` must stay.

## 2. Goal

Make the MCP reject no-src_audio calls for the **full** routing set `{cover, cover-nofsq, repaint, lego, extract, complete}`, matching the tasks it can actually route source audio to. Keep the single-source-of-truth doc invariant from the prior round.

## 3. Non-Goals

- Exposing `audio_code_string` in the MCP (out of scope; would make the requirement conditional).
- Querying the server's required-set dynamically (YAGNI; the set is a stable literal).
- Refactoring the server's function-local `_src_audio_required_tasks` to module level.
- Changing `_resolve_task_type` or `_SRC_TASK_TYPES` (the routing/accept set is correct as-is).

## 4. Design

### Change (one line + docstring)

In `mcp/acestep_mcp_server.py`:

- **Line ~320** — replace the inline literal with the existing routing constant:
  ```python
  # before:
  if not src_audio and resolved_task in {"cover", "repaint", "complete"}:
  # after:
  if not src_audio and resolved_task in _SRC_TASK_TYPES:
  ```
  Rationale: the MCP requires `src_audio` for exactly the tasks it routes source audio to (all six). No new constant, no duplication of the server's set.
- **Docstring (line ~306)** — update from "task_type cover/repaint/complete require src_audio" to list the full set: "task_type cover/cover-nofsq/repaint/lego/extract/complete require src_audio (cover-nofsq/lego/extract are server-side types; this tool surfaces them for completeness)."

`_SRC_TASK_TYPES` (line 18) is unchanged. `_resolve_task_type` is unchanged.

### Tests (`acestep_mcp_server_test.py`)

- Add a `subTest` loop over `server._SRC_TASK_TYPES` asserting each task type, when called with no `src_audio`, returns a "requires src_audio" error. This is self-updating: if the set changes, the test tracks it. (Mocks `_request` so no network.)
- Keep the existing `test_cover_without_src_audio_errors` (now subsumed by the loop, but harmless to keep or remove).
- Keep `test_src_audio_auto_cover` and the accept-set coverage in `TestResolveTaskType`.

### Doc sync (single-source-of-truth)

- `llms-full.txt` (~line 96): the "Source audio file path (cover/repaint/complete/lego)" line omits `cover-nofsq` and `extract` — update to the full six.
- `mcp/hermes-skill/SKILL.md`: the task_type table (~line 110, ~148-156) lists only text2music/cover/repaint/complete — add `cover-nofsq`/`lego`/`extract` to align with the chat-completions endpoint (which accepts all six).
- The MCP docstring change (above) is the third contract-stating spot.

### Commit

One commit: `fix(mcp): require src_audio for all source-routing task types (cover-nofsq/lego/extract)`.

## 5. Verified Facts (load-bearing)

- Server routing set `_SRC_AUDIO_TASK_TYPES = {cover, cover-nofsq, repaint, lego, extract, complete}` (`openrouter_adapter.py:740`).
- Server required (file-only) set `_src_audio_required_tasks = {cover, cover-nofsq, repaint, lego, extract}` (`generate_music_request.py:130`) — `complete` absent.
- MCP has no `audio_code_string` surface → `src_audio` is the only source path.
- Source-less `complete` → silent-target generation (`padding_utils.py:73-88`), not an error → MCP blocking it is correct.

## 6. Success Criteria

- `generate_music(prompt="x", task_type=t)` with no `src_audio` returns a "requires src_audio" error for **every** `t ∈ _SRC_TASK_TYPES` (all six), verified by a looping test.
- `complete` with `src_audio` still proceeds (accept-set unchanged).
- The 31 existing MCP tests still pass; the suite grows by the new looping test.
- `llms-full.txt` and `hermes-skill/SKILL.md` list all six src-routing task types (no drift vs. the MCP guard).
