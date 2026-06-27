# Design: MCP Reliability Fix + Doc Single-Source-of-Truth

- **Date:** 2026-06-27
- **Status:** Approved (design), revised after reviewer pass, pending implementation plan
- **Scope:** `mcp/acestep_mcp_server.py`, `llms.txt`, `llms-full.txt`, `AGENTS.md`, `mcp/hermes-skill/SKILL.md`, `acestep/api/agent_discovery_route.py`, new test file

## 1. Background & Problem

The project ships two agent-integration layers on top of the ACE-Step API server (running at `localhost:8010`):

- An **MCP server** (`mcp/acestep_mcp_server.py`) exposing 5 tools.
- **Agent docs** (`llms.txt`, `llms-full.txt`) served via `/.well-known/agent` discovery, plus `AGENTS.md` and `mcp/hermes-skill/SKILL.md`.

Three problems, confirmed by code inspection, live probing, and two rounds of adversarial review:

1. **The MCP `generate_music` is a "semi-finished" tool.** It hardcodes a model (`acestep/acestep-v15-chinese-lyric`) that **does not exist** on this server (only `acestep-v15-turbo` is loaded — confirmed via `/v1/models` and `/health`). It declares `task_type=cover/repaint` but exposes **no way to pass source audio or a repaint window**, and `task_type` defaults to `text2music`, which routes any source audio to `reference_audio` instead of `src_audio` (silent wrong-mode). It does not save audio — it returns a text description of a base64 blob it already discarded.
2. **`batch_size` would silently lose data** — completion mode encodes only `raw_audio_paths[0]`, so batch > 1 drops audio.
3. **The docs contradict each other and the API.** Verified bugs: `/v1/models` response shape documented wrong (real is a flat `data[]` list, no `default_model`); format options disagree across four files (`ogg` claimed where the native endpoint rejects it); default `audio_format` / `lm_cfg_scale` defaults wrong; the AGENTS.md tool table and the discovery route's `mcp_server.tools` list are both stale (missing `get_examples`).

## 2. Goals

- Make `generate_music` actually usable end-to-end: correct model, working cover/repaint, audio saved to disk with a returned path + metadata.
- Eliminate doc contradictions via a **single-source-of-truth** structure (reconcile in place + delegate), not a mapping-table append.
- Add the tests AGENTS.md mandates for behavior changes.
- Guard against the tool-list drift that already happened twice.

## 3. Non-Goals (explicitly out of scope)

- Async / native mode (`/release_task` + `/query_result`) for long songs. YAGNI.
- Streaming/SSE. `stream: false` stays.
- `batch_size` support. Dropped (silent data loss in completion mode).
- Advanced repaint knobs (`repaint_mode`, `repaint_strength`, `cover_noise_strength`), `thinking`, `use_format`, `sample_mode`, `shift`, `infer_method`. Rely on server defaults.
- The bash skill (`.claude/skills/acestep/SKILL.md` + `scripts/acestep.sh`) — a separate curl client that does not use MCP.
- Rewriting `_CAPTION_GUIDE` or refactoring `_request` (beyond a timeout bump). Touching `enhance_prompt`, `list_models`, `check_health`, `get_examples`.

## 4. Verified API Facts (load-bearing — implementer must match these)

### Completion mode `POST /v1/chat/completions` (`acestep/openrouter_adapter.py`, `acestep/openrouter_models.py`)

- `model` (`str`, **required**) — namespaced, e.g. `acestep/acestep-v15-turbo`.
- `messages` (required). `content` may be a `str` **or** a list of parts. For source-audio tasks:
  ```
  [{"type":"text","text": <prompt>},
   {"type":"input_audio","input_audio":{"data": <base64>, "format": <ext>}}]
  ```
  Routing (`openrouter_adapter.py` ~740): for `task_type ∈ {cover, cover-nofsq, repaint, lego, extract, complete}`, `audio[0]` → `src_audio`. For `text2music`, `audio[0]` → `reference_audio` (the wrong target for cover/repaint — hence the auto-cover rule in §5). Only the **last** user message is read.
- `audio_config`: `duration`, `format`, `bpm`, `key_scale`, `time_signature`, `vocal_language`, `instrumental`.
- Top-level: `lyrics` (str), `task_type`, `seed`, `guidance_scale`, `inference_steps`.
- Cover/repaint fields (exact names, defaults): `task_type`, `repainting_start` (float, default `0.0`), `repainting_end` (float, default `None` → interpreted server-side as "to end of audio"), `audio_cover_strength` (float, default `1.0`).

### Response shape (`_build_openrouter_response` + `_format_lm_content`)

- `choices[0].message.content` — markdown. Exact construction (`_format_lm_content`, `openrouter_adapter.py:92-123`):
  - A `## Metadata` block (only if any metadata line present), each line omitted when the value is `"N/A"`:
    - `**Caption:** {metas.prompt or metas.caption or result.prompt}`
    - `**BPM:** {bpm}`
    - `**Duration:** {duration}s`
    - `**Key:** {keyscale}`
    - `**Time Signature:** {timesignature}`
  - A `## Lyrics\n{lyrics}` block — **omitted entirely** when lyrics are absent or `[inst]`/`[instrumental]`.
  - If no parts: returns the literal `"Music generated successfully."`
- `choices[0].message.audio` — list of `{"type":"audio_url","audio_url":{"url":"data:audio/<mime>;base64,..."}}`. **Only the first** generated audio is encoded even for batch. May be `None`/empty if the file is missing (e.g. model still loading on first request).
- `seed` is **NOT present** anywhere in the response (`ResponseMessage` has `content`, `audio`, `audio_codes` only).
- The data-URL mime is one of (`_audio_to_base64_url`, `openrouter_adapter.py:75-82`): `audio/mpeg` (mp3), `audio/wav`, `audio/flac`, `audio/ogg`, `audio/mp4` (m4a), `audio/aac`. File extension must be derived via the **inverse** of this map, not the request `format`.

### `GET /v1/models`

- `{"object":"list","data":[{"id":"acestep/acestep-v15-turbo","name":...,"created":...,"input_modalities":...,"output_modalities":...,"pricing":{...},...}]}` — a **flat list** under `data`. There is **no** `default_model` key, **no** nested `models` key, and **no** `is_default` field. The primary model is emitted **first** (`openrouter_adapter.py` `list_models` emits primary, then optional secondary/tertiary). On this server: exactly one model `acestep/acestep-v15-turbo`.

### `GET /health`

- Returns `{"data":{"status":"ok","loaded_model":"acestep-v15-turbo",...}}`. `loaded_model` is the bare (un-namespaced) primary model name — a useful cross-check but not the field the MCP should use for `model`.

### Format options (per endpoint, ground truth)

- Native `/release_task` `audio_format`: `mp3, flac, wav, opus, aac, wav32` (default `flac` on the legacy handler).
- Chat-completions (`audio_config.format`): accepts `mp3, wav, flac, ogg, m4a, aac` via the mime map (default `mp3`).

### Discovery (`acestep/api/agent_discovery_route.py`)

- Serves `/.well-known/agent`, `/llms.txt`, `/llms-full.txt`. The two `.txt` files are **read from repo root at request time** (edits picked up without restart). The `/.well-known/agent` JSON body (including `mcp_server.tools`) is **hardcoded Python** (restart needed to change it).

## 5. Design

### Part 1 — `generate_music` reliability fix

**Signature** (new optional params marked `# NEW`; all backward-compatible):

```python
def generate_music(
    prompt: str,
    lyrics: str = "",
    duration: float | None = None,
    format: str = "mp3",
    bpm: int | None = None,
    key_scale: str | None = None,
    time_signature: str | None = None,   # NEW -> audio_config.time_signature
    vocal_language: str = "en",
    instrumental: bool = False,
    seed: int | None = None,
    task_type: str = "text2music",
    src_audio: str | None = None,        # NEW: local file path -> base64 multimodal
    repaint_start: float | None = None,  # NEW -> repainting_start
    repaint_end: float | None = None,    # NEW -> repainting_end (None = to-end)
    cover_strength: float | None = None, # NEW -> audio_cover_strength
    guidance_scale: float | None = None,
    inference_steps: int | None = None,
    model: str | None = None,            # NEW; None -> resolve from /v1/models
) -> str:
```

**Behavior rules:**

1. **Model.** Remove the hardcoded `acestep/acestep-v15-chinese-lyric`. If `model` is `None`, resolve from `GET /v1/models` → `data[0].id` (the primary/first-loaded model; there is no `is_default` marker, so `data[0]` is the documented choice for the primary default — valid because `list_models` emits primary first). **Cache per-process.** On any generation response that indicates a model problem, **clear the cache and retry resolution once** before surfacing the error. If `/v1/models` is empty/unreachable, raise a clear error. Always send `model` in the body (the field is required server-side).
2. **src_audio ↔ task_type interaction (prevents silent wrong-mode).** Define the source-audio task set `SRC = {cover, cover-nofsq, repaint, lego, extract, complete}`. Rules:
   - If `src_audio` is set and `task_type` is still the default `"text2music"` → **auto-set `task_type="cover"`**.
   - If `src_audio` is set and `task_type` is explicitly a non-SRC type → **raise an error** ("src_audio requires a source-audio task type").
   - If `task_type ∈ {cover, repaint, complete}` (and not auto-set) and `src_audio` is unset → **raise an error** ("this task type requires src_audio").
3. **Source audio body.** When `src_audio` is provided: read the file, base64-encode it, build `messages[0].content` as `[{"type":"text","text":prompt}, {"type":"input_audio","input_audio":{"data":<b64>, "format":<ext without dot>}}]`. Send `repainting_start`/`repainting_end`/`audio_cover_strength` when the corresponding params are set. **Require a non-empty `prompt`** when `src_audio` is set.
4. **Audio output.** After the response, decode each `choices[0].message.audio[*].audio_url.url`:
   - **Empty/None branch:** if no decodable data URL is present, return `"No audio produced (the model may still be loading). Retry in a few seconds."` **without writing a file** — distinct from the error path; not a success.
   - Otherwise, derive the extension from the response mime via the inverse map (`audio/mpeg→mp3`, `audio/wav→wav`, `audio/flac→flac`, `audio/ogg→ogg`, `audio/mp4→m4a`, `audio/aac→aac`; fallback to the request `format`). Write to `{ACESTEP_OUTPUT_DIR or ./acestep_output}/<uuid>.<ext>` where `./acestep_output` is **relative to the MCP process CWD**; `os.makedirs(..., exist_ok=True)` before writing. Filename uuid via `uuid.uuid4()`.
5. **Metadata in return.** Best-effort parse `choices[0].message.content`:
   - `## Metadata` block: regex `\*\*Caption:\*\*\s*(.+)`, `\*\*BPM:\*\*\s*(.+)`, `\*\*Duration:\*\*\s*(.+?)s`, `\*\*Key:\*\*\s*(.+)`, `\*\*Time Signature:\*\*\s*(.+)` (each optional; absent → omitted).
   - `## Lyrics` block may be absent (instrumental) — handle gracefully.
   - Return a compact text block: saved path(s) + parsed metadata. **Echo `seed` only if the caller supplied one** (for agent traceability — it is not recoverable from the response).
   - **Never re-embed base64** in the return.
6. **format validation.** If `format` is not in the chat-completions-accepted set `{mp3, wav, flac, ogg, m4a, aac}`, **raise an error listing the valid set** (no silent substitution).
7. **Timeout.** Raise the `_request` timeout from 600s to **650s** (strictly greater than the server's 600s `GENERATION_TIMEOUT`) to avoid the client timing out first and leaving a completed job unfetched. Optionally overridable via `ACESTEP_REQUEST_TIMEOUT` env.

Reused as-is: `_request`, `_headers`. Keep `stream: False`.

### Part 2 — Doc single-source-of-truth (reconcile in place + delegate)

- **`llms-full.txt` (canonical).** Fix in place:
  - `/v1/models` example → flat `data[]` list, no `default_model`, no nested `models`.
  - Two **endpoint-specific** field tables (`/release_task` vs `/v1/chat/completions`) so `audio_duration` vs `audio_config.duration`, `batch_size` (2 vs 1), `audio_format` default (`flac` vs `mp3`), and format-option sets are explicit, not contradictory.
  - `lm_cfg_scale` runtime default `2.0`.
  - Add a cover/repaint worked example (chat-completions base64 + release_task path).
  - Add an **Integration Patterns** section at the top: MCP-first (name the server + the 5 tools), HTTP fallback (completion primary; release_task for native/long).
- **`llms.txt` (thin summary).** Health check + one generate example per endpoint + model id + MCP pointer. **No normative field tables** — every canonical value delegates to `llms-full.txt`. (JSON examples are *illustrative* and may contain field names; that is acceptable as long as they are not presented as the authoritative field reference.)
- **`AGENTS.md` (MCP section only).** Tool table → 5 tools including `get_examples`. Mention Claude Code **and** Codex. Point to `llms-full.txt` for full params. Add to the PR-readiness checklist: *"If you add/remove/rename an `@mcp.tool()` in `mcp/acestep_mcp_server.py`, update this table and the discovery route's `mcp_server.tools` in the same commit."*
- **`mcp/hermes-skill/SKILL.md`.** Align the param table to **its own endpoint** (`/v1/chat/completions`, where `ogg` is valid). Add a header note: "Targets `/v1/chat/completions` only; field shapes and format options differ from `/release_task`." Add a cover/repaint example. **Replace all three `acestep/acestep-v15-chinese-lyric` occurrences** (curl examples) with the resolved model id. Do **not** rewrite the procedure.
- **`_CAPTION_GUIDE`** (`mcp/acestep_mcp_server.py:14-82`). **Unchanged in content.** Rule recorded: it must not contain endpoint URLs, API field/param names, or format-option claims; metadata *ranges* (BPM/key/duration ranges as caption-writing guidance) are explicitly **allowed** — so its existing "Metadata Ranges" section is compliant.

### Part 3 — Drift guard

- `acestep/api/agent_discovery_route.py`: `mcp_server.tools` list `4 → 5` (add `get_examples`). Requires an API server restart to take effect (the served `.txt` files do not). **Restart is confirmed safe — the service is in testing, no users.**
- New test asserting `/.well-known/agent`'s `mcp_server.tools` matches the `@mcp.tool()` functions in `mcp/acestep_mcp_server.py` (covers Python↔Python drift only; the AGENTS.md Markdown table is kept in sync via the new checklist item — not test-enforced).

### Part 4 — Tests (AGENTS.md mandates unittest-style `*_test.py`; mock `_request`, network, filesystem)

New `mcp/acestep_mcp_server_test.py`:

1. **Param→body mapping (success path):** `generate_music(prompt="x", bpm=120, key_scale="C major", instrumental=True)`; assert `audio_config.bpm/key_scale`, `lyrics=="[inst]"` injection, `task_type`, and that `model` is present.
2. **Cover multimodal body + src_audio/task_type rules:** with `src_audio` set (tiny fixture / mocked read), assert `messages[0].content` is a two-part list with an `input_audio` block and `task_type` is auto-set to `cover`; assert non-empty prompt is enforced; assert the error branches (src_audio + non-src task_type; cover/repaint without src_audio).
3. **Return shape + save:** mock a response with one `audio` data URL; use `tmp_path` as output dir; assert the tool returns the saved file path + parsed metadata, that `base64,` is **not** in the return, that the file exists, and that the extension matches the response mime.
4. **Empty-audio branch:** mock a response with `audio=None`; assert the "No audio produced" message and that **no file** is written.
5. **Metadata parse:** feed the exact `## Metadata\n**Caption:**...\n**Key:**...\n**Time Signature:**...` block (and an instrumental variant with no `## Lyrics`); assert the parsed key/value labels.
6. **Model resolution:** mock `/v1/models` returning `data[0].id`; assert that id is what's sent in `model`; assert the cache-clear-and-retry path on a model-error response.
7. **Error path:** mock `_request` → `{"error": ...}`; assert `"Generation failed:"` return.

Plus a discovery-sync test (Part 3).

## 6. Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| API server restart needed for discovery-route code change | **Confirmed no impact** — service is in testing, no users. |
| Switching model default changes output character | Current hardcoded model doesn't exist on this server, so the MCP is already mis-routed; switching to the real default is a fix, not a regression. |
| Stale model cache after operator swaps models + restarts | Cache is process-lifetime; clear + retry-once on a model-error response (§5 Part 1.1). |
| Return-shape change breaks consumers | None depend on it: bash skill + hermes use curl directly; the only reader is the LLM agent reading free text. |
| Multi-MB base64 re-entering tool output | Explicitly forbidden — return path + metadata only. |
| `data[0]` ≠ default on multi-model deployments | Documented assumption; acceptable for this single-model server and the primary-first emission order. |

## 7. Success Criteria

- `generate_music` with no audio, with lyrics, and with `src_audio` (auto-cover) all produce a saved file under the output dir and return its path + metadata.
- `generate_music` with no source audio and `task_type="cover"` raises a clear error (no silent text2music routing).
- No `base64,` substring in any `generate_music` return.
- Model sent to the server is always a real model from `/v1/models`; no `-chinese-lyric`.
- `llms.txt`, `llms-full.txt`, `AGENTS.md`, `hermes-skill/SKILL.md`, the MCP docstrings, and the discovery route agree on: tool count (5), format options (per endpoint), model id, and `/v1/models` shape.
- New test file passes (`python -m unittest mcp.acestep_mcp_server_test` and the discovery-sync test).
