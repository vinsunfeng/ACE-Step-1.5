"""ACE-Step MCP Server — Exposes music generation API as MCP tools."""

import base64
import json
import os
import re
import uuid
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

from mcp.server.fastmcp import FastMCP

API_URL = os.getenv("ACESTEP_API_URL", "http://localhost:8010").rstrip("/")
API_KEY = os.getenv("ACESTEP_API_KEY", "")
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

# CONSTRAINT: keep this guide to prose caption/lyrics writing guidance only.
# Do NOT add endpoint URLs, API field/param names, or format-option claims
# (those live in llms-full.txt). Metadata *ranges* (BPM/key/duration) as
# writing guidance are allowed.
_CAPTION_GUIDE = """\
## How to Write Captions for ACE-Step

A good caption describes the music in natural language, covering style, instruments, mood, and tempo.

### Caption Examples (good)
- "A dreamy synth-pop song with ethereal vocals and a driving bassline"
- "Chill lo-fi hip hop beat with vinyl crackle, jazz piano samples, and laid-back drums"
- "Epic orchestral battle theme with brass fanfares, timpani rolls, and string ostinatos"
- "A heartfelt Korean folk ballad with gayageum, acoustic guitar, and gentle percussion"
- "Upbeat funk track with slap bass, wah-wah guitar, tight horn stabs, and a groovy drum break"

### Caption Tips
- Be specific about instruments: "jazz piano with brush drums" > "nice music"
- Include mood/adjectives: "melancholic", "energetic", "ethereal", "raw"
- Mention genre when helpful: "bossa nova", "drum and bass", "ambient"
- Include vocal style if applicable: "breathy female vocals", "deep male baritone"

### Lyrics Format
Use section tags to structure lyrics:
```
[Intro]
[Verse 1]
First verse lyrics here
[Pre-Chorus]
Building up...
[Chorus]
The hook goes here
[Verse 2]
Second verse lyrics
[Bridge]
Something different
[Chorus]
[Outro]
```
For instrumental tracks, use `lyrics="[inst]"` or set `instrumental=True`.

### Lyrics Writing Rules
- **Length matches duration**: ~2-3 syllables per second of music.
  - 10-30s clip: 1 verse + 1 chorus (~4-8 lines)
  - 60s song: 2 verses + 2 choruses + bridge (~12-20 lines)
  - 120-240s full song: full structure with intro/outro (~24-40 lines)
- **Rhyme**: Adjacent lines should rhyme (AABB or ABAB). Chinese: 押韵脚。
- **Repetition**: Chorus should repeat at least twice. Hooks need repetition.
- **Syllable rhythm**: Keep syllable counts consistent across parallel lines.
  - Good: "Walking in the rain (5) / Feeling so much pain (5)"
  - Bad: "Walking in the rain (5) / I feel pain (3)"
- **Language match**: Set vocal_language to match the lyrics language.
  - Chinese: zh (supports 粤语 with vocal_language="yue")
  - Japanese: ja (hiragana/katakana preferred over kanji)
  - Korean: ko (hangul only)
- **Avoid**: Overly long words, tongue twisters, too many consecutive
  consonants — these cause vocal artifacts.
- **Section balance**: Chorus = most memorable part. Verse = storytelling.
  Bridge = contrast (different melody/feel).

### Metadata Ranges
- BPM: 60-200 (60=ballad, 85=lo-fi, 120=pop, 140=EDM, 170=drum&bass)
- Key: "C major", "A minor", "F# major", "Bb minor", etc.
- Duration: 10-300 seconds (short clips: 10-30, full song: 120-240)
- Time signature: "4/4" (default), "3/4" (waltz), "6/8" (ballad)
- Languages: en, zh, ja, ko, fr, de, es, pt, it, ru, bn, ar, hi, yue, and 35 more

### Workflow
1. Write a detailed caption (or let the user describe what they want)
2. Write lyrics with section tags if vocals are needed
3. Set appropriate BPM, key, duration, vocal_language
4. Call generate_music with these parameters
"""

mcp = FastMCP("acestep", instructions=_CAPTION_GUIDE)


def _headers(content_type: str = "application/json") -> dict[str, str]:
    h = {"Content-Type": content_type, "Accept": "application/json"}
    if API_KEY:
        h["Authorization"] = f"Bearer {API_KEY}"
    return h


def _request(method: str, path: str, body: dict | None = None) -> dict:
    url = f"{API_URL}{path}"
    data = json.dumps(body).encode() if body else None
    req = Request(url, data=data, headers=_headers(), method=method)
    try:
        with urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            data = json.loads(resp.read())
            if data.get("error"):
                return {"error": str(data["error"])}
            return data
    except HTTPError as e:
        err_body = e.read().decode(errors="replace")
        return {"error": f"HTTP {e.code}: {err_body[:500]}"}
    except URLError as e:
        return {"error": f"Connection failed: {e.reason}"}


def _clear_model_cache() -> None:
    """Clear the cached resolved model id."""
    global _model_cache
    _model_cache = None


def _resolve_model(force: bool = False) -> str:
    """Resolve the primary model id, cached.

    Primary source: GET /v1/models -> data[0].id (flat list, primary first;
    no is_default marker exists). Cold-start fallback: when /v1/models is empty
    (model lazy-loads on first generation), use GET /health's loaded_model,
    which reports the primary model name even before load. ``force`` clears the
    cache first (retry path after a stale-model error).
    """
    global _model_cache
    if force:
        _clear_model_cache()
    if _model_cache is None:
        model_id = None
        r = _request("GET", "/v1/models")
        if not r.get("error"):
            data = r.get("data", [])
            if isinstance(data, list) and data:
                first = data[0]
                model_id = first.get("id") if isinstance(first, dict) else str(first)
        if model_id is None:
            h = _request("GET", "/health")
            if h.get("error"):
                raise RuntimeError(f"Could not resolve model: {h['error']}")
            h_data = h.get("data") if isinstance(h, dict) else None
            loaded = h_data.get("loaded_model") if isinstance(h_data, dict) else None
            if not loaded:
                raise RuntimeError("No models available (/v1/models empty and /health has no loaded_model)")
            loaded = str(loaded)
            model_id = loaded if loaded.startswith("acestep/") else f"acestep/{loaded}"
        _model_cache = model_id
    return _model_cache


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


def _save_audio(audio_list, out_dir: str) -> list:
    """Decode base64 data URLs into files in out_dir. Returns saved paths.

    Returns [] if no decodable audio (caller treats empty as 'no audio').
    Extension is derived from the response mime via _MIME_TO_EXT (fallback mp3).
    """
    os.makedirs(out_dir, exist_ok=True)
    saved = []
    for a in audio_list or []:
        url = a.get("audio_url", {}).get("url", "") if isinstance(a, dict) else ""
        if not url.startswith("data:") or "," not in url:
            continue
        header, b64data = url.split(",", 1)
        try:
            raw = base64.b64decode(b64data, validate=False)
        except Exception:
            continue
        mime = header.split(":")[1].split(";")[0]
        ext = _MIME_TO_EXT.get(mime, "mp3")
        path = os.path.join(out_dir, f"{uuid.uuid4().hex}.{ext}")
        with open(path, "wb") as f:
            f.write(raw)
        saved.append(path)
    return saved


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
    m = re.search(r"## Lyrics\s*\n(.+?)(?=^## |\Z)", content, re.DOTALL | re.MULTILINE)
    if m:
        meta["lyrics"] = m.group(1).strip()
    return meta


@mcp.tool()
def check_health() -> str:
    """Check if the ACE-Step API server is healthy."""
    r = _request("GET", "/health")
    if r.get("error"):
        return f"API unhealthy: {r['error']}"
    status = r.get("data", {}).get("status", "unknown")
    model = r.get("data", {}).get("loaded_model", "none")
    init = r.get("data", {}).get("models_initialized", False)
    return f"API healthy (status={status}, model={model}, initialized={init})"


@mcp.tool()
def list_models() -> str:
    """List available ACE-Step music generation models."""
    r = _request("GET", "/v1/models")
    if r.get("error"):
        return f"Error: {r['error']}"
    model_list = r.get("data", [])
    if isinstance(model_list, list) and model_list:
        lines = ["Available models:"]
        for m in model_list:
            if isinstance(m, dict):
                default = " (default)" if m.get("is_default") else ""
                lines.append(f"  - {m.get('id', m.get('name', 'unknown'))}{default}")
            else:
                lines.append(f"  - {m}")
        return "\n".join(lines)
    return "No models loaded yet. Models are lazy-loaded on first request."


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

    if src_audio and not os.path.isfile(src_audio):
        return f"src_audio file not found: {src_audio}"

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


@mcp.tool()
def enhance_prompt(
    prompt: str,
    lyrics: str = "",
    temperature: float = 0.85,
) -> str:
    """Use ACE-Step's LLM to enhance a music prompt and structure lyrics.

    Returns structured caption, lyrics, BPM, key, duration, and language.
    """
    body = {"prompt": prompt, "lyrics": lyrics, "temperature": temperature}
    r = _request("POST", "/format_input", body)

    if r.get("error"):
        return f"Enhancement failed: {r['error']}"

    data = r.get("data", {})
    if not data:
        return f"No enhancement data. Response: {json.dumps(r, indent=2)[:500]}"

    lines = ["Enhanced music parameters:"]
    if "caption" in data:
        lines.append(f"  Caption: {data['caption']}")
    if "lyrics" in data:
        lines.append(f"  Lyrics: {data['lyrics']}")
    if "bpm" in data:
        lines.append(f"  BPM: {data['bpm']}")
    if "key_scale" in data:
        lines.append(f"  Key: {data['key_scale']}")
    if "time_signature" in data:
        lines.append(f"  Time sig: {data['time_signature']}")
    if "duration" in data:
        lines.append(f"  Duration: {data['duration']}s")
    if "vocal_language" in data:
        lines.append(f"  Language: {data['vocal_language']}")
    return "\n".join(lines)


@mcp.tool()
def get_examples(style: str = "full") -> str:
    """Get example music generation parameters from ACE-Step's sample pool.

    Useful when you need inspiration or want to see the expected format
    for captions, lyrics, and metadata.

    Args:
        style: "simple" for short descriptions, "full" for complete examples
            with lyrics, BPM, key, and duration.
    """
    sample_type = "simple_mode" if style == "simple" else "custom_mode"
    r = _request("POST", "/create_random_sample", {"sample_type": sample_type})

    if r.get("error"):
        return f"Failed to get examples: {r['error']}"

    data = r.get("data", {})
    if not data:
        return "No example data available."

    lines = ["Example generation parameters:"]
    if "caption" in data:
        lines.append(f"  Caption: {data['caption']}")
    if "description" in data:
        lines.append(f"  Description: {data['description']}")
    if "lyrics" in data:
        lyrics = data["lyrics"]
        if len(lyrics) > 300:
            lyrics = lyrics[:300] + "..."
        lines.append(f"  Lyrics: {lyrics}")
    if "bpm" in data:
        lines.append(f"  BPM: {data['bpm']}")
    if "keyscale" in data:
        lines.append(f"  Key: {data['keyscale']}")
    if "duration" in data:
        lines.append(f"  Duration: {data['duration']}s")
    if "language" in data:
        lines.append(f"  Language: {data['language']}")
    if "vocal_language" in data:
        lines.append(f"  Vocal language: {data['vocal_language']}")
    if "timesignature" in data:
        lines.append(f"  Time sig: {data['timesignature']}")
    if "instrumental" in data:
        lines.append(f"  Instrumental: {data['instrumental']}")
    lines.append("\nCall get_examples again for more samples.")
    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run(transport="stdio")
