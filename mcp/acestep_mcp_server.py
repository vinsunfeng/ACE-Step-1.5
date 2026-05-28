"""ACE-Step MCP Server — Exposes music generation API as MCP tools."""

import base64
import json
import os
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

from mcp.server.fastmcp import FastMCP

API_URL = os.getenv("ACESTEP_API_URL", "http://localhost:8010").rstrip("/")
API_KEY = os.getenv("ACESTEP_API_KEY", "")

mcp = FastMCP("acestep", instructions="ACE-Step music generation. Use generate_music to create audio from text.")


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
        with urlopen(req, timeout=600) as resp:
            data = json.loads(resp.read())
            if data.get("error"):
                return {"error": str(data["error"])}
            return data
    except HTTPError as e:
        err_body = e.read().decode(errors="replace")
        return {"error": f"HTTP {e.code}: {err_body[:500]}"}
    except URLError as e:
        return {"error": f"Connection failed: {e.reason}"}


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
    vocal_language: str = "en",
    instrumental: bool = False,
    seed: int | None = None,
    task_type: str = "text2music",
    guidance_scale: float | None = None,
    inference_steps: int | None = None,
) -> str:
    """Generate music from a text description.

    Args:
        prompt: Description of the music style, mood, instruments.
        lyrics: Lyrics with section tags [Verse], [Chorus], etc. Use [inst] for instrumental.
        duration: Target duration in seconds.
        format: Output format — mp3, wav, flac, ogg.
        bpm: Beats per minute (60-200).
        key_scale: Musical key, e.g. "C major", "A minor".
        vocal_language: Language code — en, zh, ja, ko, etc.
        instrumental: True for instrumental only (no vocals).
        seed: Random seed for reproducibility.
        task_type: Generation type — text2music, cover, repaint, complete.
        guidance_scale: Classifier-free guidance (higher = more prompt adherence).
        inference_steps: Diffusion steps (more = higher quality, slower).
    """
    if instrumental and not lyrics:
        lyrics = "[inst]"

    audio_config: dict = {"format": format, "vocal_language": vocal_language}
    if duration is not None:
        audio_config["duration"] = duration
    if bpm is not None:
        audio_config["bpm"] = bpm
    if key_scale is not None:
        audio_config["key_scale"] = key_scale
    if instrumental:
        audio_config["instrumental"] = True

    body: dict = {
        "model": "acestep/acestep-v15-chinese-lyric",
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "audio_config": audio_config,
        "task_type": task_type,
    }
    if lyrics:
        body["lyrics"] = lyrics
    if seed is not None:
        body["seed"] = seed
    if guidance_scale is not None:
        body["guidance_scale"] = guidance_scale
    if inference_steps is not None:
        body["inference_steps"] = inference_steps

    r = _request("POST", "/v1/chat/completions", body)

    if r.get("error"):
        return f"Generation failed: {r['error']}"

    choices = r.get("choices", [])
    if not choices:
        return f"No output. Full response: {json.dumps(r, indent=2)[:1000]}"

    msg = choices[0].get("message", {})
    content = msg.get("content", "")
    audio_list = msg.get("audio", [])

    result_parts = []
    if content:
        result_parts.append(content)

    if audio_list:
        for i, a in enumerate(audio_list):
            url = a.get("audio_url", {}).get("url", "")
            if url.startswith("data:"):
                # Extract mime and size info
                header, _ = url.split(",", 1)
                mime = header.split(":")[1].split(";")[0] if ":" in header else "unknown"
                b64_len = len(url.split(",", 1)[1]) if "," in url else 0
                approx_bytes = b64_len * 3 // 4
                size_mb = approx_bytes / (1024 * 1024)
                result_parts.append(
                    f"\nAudio #{i + 1}: {mime}, ~{size_mb:.1f} MB base64 data URL"
                )
            else:
                result_parts.append(f"\nAudio #{i + 1}: {url}")
        result_parts.append(
            "\nThe audio is a base64 data URL in the 'audio' field. "
            "Decode and save to play."
        )
    else:
        result_parts.append("\nNo audio in response (models may still be loading).")

    return "\n".join(result_parts)


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


if __name__ == "__main__":
    mcp.run(transport="stdio")
