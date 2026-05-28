# CLAUDE.md — ACE-Step 1.5

ACE-Step 1.5 is an AI music generation system. It uses a Diffusion Transformer (DiT) with an optional 5Hz Language Model for metadata completion.

## Music Generation

Use the MCP server at `mcp/acestep_mcp_server.py` to generate music. The MCP tools are:
- `generate_music` — Create audio from text descriptions
- `list_models` — Check available models
- `enhance_prompt` — Use LLM to structure prompt/lyrics
- `check_health` — Verify API status

### Example

```
generate_music(
    prompt="A chill lo-fi hip hop beat with vinyl crackle and jazz piano",
    lyrics="[inst]",
    duration=30,
    format="mp3"
)
```

### Lyrics Format

Use section tags: `[Verse]`, `[Chorus]`, `[Bridge]`, `[Intro]`, `[Outro]`.
For instrumental: `lyrics="[inst]"` or `instrumental=True`.

### Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `ACESTEP_API_URL` | `http://localhost:8010` | API endpoint |
| `ACESTEP_API_KEY` | (none) | API authentication key |

## Development Commands

```bash
# Install (in ROCm container)
pip install -e . --no-deps

# Run tests
python -m pytest acestep/rocm_compat_test.py -v
python -m pytest acestep/gpu_config_effective_free_vram_test.py -v

# Start API server (container)
python -m acestep.api_server --host 0.0.0.0 --port 8010 --api-key acestep-rocm
```

## MCP Server Configuration

To add ACE-Step MCP server to Claude Code settings:

```json
{
  "mcpServers": {
    "acestep": {
      "command": "python",
      "args": ["mcp/acestep_mcp_server.py"],
      "env": {
        "ACESTEP_API_URL": "http://localhost:8010",
        "ACESTEP_API_KEY": "${ACESTEP_API_KEY}"
      }
    }
  }
}
```

## ROCm / AMD GPU

See `README-ROCm.md` and `docs/rocm-container-setup.md` for AMD GPU setup.
- Container: `acestep-rocm` (toolbox, `docker.io/kyuz0/vllm-therock-gfx1151:stable`)
- GPU: AMD Ryzen AI MAX+ 395 (Strix Halo, gfx1151)
- Default dtype: float32 (consumer RDNA), backend: PyTorch (pt)
