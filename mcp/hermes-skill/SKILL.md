---
name: acestep
description: Generate music with ACE-Step 1.5 — text to full songs (vocals + instruments) via REST API
version: 1.0.0
metadata:
  hermes:
    tags: [music, audio, generation, creative]
    category: creative
    config:
      - key: acestep.api_url
        description: "ACE-Step API base URL"
        prompt: "Enter the ACE-Step API URL (e.g. http://localhost:8010)"
      - key: acestep.api_key
        description: "ACE-Step API key (optional)"
        prompt: "Enter the API key (leave empty if no auth)"
---

# ACE-Step Music Generation

## When to Use

Use this skill when the user asks to:
- Generate music or songs from text descriptions
- Create instrumental tracks or beats
- Make cover versions or remix audio
- Enhance or structure music prompts and lyrics

## API Connection

The ACE-Step API runs at the configured URL with optional Bearer token auth.

Test connectivity:
```bash
curl -s {{API_URL}}/health -H "Authorization: Bearer {{API_KEY}}"
```

## Core Procedure: Generate Music

### 1. Simple Generation (one-shot)

```bash
curl -X POST {{API_URL}}/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer {{API_KEY}}' \
  -d '{
    "model": "acestep/acestep-v15-chinese-lyric",
    "messages": [{"role": "user", "content": "DESCRIPTION_HERE"}],
    "stream": false,
    "audio_config": {
      "duration": 30,
      "format": "mp3",
      "vocal_language": "en"
    }
  }'
```

### 2. With Lyrics

```bash
curl -X POST {{API_URL}}/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer {{API_KEY}}' \
  -d '{
    "model": "acestep/acestep-v15-chinese-lyric",
    "messages": [{"role": "user", "content": "DESCRIPTION_HERE"}],
    "stream": false,
    "lyrics": "[Verse]\nLyrics here\n[Chorus]\nChorus lyrics",
    "audio_config": {
      "duration": 45,
      "format": "mp3",
      "vocal_language": "en"
    }
  }'
```

### 3. Instrumental

```bash
curl -X POST {{API_URL}}/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer {{API_KEY}}' \
  -d '{
    "model": "acestep/acestep-v15-chinese-lyric",
    "messages": [{"role": "user", "content": "DESCRIPTION_HERE"}],
    "stream": false,
    "lyrics": "[inst]",
    "audio_config": {
      "duration": 30,
      "format": "mp3",
      "instrumental": true
    }
  }'
```

## Parameters Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` (in messages) | string | required | Music style/mood description |
| `lyrics` | string | "" | Lyrics with [Verse], [Chorus] tags. [inst] = instrumental |
| `duration` | float | auto | Duration in seconds |
| `format` | string | "mp3" | mp3, wav, flac, ogg, aac |
| `bpm` | int | auto | Beats per minute (60-200) |
| `key_scale` | string | auto | Key, e.g. "C major", "A minor" |
| `vocal_language` | string | "en" | en, zh, ja, ko, fr, de, es, etc. |
| `seed` | int | random | Seed for reproducibility |
| `guidance_scale` | float | 7.0 | Higher = more prompt adherence |
| `inference_steps` | int | 8 | More steps = higher quality, slower |

## Response Format

Audio is returned as base64 data URL in `choices[0].message.audio[0].audio_url.url`.

To save audio:
```bash
# Extract base64 from response, decode to file
echo "RESPONSE" | python3 -c "
import sys, json, base64
r = json.load(sys.stdin)
url = r['choices'][0]['message']['audio'][0]['audio_url']['url']
header, b64data = url.split(',', 1)
with open('output.mp3', 'wb') as f:
    f.write(base64.b64decode(b64data))
print('Saved output.mp3')
"
```

## Enhance Prompt (Optional)

Use the LLM to structure raw text into proper music parameters:

```bash
curl -X POST {{API_URL}}/format_input \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer {{API_KEY}}' \
  -d '{"prompt": "chill jazz", "lyrics": "walking in the rain"}'
```

Returns structured caption, lyrics, BPM, key, duration.

## Task Types

| Type | Use For |
|------|---------|
| `text2music` | Text to new music (default) |
| `cover` | Cover with source audio + new style |
| `repaint` | Regenerate a section of audio |
| `complete` | Continue from existing audio |

## Pitfalls

- **Model not loaded**: First request triggers model download. Subsequent requests are fast. If you get "Model not initialized", wait and retry.
- **Timeout**: Generation takes 10-60s depending on duration and steps. The curl timeout should be at least 600s.
- **Audio size**: A 30s MP3 is ~500KB base64 (~700KB raw). Longer durations produce larger responses.
- **Language**: Default is English. For Chinese/Japanese/Korean songs, set `vocal_language` accordingly.
- **Large responses**: The response contains the full audio as base64. For very long songs, consider using `/release_task` + `/query_result` for async processing.

## Verification

After generation:
1. Check response contains `choices[0].message.audio`
2. Verify the audio data URL starts with `data:audio/`
3. Decode and save the file, play to confirm audio quality
