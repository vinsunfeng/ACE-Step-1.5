"""Session artifact helpers for generated-source repaint."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch


_LEGACY_RETAKE_MODE_ALIASES = {"retake", "most_natural", "most-natural", "most natural"}


def normalize_repaint_mode_alias(mode: str) -> str:
    """Normalize repaint mode aliases to supported public modes.

    Args:
        mode: Requested repaint mode string.

    Returns:
        Canonical public repaint mode string.
    """
    requested = (mode or "auto").strip().lower()
    return "balanced" if requested in _LEGACY_RETAKE_MODE_ALIASES else requested


def resolve_repaint_mode(mode: str, source_session_dir: Optional[str] = None) -> str:
    """Resolve repaint mode defaults.

    Args:
        mode: Requested repaint mode.
        source_session_dir: Unused compatibility parameter.

    Returns:
        Effective repaint mode.
    """
    _ = source_session_dir
    requested = normalize_repaint_mode_alias(mode)
    return "balanced" if requested == "auto" else requested


def load_source_session_track(session_dir: str, track_index: int = 1) -> dict[str, Any]:
    """Load generated-source artifacts used by session-backed repaint.

    Args:
        session_dir: Directory containing session artifacts.
        track_index: One-based track index.

    Returns:
        Mapping containing params, optional LM metadata, and final latents.

    Raises:
        FileNotFoundError: If required artifact files are missing.
        ValueError: If required latents are unavailable.
    """
    root = Path(session_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"source_session_dir does not exist: {root}")
    index = max(1, int(track_index))
    params_path = root / f"{index:02d}_params.json"
    latents_path = root / f"{index:02d}_latents.npy"
    if not params_path.exists():
        raise FileNotFoundError(f"source session params not found: {params_path}")
    if not latents_path.exists():
        raise FileNotFoundError(f"source session latents not found: {latents_path}")

    params = _read_json(params_path)
    lm_metadata_path = root / "lm_metadata.json"
    lm_metadata = _read_json(lm_metadata_path) if lm_metadata_path.exists() else {}
    latents_np = np.load(latents_path).astype(np.float32)
    if latents_np.ndim != 2:
        raise ValueError("source session latents must be shaped [T, C]")

    return {
        "session_dir": str(root),
        "track_index": index,
        "params": params,
        "lm_metadata": lm_metadata,
        "latents": torch.from_numpy(latents_np),
        "duration": float(latents_np.shape[0]) / 25.0,
    }


def _read_json(path: Path) -> dict[str, Any]:
    """Read a UTF-8 JSON object."""
    with path.open(encoding="utf-8") as file:
        return json.load(file)
