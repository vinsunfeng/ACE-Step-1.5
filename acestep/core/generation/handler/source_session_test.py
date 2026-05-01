"""Tests for session-backed repaint artifact helpers."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from acestep.core.generation.handler.source_session import (
    load_source_session_track,
    normalize_repaint_mode_alias,
    resolve_repaint_mode,
)
from acestep.core.generation.handler.source_session_save import (
    can_save_generation_session_artifacts,
    save_generation_session_artifacts,
)


class SourceSessionTests(unittest.TestCase):
    """Verify generated-source session loading and mode resolution."""

    def test_auto_mode_resolves_to_balanced_with_or_without_session(self):
        """Auto mode should keep the standard repaint path."""
        self.assertEqual("balanced", resolve_repaint_mode("auto", None))
        self.assertEqual("balanced", resolve_repaint_mode("auto", "/tmp/session"))
        self.assertEqual("balanced", resolve_repaint_mode("most natural", "/tmp/session"))
        self.assertEqual("aggressive", resolve_repaint_mode("aggressive", "/tmp/session"))

    def test_legacy_retake_alias_normalizes_to_balanced(self):
        """Legacy retake strings should not expose a separate repaint path."""
        self.assertEqual("balanced", normalize_repaint_mode_alias("retake"))
        self.assertEqual("balanced", normalize_repaint_mode_alias("most_natural"))

    def test_load_source_track_does_not_require_audio_codes(self):
        """Source repaint tracks only require saved final latents."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "01_params.json").write_text("{}", encoding="utf-8")
            np.save(root / "01_latents.npy", np.ones((4, 2), dtype=np.float32))

            source = load_source_session_track(tmp, 1)

            self.assertEqual((4, 2), tuple(source["latents"].shape))

    def test_load_source_track_returns_latents(self):
        """Valid source artifacts should load into source repaint mapping."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            params = {"cot_caption": "source caption", "lyrics": "la"}
            (root / "01_params.json").write_text(json.dumps(params), encoding="utf-8")
            np.save(root / "01_latents.npy", np.ones((50, 3), dtype=np.float32))

            source = load_source_session_track(tmp, 1)

            self.assertEqual((50, 3), tuple(source["latents"].shape))
            self.assertAlmostEqual(2.0, source["duration"])

    def test_save_session_artifacts_requires_latents(self):
        """Reusable session persistence should not write incomplete artifacts."""
        result = SimpleNamespace(
            success=True,
            audios=[{"params": {}}],
            extra_outputs={},
        )

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "pred_latents"):
                save_generation_session_artifacts(result=result, session_dir=tmp)

    def test_save_session_artifacts_writes_loadable_track(self):
        """Saved generation artifacts should be directly loadable for repaint."""
        result = SimpleNamespace(
            success=True,
            audios=[{"params": {"caption": "source"}}],
            extra_outputs={"pred_latents": torch.ones(1, 6, 3)},
        )

        with tempfile.TemporaryDirectory() as tmp:
            save_generation_session_artifacts(result=result, session_dir=tmp)
            source = load_source_session_track(tmp, 1)

            self.assertEqual((6, 3), tuple(source["latents"].shape))

    def test_can_save_session_artifacts_is_capability_based(self):
        """Any task result with final latents can be a source."""
        result = SimpleNamespace(
            success=True,
            audios=[{"params": {"task_type": "cover"}}],
            extra_outputs={"pred_latents": torch.ones(1, 6, 3)},
        )

        self.assertTrue(can_save_generation_session_artifacts(result))

    def test_can_save_session_artifacts_rejects_missing_latents(self):
        """Session-backed repaint needs saved final latents."""
        result = SimpleNamespace(
            success=True,
            audios=[{"params": {"audio_codes": "<|audio_code_1|>"}}],
            extra_outputs={},
        )

        self.assertFalse(can_save_generation_session_artifacts(result))


if __name__ == "__main__":
    unittest.main()
