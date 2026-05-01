"""Tests for generation mode repaint UI wiring helpers."""

import tempfile
import unittest

from acestep.ui.gradio.events.wiring.generation_mode_wiring import (
    _on_source_session_dir_change,
)


class RepaintModeChoiceTests(unittest.TestCase):
    """Verify repaint mode choices remain stable."""

    def test_legacy_most_natural_value_resets_without_session_folder(self):
        """Legacy most natural should not appear in choices."""
        update = _on_source_session_dir_change("", "most natural")

        self.assertNotIn("most natural", update["choices"])
        self.assertEqual("auto", update["value"])

    def test_most_natural_hidden_with_existing_session_folder(self):
        """Generated sessions are hidden state, not a separate user-facing mode."""
        with tempfile.TemporaryDirectory() as tmp:
            update = _on_source_session_dir_change(tmp, "balanced")

        self.assertNotIn("most natural", update["choices"])
        self.assertEqual("balanced", update["value"])


if __name__ == "__main__":
    unittest.main()
