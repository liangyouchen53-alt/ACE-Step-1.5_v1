"""Verify quick setup uses the actual adapter and fails instead of claiming success."""

import unittest
from unittest.mock import MagicMock, patch

import gradio as gr

from .piano_quick import ADAPTER, prepare_piano


class QuickPianoTests(unittest.TestCase):
    """Cover actual settings, empty captions and missing/load-failed adapters."""

    def test_load_and_preserve_caption(self):
        handler = MagicMock(lora_loaded=True, use_lora=True)
        handler.add_lora.return_value = "✅ Loaded"
        handler.set_active_lora_adapter.return_value = "✅ Selected"
        settings = prepare_piano(handler, "  Soft solo piano.  ", 15)
        self.assertEqual(settings["captions"], "Soft solo piano.")
        self.assertEqual(settings["audio_duration"], 15)
        self.assertFalse(settings["think_checkbox"])
        self.assertTrue(settings["use_lora_checkbox"])
        handler.add_lora.assert_called_once_with(str(ADAPTER), adapter_name="quick_piano")
        handler.set_use_lora.assert_called_once_with(True)
        self.assertIn("piano", prepare_piano(handler, "", 30)["captions"])

    def test_failure_is_visible(self):
        handler = MagicMock(lora_loaded=False)
        handler.add_lora.return_value = "❌ Failed"
        with self.assertRaises(gr.Error):
            prepare_piano(handler, "", 30)

    @patch("pathlib.Path.is_file", return_value=False)
    def test_missing_adapter_uses_base_and_disables_previous_lora(self, exists):
        handler = MagicMock()
        settings = prepare_piano(handler, "平靜鋼琴", 30)
        handler.set_use_lora.assert_called_once_with(False)
        handler.add_lora.assert_not_called()
        self.assertFalse(settings["use_lora_checkbox"])
        self.assertEqual(settings["lora_path"], "")
        self.assertEqual(settings["lora_scale_slider"], 0)
        self.assertEqual(settings["captions"], "平靜鋼琴")


if __name__ == "__main__":
    unittest.main()
