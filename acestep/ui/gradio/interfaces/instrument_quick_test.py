"""Test instrument isolation and truthful model availability reporting."""

import unittest
from unittest.mock import MagicMock, patch

import gradio as gr

from .instrument_quick import ROOT, prepare_tongue
ADAPTER = ROOT / "training/singing_bowl_expanded/lora/final"


class InstrumentTests(unittest.TestCase):
    """Prevent piano adapter leakage and false drum model success reports."""

    @patch("pathlib.Path.is_file", return_value=False)
    def test_untrained_disables_piano(self, exists):
        handler = MagicMock(use_lora=False)
        settings, mode = prepare_tongue(handler, "", 30)
        handler.set_use_lora.assert_called_once_with(False)
        handler.add_lora.assert_not_called()
        self.assertFalse(settings["use_lora_checkbox"])
        self.assertIn("尚未訓練", mode)
        self.assertIn("singing bowl", settings["captions"])

    @patch("pathlib.Path.is_file", return_value=True)
    def test_trained_loads_separate_adapter(self, exists):
        handler = MagicMock(use_lora=True, lora_loaded=True)
        handler.add_lora.return_value = "✅ Loaded"
        handler.set_active_lora_adapter.return_value = "✅ Selected"
        settings, mode = prepare_tongue(handler, "  Soft drum  ", 15)
        handler.add_lora.assert_called_once_with(str(ADAPTER), adapter_name="quick_singing_bowl_expanded")
        self.assertTrue(settings["use_lora_checkbox"])
        self.assertEqual(settings["captions"], "Soft drum")
        self.assertEqual(settings["audio_duration"], 15)

    @patch("pathlib.Path.is_file", return_value=True)
    def test_failed_load_does_not_reuse_previous_success(self, exists):
        handler = MagicMock(lora_loaded=True)
        handler.add_lora.return_value = "❌ broken adapter"
        with self.assertRaises(gr.Error):
            prepare_tongue(handler, "", 30)
        handler.set_use_lora.assert_not_called()

    @patch("pathlib.Path.is_file", return_value=True)
    def test_reapply_selects_existing_named_adapter(self, exists):
        handler = MagicMock(lora_loaded=True, use_lora=True)
        handler._active_loras = {"quick_singing_bowl_expanded": 1.0, "quick_piano": 1.0}
        handler.set_active_lora_adapter.return_value = "✅ Selected"
        prepare_tongue(handler, "", 30)
        handler.add_lora.assert_not_called()
        handler.set_active_lora_adapter.assert_called_once_with("quick_singing_bowl_expanded")


if __name__ == "__main__":
    unittest.main()


