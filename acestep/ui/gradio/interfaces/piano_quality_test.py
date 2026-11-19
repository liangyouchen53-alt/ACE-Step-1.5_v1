"""Regression checks for the explicit piano preset."""

import unittest

import gradio as gr

from .piano_quality import create_piano_quality_controls, piano_settings


class PianoQualityTest(unittest.TestCase):
    """Check duration fidelity, lossless output and mode isolation."""

    def test_parameters(self):
        settings = piano_settings()
        self.assertEqual(settings["audio_duration"], 30)
        self.assertFalse(settings["duration_auto"])
        self.assertFalse(settings["use_cot_caption"])
        self.assertEqual(settings["audio_format"], "flac")
        self.assertEqual(settings["lyrics"], "[Instrumental]")
        self.assertNotIn("use_lora_checkbox", settings)

    def test_mode_guard_and_outputs(self):
        with gr.Blocks() as demo:
            components = {key: gr.State(value) for key, value in piano_settings().items()}
            components["generation_mode"] = gr.State("Custom")
            create_piano_quality_controls(components)
        callback = next(fn.fn for fn in demo.fns.values() if fn.api_name == "apply_piano_quality")
        result = callback("Custom")
        self.assertEqual(result[:-1], list(piano_settings().values()))
        for mode in ("Simple", "Cover", "Repaint"):
            self.assertTrue(all(value == gr.skip() for value in callback(mode)[:-1]))


if __name__ == "__main__":
    unittest.main()
