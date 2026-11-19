"""Test duration boundaries and reference validation without model loading."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from .music_conditioning import timing_settings, validate_reference


class ConditioningTests(unittest.TestCase):
    """Cover effective settings and rejected audio inputs."""

    def test_timing_and_invalid_duration(self):
        settings, caption = timing_settings(30, "保留餘韻")
        self.assertEqual(settings["audio_duration"], 30)
        self.assertEqual(settings["fade_out_duration"], 0.25)
        self.assertIn("last 5 seconds", caption)
        for value in (0, 301, float("nan")):
            with self.assertRaises(ValueError):
                timing_settings(value, "保留餘韻")

    def test_reference_audio(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "reference.wav"
            rate = 8000
            wave = 0.1 * np.sin(2 * np.pi * 220 * np.arange(4 * rate) / rate)
            sf.write(path, wave, rate)
            self.assertEqual(validate_reference(path), str(path))
            sf.write(path, np.zeros(4 * rate), rate)
            with self.assertRaises(ValueError):
                validate_reference(path)
            sf.write(path, wave[:rate], rate)
            with self.assertRaises(ValueError):
                validate_reference(path)
        self.assertIsNone(validate_reference(None))

