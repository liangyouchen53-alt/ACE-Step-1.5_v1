"""Tests for excluding unsuitable audio before building the training dataset."""

import unittest

import numpy as np

from prepare import valid_audio


class AudioChecks(unittest.TestCase):
    """Ensure valid low-level audio passes while silent/damaged audio is rejected."""

    def test_valid_audio(self):
        self.assertTrue(valid_audio(np.full((48000, 2), 0.05, dtype=np.float32)))

    def test_invalid_audio(self):
        for value in (0, 1, np.nan, np.inf):
            with self.subTest(value=value):
                self.assertFalse(valid_audio(np.full((48000, 2), value, dtype=np.float32)))


if __name__ == "__main__":
    unittest.main()
