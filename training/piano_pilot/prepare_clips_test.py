"""Unit tests for pilot clip selection without downloading data or using a GPU."""
import unittest
import numpy as np
from prepare_clips import choose_windows


class ClipSelectionTests(unittest.TestCase):
    """Reject unsuitable signal windows and select quiet nonoverlapping clips."""

    def test_silence_is_excluded(self):
        self.assertEqual(choose_windows(np.zeros((500, 2), dtype=np.float32), 10, 4), [])

    def test_quiet_windows_are_ranked_first(self):
        audio = np.zeros((600, 2), dtype=np.float32)
        for index, amplitude in enumerate((0.1, 0.2, 0.05, 0.15), start=1):
            audio[index * 100:(index + 1) * 100] = amplitude
        windows = choose_windows(audio, 10, 2)
        self.assertEqual([start for _, start in windows], [300, 100])

    def test_clipped_audio_is_excluded(self):
        self.assertEqual(choose_windows(np.ones((500, 2), dtype=np.float32), 10, 4), [])


if __name__ == '__main__':
    unittest.main()
