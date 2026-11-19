"""Check explicit performance composition and invalid input rejection."""

import unittest

from .bowl_performance import performance_caption


class PerformanceTests(unittest.TestCase):
    """Keep timbre description intact while varying requested gestures."""

    def test_prompt_only_does_not_insert_strikes_or_pauses(self):
        caption = "連續磨缽，穩定延音"
        self.assertEqual(performance_caption(caption, "依提示詞", "依提示詞"), caption)

    def test_rubbing_and_spacing(self):
        caption = performance_caption("Metal bowl.", "沿缽緣摩擦", "稀疏")
        self.assertIn("Metal bowl.", caption)
        self.assertIn("rim rubbing", caption)
        self.assertNotIn("mallet", caption)
        self.assertIn("pauses", caption)

    def test_invalid_setting(self):
        with self.assertRaises(ValueError):
            performance_caption("", "unknown", "稀疏")
