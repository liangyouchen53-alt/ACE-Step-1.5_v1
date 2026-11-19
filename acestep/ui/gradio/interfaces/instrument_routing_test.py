"""Test prompt routing and conservative mixed-instrument fallback."""

import unittest

from .instrument_routing import BASE, detect_instrument


class RoutingTests(unittest.TestCase):
    """Protect instrument choices across common Chinese and English prompts."""

    def test_explicit_instruments(self):
        self.assertEqual(detect_instrument("平靜頌缽，沒有鋼琴"), "頌缽")
        self.assertEqual(detect_instrument("Solo piano"), "鋼琴")
        self.assertEqual(detect_instrument("Singing bowl without piano"), "頌缽")

    def test_mixed_and_unknown(self):
        for caption in ("鋼琴與頌缽", "鋼琴加吉他", "放鬆音樂", "guitar solo"):
            self.assertEqual(detect_instrument(caption), BASE)
