"""Guard explicit instrument choices against unwanted LM rewrites."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from .music_intent import resolve_prompt, bowl_caption
from .instrument_routing import BASE, detect_instrument


class IntentTests(unittest.TestCase):
    """Exercise aliases, exclusions, mixed requests and optional LM fallback."""

    def test_bowl_aliases_and_negatives(self):
        for text in ("宋缽放鬆音樂", "頌鉢，不要鋼琴、吉他", "颂钵",
                     "Tibetan singing bowl without drums and bass", "Himalayan bowl"):
            with self.subTest(text=text):
                self.assertEqual(detect_instrument(text), "頌缽")

    def test_mixed_requests(self):
        self.assertEqual(detect_instrument("頌缽加鋼琴"), BASE)
        self.assertEqual(detect_instrument("solo guitar without piano"), BASE)

    def test_explicit_bypasses_lm(self):
        text = "頌缽，持續摩擦，不要鋼琴"
        result = resolve_prompt(text, None, 30)
        self.assertEqual(result[:2], (text, "頌缽"))
        self.assertIn(text, bowl_caption(text))
        self.assertNotIn("strikes", bowl_caption(text))

    def test_unavailable_and_empty(self):
        self.assertEqual(resolve_prompt("放鬆音樂", None, 30)[1], BASE)
        with self.assertRaises(ValueError):
            resolve_prompt(" ", None, 30)

    @patch("acestep.inference.format_sample")
    def test_lm_suggests_only_when_unspecified(self, formatter):
        formatter.return_value = SimpleNamespace(success=True, caption="Soft solo piano")
        handler = SimpleNamespace(llm_initialized=True)
        self.assertEqual(resolve_prompt("平靜的音樂", handler, 30)[1], "鋼琴")
        formatter.return_value = SimpleNamespace(success=True, caption="Jazz saxophone, piano and bass")
        self.assertEqual(resolve_prompt("平靜的音樂", handler, 30)[:2], ("平靜的音樂", BASE))
        self.assertEqual(formatter.call_args.kwargs["user_metadata"]["duration"], 30)
        self.assertEqual(resolve_prompt("不要鋼琴，放鬆就好", handler, 30)[1], BASE)
        formatter.return_value = SimpleNamespace(success=False, caption="")
        self.assertEqual(resolve_prompt("平靜的音樂", handler, 30)[0], "平靜的音樂")
