"""Ensure quick generation waits for successful prompt preparation."""

import unittest
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from .generation_run_wiring import register_generation_run_handlers


class RunWiringTests(unittest.TestCase):
    """Protect the quick success gate and the original advanced button."""

    @patch("acestep.ui.gradio.events.wiring.generation_run_wiring.res_h")
    def test_both_entrypoints_require_success(self, handlers):
        generation = defaultdict(MagicMock)
        quick = MagicMock()
        generation["quick_generate_event"] = quick
        context = SimpleNamespace(generation_section=generation,
                                  results_section=defaultdict(MagicMock),
                                  dit_handler=MagicMock(), llm_handler=MagicMock())
        register_generation_run_handlers(context)
        quick.success.assert_called_once()
        quick.then.assert_not_called()
        quick.success.return_value.success.assert_called_once()
        generation["generate_btn"].click.assert_called_once()
        generation["generate_btn"].click.return_value.success.assert_called_once()

    @patch("acestep.ui.gradio.events.wiring.generation_run_wiring.res_h")
    def test_standard_interface_unchanged(self, handlers):
        generation = defaultdict(MagicMock)
        context = SimpleNamespace(generation_section=generation,
                                  results_section=defaultdict(MagicMock),
                                  dit_handler=MagicMock(), llm_handler=MagicMock())
        register_generation_run_handlers(context)
        generation["generate_btn"].click.assert_called_once()
