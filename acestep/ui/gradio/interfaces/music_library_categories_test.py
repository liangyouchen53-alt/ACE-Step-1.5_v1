"""Category switching and deletion isolation regressions."""

import tempfile
import unittest
from pathlib import Path

import gradio as gr

from .music_library import create_music_library
from .music_library_categories import ALL, GENERAL, TESTS, filter_music


class CategoryTests(unittest.TestCase):
    """Preserve normal songs while separating known model test outputs."""

    def test_classification(self):
        normal = ["batch_123/song.wav", "my_test_song.wav"]
        tests = ["piano_expanded_comparison\\original\\a.flac",
                 "model_tests/new.wav",
                 "batch_1794909050/d840c0e8-085d-aca9-df56-f85cfda2b94b.flac"]
        self.assertEqual(filter_music(normal + tests, GENERAL), normal)
        self.assertEqual(filter_music(normal + tests, TESTS), tests)
        self.assertEqual(filter_music(normal + tests, ALL), normal + tests)
        with self.assertRaises(ValueError):
            filter_music(normal, "invalid")

    def test_switch_and_delete_are_scoped(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "song.wav").write_bytes(b"audio")
            (root / "model_tests").mkdir()
            test = root / "model_tests/test.wav"
            test.write_bytes(b"audio")
            relative = str(test.relative_to(root))
            with gr.Blocks() as demo:
                create_music_library(str(root), demo=demo)
            callbacks = {fn.fn.__name__: fn.fn for fn in demo.fns.values()}
            refresh = callbacks["refresh_music"]
            delete = callbacks["delete_selected_music"]
            self.assertEqual(refresh()[0]["choices"], ["song.wav"])
            switched = refresh(TESTS)
            self.assertEqual(switched[0]["choices"], [relative])
            self.assertEqual(switched[5]["choices"], [relative])
            self.assertIsNone(switched[1])
            self.assertEqual(switched[5]["value"], [])
            self.assertFalse(switched[6])
            rejected = delete([relative], True, GENERAL)
            self.assertIn("重新勾選", rejected[4])
            self.assertTrue(test.exists())
            removed = delete([relative], True, TESTS)
            self.assertEqual(removed[0]["choices"], [])
            self.assertEqual(refresh(TESTS)[5]["choices"], [])
            self.assertTrue((root / "song.wav").exists())
            self.assertEqual(refresh(ALL)[0]["choices"], ["song.wav"])


if __name__ == "__main__":
    unittest.main()
