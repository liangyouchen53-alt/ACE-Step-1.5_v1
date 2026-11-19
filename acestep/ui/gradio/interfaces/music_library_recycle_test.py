"""Validate recycle scope and shared sidecar preservation without touching user audio."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from .music_library_recycle import recycle_music, recycle_empty_folders, recycle_output_contents


class RecycleTests(unittest.TestCase):
    """Test prevalidation, shared metadata and companion selection."""

    @patch("acestep.ui.gradio.interfaces.music_library_recycle.recycle_files")
    def test_empty_folders_and_complete_output(self, recycle):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nested = root / "empty" / "nested"
            nested.mkdir(parents=True)
            keep = root / "keep"
            keep.mkdir()
            (keep / "artifact.vtt").write_text("test")
            recycle.side_effect = lambda paths: [p.rmdir() for p in paths]
            self.assertEqual(recycle_empty_folders(root), 2)
            self.assertTrue(root.exists())
            self.assertTrue(keep.exists())
            recycle.side_effect = None
            recycle_output_contents(root)
            self.assertEqual(recycle.call_args.args[0], [keep])
            self.assertNotIn(root, recycle.call_args.args[0])

    @patch("acestep.ui.gradio.interfaces.music_library_recycle.recycle_files")
    def test_companions_and_shared_formats(self, recycle):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("a.wav", "a.mp3", "a.json", "b.wav", "b.session.npz", "keep.wav"):
                (root / name).write_bytes(b"test")
            recycle_music(root, ["a.wav", "b.wav"])
            paths = recycle.call_args.args[0]
            self.assertNotIn(root / "a.json", paths)
            self.assertIn(root / "b.session.npz", paths)
            self.assertNotIn(root / "keep.wav", paths)
            recycle_music(root, ["a.wav", "a.mp3"])
            self.assertIn(root / "a.json", recycle.call_args.args[0])

    @patch("acestep.ui.gradio.interfaces.music_library_recycle.recycle_files")
    def test_invalid_paths_never_start_recycle(self, recycle):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.wav").write_bytes(b"test")
            for name in ("../outside.wav", "weights.safetensors"):
                with self.assertRaises(ValueError):
                    recycle_music(root, ["a.wav", name])
            recycle.assert_not_called()
            self.assertIn("沒有", recycle_music(root, []))


if __name__ == "__main__":
    unittest.main()
