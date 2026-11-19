"""Regression tests for generated music discovery and scoped file access."""

import tempfile
import unittest
from pathlib import Path

from .music_library import list_music, select_music
from .music_library import create_music_library
import gradio as gr
from .music_library_delete import delete_music


class MusicLibraryTests(unittest.TestCase):
    """Keep audio selection inside the configured output directory."""

    def test_lists_audio_and_reads_matching_metadata(self):
        """Nested output audio is selectable with its matching parameter file."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            batch = root / "batch_1"
            batch.mkdir()
            audio = batch / "piano.wav"
            audio.write_bytes(b"audio")
            metadata = audio.with_suffix(".json")
            metadata.write_text('{"caption": "平靜鋼琴"}', encoding="utf-8")
            self.assertEqual(list_music(root), [str(audio.relative_to(root))])
            preview, files, text = select_music(root, str(audio.relative_to(root)))
            self.assertEqual(preview, str(audio.resolve()))
            self.assertEqual(files, [str(audio.resolve()), str(metadata.resolve())])
            self.assertIn("平靜鋼琴", text)

    def test_rejects_parent_directory_and_non_audio(self):
        """Caller-supplied paths cannot access other files on the computer."""
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            root = parent / "outputs"
            root.mkdir()
            (parent / "outside.wav").write_bytes(b"audio")
            (root / "params.json").write_text("{}", encoding="utf-8")
            for name in ("../outside.wav", "params.json", "missing.wav"):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    select_music(root, name)

    def test_empty_selection_and_missing_metadata(self):
        """An empty library and audio without metadata remain usable."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(list_music(root), [])
            self.assertEqual(select_music(root, None), (None, [], ""))
            audio = root / "sample.mp3"
            audio.write_bytes(b"audio")
            preview, files, text = select_music(root, "sample.mp3")
            self.assertEqual(preview, str(audio.resolve()))
            self.assertEqual(files, [str(audio.resolve())])
            self.assertIn("沒有附帶", text)

    def test_delete_selected_audio_and_sidecars_only(self):
        """Delete selected audio with parameters/subtitles and preserve other songs."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("piano.wav", "piano.json", "piano.lrc", "piano.vtt", "keep.wav"):
                (root / name).write_bytes(b"test")
            message = delete_music(root, ["piano.wav", "piano.wav"], True)
            self.assertIn("已刪除 1 個", message)
            self.assertEqual(list_music(root), ["keep.wav"])
            self.assertFalse((root / "piano.json").exists())
            self.assertFalse((root / "piano.lrc").exists())
            self.assertFalse((root / "piano.vtt").exists())

    def test_delete_requires_confirmation(self):
        """A selection without confirmation cannot change files."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "piano.wav"
            audio.write_bytes(b"audio")
            self.assertIn("先勾選", delete_music(root, [], True))
            self.assertIn("確認", delete_music(root, ["piano.wav"], False))
            self.assertTrue(audio.exists())

    def test_invalid_selection_cannot_partially_delete(self):
        """Validate every path before mutating even the first selected song."""
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            root = parent / "outputs"
            root.mkdir()
            audio = root / "piano.wav"
            audio.write_bytes(b"audio")
            outside = parent / "outside.wav"
            outside.write_bytes(b"audio")
            for name in ("../outside.wav", "weights.safetensors", str(outside)):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    delete_music(root, ["piano.wav", name], True)
                self.assertTrue(audio.exists())
                self.assertTrue(outside.exists())

    def test_shared_metadata_survives_and_stale_selection_is_safe(self):
        """Retain JSON for another format and tolerate already-removed audio."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("piano.wav", "piano.mp3", "piano.json"):
                (root / name).write_bytes(b"test")
            delete_music(root, ["piano.wav"], True)
            self.assertTrue((root / "piano.json").exists())
            message = delete_music(root, ["piano.wav"], True)
            self.assertIn("已不存在", message)
            delete_music(root, ["piano.mp3"], True)
            self.assertFalse((root / "piano.json").exists())

    def test_delete_and_page_load_refresh_both_lists(self):
        """Deletion and subsequent visits must read live files, not startup choices."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "remove.wav").write_bytes(b"audio")
            (root / "keep.wav").write_bytes(b"audio")
            with gr.Blocks() as demo:
                with gr.Tabs():
                    with gr.Tab("音樂") as tab:
                        create_music_library(str(root), demo=demo, tab=tab)
            callbacks = {block.fn.__name__: block.fn for block in demo.fns.values()}
            removed = callbacks["delete_selected_music"](["remove.wav"], True)
            self.assertEqual(removed[0]["choices"], ["keep.wav"])
            self.assertEqual(removed[5]["choices"], ["keep.wav"])
            self.assertIsNone(removed[0]["value"])
            self.assertIsNone(removed[1])
            refreshed = callbacks["refresh_music"]()
            self.assertEqual(refreshed[0]["choices"], ["keep.wav"])
            self.assertEqual(refreshed[5]["choices"], ["keep.wav"])
            (root / "new.wav").write_bytes(b"audio")
            refreshed = callbacks["refresh_music"]()
            self.assertIn("new.wav", refreshed[0]["choices"])
            targets = [target[1] for block in demo.fns.values() for target in block.targets]
            self.assertIn("load", targets)
            self.assertIn("select", targets)


if __name__ == "__main__":
    unittest.main()
