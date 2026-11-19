"""Check migrated manifests without requiring a Spark GPU."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from scripts.spark import normalize_manifests


class ManifestTests(unittest.TestCase):
    """Cover Windows separator migration and traversal rejection."""

    def test_relative_windows_paths(self):
        with tempfile.TemporaryDirectory() as folder:
            data = Path(folder).resolve()
            (data / "clips").mkdir()
            (data / "clips/a.wav").touch()
            for split in ("train", "heldout"):
                (data / f"{split}.json").write_text(json.dumps(
                    {"samples": [{"audio_path": "clips\\a.wav"}]}), encoding="utf-8")
            normalize_manifests(data)
            self.assertEqual(json.loads((data / "train.json").read_text())["samples"][0]
                             ["audio_path"], "clips/a.wav")

    def test_outside_paths_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            data = Path(folder).resolve()
            for value in ("../outside.wav", "C:/old/a.wav"):
                (data / "train.json").write_text(json.dumps(
                    {"samples": [{"audio_path": value}]}), encoding="utf-8")
                with self.assertRaises(ValueError):
                    normalize_manifests(data)

    def test_linux_recycle_uses_trash_command(self):
        from acestep.ui.gradio.interfaces.music_library_recycle import recycle_files
        with patch("acestep.ui.gradio.interfaces.music_library_recycle.os.name", "posix"), \
                patch("acestep.ui.gradio.interfaces.music_library_recycle.shutil.which",
                      return_value="/usr/bin/gio"), \
                patch("acestep.ui.gradio.interfaces.music_library_recycle.subprocess.run") as run:
            recycle_files(["/music/a.flac"])
            run.assert_called_once_with(["/usr/bin/gio", "trash", "--", "/music/a.flac"],
                                        check=True)
