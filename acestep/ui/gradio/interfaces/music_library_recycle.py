"""Recycle a bounded list of generated audio and its unshared companion files."""

import ctypes
import os
from pathlib import Path
from ctypes import wintypes

from .music_library_delete import AUDIO_EXTENSIONS


def recycle_empty_folders(root: Path) -> int:
    """Recycle empty descendants bottom-up, preserving the output root and links."""
    root = root.resolve()
    count = 0
    folders = sorted(root.rglob("*"), key=lambda p: len(p.parts), reverse=True)
    for folder in folders:
        if (folder.is_symlink() or not folder.is_dir()
                or not folder.resolve().is_relative_to(root)
                or any(p.is_symlink() for p in folder.parents if p != root)):
            continue
        if not any(folder.iterdir()):
            recycle_files([folder])
            count += 1
    return count


def recycle_output_contents(root: Path) -> str:
    """Recycle all output children, including orphaned artifacts, but preserve root."""
    root = root.resolve()
    children = list(root.iterdir())
    for path in root.rglob("*"):
        if path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction()):
            raise ValueError("資料夾中有連結，請先在檔案總管檢查後再清空。")
        if not path.resolve().is_relative_to(root):
            raise ValueError("資料夾超出音樂輸出範圍，已停止清空。")
    if children:
        recycle_files(children)
    return "已清空音樂輸出資料夾，包含音檔、生成資料與子資料夾；可至資源回收筒還原。"


def recycle_files(paths: list[Path]) -> None:
    """Ask the Windows shell to recycle explicit files; surface cancellation/errors."""
    if os.name != "nt":
        raise OSError("一鍵清空目前只支援 Windows 資源回收筒。")

    class FileOperation(ctypes.Structure):
        _fields_ = [
            ("hwnd", wintypes.HWND), ("wFunc", wintypes.UINT),
            ("pFrom", wintypes.LPCWSTR), ("pTo", wintypes.LPCWSTR),
            ("fFlags", wintypes.WORD), ("fAnyOperationsAborted", wintypes.BOOL),
            ("hNameMappings", ctypes.c_void_p), ("lpszProgressTitle", wintypes.LPCWSTR),
        ]

    source = ctypes.create_unicode_buffer("\0".join(str(p) for p in paths) + "\0\0")
    operation = FileOperation()
    operation.wFunc = 3  # FO_DELETE
    operation.pFrom = ctypes.cast(source, wintypes.LPCWSTR)
    operation.fFlags = 0x40 | 0x10 | 0x4 | 0x400  # recycle, no confirmation/progress/error UI
    function = ctypes.windll.shell32.SHFileOperationW
    function.argtypes = [ctypes.POINTER(FileOperation)]
    function.restype = ctypes.c_int
    result = function(ctypes.byref(operation))
    if result or operation.fAnyOperationsAborted:
        raise OSError(f"部分檔案可能未移入資源回收筒，請重新整理清單（代碼 {result}）。")


def recycle_music(root: Path, names: list[str]) -> str:
    """Validate the entire selection and recycle only scoped audio and companion files."""
    root = root.resolve()
    audio = set()
    for name in names:
        original = root / name
        path = original.resolve()
        if (not path.is_relative_to(root) or path.suffix.lower() not in AUDIO_EXTENSIONS
                or original.is_symlink()
                or any(p.is_symlink() for p in original.parents if p != root)):
            raise ValueError("只能清理音樂資料夾中的一般音樂檔案。")
        if path.is_file():
            audio.add(path)
    if not audio:
        count = recycle_empty_folders(root)
        return f"目前分類沒有可清理的音樂；已整理 {count} 個空資料夾。"
    files = set(audio)
    for path in audio:
        if any(path.with_suffix(ext).exists() and path.with_suffix(ext) not in audio
               for ext in AUDIO_EXTENSIONS):
            continue
        for suffix in (".json", ".lrc", ".vtt", ".repaint_latents.npy", ".session.npz"):
            companion = path.with_suffix(suffix)
            if (companion.is_file() and not companion.is_symlink()
                    and companion.resolve().is_relative_to(root)):
                files.add(companion)
    recycle_files(sorted(files))
    count = recycle_empty_folders(root)
    return f"已將 {len(audio)} 首音樂、相關生成資料與 {count} 個空資料夾移至資源回收筒。清空資源回收筒後才會釋放磁碟空間。"
