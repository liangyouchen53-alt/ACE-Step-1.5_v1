"""Delete explicitly selected generated audio within the fixed library root."""

from pathlib import Path

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a"}


def delete_music(root: Path, selected: list[str] | None, confirmed: bool) -> str:
    """Validate the full selection before deleting audio and unshared sidecars."""
    if not selected:
        return "請先勾選要刪除的音樂。"
    if not confirmed:
        return "請先勾選「我確認永久刪除所選音樂」。"
    root = root.resolve()
    paths = []
    for name in dict.fromkeys(selected):
        original = root / name
        path = original.resolve()
        if not path.is_relative_to(root) or path.suffix.lower() not in AUDIO_EXTENSIONS:
            raise ValueError("只能刪除音樂資料夾內的音樂檔案。")
        if original.is_symlink() or any(parent.is_symlink() for parent in original.parents if parent != root):
            raise ValueError("不能透過捷徑或符號連結刪除檔案。")
        paths.append(path)
    deleted, missing, freed, errors = 0, 0, 0, []
    for path in paths:
        if not path.is_file():
            missing += 1
            continue
        try:
            size = path.stat().st_size
            path.unlink()
            freed += size
            deleted += 1
        except OSError:
            errors.append(path.name)
            continue
        # A WAV and MP3 of the same song can share parameter/subtitle files.
        if any(path.with_suffix(ext).exists() for ext in AUDIO_EXTENSIONS):
            continue
        for suffix in (".json", ".lrc", ".vtt"):
            sidecar = path.with_suffix(suffix)
            if sidecar.is_symlink() or not sidecar.resolve().is_relative_to(root):
                continue
            if sidecar.is_file():
                try:
                    size = sidecar.stat().st_size
                    sidecar.unlink()
                    freed += size
                except OSError:
                    errors.append(sidecar.name)
    message = f"已刪除 {deleted} 個音樂檔案，釋放約 {freed / 1024**2:.2f} MB。"
    if missing:
        message += f" {missing} 個檔案已不存在。"
    if errors:
        message += " 部分檔案無法刪除（可能使用中）：" + "、".join(errors)
    return message
