"""Browse generated audio within the configured local output directory."""

import os
from pathlib import Path

import gradio as gr

from .music_library_delete import delete_music
from .music_library_recycle import recycle_music, recycle_output_contents
from .music_library_categories import ALL, CATEGORIES, GENERAL, filter_music

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a"}


def list_music(root: Path) -> list[str]:
    """Return audio paths relative to root, newest first, excluding outside links."""
    root = root.resolve()
    files = [
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
        and path.resolve().is_relative_to(root)
    ]
    return [str(path.relative_to(root)) for path in sorted(
        files, key=lambda path: path.stat().st_mtime, reverse=True
    )]


def select_music(root: Path, relative_path: str | None) -> tuple:
    """Return audio, download files and metadata for an existing library item."""
    if not relative_path:
        return None, [], ""
    root = root.resolve()
    path = (root / relative_path).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("找不到音樂檔案，請重新整理清單。")
    if path.suffix.lower() not in AUDIO_EXTENSIONS:
        raise ValueError("此檔案不是支援的音樂格式。")
    metadata = path.with_suffix(".json")
    files = [str(path)]
    text = "此音樂沒有附帶的生成資料。"
    if metadata.is_file() and metadata.resolve().is_relative_to(root):
        files.append(str(metadata))
        text = metadata.read_text(encoding="utf-8-sig")
    return str(path), files, text


def create_music_library(output_dir: str, demo=None, tab=None) -> None:
    """Build a local music browser with preview, download and folder access."""
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    gr.Markdown("### 已生成的音樂\n生成完成後，按「重新整理清單」查看最新音樂。")
    gr.Textbox(value=str(root), label="音樂儲存資料夾", interactive=False)
    category = gr.Radio(choices=CATEGORIES, value=GENERAL, label="音樂分類")
    gr.Markdown("預設顯示一般生成音樂；模型比較與驗證音檔請切換至「模型測試音樂」。")
    with gr.Row():
        refresh = gr.Button("🔄 重新整理清單")
        open_folder = gr.Button("📂 在檔案總管開啟資料夾")
    status = gr.Markdown()
    clear_category = gr.Button("🗑️ 一鍵清空目前分類（移至資源回收筒）", variant="stop")
    gr.Markdown("按下即清空目前分類音樂並整理空資料夾。選「全部音樂」會清空輸出資料夾內所有檔案與子資料夾。可至 Windows 資源回收筒還原。")
    choices = filter_music(list_music(root), GENERAL)
    music = gr.Dropdown(choices=choices, label="選擇音樂", interactive=True)
    preview = gr.Audio(label="試聽音樂", type="filepath", interactive=False)
    downloads = gr.File(label="下載音樂與生成資料", file_count="multiple", interactive=False)
    with gr.Accordion("查看生成資料", open=False):
        details = gr.Textbox(label="生成參數", lines=12, interactive=False)
    with gr.Accordion("🗑️ 清理音樂", open=False):
        gr.Markdown("勾選要刪除的音樂。會永久刪除音檔及未被其他格式共用的生成資料、字幕，無法復原。")
        selected = gr.CheckboxGroup(choices=choices, label="要刪除的音樂")
        confirmed = gr.Checkbox(label="我確認永久刪除所選音樂", value=False)
        delete = gr.Button("🗑️ 刪除勾選的音樂", variant="stop")

    def refresh_music(current_category=GENERAL):
        """Refresh library choices and clear the previous selection and preview."""
        all_music = list_music(root)
        choices = filter_music(all_music, current_category)
        return (
            gr.update(choices=choices, value=None), None, [], "",
            f"{current_category}：{len(choices)} 首｜資料夾共有 {len(all_music)} 個音檔。",
            gr.update(choices=choices, value=[]), False,
        )

    def delete_selected_music(selection, confirmation, current_category=GENERAL):
        """Delete only confirmed selections, then refresh every library control."""
        if selection and confirmation:
            visible = set(filter_music(list_music(root), current_category))
            if any(item not in visible for item in selection):
                updates = list(refresh_music(current_category))
                updates[4] = "清單已變更，請重新勾選目前分類中的音樂。"
                return tuple(updates)
        message = delete_music(root, selection, confirmation)
        if not selection or not confirmation:
            return gr.skip(), gr.skip(), gr.skip(), gr.skip(), message, gr.skip(), gr.skip()
        updates = list(refresh_music(current_category))
        updates[4] = message
        return tuple(updates)

    def open_output_folder():
        """Open only the fixed output directory on this local Windows computer."""
        try:
            os.startfile(str(root))
        except OSError as exc:
            return f"無法開啟資料夾：{exc}"
        return "已在這台電腦的檔案總管開啟音樂資料夾。"

    def clear_current_category(current_category):
        """Recycle only the current category and always refresh both lists."""
        names = filter_music(list_music(root), current_category)
        try:
            message = (recycle_output_contents(root) if current_category == ALL
                       else recycle_music(root, names))
        except (OSError, ValueError) as exc:
            message = str(exc)
        updates = list(refresh_music(current_category))
        updates[4] = message
        return tuple(updates)

    library_outputs = [music, preview, downloads, details, status, selected, confirmed]
    clear_category.click(clear_current_category, inputs=[category], outputs=library_outputs,
                         api_name="recycle_music_category")
    refresh.click(refresh_music, inputs=[category], outputs=library_outputs)
    category.change(refresh_music, inputs=[category], outputs=library_outputs)
    if demo is not None:
        demo.load(refresh_music, inputs=[category], outputs=library_outputs, api_name="refresh_music_library")
    if tab is not None:
        tab.select(refresh_music, inputs=[category], outputs=library_outputs, api_name=False)
    delete.click(
        delete_selected_music, inputs=[selected, confirmed, category], outputs=library_outputs,
        api_name="delete_generated_music",
    )
    open_folder.click(open_output_folder, outputs=[status], api_name=False)
    music.change(
        lambda choice: select_music(root, choice), inputs=[music],
        outputs=[preview, downloads, details], api_name="select_generated_music",
    )
