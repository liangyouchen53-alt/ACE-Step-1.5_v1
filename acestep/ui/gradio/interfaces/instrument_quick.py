"""Explicit model selection for the local steel tongue drum workflow."""

from pathlib import Path

import gradio as gr

from .piano_quality import piano_settings
from .instrument_adapter import activate_adapter

PIANO = "鋼琴"
TONGUE = "頌缽"
ROOT = Path(__file__).resolve().parents[4]
ADAPTER = ROOT / "training/singing_bowl_expanded/lora/final"
if not (ADAPTER / "adapter_model.safetensors").is_file():
    ADAPTER = ROOT / "training/singing_bowl/lora/final"
CAPTION = ('Solo acoustic singing bowl. Deep humming resonance, shimmering metallic overtones and long natural decays. Meditative instrumental solo.')


def resolve_adapter():
    """Resolve the newest prepared model on each apply, including after training."""
    expanded = ROOT / "training/singing_bowl_expanded/lora/final"
    return expanded if (expanded / "adapter_model.safetensors").is_file() else ADAPTER


def selection_description(instrument):
    """Show whether a trained drum adapter is available, without claiming it is loaded."""
    if instrument == PIANO:
        return piano_settings()["captions"], "已選鋼琴，請按「套用樂器與設定」後再生成。"
    if instrument != TONGUE:
        raise gr.Error("未知的樂器選項。")
    mode = "已找到頌缽微調模型" if (ADAPTER / "adapter_model.safetensors").is_file() else (
        "頌缽尚未訓練：套用後使用原始模型試聽，並關閉鋼琴微調"
    )
    return CAPTION, mode + "。請按「套用樂器與設定」後再生成。"


def prepare_tongue(handler, caption, seconds):
    """Use the drum adapter if present; otherwise explicitly disable existing adapters."""
    adapter = resolve_adapter()
    available = (adapter / "adapter_model.safetensors").is_file()
    if available:
        name = "quick_singing_bowl_expanded" if "singing_bowl_expanded" in str(adapter) else "quick_singing_bowl"
        activate_adapter(handler, adapter, name)
        mode = f"頌缽微調模型（{adapter.parent.parent.name}）"
    else:
        handler.set_use_lora(False)
        mode = "原始模型試聽（頌缽尚未訓練；鋼琴微調已關閉）"
    if bool(handler.use_lora) != available:
        raise gr.Error("樂器模型切換失敗，請重新啟動網站。")
    settings = piano_settings()
    settings.update(captions=caption.strip() or CAPTION, audio_duration=int(seconds),
                    bpm=None, bpm_auto=True, fade_out_duration=0.25,
                    lora_path=str(adapter) if available else "",
                    use_lora_checkbox=available, lora_scale_slider=1.0)
    return settings, mode


