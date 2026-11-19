"""A short, guided setup for the local solo-piano workflow."""

from pathlib import Path

import gradio as gr
from acestep.constants import DEFAULT_DIT_INSTRUCTION

from .piano_quality import piano_settings
from .instrument_adapter import activate_adapter
from .bowl_performance import TECHNIQUES, DENSITIES, performance_caption
from .music_conditioning import ENDINGS, timing_settings, validate_reference
from .instrument_routing import AUTO, BASE, detect_instrument
from .instrument_quick import PIANO, TONGUE, prepare_tongue, selection_description
from .music_intent import resolve_prompt, bowl_caption

ADAPTER = Path(__file__).resolve().parents[4] / "training/my_piano/lora/final"


def prepare_piano(handler, caption: str, seconds: int) -> dict:
    """Use the piano adapter when installed, otherwise disable LoRA for base inference."""
    available = (ADAPTER / "adapter_model.safetensors").is_file()
    if available:
        activate_adapter(handler, ADAPTER, "quick_piano")
    else:
        handler.set_use_lora(False)
    settings = piano_settings()
    settings["captions"] = caption.strip() or settings["captions"]
    settings["audio_duration"] = int(seconds)
    settings.update(lora_path=str(ADAPTER) if available else "", use_lora_checkbox=available,
                    lora_scale_slider=1.0 if available else 0.0)
    return settings


def create_quick_inputs() -> dict:
    """Render the three primary decisions before the detailed controls."""
    gr.Markdown("### 今天，想聽什麼？\n描述樂器、心情與演奏方式，其餘交給系統。")
    with gr.Accordion("樂器選擇（通常不必調整）", open=False):
        instrument = gr.Radio([AUTO, PIANO, TONGUE], value=AUTO, label="樂器選擇方式")
    caption = gr.Textbox(
        label="想聽什麼樣的音樂？", value="平靜、放鬆的純鋼琴音樂", lines=3,
        info="例如：平靜的頌缽，持續沿缽緣摩擦，音色低沉、音量平穩。",
        elem_id="music-prompt",
    )
    seconds = gr.Slider(5, 300, value=30, step=1, label="音樂長度（秒）")
    button = gr.Button("生成我的音樂", variant="primary", size="lg", elem_id="music-create")
    status = gr.Markdown("準備好了。直接輸入描述，按一下就開始。", elem_id="music-status")
    instrument.change(lambda choice: (gr.skip(), "將依提示詞選擇模型，請按套用。")
                      if choice == AUTO else selection_description(choice),
                      inputs=instrument, outputs=[caption, status],
                      api_name=False)
    return dict(instrument=instrument, caption=caption, seconds=seconds, button=button, status=status)


def wire_quick_inputs(quick: dict, components: dict, handler, llm_handler=None):
    """Switch mode before applying values so mode-change events cannot erase them."""
    keys = list(piano_settings()) + ["lora_path", "use_lora_checkbox", "lora_scale_slider",
                                   "seed", "random_seed_checkbox", "task_type",
                                   "instruction_display_gen", "autogen_checkbox"]

    with gr.Accordion("頌缽演奏與音色（選填）", open=False):
        reference = gr.Audio(label="音色參考（選填：乾淨的頌缽錄音）", type="filepath")
        technique = gr.Radio(list(TECHNIQUES), value="依提示詞", label="演奏方式")
        density = gr.Radio(list(DENSITIES), value="依提示詞", label="演奏疏密")
        strength = gr.Slider(0, 1, value=0.8, step=0.1, label="頌缽微調強度",
                             info="0 使用底座音色；1 完整套用微調。僅對頌缽生效。")
        gr.Markdown("演奏設定透過文字條件引導，無法保證精確敲擊時間。請配合參考音訊試聽。")
        ending = gr.Radio(list(ENDINGS), value="保留餘韻", label="頌缽收尾方式")
    with gr.Accordion("重現與比較", open=False):
        use_lm = gr.Checkbox(value=True, label="未指定樂器時，讓 LM 建議配器")
        fixed = gr.Checkbox(value=False, label="固定種子（比較音色設定時使用）")
        seed_value = gr.Number(value=20261118, precision=0, label="種子")

    def apply(caption, seconds, instrument=PIANO, reference_path=None,
              technique="依提示詞", density="依提示詞", strength=0.8,
              ending="保留餘韻", fixed=False, seed_value=20261118, use_lm=True):
        """Apply validated settings only after successful adapter loading."""
        automatic = instrument == AUTO
        decision = "手動指定樂器"
        try:
            timed, ending_caption = timing_settings(seconds, ending)
            if automatic:
                caption, instrument, decision = resolve_prompt(caption, llm_handler, seconds, use_lm)
            reference_path = validate_reference(reference_path) if instrument == TONGUE else None
            seed_number = int(seed_value)
            if not 0 <= seed_number <= 4294967295:
                raise ValueError("種子需介於 0～4294967295。")
        except (ValueError, TypeError, OverflowError) as exc:
            raise gr.Error(str(exc)) from exc
        if instrument == PIANO:
            settings = prepare_piano(handler, caption, seconds)
            mode = "鋼琴微調模型" if settings["use_lora_checkbox"] else (
                "原始模型生成鋼琴（本機未安裝鋼琴 LoRA）")
        elif instrument == TONGUE:
            caption = bowl_caption(caption)
            settings, mode = prepare_tongue(handler, caption, seconds)
            settings["captions"] = performance_caption(settings["captions"], technique, density)
            settings.update(timed)
            settings["captions"] += " " + ending_caption
            if settings["use_lora_checkbox"]:
                handler.set_lora_scale(float(strength))
                settings["lora_scale_slider"] = float(strength)
            mode += f"｜{technique}／{density}｜微調強度 {float(strength):.1f}"
        elif instrument == BASE:
            handler.set_use_lora(False)
            settings = piano_settings()
            settings.update(captions=caption.strip(), audio_duration=int(seconds),
                            bpm=None, bpm_auto=True, lora_path="", use_lora_checkbox=False,
                            lora_scale_slider=0)
            mode = "原始模型（混合樂器或未辨識出單一微調樂器）"
        else:
            raise gr.Error("未知的樂器選項。")
        if instrument == TONGUE and reference_path:
            settings["reference_audio"] = reference_path
            mode += "｜已套用音色參考（全局音色引導）"
        settings.update(seed=str(seed_number) if fixed else "-1", random_seed_checkbox=not fixed)
        settings.update(task_type="text2music", instruction_display_gen=DEFAULT_DIT_INSTRUCTION,
                        autogen_checkbox=False)
        if automatic:
            mode = "自動判斷：" + mode
        return [settings[key] for key in keys] + [
            f"✅ {mode}｜{int(seconds)} 秒｜{decision}。生成進度請見下方結果。"
        ]

    return quick["button"].click(
        lambda: "Custom", outputs=[components["generation_mode"]], api_name=False,
    ).success(
        apply, inputs=[quick["caption"], quick["seconds"], quick["instrument"], reference,
                       technique, density, strength, ending, fixed, seed_value, use_lm],
        outputs=[components[key] for key in keys] + [quick["status"]],
        api_name="prepare_quick_piano",
    )



