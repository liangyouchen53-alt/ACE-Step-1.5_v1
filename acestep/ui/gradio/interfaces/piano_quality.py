"""Explicit, reproducible solo-piano settings for the local generation UI."""

import gradio as gr


def piano_settings() -> dict:
    """Return conservative Turbo settings without LM caption rewriting."""
    return {
        "captions": (
            "A 30-second solo acoustic grand piano miniature, calm and relaxing. "
            "Slow lyrical melody at 65 BPM, soft touch, warm natural piano tone, "
            "gentle legato phrasing, sparse left-hand accompaniment and spacious pauses. "
            "Intimate clean recording with subtle room ambience and restrained dynamics. "
            "A simple resolving final phrase, followed by the natural decay of the last chord. "
            "Piano is the only instrument. No vocals, humming, drums, bass, strings or synths."
        ),
        "lyrics": "[Instrumental]",
        "instrumental_checkbox": True,
        "audio_duration": 30,
        "duration_auto": False,
        "bpm": 65,
        "bpm_auto": False,
        "batch_size_input": 1,
        "think_checkbox": False,
        "use_cot_caption": False,
        "use_cot_metas": False,
        "use_cot_language": False,
        "inference_steps": 8,
        "shift": 3.0,
        "custom_timesteps": "",
        "audio_format": "flac",
        "enable_normalization": True,
        "normalization_db": -1.0,
        "fade_in_duration": 0.0,
        "fade_out_duration": 1.5,
        "latent_shift": 0.0,
        "latent_rescale": 1.0,
        "reference_audio": None,
        "src_audio": None,
        "text2music_audio_code_string": "",
        "infer_method": "ode",
        "sampler_mode": "euler",
    }


def create_piano_quality_controls(components: dict) -> None:
    """Add an opt-in preset; leave existing prompts intact until clicked."""
    with gr.Accordion("🎹 平靜鋼琴生成設定", open=False):
        gr.Markdown(
            "請先切換至 **Custom（自訂）** 模式，使用 **Turbo** 模型，再套用設定。"
            "會取代提示詞及相關參數：30 秒、65 BPM、純鋼琴、FLAC 無損輸出與 1.5 秒淡出。"
            "關閉 LM 改寫以保留樂器描述；1.7B 模型仍可供其他模式使用。\n\n"
            "淡出能減少突兀截斷，但不保證樂句完整。此設定不會載入或訓練 LoRA；"
            "比較原模型時請先關閉 LoRA。不同種子仍需試聽挑選。"
        )
        button = gr.Button("套用 30 秒平靜鋼琴設定")
        status = gr.Markdown()

    keys = list(piano_settings())

    def apply(mode):
        """Only apply text-to-music parameters in Custom mode."""
        if mode != "Custom":
            return [gr.skip() for _ in keys] + ["請先切換至 Custom（自訂）模式。"]
        return list(piano_settings().values()) + ["已套用。請按「生成音樂」並試聽結果。"]

    button.click(
        apply, inputs=[components["generation_mode"]],
        outputs=[components[key] for key in keys] + [status], api_name="apply_piano_quality",
    )
