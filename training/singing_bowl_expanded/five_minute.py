"""Generate matched piano samples with fresh models for each adapter condition."""

import gc
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(__file__).resolve().parent
os.environ["HF_HOME"] = str(ROOT / ".hf-cache")
os.environ["HF_MODULES_CACHE"] = str(ROOT / ".hf-cache/modules")

import numpy as np
import soundfile as sf
import torch

from acestep.handler import AceStepHandler
from acestep.inference import GenerationConfig, GenerationParams, generate_music
from acestep.llm_inference import LLMHandler
from acestep.ui.gradio.interfaces.piano_quality import piano_settings


def main():
    """Generate one continuous five-minute trial without stitching or repetition."""
    handler = AceStepHandler()
    status, ready = handler.initialize_service(
        project_root=str(ROOT), config_path="acestep-v15-turbo", device="cuda",
        offload_to_cpu=True, offload_dit_to_cpu=True, compile_model=False,
        quantization=None, use_flash_attention=False)
    if not ready:
        raise RuntimeError(status)
    message = handler.load_lora(str(DATA / "lora/final"))
    if not message.startswith("✅") or not handler.lora_loaded:
        raise RuntimeError(message)
    handler.set_use_lora(True)
    handler.set_lora_scale(0.8)
    params = GenerationParams(
        caption="Solo acoustic singing bowl, continuous gentle circular rim rubbing. Deep steady humming resonance, shimmering metallic overtones and smooth sustained tones. Calm unhurried instrumental meditation, gradually evolving timbre and restrained dynamics. Let the final resonance settle naturally at the end.",
        lyrics="[Instrumental]", instrumental=True, duration=300, bpm=None,
        thinking=False, use_cot_caption=False, use_cot_metas=False,
        use_cot_language=False, inference_steps=8, shift=3, seed=20261118,
        enable_normalization=True, normalization_db=-1, fade_out_duration=3)
    config = GenerationConfig(batch_size=1, audio_format="flac", use_random_seed=False)
    result = generate_music(handler, LLMHandler(), params, config,
                           save_dir=str(ROOT / "gradio_outputs/five_minute_trial"))
    if not result.success:
        raise RuntimeError(result.error)
    path = result.audios[0]["path"]
    audio, rate = sf.read(path, always_2d=True)
    report = {"path": path, "seconds": len(audio)/rate, "sample_rate": rate,
              "finite": bool(np.isfinite(audio).all()),
              "peak": float(np.abs(audio).max()), "human_reviewed": False}
    if abs(report["seconds"] - 300) > 0.1 or not report["finite"]:
        raise ValueError(report)
    (DATA / "five_minute_result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)

if __name__ == "__main__":
    main()
