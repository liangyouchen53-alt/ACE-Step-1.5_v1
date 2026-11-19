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
    """Save 2 seeds x 3 models, technical checks and an unfilled listening scorecard."""
    adapters = {"original": None, "pilot": ROOT / "training/piano_pilot/lora/final",
                "expanded": DATA / "lora/final"}
    report = []
    for label, adapter in adapters.items():
        handler = AceStepHandler()
        status, ready = handler.initialize_service(
            project_root=str(ROOT), config_path="acestep-v15-turbo", device="cuda",
            offload_to_cpu=True, offload_dit_to_cpu=True, compile_model=False,
            quantization=None, use_flash_attention=False,
        )
        if not ready:
            raise RuntimeError(status)
        if adapter:
            print(handler.load_lora(str(adapter)), flush=True)
            if not handler.lora_loaded:
                raise RuntimeError("Adapter failed to load")
            handler.set_use_lora(True)
            handler.set_lora_scale(1.0)
        for seed in (20261117, 20261118):
            params = GenerationParams(
                caption=piano_settings()["captions"], lyrics="[Instrumental]",
                instrumental=True, duration=30, bpm=65, thinking=False,
                use_cot_caption=False, use_cot_metas=False, use_cot_language=False,
                inference_steps=8, shift=3, seed=seed, enable_normalization=True,
                normalization_db=-1.0, fade_out_duration=1.5,
            )
            config = GenerationConfig(batch_size=1, audio_format="flac", use_random_seed=False)
            output = ROOT / "gradio_outputs/piano_expanded_comparison" / label / str(seed)
            result = generate_music(handler, LLMHandler(), params, config, save_dir=str(output))
            if not result.success:
                raise RuntimeError(result.error)
            path = result.audios[0]["path"]
            audio, rate = sf.read(path, always_2d=True)
            peak = float(np.abs(audio).max())
            finite = bool(np.isfinite(audio).all())
            if not finite or peak >= 1 or len(audio) / rate != 30:
                raise ValueError("Generated audio failed technical checks")
            report.append({"model": label, "seed": seed, "path": path,
                           "seconds": len(audio) / rate, "sample_rate": rate,
                           "peak": peak, "finite": finite,
                           "rms": float(np.sqrt(np.mean(audio * audio))),
                           "tail_peak": float(np.abs(audio[-1]).max())})
            print(label, seed, "OK", flush=True)
            (DATA / "comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        del handler
        gc.collect()
        torch.cuda.empty_cache()
    differences = []
    for seed in (20261117, 20261118):
        paths = {row["model"]: row["path"] for row in report if row["seed"] == seed}
        baseline, rate = sf.read(paths["original"], always_2d=True)
        for label in ("pilot", "expanded"):
            audio, candidate_rate = sf.read(paths[label], always_2d=True)
            assert candidate_rate == rate and audio.shape == baseline.shape
            difference = float(np.abs(audio - baseline).mean())
            assert difference > 0, "Adapter did not change the generated waveform"
            differences.append({"seed": seed, "model": label,
                                "mean_absolute_waveform_difference": difference})
    (DATA / "differences.json").write_text(json.dumps(differences, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
