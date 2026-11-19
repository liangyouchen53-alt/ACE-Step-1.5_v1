"""Render matched baseline/LoRA samples without a language model."""

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PILOT = Path(__file__).resolve().parent
os.environ["HF_HOME"] = str(ROOT / ".hf-cache")
os.environ["HF_MODULES_CACHE"] = str(ROOT / ".hf-cache" / "modules")

from acestep.handler import AceStepHandler
from acestep.inference import GenerationConfig, GenerationParams, generate_music
from acestep.llm_inference import LLMHandler


def main():
    """Save identical-seed before/after examples and objective file checks."""
    import numpy as np
    import soundfile as sf

    adapter = PILOT / "lora" / "final" / "adapter_model.safetensors"
    if not adapter.exists():
        raise FileNotFoundError("No trained LoRA found. Finish training first.")
    handler = AceStepHandler()
    status, ready = handler.initialize_service(
        project_root=str(ROOT), config_path="acestep-v15-turbo", device="cuda",
        offload_to_cpu=True, offload_dit_to_cpu=True, compile_model=False,
        quantization=None, use_flash_attention=False,
    )
    if not ready:
        raise RuntimeError(status)
    params = GenerationParams(
        caption="Solo acoustic piano only, gentle calm melody, quiet dynamics, sparse arrangement, no other instruments, no drums, no vocals, no humming.",
        lyrics="[Instrumental]", instrumental=True, duration=30, bpm=65,
        thinking=False, use_cot_caption=False, use_cot_metas=False,
        use_cot_language=False, inference_steps=8, shift=3, seed=20261117,
    )
    config = GenerationConfig(batch_size=1, audio_format="wav", use_random_seed=False)
    report = {}
    for label in ("baseline", "lora"):
        if label == "lora":
            print(handler.load_lora(str(adapter.parent)), flush=True)
            if not handler.lora_loaded:
                raise RuntimeError("LoRA failed to load")
            handler.set_use_lora(True)
            handler.set_lora_scale(1.0)
        output = ROOT / "gradio_outputs" / "piano_pilot_comparison" / label
        result = generate_music(handler, LLMHandler(), params, config, save_dir=str(output))
        if not result.success:
            raise RuntimeError(result.error)
        path = result.audios[0]["path"]
        audio, rate = sf.read(path, always_2d=True)
        report[label] = {
            "path": path, "seconds": len(audio) / rate, "sample_rate": rate,
            "finite": bool(np.isfinite(audio).all()),
            "rms": float(np.sqrt(np.mean(audio * audio))),
        }
        print(label, report[label], flush=True)
    report["limitations"] = "Short pilot only. Audio validity does not prove better quality, calm mood or absence of vocals. Listen to both samples."
    (PILOT / "comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

