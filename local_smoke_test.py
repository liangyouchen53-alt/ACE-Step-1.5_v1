"""Generate one reproducible 30-second instrumental sample on a local GPU."""
import os
import json
import time
from pathlib import Path

os.environ["HF_HOME"] = str(Path(__file__).resolve().parent / ".hf-cache")
os.environ["HF_MODULES_CACHE"] = str(Path(__file__).resolve().parent / ".hf-cache" / "modules")
from acestep.handler import AceStepHandler
from acestep.inference import GenerationConfig, GenerationParams, generate_music
from acestep.llm_inference import LLMHandler


def main():
    """Initialize the offloaded 2B model and save audio plus test metadata."""
    root = Path(__file__).resolve().parent
    output = root / "outputs" / "local_smoke_test"
    output.mkdir(parents=True, exist_ok=True)
    handler = AceStepHandler()
    status, ready = handler.initialize_service(
        project_root=str(root), config_path="acestep-v15-turbo", device="cuda",
        use_flash_attention=False, compile_model=False,
        offload_to_cpu=True, offload_dit_to_cpu=True, quantization=None,
    )
    print(status, flush=True)
    if not ready:
        raise RuntimeError(status)
    params = GenerationParams(
        caption="Instrumental ambient piano music, gentle warm piano, soft strings, calm mood, sparse arrangement, no vocals, no humming, no choir, no speech.",
        lyrics="[Instrumental]", instrumental=True, bpm=70,
        duration=30, keyscale="C major", timesignature="4",
        thinking=False, inference_steps=8, shift=3.0, seed=20261117,
        use_cot_caption=False, use_cot_metas=False, use_cot_language=False,
    )
    config = GenerationConfig(batch_size=1, audio_format="wav", use_random_seed=False)
    started = time.perf_counter()
    result = generate_music(handler, LLMHandler(), params, config, save_dir=str(output))
    report = {
        "success": result.success, "error": result.error,
        "generation_seconds": round(time.perf_counter() - started, 2),
        "audio_paths": [item["path"] for item in result.audios],
        "status": result.status_message,
    }
    (output / "test_result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if not result.success:
        raise RuntimeError(result.error)


if __name__ == "__main__":
    main()

