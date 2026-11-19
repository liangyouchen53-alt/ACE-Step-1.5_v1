"""Audit pilot provenance, saved weights and generated comparison files."""

import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parent


def main():
    """Write an evidence report; do not equate file checks with music quality."""
    clips = json.loads((ROOT / "clips.json").read_text(encoding="utf-8"))
    training = [item for item in clips if item["split"] == "train"]
    heldout = [item for item in clips if item["split"] == "heldout"]
    assert {x["source_id"] for x in training}.isdisjoint(x["source_id"] for x in heldout)
    for item in clips:
        audio, rate = sf.read(ROOT / item["audio_path"])
        assert len(audio) / rate == 10 and np.isfinite(audio).all()
    adapter = ROOT / "lora" / "final" / "adapter_model.safetensors"
    weights = load_file(str(adapter))
    assert all(torch.isfinite(value).all() for value in weights.values())
    updates = [value for key, value in weights.items() if "lora_B" in key]
    assert updates and all(torch.count_nonzero(value) > 0 for value in updates)
    checkpoints = list((ROOT / "lora" / "checkpoints").glob("*/training_state.pt"))
    states = [torch.load(str(path), map_location="cpu", weights_only=True) for path in checkpoints]
    latest = max(states, key=lambda state: state["global_step"])
    with adapter.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    report = {
        "training_clips": len(training), "heldout_clips": len(heldout),
        "clip_seconds": 10, "recording_split_disjoint": True,
        "base_model": "acestep-v15-turbo", "adapter_path": str(adapter),
        "adapter_bytes": adapter.stat().st_size, "adapter_sha256": digest,
        "all_adapter_values_finite": True, "nonzero_B_tensors": len(updates),
        "completed_epochs": latest["epoch"], "optimizer_steps": latest["global_step"],
        "checkpoint_count": len(states), "optimizer_state_saved": "optimizer_state_dict" in latest,
        "scheduler_state_saved": "scheduler_state_dict" in latest,
        "quality_assessment": "Not established. Heldout audio is excluded from training but no heldout loss or listening evaluation was performed.",
    }
    comparison_path = ROOT / "comparison.json"
    if comparison_path.exists():
        comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
        for key in ("baseline", "lora"):
            audio, rate = sf.read(comparison[key]["path"])
            assert np.isfinite(audio).all() and len(audio) / rate == 30
        report["comparison"] = comparison
    (ROOT / "verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
