"""Verify raw clip audio and exact preprocessed tensor membership before training."""

import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

DATA = Path(__file__).resolve().parent


def main():
    """Reject missing/stale tensors, invalid waveforms and nonfinite tensors."""
    counts = {}
    for split in ("train", "heldout"):
        samples = json.loads((DATA / f"{split}.json").read_text(encoding="utf-8"))["samples"]
        expected = {Path(row["audio_path"]).stem for row in samples}
        tensor_paths = sorted((DATA / "tensors" / split).glob("*.pt"))
        if expected != {p.stem for p in tensor_paths}:
            raise ValueError("Missing or unexpected tensors: " + split)
        for sample in samples:
            audio, rate = sf.read(DATA / sample["audio_path"], always_2d=True)
            if (len(audio)/rate != 20 or not np.isfinite(audio).all()
                    or not 0 < np.abs(audio).max() < 1):
                raise ValueError("Invalid audio: " + sample["audio_path"])
        for path in tensor_paths:
            tensors = torch.load(path, map_location="cpu", weights_only=True)
            for key in ("target_latents", "attention_mask", "encoder_hidden_states",
                        "encoder_attention_mask", "context_latents"):
                if not torch.isfinite(tensors[key]).all():
                    raise ValueError("Nonfinite tensor: " + str(path))
        counts[split] = len(samples)
    if {p.stem for p in (DATA / "clips/train").glob("*.wav")} & {
            p.stem for p in (DATA / "clips/heldout").glob("*.wav")}:
        raise ValueError("Overlapping clip names")
    report = {"counts": counts, "all_audio_valid": True, "all_tensors_finite": True,
              "tensor_membership_verified": True, "training_started": False}
    (DATA / "verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
