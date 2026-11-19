"""Audit expanded adapter artifacts and ensure evaluation data never entered training."""

import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from prepare import validate_splits

DATA = Path(__file__).resolve().parent


def file_hash(path):
    """Compute a streaming SHA256 digest."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    """Validate tensor membership, finite adapter values and resumable checkpoints."""
    records = json.loads((DATA / "sources.json").read_text(encoding="utf-8"))
    validate_splits(records)
    clips = json.loads((DATA / "clips.json").read_text(encoding="utf-8"))
    for split in ("train", "heldout"):
        expected = {Path(c["audio_path"]).stem for c in clips if c["split"] == split}
        actual = {p.stem for p in (DATA / "tensors" / split).glob("*.pt")}
        assert expected == actual, (split, expected ^ actual)
    adapter = DATA / "lora/final/adapter_model.safetensors"
    weights = load_file(str(adapter))
    assert all(torch.isfinite(tensor).all() for tensor in weights.values())
    nonzero = sum(bool(torch.count_nonzero(tensor)) for name, tensor in weights.items()
                  if "lora_B" in name)
    assert nonzero > 0
    checkpoints = sorted((DATA / "lora/checkpoints").glob("epoch_*/training_state.pt"))
    assert len(checkpoints) == 6
    state = torch.load(checkpoints[-1], map_location="cpu", weights_only=True)
    assert state["epoch"] == 6
    assert "optimizer_state_dict" in state and "scheduler_state_dict" in state
    old = DATA.parent / "piano_pilot/lora/final/adapter_model.safetensors"
    old_unchanged = file_hash(old) == "831d41fa663cd585491e875948a507b286038fe54dcee6aa2cd7c9dfe05d32a2"
    assert old_unchanged
    report = {
        "adapter_bytes": adapter.stat().st_size, "adapter_sha256": file_hash(adapter),
        "tensor_count": len(weights), "nonzero_B_tensors": nonzero,
        "all_adapter_values_finite": True, "epochs": state["epoch"],
        "optimizer_steps": state["global_step"], "checkpoints": len(checkpoints),
        "train_heldout_membership_verified": True, "old_adapter_unchanged": old_unchanged,
        "quality_improvement_proven": False,
    }
    (DATA / "verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
