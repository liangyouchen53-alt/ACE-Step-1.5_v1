"""Build lossless 20-second piano clips with recording/work-level holdout."""

import json
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent
SECONDS = 20
CAPTION = "Solo acoustic piano, classical piano performance, no vocals, no other instruments."


def choose_windows(audio, rate, count):
    """Select quiet nonoverlapping windows, rejecting silence, clipping and NaNs."""
    width = SECONDS * rate
    candidates = []
    for start in range(width, len(audio) - width, width):
        clip = audio[start:start + width]
        if not np.isfinite(clip).all():
            continue
        rms = float(np.sqrt(np.mean(clip * clip)))
        peak = float(np.abs(clip).max())
        if 0.003 < rms and 0.025 < peak < 0.999:
            candidates.append((rms, start))
    return sorted(candidates)[:count]


def validate_splits(records):
    """Reject duplicate source audio or named works crossing the holdout boundary."""
    for key in ("sha256", "work"):
        train = {r[key] for r in records if r["split"] == "train"}
        heldout = {r[key] for r in records if r["split"] == "heldout"}
        if train & heldout:
            raise ValueError(f"Split leakage: {key}: {train & heldout}")


def main():
    """Prepare separate training/evaluation manifests and retain exact source offsets."""
    records = json.loads((ROOT / "sources.json").read_text(encoding="utf-8"))
    validate_splits(records)
    samples = {"train": [], "heldout": []}
    manifest = []
    for record in records:
        audio, rate = sf.read(record["audio_path"], dtype="float32", always_2d=True)
        split = record["split"]
        windows = choose_windows(audio, rate, 4 if split == "heldout" else 8)
        if len(windows) < 2:
            raise ValueError(f"Insufficient valid windows: {record['id']}")
        for index, (rms, start) in enumerate(windows):
            path = ROOT / "clips" / split / f"{record['id']}_{index:02d}.wav"
            path.parent.mkdir(parents=True, exist_ok=True)
            clip = audio[start:start + SECONDS * rate].copy()
            fade = int(0.02 * rate)
            clip[:fade] *= np.linspace(0, 1, fade)[:, None]
            clip[-fade:] *= np.linspace(1, 0, fade)[:, None]
            sf.write(path, clip, rate, subtype="PCM_16")
            sample = {
                "audio_path": str(path.relative_to(ROOT)), "caption": CAPTION,
                "lyrics": "[Instrumental]", "is_instrumental": True,
                "duration": SECONDS, "language": "unknown", "bpm": None,
                "keyscale": "", "timesignature": "",
            }
            samples[split].append(sample)
            manifest.append({**sample, "source_id": record["id"], "split": split,
                             "dataset": record["dataset"], "work": record["work"],
                             "start_seconds": start / rate, "rms": rms,
                             "license": record["license"],
                             "mood_review": "Unreviewed; low RMS is not proof of calm mood."})
    for split, items in samples.items():
        dataset = {"metadata": {"name": "piano_expanded_" + split,
                                "all_instrumental": True, "genre_ratio": 0}, "samples": items}
        (ROOT / f"{split}.json").write_text(
            json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    (ROOT / "clips.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print({split: len(items) for split, items in samples.items()}, flush=True)


if __name__ == "__main__":
    main()
