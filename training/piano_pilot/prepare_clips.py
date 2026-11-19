"""Prepare quiet, nonsilent piano candidates with recording-level holdout."""

import json
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent
CLIP_SECONDS = 10
CAPTION = "Solo acoustic piano, classical piano performance, no vocals, no other instruments."


def choose_windows(audio, sample_rate, count):
    """Rank non-overlapping windows by RMS, rejecting silence and very loud peaks."""
    width = CLIP_SECONDS * sample_rate
    candidates = []
    for start in range(width, len(audio) - width, width):
        clip = audio[start:start + width]
        rms = float(np.sqrt(np.mean(clip * clip)))
        peak = float(np.max(np.abs(clip)))
        if rms > 0.003 and peak > 0.025 and peak < 0.999:
            candidates.append((rms, start))
    return sorted(candidates)[:count]


def main():
    """Write 24 training clips and four heldout clips plus source manifests."""
    records = json.loads((ROOT / "sources.json").read_text(encoding="utf-8"))
    samples, manifest = [], []
    for record in records:
        split = record["pilot_split"]
        output = ROOT / "clips" / split
        output.mkdir(parents=True, exist_ok=True)
        audio, rate = sf.read(ROOT / record["local_audio"], dtype="float32", always_2d=True)
        count = 4 if split == "heldout" else 8
        windows = choose_windows(audio, rate, count)
        if len(windows) < count:
            raise RuntimeError(f"Not enough nonsilent windows: {record['id']}")
        for index, (rms, start) in enumerate(windows):
            path = output / f"maestro_{record['id']}_{index:02d}.wav"
            clip = audio[start:start + CLIP_SECONDS * rate].copy()
            fade = min(int(rate * 0.02), len(clip) // 2)
            clip[:fade] *= np.linspace(0, 1, fade)[:, None]
            clip[-fade:] *= np.linspace(1, 0, fade)[:, None]
            sf.write(path, clip, rate, subtype="PCM_16")
            sample = {
                "audio_path": str(path.relative_to(ROOT)), "caption": CAPTION,
                "lyrics": "[Instrumental]", "is_instrumental": True,
                "duration": CLIP_SECONDS, "language": "unknown",
                "bpm": None, "keyscale": "", "timesignature": "",
            }
            if split == "train":
                samples.append(sample)
            manifest.append({
                **sample, "split": split, "source_id": record["id"],
                "source_title": record["canonical_title"], "start_seconds": start / rate,
                "source_rms": rms, "license": record["license"],
                "review_status": "Automatically selected quiet candidates; not human-audited for mood.",
            })
    dataset = {"metadata": {"name": "maestro_piano_pilot", "all_instrumental": True,
                            "genre_ratio": 0}, "samples": samples}
    for name, data in (("dataset.json", dataset), ("clips.json", manifest)):
        (ROOT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Prepared {len(samples)} training and {len(manifest)-len(samples)} heldout clips.")


if __name__ == "__main__":
    main()
