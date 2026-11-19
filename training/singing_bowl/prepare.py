"""Prepare a small singing bowl timbre pilot from documented CC0 previews."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

DATA = Path(__file__).resolve().parent
CAPTION = "Solo singing bowl, struck and rubbed metal bowl, sustained harmonic resonance and natural decay."


def valid_audio(audio):
    """Reject silent, corrupt or persistently clipped windows."""
    return (np.isfinite(audio).all() and float(np.sqrt(np.mean(audio ** 2))) > 0.001
            and float(np.mean(np.abs(audio) >= 0.999)) < 0.001)


def main():
    """Split by source recording and cut nonoverlapping 20-second windows."""
    sources = json.loads((DATA / "sources.json").read_text(encoding="utf-8"))
    rows = {"train": [], "heldout": []}
    manifest, rejected = [], []
    for source in sources:
        path = DATA / source["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != source["sha256"]:
            raise ValueError("Source checksum mismatch")
        audio, rate = sf.read(path, dtype="float32", always_2d=True)
        for start in range(0, len(audio) - 20 * rate + 1, 20 * rate):
            clip = audio[start:start + 20 * rate].copy()
            if not valid_audio(clip):
                rejected.append({"source": source["id"], "start_seconds": start / rate})
                continue
            divisor = math.gcd(rate, 48000)
            clip = resample_poly(clip, 48000 // divisor, rate // divisor, axis=0)
            if clip.shape[1] == 1:
                clip = np.repeat(clip, 2, axis=1)
            peak = float(np.abs(clip).max())
            if peak > 0.99:
                clip *= 0.99 / peak
            clip[:960] *= np.linspace(0, 1, 960)[:, None]
            clip[-960:] *= np.linspace(1, 0, 960)[:, None]
            target = DATA / "clips" / source["split"] / f"td_{source['id']}_{start//rate:04d}.wav"
            target.parent.mkdir(parents=True, exist_ok=True)
            sf.write(target, clip, 48000, subtype="PCM_16")
            row = {"audio_path": str(target.relative_to(DATA)), "caption": CAPTION,
                   "lyrics": "[Instrumental]", "is_instrumental": True, "duration": 20,
                   "language": "unknown", "bpm": None, "keyscale": "", "timesignature": ""}
            rows[source["split"]].append(row)
            manifest.append({**row, "source_id": source["id"], "split": source["split"],
                             "start_seconds": start / rate, "license": source["license"]})
    for split, samples in rows.items():
        if not samples:
            raise ValueError("No valid clips: " + split)
        dataset = {"metadata": {"name": "singing_bowl_pilot_" + split,
                                "all_instrumental": True, "genre_ratio": 0}, "samples": samples}
        (DATA / f"{split}.json").write_text(json.dumps(dataset, indent=2), encoding="utf-8")
    (DATA / "clips.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    summary = {"source_recordings": len(sources), "train_clips": len(rows["train"]),
               "heldout_clips": len(rows["heldout"]), "clip_seconds": 20,
               "sample_rate": 48000, "rejected_windows": rejected,
               "purpose": "small timbre pilot, not a full music performance corpus",
               "source_quality": "lossy public HQ MP3 previews", "human_listening_review": False}
    (DATA / "preparation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

