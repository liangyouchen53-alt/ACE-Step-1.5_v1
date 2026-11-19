"""Prepare user-provided bowl recordings without modifying original downloads."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

DATA = Path(__file__).resolve().parent
INPUT = DATA.parent / "singing_bowl_new_sources/manual"


def main():
    """Group by creator, remove exact duplicates and validate nonoverlapping windows."""
    sources, manifest, excluded, seen = [], [], [], set()
    samples = {"train": [], "heldout": []}
    for path in sorted(INPUT.glob("*.mp3")):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        reason = next((word for word in ("sitar", "handpan", "crystal")
                       if word in path.stem.lower()), None)
        if reason or digest in seen:
            excluded.append({"file": path.name, "reason": reason or "exact duplicate"})
            continue
        seen.add(digest)
        audio, rate = sf.read(path, dtype="float32", always_2d=True)
        if audio.shape[1] not in (1, 2) or not np.isfinite(audio).all():
            raise ValueError("Invalid source: " + path.name)
        author = path.stem.split("-")[0]
        # All recordings by the same creator stay in a single split.
        split = "heldout" if author == "kalsstockmedia" else "train"
        source_id = digest[:12]
        sources.append({"id": source_id, "file": str(path), "author_from_filename": author,
                        "sha256": digest, "seconds": len(audio)/rate, "split": split,
                        "license": "user-provided; not verified", "human_reviewed": False})
        for start in range(0, len(audio)-20*rate+1, 20*rate):
            clip = audio[start:start+20*rate].copy()
            rms = float(np.sqrt(np.mean(clip**2)))
            if rms < 0.001 or float(np.mean(np.abs(clip) >= .999)) >= .001:
                continue
            divisor = math.gcd(rate, 48000)
            clip = resample_poly(clip, 48000//divisor, rate//divisor, axis=0)
            if clip.shape[1] == 1:
                clip = np.repeat(clip, 2, axis=1)
            peak = float(np.abs(clip).max())
            if peak > .99:
                clip *= .99/peak
            clip[:960] *= np.linspace(0, 1, 960)[:, None]
            clip[-960:] *= np.linspace(1, 0, 960)[:, None]
            target = DATA / "clips" / split / f"bowl_{source_id}_{start//rate:05d}.wav"
            target.parent.mkdir(parents=True, exist_ok=True)
            sf.write(target, clip, 48000, subtype="PCM_16")
            row = {"audio_path": str(target.relative_to(DATA)),
                   "caption": "Singing bowl meditation, resonant bowl tones and natural decay.",
                   "lyrics": "[Instrumental]", "is_instrumental": True, "duration": 20,
                   "language": "unknown", "bpm": None, "keyscale": "", "timesignature": ""}
            samples[split].append(row)
            manifest.append({**row, "source_id": source_id, "split": split,
                             "start_seconds": start/rate, "rms": rms})
    for split, rows in samples.items():
        if not rows:
            raise ValueError("Empty split: " + split)
        payload = {"metadata": {"name": "manual_bowls_"+split,
                                "all_instrumental": True, "genre_ratio": 0}, "samples": rows}
        (DATA/f"{split}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    summary = {"source_recordings": len(sources), "train_clips": len(samples["train"]),
               "heldout_clips": len(samples["heldout"]), "clip_seconds": 20,
               "sample_rate": 48000, "excluded": excluded,
               "human_listening_review": False, "licenses_verified": False}
    for filename, payload in (("sources.json", sources), ("clips.json", manifest),
                              ("preparation.json", summary)):
        (DATA/filename).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
