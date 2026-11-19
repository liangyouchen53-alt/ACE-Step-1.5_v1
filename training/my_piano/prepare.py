"""Prepare the complete Goldberg release and independent existing holdout clips."""

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

DATA = Path(__file__).resolve().parent
DOWNLOAD = DATA.parent / "downloaded_piano"
CAPTION = "Solo acoustic grand piano, classical piano performance, no vocals, no other instruments."


def valid_audio(audio):
    """Reject silence, nonfinite audio and sustained full-scale samples."""
    if not np.isfinite(audio).all():
        return False
    rms = float(np.sqrt(np.mean(audio * audio)))
    clipped = float(np.mean(np.abs(audio) >= 0.999))
    return rms > 0.001 and clipped < 0.001


def main():
    """Cut nonoverlapping 20-second clips, resample and write auditable manifests."""
    sources = json.loads((DOWNLOAD / "manifest.json").read_text(encoding="utf-8"))
    rows, manifest, rejected = [], [], []
    for index, source in enumerate(sources, 1):
        path = DOWNLOAD / source["file"]
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "md5").hexdigest() != source["md5"]:
                raise ValueError("Source checksum mismatch: " + path.name)
        accepted = 0
        with sf.SoundFile(path) as audio_file:
            rate = audio_file.samplerate
            for start in range(0, len(audio_file) - 20 * rate + 1, 20 * rate):
                audio_file.seek(start)
                audio = audio_file.read(20 * rate, dtype="float32", always_2d=True)
                if not valid_audio(audio):
                    rejected.append({"source": source["file"], "start_seconds": start/rate})
                    continue
                # This release is 96 kHz; preserve stereo and use anti-aliased downsampling.
                if rate != 48000:
                    import math
                    divisor = math.gcd(rate, 48000)
                    audio = resample_poly(audio, 48000 // divisor, rate // divisor, axis=0)
                peak = float(np.abs(audio).max())
                if peak > 0.99:
                    audio *= 0.99 / peak
                fade = 960
                audio[:fade] *= np.linspace(0, 1, fade)[:, None]
                audio[-fade:] *= np.linspace(1, 0, fade)[:, None]
                output = DATA / "clips/train" / f"goldberg_{index:02d}_{start//rate:04d}.wav"
                output.parent.mkdir(parents=True, exist_ok=True)
                sf.write(output, audio, 48000, subtype="PCM_16")
                sample = {"audio_path": str(output.relative_to(DATA)), "caption": CAPTION,
                          "lyrics": "[Instrumental]", "is_instrumental": True,
                          "duration": 20, "language": "unknown", "bpm": None,
                          "keyscale": "", "timesignature": ""}
                rows.append(sample)
                manifest.append({**sample, "split": "train", "source": source["file"],
                                 "source_md5": source["md5"], "start_seconds": start/rate,
                                 "license": "CC0 1.0", "mood_review": "Not human-rated"})
                accepted += 1
        if not accepted:
            raise ValueError("No valid clips from " + path.name)
        print(f"Source {index}/{len(sources)}: {accepted} clips", flush=True)
    previous = DATA.parent / "piano_expanded"
    heldout = []
    for row in json.loads((previous / "clips.json").read_text(encoding="utf-8")):
        if row["split"] != "heldout":
            continue
        source = previous / row["audio_path"]
        target = DATA / "clips/heldout" / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        item = {k: row[k] for k in ("caption", "lyrics", "is_instrumental", "duration",
                                   "language", "bpm", "keyscale", "timesignature")}
        item["audio_path"] = str(target.relative_to(DATA))
        heldout.append(item)
        manifest.append({**row, "audio_path": item["audio_path"]})
    for split, samples in (("train", rows), ("heldout", heldout)):
        dataset = {"metadata": {"name": "goldberg_full_" + split,
                                "all_instrumental": True, "genre_ratio": 0}, "samples": samples}
        (DATA / f"{split}.json").write_text(json.dumps(dataset, indent=2), encoding="utf-8")
    (DATA / "clips.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    summary = {"source_recordings": len(sources), "train_clips": len(rows),
               "train_minutes": len(rows)/3, "heldout_clips": len(heldout),
               "sample_rate": 48000, "clip_seconds": 20, "rejected_windows": rejected,
               "training_started": False}
    (DATA / "preparation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
