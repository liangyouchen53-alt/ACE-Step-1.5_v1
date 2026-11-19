"""Download four official MAESTRO recordings using ZIP HTTP range requests."""

import hashlib
import json
from pathlib import Path

import requests
from remotezip import RemoteZip

ROOT = Path(__file__).resolve().parent
BASE = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/"
IDS = ["45", "75", "74"]


def main():
    """Cache a small, traceable, noncommercial piano pilot dataset."""
    metadata_path = ROOT / "maestro-v3.0.0.json"
    if not metadata_path.exists():
        response = requests.get(BASE + metadata_path.name, timeout=60)
        response.raise_for_status()
        metadata_path.write_bytes(response.content)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    chopin = next(i for i, title in metadata["canonical_title"].items()
                  if title == "Nocturne in C-Sharp Minor, Op. Posth."
                  and metadata["split"][i] == "train")
    records = []
    raw = ROOT / "raw"
    raw.mkdir(exist_ok=True)
    with RemoteZip(BASE + "maestro-v3.0.0.zip", timeout=120) as archive:
        for index in [*IDS, chopin]:
            row = {key: values[index] for key, values in metadata.items()}
            relative = row["audio_filename"]
            name = "maestro-v3.0.0/" + relative
            destination = raw / f"maestro_{index}.wav"
            info = archive.getinfo(name)
            if not destination.exists() or destination.stat().st_size != info.file_size:
                print(f"Downloading {index}: {row['canonical_title']}", flush=True)
                partial = destination.with_suffix(".partial")
                with archive.open(name) as source, partial.open("wb") as target:
                    while chunk := source.read(1024 * 1024):
                        target.write(chunk)
                partial.replace(destination)
            row.update({
                "id": index, "local_audio": str(destination.relative_to(ROOT)),
                "pilot_split": "heldout" if index == "74" else "train",
                "source_url": BASE + "maestro-v3.0.0.zip",
                "license": "CC BY-NC-SA 4.0", "zip_crc32": info.CRC,
                "sha256": hashlib.file_digest(destination.open("rb"), "sha256").hexdigest(),
            })
            records.append(row)
            print(f"Ready {destination.name}: {destination.stat().st_size / 1e6:.1f} MB", flush=True)
    (ROOT / "sources.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (ROOT / "DATA_LICENSE.txt").write_text(
        "MAESTRO v3.0.0, Google LLC; International Piano-e-Competition performances.\n"
        "Dataset: https://magenta.withgoogle.com/datasets/maestro\n"
        "License: CC BY-NC-SA 4.0 (noncommercial; attribution; share-alike).\n"
        "https://creativecommons.org/licenses/by-nc-sa/4.0/\n"
        "Citation: Hawthorne et al., Enabling Factorized Piano Music Modeling and "
        "Generation with the MAESTRO Dataset, ICLR 2019.\n"
        "Pilot clips are excerpts; sources.json and clips.json preserve provenance.\n"
        "This pilot is for local noncommercial research; no commercial clearance is claimed.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

