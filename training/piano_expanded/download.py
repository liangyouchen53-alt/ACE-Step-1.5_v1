"""Fetch bounded, attributed piano subsets from official/public-domain sources."""

import hashlib
import json
from pathlib import Path
from urllib.parse import quote

import requests
from remotezip import RemoteZip

ROOT = Path(__file__).resolve().parent
SMD_URL = "https://zenodo.org/api/records/13753319/files/SMD-piano_v2.zip/content"
SMD_WORKS = {
    "Bach_BWV871-01": "train", "Bach_BWV875-01": "train",
    "Beethoven_Op027No1-01": "train", "Beethoven_Op027No1-03": "train",
    "Brahms_Op010No2": "train", "Chopin_Op028-04": "train",
    "Chopin_Op028-15": "train", "Chopin_Op048No1": "train",
    "Beethoven_Op031No2-02": "heldout", "Bach_BWV849-01": "heldout",
}


def digest(path):
    """Hash an audio file without retaining it in memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    """Download selected lossless recordings; preserve hashes and license evidence."""
    raw = ROOT / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    records = []
    previous = ROOT.parent / "piano_pilot"
    for old in json.loads((previous / "sources.json").read_text(encoding="utf-8")):
        records.append({
            "id": "maestro_" + old["id"], "dataset": "MAESTRO v3",
            "audio_path": str((previous / old["local_audio"]).resolve()),
            "split": old["pilot_split"], "work": old["canonical_title"],
            "license": old["license"], "source_url": old["source_url"],
            "sha256": old["sha256"],
        })
    with RemoteZip(SMD_URL, timeout=120) as archive:
        names = archive.namelist()
        for work, split in SMD_WORKS.items():
            name = next(n for n in names if n.startswith("wav_44100_stereo/" + work + "_")
                        and n.endswith(".wav"))
            info = archive.getinfo(name)
            path = raw / Path(name).name
            if not path.exists() or path.stat().st_size != info.file_size:
                print("Downloading SMD", work, flush=True)
                partial = path.with_suffix(".partial")
                with archive.open(name) as source, partial.open("wb") as target:
                    while chunk := source.read(1024 * 1024):
                        target.write(chunk)
                partial.replace(path)
            records.append({
                "id": "smd_" + work, "dataset": "SMD v2", "work": work,
                "audio_path": str(path.resolve()), "split": split,
                "source_url": SMD_URL, "archive_member": name,
                "license": "CC BY-NC-SA 3.0", "sha256": digest(path),
                "zip_crc32": info.CRC,
            })
    response = requests.get("https://archive.org/metadata/OpenGoldbergVariations", timeout=60)
    response.raise_for_status()
    metadata = response.json()
    assert "publicdomain/zero/1.0" in metadata["metadata"]["licenseurl"]
    (ROOT / "goldberg_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    for track in ("01", "14"):
        entry = next(f for f in metadata["files"]
                     if f["name"].endswith(".flac") and f" - {track} " in f["name"])
        path = raw / f"goldberg_{track}.flac"
        url = "https://archive.org/download/OpenGoldbergVariations/" + quote(entry["name"])
        if not path.exists() or path.stat().st_size != int(entry["size"]):
            print("Downloading Goldberg", track, flush=True)
            with requests.get(url, stream=True, timeout=(30, 120)) as stream:
                stream.raise_for_status()
                partial = path.with_suffix(".partial")
                with partial.open("wb") as target:
                    for chunk in stream.iter_content(1024 * 1024):
                        target.write(chunk)
                partial.replace(path)
        with path.open("rb") as stream:
            assert hashlib.file_digest(stream, "md5").hexdigest() == entry["md5"]
        records.append({
            "id": "goldberg_" + track, "dataset": "Open Goldberg Variations",
            "work": "Bach Goldberg Variations BWV988", "split": "train",
            "audio_path": str(path.resolve()), "source_url": url,
            "license": "CC0 1.0", "sha256": digest(path),
            "performer": "Kimiko Ishizaka", "title": entry["name"],
        })
    (ROOT / "sources.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Ready:", len(records), "recordings", flush=True)


if __name__ == "__main__":
    main()
