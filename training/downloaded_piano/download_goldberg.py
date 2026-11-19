"""Download and checksum the complete CC0 Open Goldberg FLAC release."""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote

import requests
import soundfile as sf

ROOT = Path(__file__).resolve().parent


def download(entry):
    """Fetch one original FLAC and verify its release checksum and decodability."""
    folder = ROOT / "OpenGoldberg_CC0"
    folder.mkdir(exist_ok=True)
    path = folder / entry["name"]
    url = "https://archive.org/download/OpenGoldbergVariations/" + quote(entry["name"])
    for attempt in range(3):
        try:
            if not path.exists():
                with requests.get(url, stream=True, timeout=(30, 120)) as response:
                    response.raise_for_status()
                    partial = path.with_suffix(".partial")
                    with partial.open("wb") as target:
                        for chunk in response.iter_content(1024 * 1024):
                            target.write(chunk)
                    partial.replace(path)
            with path.open("rb") as stream:
                checksum = hashlib.file_digest(stream, "md5").hexdigest()
            if checksum != entry["md5"]:
                raise ValueError("Checksum mismatch: " + path.name)
            info = sf.info(path)
            print("Verified:", path.name, flush=True)
            return {"file": str(path.relative_to(ROOT)), "url": url,
                    "md5": checksum, "bytes": path.stat().st_size,
                    "seconds": info.duration, "sample_rate": info.samplerate,
                    "license": "CC0 1.0", "performer": "Kimiko Ishizaka"}
        except requests.RequestException:
            if attempt == 2:
                raise


def main():
    """Save release evidence and a complete, validated audio inventory."""
    response = requests.get("https://archive.org/metadata/OpenGoldbergVariations", timeout=60)
    response.raise_for_status()
    metadata = response.json()
    assert "publicdomain/zero/1.0" in metadata["metadata"]["licenseurl"]
    (ROOT / "release_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    entries = [f for f in metadata["files"] if f["name"].lower().endswith(".flac")]
    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(download, entries))
    (ROOT / "manifest.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print("COMPLETE", len(records), "files", sum(r["bytes"] for r in records), "bytes", flush=True)


if __name__ == "__main__":
    main()
