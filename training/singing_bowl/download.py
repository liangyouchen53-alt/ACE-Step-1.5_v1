"""Download public CC0 previews with provenance, without requiring an account."""

import hashlib
import json
import re
from pathlib import Path

import requests
import soundfile as sf

DATA = Path(__file__).resolve().parent
SOURCES = [
    ("573805", "hollandm", "train"),
    ("439235", "zambolino", "heldout"),
]


def main():
    """Save license evidence, original preview bytes and checksums."""
    (DATA / "sources").mkdir(parents=True, exist_ok=True)
    rows = []
    for sound_id, author, split in SOURCES:
        page = f"https://freesound.org/people/{author}/sounds/{sound_id}/"
        response = requests.get(page, timeout=60)
        response.raise_for_status()
        html = response.text
        if "creativecommons.org/publicdomain/zero/1.0" not in html:
            raise ValueError("CC0 license evidence missing: " + page)
        matches = re.findall(r'https://cdn\.freesound\.org/previews/[^\s\"<>]+-hq\.mp3', html)
        if not matches:
            raise ValueError("Public HQ preview missing: " + page)
        url = matches[0]
        path = DATA / "sources" / f"{sound_id}.mp3"
        if not path.exists():
            audio = requests.get(url, timeout=120)
            audio.raise_for_status()
            sf.info(__import__("io").BytesIO(audio.content))
            path.write_bytes(audio.content)
        info = sf.info(path)
        (DATA / "sources" / f"{sound_id}.html").write_text(html, encoding="utf-8")
        rows.append({"id": sound_id, "author": author, "split": split,
                     "page": page, "download_url": url, "license": "CC0-1.0",
                     "file": str(path.relative_to(DATA)), "format": "lossy HQ MP3 preview",
                     "seconds": info.duration, "sample_rate": info.samplerate,
                     "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    (DATA / "sources.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()

