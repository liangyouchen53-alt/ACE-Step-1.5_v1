#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ "$(uname -s)" != Linux || "$(uname -m)" != aarch64 ]]; then
  echo '此安裝入口供 DGX Spark Linux ARM64 使用。' >&2
  exit 1
fi
command -v uv >/dev/null || { echo '請先安裝 uv：https://docs.astral.sh/uv/getting-started/installation/'; exit 1; }
command -v ffmpeg >/dev/null || { echo '請先執行 sudo apt install ffmpeg libsndfile1 libglib2.0-bin'; exit 1; }
uv sync --frozen --python 3.11
uv run --frozen python scripts/spark.py check
echo '安裝完成。下載模型：uv run --frozen acestep-download'
echo '啟動網站：bash start_spark.sh'
