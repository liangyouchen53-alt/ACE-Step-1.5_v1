"""Validated controls using existing ACE-Step conditioning, not new model branches."""

import math

import numpy as np
import soundfile as sf

ENDINGS = {"保留餘韻": 0.25, "柔和淡出": 2.0}


def timing_settings(seconds, ending):
    """Return exact duration and bounded fade plus a soft ending request."""
    seconds = float(seconds)
    if not math.isfinite(seconds) or not 5 <= seconds <= 300 or ending not in ENDINGS:
        raise ValueError("長度需介於 5～300 秒，並選擇有效收尾方式。")
    tail = min(5.0, seconds / 3)
    return {
        "audio_duration": seconds, "duration_auto": False,
        "fade_out_duration": min(ENDINGS[ending], seconds / 4),
    }, f"Let the final resonance decay during the last {tail:g} seconds, with no new strikes near the end."


def validate_reference(path):
    """Reject corrupt, silent, nonfinite or excessively long reference audio."""
    if not path:
        return None
    try:
        info = sf.info(path)
        if not 3 <= info.duration <= 120 or info.channels not in (1, 2):
            raise ValueError("參考音訊需為 3～120 秒、單聲道或雙聲道。")
        audio, _ = sf.read(path, dtype="float32", always_2d=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError("無法讀取參考音訊，請使用 WAV 或 FLAC。") from exc
    if not np.isfinite(audio).all() or np.sqrt(np.mean(audio ** 2)) < 0.0001:
        raise ValueError("參考音訊為靜音或含有無效數值。")
    return str(path)

