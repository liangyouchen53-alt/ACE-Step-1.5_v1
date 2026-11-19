"""Conservative prompt-based adapter routing; ambiguous requests use the base model."""

import re

AUTO = "依提示詞自動判斷"
BASE = "原始模型"
PATTERNS = {
    "鋼琴": r"鋼琴|钢琴|\bpiano\b",
    "頌缽": r"[頌颂宋][缽钵鉢]|藏缽|西藏音缽|\b(?:singing|tibetan|himalayan|crystal)\s*[- ]?bowls?\b",
    BASE: r"吉他|小提琴|大提琴|長笛|薩克斯|手碟|空靈鼓|鼓組|合成器|"
          r"\b(?:guitars?|violin|cello|flute|saxophone|handpan|drums?|synthesizers?|accordion|bass|strings)\b",
}


def mentioned_instruments(caption):
    """Return positively requested families, excluding local negated lists."""
    text = caption.lower()
    names = "|".join(PATTERNS.values())
    item = rf"(?:{names})"
    text = re.sub(
        rf"(?:\bno\b|\bwithout\b|不要|沒有|不含|無|避免)\s*"
        rf"(?:任何|使用|加入)?\s*{item}(?:\s*(?:、|,|和|與|及|或|and|or)\s*{item})*",
        "", text,
    )
    return {name for name, pattern in PATTERNS.items() if re.search(pattern, text)}


def detect_instrument(caption):
    """Detect explicit instrument names, excluding simple negative requests."""
    names = mentioned_instruments(caption)
    return next(iter(names)) if len(names) == 1 else BASE
