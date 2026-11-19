"""Separate requested bowl performance from timbre conditioning."""

TECHNIQUES = {
    "依提示詞": "",
    "輕柔敲擊": "Gentle mallet strikes followed by natural ringing decay.",
    "沿缽緣摩擦": "Continuous circular rim rubbing, sustained humming resonance.",
    "敲擊與摩擦交替": "Alternating gentle mallet strikes and circular rim rubbing.",
}
DENSITIES = {
    "依提示詞": "",
    "稀疏": "Very sparse events with long quiet pauses and spacious natural decay.",
    "適中": "Unhurried events with clear space for resonance between gestures.",
}


def performance_caption(caption, technique, density):
    """Compose explicit requests without claiming precise event scheduling."""
    if technique not in TECHNIQUES or density not in DENSITIES:
        raise ValueError("未知的頌缽演奏設定")
    return " ".join(part for part in (caption.strip(), TECHNIQUES[technique],
                                      DENSITIES[density]) if part)
