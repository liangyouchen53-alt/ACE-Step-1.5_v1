"""Protect explicit instrument intent; use the local LM only for unspecified requests."""

from .instrument_routing import BASE, detect_instrument, mentioned_instruments


def resolve_prompt(caption, llm_handler, seconds, use_lm=True):
    """Return caption, adapter family and an honest explanation of the decision."""
    caption = caption.strip()
    if not caption:
        raise ValueError("請先描述想聽的音樂。")
    if mentioned_instruments(caption):
        return caption, detect_instrument(caption), "保留你指定的樂器"
    if not use_lm or not getattr(llm_handler, "llm_initialized", False):
        return caption, BASE, "未指定樂器，使用原始模型；LM 未啟用或尚未初始化"
    from acestep.inference import format_sample

    result = format_sample(
        llm_handler, caption, "[Instrumental]",
        user_metadata={"duration": float(seconds), "language": "unknown"},
        temperature=0.3,
    )
    if not result.success or not result.caption.strip():
        return caption, BASE, "LM 整理失敗，保留原始描述"
    # Do not let an inferred solo adapter override explicit exclusions.
    proposed = detect_instrument(result.caption)
    if any(word in caption.lower() for word in ("不要", "沒有", "不含", "無", "without", "no ")):
        return caption, BASE, "保留原始排除條件，未採用 LM 擴寫"
    if proposed == BASE:
        return caption, BASE, "LM 未提出單一主題，保留原始描述"
    anchor = "Solo acoustic piano. " if proposed == "鋼琴" else "Solo acoustic singing bowl. "
    return anchor + caption, proposed, "LM 建議配器（未指定樂器；保留原始風格描述）"


def bowl_caption(caption):
    """Anchor bilingual bowl timbre without inventing a playing technique."""
    hints = []
    rubbing = any(word in caption for word in ("摩擦", "磨缽", "磨缽緣", "繞缽"))
    striking = any(word in caption for word in ("敲擊", "輕敲", "敲缽"))
    # Leave negated/complex performance instructions untouched rather than guessing.
    negated = any(word in caption for word in ("不要", "沒有", "不含", "避免", "不敲"))
    if not negated:
        if rubbing and striking:
            hints.append("Mallet strikes and circular rim rubbing.")
        elif rubbing:
            hints.append("Continuous gentle circular rim rubbing with sustained resonance.")
        elif striking:
            hints.append("Gentle mallet strikes with natural ringing decay.")
    if any(word in caption for word in ("平靜", "放鬆", "平穩")):
        hints.append("Calm, restrained dynamics and smooth gradual transitions.")
    return (
        "Solo acoustic singing bowl, resonant bowl tones with metallic overtones "
        "and natural sustained decay. Instrumental only. " + " ".join(hints)
        + " " + caption.strip()
    )
