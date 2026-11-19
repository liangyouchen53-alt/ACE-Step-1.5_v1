"""Classify known evaluation outputs without moving audio or breaking report links."""

GENERAL = "一般生成音樂"
TESTS = "模型測試音樂"
ALL = "全部音樂"
CATEGORIES = [GENERAL, TESTS, ALL]
TEST_DIRECTORIES = {"piano_pilot_comparison", "piano_expanded_comparison", "model_tests"}
TEST_FILES = {
    "batch_1794909050/d840c0e8-085d-aca9-df56-f85cfda2b94b.flac",
}


def filter_music(paths: list[str], category: str) -> list[str]:
    """Keep the requested category; unknown ordinary files remain visible by default."""
    if category not in CATEGORIES:
        raise ValueError("不支援的音樂分類。")
    result = []
    for path in paths:
        normalized = path.replace("\\", "/")
        is_test = normalized.split("/", 1)[0] in TEST_DIRECTORIES or normalized in TEST_FILES
        if category == ALL or (category == TESTS) == is_test:
            result.append(path)
    return result
