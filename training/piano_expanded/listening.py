"""Create a local, label-hidden listening sheet without inventing ratings."""

import csv
import html
import json
import os
import random
from pathlib import Path
from urllib.parse import quote

DATA = Path(__file__).resolve().parent


def main():
    """Randomize condition labels per seed; leave all subjective ratings empty."""
    records = json.loads((DATA / "comparison.json").read_text(encoding="utf-8"))
    sections, key, ratings = [], [], []
    names = {"original": "原模型", "pilot": "上一版 LoRA", "expanded": "擴充版 LoRA"}
    for seed in sorted({r["seed"] for r in records}):
        rows = [r for r in records if r["seed"] == seed]
        random.Random(seed).shuffle(rows)
        cards = []
        for letter, row in zip("ABC", rows):
            path = Path(row["path"])
            if not path.exists():
                raise FileNotFoundError(path)
            src = quote(os.path.relpath(path, DATA).replace("\\", "/"))
            label = f"{seed}-{letter}"
            key.append({"label": label, "model": row["model"], "path": row["path"]})
            ratings.append([label, "", "", "", "", "", ""])
            cards.append(
                f'<article><h3>樣本 {letter}</h3><audio controls preload="none" src="{src}"></audio>'
                '<p>在 CSV 表填寫：純鋼琴程度、平靜感、音色自然度、旋律連貫、結尾完整度。</p>'
                f'<details><summary>評分後再揭曉</summary>{html.escape(names[row["model"]])}'
                '</details></article>'
            )
        sections.append(f"<h2>固定種子 {seed}</h2>" + "".join(cards))
    page = '''<!doctype html><html lang="zh-Hant"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>純鋼琴模型試聽比較</title><style>
body{font:17px/1.7 system-ui;background:#f4f6f8;color:#182435;max-width:900px;margin:36px auto;padding:0 20px}
article{background:white;border:1px solid #dbe1e8;border-radius:12px;padding:20px;margin:14px 0}
audio{width:100%}summary{cursor:pointer}h1,h2{line-height:1.3}
</style><h1>純鋼琴模型試聽比較</h1>
<p>每組使用相同提示詞、種子、30 秒長度與後處理，只改變模型權重。A／B／C 的順序已打亂。</p>
<p>先在相同音量下聽完，再填寫「試聽評分.csv」：各項 1 分最差、5 分最好。額外樂器、人聲、爆音請寫入備註。
這是待填寫的試聽工具，並非已完成的人類盲測；不要把較大聲直接視為較佳。每組只有一個種子，兩組不足以涵蓋所有結果。</p>
<p>若清除「我的音樂」中的比較檔案，本頁對應播放器就會失效；可重新執行 compare 產生新樣本。</p>
'''
    (DATA / "試聽比較.html").write_text(page + "".join(sections) + "</html>", encoding="utf-8")
    (DATA / "listening_key.json").write_text(json.dumps(key, indent=2), encoding="utf-8")
    with (DATA / "試聽評分.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["樣本", "純鋼琴程度1至5", "平靜感1至5", "音色自然度1至5",
                         "旋律連貫1至5", "結尾完整度1至5", "備註"])
        writer.writerows(ratings)


if __name__ == "__main__":
    main()
