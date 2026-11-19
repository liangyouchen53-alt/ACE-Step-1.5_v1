"""Render a Traditional Chinese report from measured artifacts, without invented ratings."""

import json
from collections import defaultdict
from pathlib import Path

DATA = Path(__file__).resolve().parent


def read(name):
    """Read a completed experiment artifact."""
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def link(name, label=None):
    """Produce an absolute local artifact link."""
    return f"[{label or name}](<{(DATA / name).as_posix()}>)"


def main():
    """Build results tables directly from audits and paired evaluation outputs."""
    audit, verify = read("data_audit.json"), read("verification.json")
    evaluation, comparison = read("evaluation.json"), read("comparison.json")
    results = evaluation["results"]
    baseline = results["original"]["mean_mse"]
    labels = {"original": "原模型", "pilot": "上一版 LoRA", "expanded": "擴充版 LoRA"}
    table = "\n".join(
        f"| {labels[k]} | {v['mean_mse']:.6f} | {(v['mean_mse']/baseline-1)*100:+.3f}% |"
        for k, v in results.items()
    )
    clip_sources = {Path(c["audio_path"]).stem: c["source_id"] for c in read("clips.json")}
    grouped = {}
    for label, result in results.items():
        groups = defaultdict(list)
        for row in result["clips"]:
            groups[clip_sources[row["clip"]]].append(row["mse"])
        grouped[label] = {k: sum(v)/len(v) for k, v in groups.items()}
    source_table = "\n".join(
        f"| {source} | {grouped['original'][source]:.6f} | "
        f"{grouped['pilot'][source]:.6f} | {grouped['expanded'][source]:.6f} |"
        for source in sorted(grouped["original"])
    )
    checkpoints = sorted((DATA / "lora/checkpoints").glob("epoch_*"))
    training_table = "\n".join(
        f"| {p.name.split('_')[1]} | {p.name.split('_')[-1]} |" for p in checkpoints
    )
    paired_pilot = (results["expanded"]["mean_mse"] / results["pilot"]["mean_mse"] - 1) * 100
    report = f"""# 平靜純鋼琴模型：資料擴充與改善評估報告

報告日期：2026-11-18；訓練於 2026-11-17 執行。目標：ACE-Step 1.5 生成 30 秒、平靜、無人聲的純鋼琴。

已完成資料擴充、6 輪 LoRA 訓練及三版本比較。**這是完成實驗的報告，不代表已證明新模型的聽感更好。**
本輪只更新 Turbo 音樂生成模型的 LoRA，沒有訓練 1.7B 語言模型；原模型與上一版權重保留。

## 1. 找到並使用的資料

新增 [SMD v2](https://www.audiolabs-erlangen.de/resources/MIR/SMD/midi) 原始立體聲鋼琴錄音，以及 [Open Goldberg Variations](https://opengoldbergvariations.org/) 的公共領域鋼琴錄音；另沿用既有 [MAESTRO v3](https://magenta.withgoogle.com/datasets/maestro) 子集。
SMD 為 CC BY-NC-SA 3.0、MAESTRO 為 CC BY-NC-SA 4.0；Open Goldberg 為 CC0 1.0。本輪整體以非商業本機研究為範圍，未宣稱模型已取得商業用途授權。

完整候選資料集、未採用原因與官方連結：{link('資料集調查.md')}。來源署名與改作方式：{link('DATA_LICENSE.md')}。

| 項目 | 上一版試跑 | 本輪 |
|---|---:|---:|
| 實際使用的錄音來源集合 | 1 | 3 |
| 訓練錄音數 | 3 | 13 |
| 訓練片段數 | 24 | 93 |
| 每片段秒數 | 10 | 20 |
| 不重複訓練音訊長度 | 4 分鐘 | 31 分鐘（7.75 倍） |
| 保留評估錄音數 | 1 | 3 |
| 保留評估片段數 | 4 | 12 |
| 是否實際測量保留資料 MSE | 否 | 是 |

本輪訓練片段組成：MAESTRO 24、SMD 53、Open Goldberg 16。新下載原始音訊約 {audit['new_download_bytes']/1e6:.1f} MB，沒有下載整個百 GB 資料庫。
來源清單：{link('sources.json')}；切片起點與授權：{link('clips.json')}；資料檢查：{link('data_audit.json')}。

## 2. 資料品質與分割

只使用原始 WAV／FLAC 鋼琴錄音；擷取不重疊的 20 秒片段，首尾各做 20 毫秒淡化，排除靜音、接近削波及非有限數值。105 個片段均通過音訊檢查與預處理。
13 筆錄音用於訓練，另外 3 筆用於評估；來源音訊 SHA256 與命名曲目沒有跨分割重複，張量檔案也與兩份資料清單逐一相符。
本輪未對底座模型預訓練資料做去污染審計，因此只能說保留曲目未進入本次／前次 LoRA 訓練，不能保證底座從未見過它們。

安靜片段以 RMS 排序，不代表已完成情緒審核。訓練描述維持「古典鋼琴獨奏、無人聲」，未替未經聆聽的資料編造 BPM 或平靜情緒標籤。

## 3. 訓練實際結果

- 硬體：RTX 5060，約 8 GB VRAM；本輪沒有發生 OOM。數次觀察值為 6,960–7,612 MiB，這不是全程峰值；後段訓練速度明顯變慢，不應直接推估可以擴大批次或片長。
- 底座：acestep-v15-turbo；另起一個全新 LoRA，不是接續上一版。
- rank 8、alpha 16；batch 1、梯度累積 4；bf16、Adafactor、學習率 0.00005、warmup 6 steps。
- 梯度檢查點與 CPU offload；共 {verify['epochs']} epochs、{verify['optimizer_steps']} 次最佳化更新，保存 {verify['checkpoints']} 個可續訓檢查點。
- 本輪訓練程式回報總耗時 14 分 55 秒；訓練 seed 42。輪數以檢查點與狀態檔驗證，未採用上游摘要畫面誤顯示的「0 / 6」。
- 最終權重 {verify['adapter_bytes']/1e6:.2f} MB；所有值有限，{verify['nonzero_B_tensors']} 個 LoRA B 張量非零；舊版 SHA256 未變。

| Epoch | 訓練平均 loss |
|---|---:|
{training_table}

以上各輪包含隨機噪聲／時間步，不宜視為精確可比的品質分數。這次同時增加資料、延長片長並改變訓練更新次數，不能把模型差異單獨歸因於新資料集。

## 4. 相同保留資料的數值比較

12 個 20 秒片段，每片段使用相同的 2 個噪聲種子與 3 個固定時間步（0.2、0.5、0.8），共每模型 72 次條件式 flow MSE 測量。評估模式關閉 dropout；三個版本使用相同輸入。**數值越低越好，但只是代理指標。**

| 模型 | 平均 flow MSE | 相對原模型變化（負數較低） |
|---|---:|---:|
{table}

擴充版相對上一版的 MSE 變化為 **{paired_pilot:+.3f}%**。

| 保留錄音 | 原模型 | 上一版 | 擴充版 |
|---|---:|---:|---:|
{source_table}

原始逐片段與逐測量數值：{link('evaluation.json')}。只有 3 首獨立錄音，72 次測量不能當成 72 首獨立樣本；沒有統計顯著性結論。這個指標也不能證明無人聲、平靜程度或旋律自然度。

## 5. 生成與試聽比較

使用完全相同的英文平靜鋼琴提示詞、30 秒、65 BPM、8 步、shift 3、關閉 LM 思考與改寫、FLAC、-1 dB 峰值正規化與 1.5 秒淡出。
種子為 20261117 與 20261118，原模型／上一版／擴充版各產生兩首，共 {len(comparison)} 首。每個版本重新載入底座，避免前一個 LoRA 殘留。
所有音檔都是 30 秒、48 kHz，數值有效，輸出峰值低於滿刻度。部分音訊在正規化前峰值達 1.0；正規化後未超界不能證明原始生成沒有失真，仍需試聽。峰值與尾端淡化是共同後處理結果，不能算作新版模型特有的改善。

請開啟 {link('試聽比較.html')}，以同一播放音量比較隱藏名稱的 A／B／C，再填寫 {link('試聽評分.csv')}。尚未有人類評分，沒有填入虛構 MOS、情緒分數或偏好勝率。
生成紀錄：{link('comparison.json')}。音檔也可在網站「我的音樂」中的 piano_expanded_comparison 資料夾找到。

## 6. 結論與下一步

已改善資料多樣性、上下文長度、來源追溯與評估完整性，並交付可運行的新權重。是否更符合「平靜純鋼琴」，仍以試聽結果決定；本輪未把新版設為網站預設 LoRA。
優先補人工標註：剔除快速、激烈及有干擾的片段，為平靜程度、演奏密度、音色和樂句結束標籤；再逐步增加獨立曲目，避免只切出更多相似片段。
若依這份保留資料調參，它就應被當成驗證集，後續還要建立新的最終測試集。擴大至 30 秒片段或更高訓練量前，先用這份評分表確認方向。

新版權重資料夾：{link('lora/final')}。可在網站 LoRA 設定載入此資料夾；本次比較強度為 1.0。
無需 GPT 的重跑指令與操作步驟：{link('操作說明.md')}。權重校驗與完整性紀錄：{link('verification.json')}。
"""
    (DATA / "改善報告.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
