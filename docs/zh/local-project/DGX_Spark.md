# NVIDIA DGX Spark 執行版本

適用 GB10、Linux ARM64（aarch64）。沿用專案已鎖定的 PyTorch 2.10 CUDA 13.0 ARM64 相依套件。不要複製 Windows 的 `.venv`、`.python` 或 CUDA 套件到 Spark。

## 安裝與啟動

在 Spark 終端執行（私人 repo 請先登入 GitHub）：

```bash
git clone https://github.com/liangyouchen53-alt/ACE-Step-1.5_v1.git
cd ACE-Step-1.5_v1
sudo apt update
sudo apt install -y ffmpeg libsndfile1 libglib2.0-bin
```

依 [uv 官方安裝說明](https://docs.astral.sh/uv/getting-started/installation/)安裝 uv，再執行：

```bash
bash setup_spark.sh
uv run --frozen acestep-download
bash start_spark.sh
```

安裝腳本會先用現有 `uv.lock` 建立 Python 3.11 環境，再實際執行 CUDA BF16 矩陣運算；失敗會停止，請保留錯誤訊息，確認主機驅動與套件相容性。首次下載模型需要網路與足夠磁碟空間。

網站預設只監聽 Spark 的 `127.0.0.1:7860`。若從 Windows 操作，在 Windows 終端建立 SSH 通道：

```bash
ssh -L 7861:127.0.0.1:7860 <帳號>@<Spark位址>
```

然後在 Windows 瀏覽器開啟 `http://127.0.0.1:7861`。Spark 本機瀏覽器直接開 `http://127.0.0.1:7860`。網站保持左右雙欄、繁體中文與生成／音樂清單功能。Linux 清空功能透過 `gio trash` 送入 Spark 使用者回收筒；若無法使用會回報錯誤。

## 搬移已訓練模型及資料

GitHub 不包含模型、原始音訊、切片或張量。請透過 SFTP／scp 另行搬移需要的目錄，保持相對位置：

- `training/my_piano/lora/final/`：鋼琴快捷功能所需模型。
- `training/singing_bowl_expanded/lora/final/`：目前快捷頌缽模型。
- `training/singing_bowl_manual/`：若要沿用已處理的資料，搬移此目錄的 `clips/`、JSON 與 `tensors/`；原始 MP3 另在 `training/singing_bowl_new_sources/manual/`。
- `checkpoints/`：可搬移現有底座權重，或在 Spark 重新下載。

請勿公開尚未確認授權的手動下載素材。沒有搬移鋼琴或頌缽 LoRA 時，快捷流程會關閉先前 LoRA，使用原始模型並顯示狀態；仍可依提示詞生成。

## 頌缽訓練

網站終端先按 Ctrl+C。若搬移的是原始音檔，可重新整理切片：

```bash
uv run --frozen python training/singing_bowl_manual/prepare.py
```

若已搬移完整 `clips/` 與 JSON，可直接執行預處理。入口會轉換舊 Windows JSON 的反斜線路徑，並檢查檔案存在：

```bash
uv run --frozen python scripts/spark.py preprocess --dataset singing_bowl_manual
uv run --frozen python scripts/spark.py train --dataset singing_bowl_manual --epochs 4
```

完成後輸出 `training/singing_bowl_manual/lora/`。續訓使用 `resume --resume-from <含training_state.pt的checkpoint目錄>`，epochs 是總輪數目標。新訓練不自動取代網站快捷模型，需選擇新 LoRA 或調整快捷路徑。

## 驗證範圍

此版本在 Windows 做過入口邏輯與路徑處理檢查；尚未連上實體 DGX Spark 完成安裝、音樂生成或訓練。Spark 上需以 `setup_spark.sh` 的 CUDA 檢查及首次短音訊生成確認可用性。保留 PT 後端、CPU offload，先以 30 秒、單首測試，再延長至 300 秒。未宣稱 Spark 效能或生成音色已實機驗證。

參考：[NVIDIA DGX Spark 移植指南](https://docs.nvidia.com/dgx/dgx-spark-porting-guide/dgx-spark-porting-guide.pdf)。
