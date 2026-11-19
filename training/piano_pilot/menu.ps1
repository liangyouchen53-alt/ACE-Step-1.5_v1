$ErrorActionPreference = 'Stop'
$taskRunner = Join-Path $PSScriptRoot 'run.ps1'
while ($true) {
    Write-Host ''
    Write-Host '=== 純鋼琴 LoRA 訓練工具（不需要 GPT） ==='
    Write-Host '1. 下載範例資料並切片（需要網路；已下載會重用）'
    Write-Host '2. 預處理音樂（會停止音樂網站）'
    Write-Host '3. 開始小規模訓練（預設 4 epochs）'
    Write-Host '4. 從檢查點繼續訓練'
    Write-Host '5. 生成訓練前後比較音樂'
    Write-Host '6. 查看 GPU 與訓練權重'
    Write-Host '7. 開啟繁體中文操作手冊'
    Write-Host '8. 開啟音樂生成網站'
    Write-Host '0. 離開'
    $taskChoice = Read-Host '輸入選項'
    try {
        switch ($taskChoice) {
            '1' { & $taskRunner -Action prepare }
            '2' { & $taskRunner -Action preprocess }
            '3' {
                $taskEpochs = Read-Host '輸入總 epochs，直接 Enter 使用 4'
                if (-not $taskEpochs) { $taskEpochs = '4' }
                & $taskRunner -Action train -Epochs ([int]$taskEpochs)
            }
            '4' {
                $taskResume = Read-Host '貼上含 training_state.pt 的檢查點資料夾路徑'
                $taskEpochs = Read-Host '輸入訓練到第幾個 epoch（總目標，需大於已完成數）'
                & $taskRunner -Action resume -ResumeFrom $taskResume.Trim('"') -Epochs ([int]$taskEpochs)
            }
            '5' { & $taskRunner -Action compare }
            '6' { & $taskRunner -Action status }
            '7' { Start-Process -FilePath (Join-Path $PSScriptRoot '獨立訓練操作手冊.md') }
            '8' {
                $taskRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
                & (Join-Path $taskRoot 'start_local_5060.bat')
            }
            '0' { return }
            default { Write-Host '請輸入 0 到 8。' }
        }
    } catch { Write-Host $_.Exception.Message -ForegroundColor Red }
}
