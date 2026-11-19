param(
    [ValidateSet('prepare','preprocess','train','resume','compare','status','stop-web')]
    [string]$Action = 'status',
    [int]$Epochs = 4,
    [string]$ResumeFrom = ''
)
$ErrorActionPreference = 'Stop'
$taskRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$taskPython = Join-Path $taskRoot '.venv\Scripts\python.exe'
Set-Location -LiteralPath $taskRoot
$env:PYTHONUTF8 = '1'
$env:HF_HOME = Join-Path $taskRoot '.hf-cache'
$env:HF_MODULES_CACHE = Join-Path $env:HF_HOME 'modules'
$env:TOKENIZERS_PARALLELISM = 'false'

$taskData = 'training/piano_pilot'

function Stop-MusicWebsite {
    $taskListener = Get-NetTCPConnection -LocalPort 7860 -State Listen -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($taskListener) {
        $taskServer = Get-Process -Id $taskListener.OwningProcess -ErrorAction SilentlyContinue
        if (-not $taskServer) { return }
        $taskServerPath = $taskServer.Path
        if ([string]::IsNullOrWhiteSpace($taskServerPath)) {
            $taskProcessInfo = Get-CimInstance Win32_Process -Filter "ProcessId = $($taskServer.Id)"
            $taskServerPath = $taskProcessInfo.ExecutablePath
        }
        if ([string]::IsNullOrWhiteSpace($taskServerPath)) {
            throw 'Cannot verify the website process path. Close the ACE-Step website terminal manually, then retry training.'
        }
        $taskAllowed = $taskServerPath -eq $taskPython -or
            $taskServerPath.StartsWith((Join-Path $taskRoot '.python\'), [StringComparison]::OrdinalIgnoreCase)
        if (-not $taskAllowed) { throw 'Port 7860 belongs to another program; nothing was stopped.' }
        Stop-Process -Id $taskServer.Id
        Write-Host 'ACE-Step website stopped to free GPU memory.'
    }
}

function Invoke-PilotPython([string[]]$TaskArguments, [string]$LogName) {
    $taskLog = Join-Path $PSScriptRoot $LogName
    $ErrorActionPreference = 'Continue'
    & $taskPython -X utf8 @TaskArguments 2>&1 | Tee-Object -FilePath $taskLog
    $ErrorActionPreference = 'Stop'
    if ($LASTEXITCODE -ne 0) { throw "Command failed. See $taskLog" }
}

switch ($Action) {
    'prepare' {
        Invoke-PilotPython @("$taskData/download_maestro.py") 'download.log'
        Invoke-PilotPython @("$taskData/prepare_clips.py") 'prepare.log'
    }
    'stop-web' { Stop-MusicWebsite }
    'preprocess' {
        Stop-MusicWebsite
        Invoke-PilotPython @('train.py','--yes','--plain','fixed',
            '--checkpoint-dir','checkpoints','--model-variant','turbo',
            '--dataset-dir',"$taskData/tensors",'--output-dir',"$taskData/lora",'--preprocess',
            '--dataset-json',"$taskData/dataset.json",'--audio-dir',"$taskData/clips/train",
            '--tensor-output',"$taskData/tensors",'--max-duration','10',
            '--device','cuda','--precision','bf16') 'preprocess.log'
    }
    { $_ -in 'train','resume' } {
        Stop-MusicWebsite
        if ($Action -eq 'train' -and (Test-Path (Join-Path $PSScriptRoot 'lora\final\adapter_model.safetensors'))) {
            throw 'Existing LoRA found. Use resume, or rename the lora folder to archive it before starting a new run.'
        }
        $taskArguments = @('train.py','--yes','--plain','fixed',
            '--checkpoint-dir','checkpoints','--model-variant','turbo',
            '--dataset-dir',"$taskData/tensors",'--output-dir',"$taskData/lora",
            '--device','cuda','--precision','bf16','--rank','8','--alpha','16',
            '--batch-size','1','--gradient-accumulation','4','--epochs',"$Epochs",
            '--lr','0.00005','--warmup-steps','2','--optimizer-type','adafactor',
            '--gradient-checkpointing','--offload-encoder','--num-workers','0',
            '--save-every','1','--log-every','1','--sample-every-n-epochs','0')
        if ($Action -eq 'resume') {
            if (-not $ResumeFrom) { throw 'Pass -ResumeFrom with a checkpoint folder.' }
            $taskArguments += @('--resume-from',$ResumeFrom)
        }
        Invoke-PilotPython $taskArguments ('train-' + (Get-Date -Format 'yyyyMMdd-HHmmss') + '.log')
    }
    'compare' {
        Stop-MusicWebsite
        Invoke-PilotPython @("$taskData/compare.py") 'compare.log'
    }
    'status' {
        Get-ChildItem (Join-Path $PSScriptRoot 'lora') -Recurse -Filter 'adapter_model.safetensors' -ErrorAction SilentlyContinue |
            Select-Object FullName,Length,LastWriteTime
        nvidia-smi
    }
}



