param(
    [ValidateSet('preprocess','train','resume','status')]
    [string]$Action = 'status',
    [int]$Epochs = 4,
    [string]$ResumeFrom = ''
)
$ErrorActionPreference = 'Stop'
$taskRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$taskPython = Join-Path $taskRoot '.venv\Scripts\python.exe'
Set-Location -LiteralPath $taskRoot
$env:PYTHONUTF8 = '1'
$env:PYTHONUNBUFFERED = '1'
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
$OutputEncoding = [Console]::OutputEncoding
$env:HF_HOME = Join-Path $taskRoot '.hf-cache'
$env:HF_MODULES_CACHE = Join-Path $env:HF_HOME 'modules'
$env:TOKENIZERS_PARALLELISM = 'false'
$taskData = 'training/singing_bowl'
function Invoke-LocalPython([string[]]$TaskArguments, [string]$LogName) {
    $ErrorActionPreference = 'Continue'
    & $taskPython -u -X utf8 @TaskArguments 2>&1 |
        ForEach-Object { $_.ToString() } |
        Tee-Object -FilePath (Join-Path $PSScriptRoot $LogName)
    $ErrorActionPreference = 'Stop'
    if ($LASTEXITCODE -ne 0) { throw "Failed. See $LogName" }
}
if ($Action -eq 'status') {
    Get-Content (Join-Path $PSScriptRoot 'preparation.json')
    Get-ChildItem (Join-Path $PSScriptRoot 'tensors/train') -Filter '*.pt' | Measure-Object
    exit
}
& (Join-Path $taskRoot 'training/piano_pilot/run.ps1') -Action stop-web
if ($Action -eq 'preprocess') {
    foreach ($taskSplit in @('train','heldout')) {
        Invoke-LocalPython @('train.py','--yes','--plain','fixed',
            '--checkpoint-dir','checkpoints','--model-variant','turbo',
            '--dataset-dir',"$taskData/tensors/$taskSplit",'--output-dir',"$taskData/lora",
            '--preprocess','--dataset-json',"$taskData/$taskSplit.json",
            '--audio-dir',"$taskData/clips/$taskSplit",'--tensor-output',"$taskData/tensors/$taskSplit",
            '--max-duration','20','--device','cuda','--precision','bf16') "preprocess-$taskSplit.log"
    }
    exit
}
if ($Epochs -lt 1) { throw 'Epochs must be positive.' }
if ($Action -eq 'train' -and (Test-Path "$taskData/lora")) {
    throw 'Existing lora folder found. Use resume or archive the folder before starting again.'
}
Invoke-LocalPython @("$taskData/verify.py") 'verify.log'
$taskArguments = @('train.py','--yes','--plain','fixed',
    '--checkpoint-dir','checkpoints','--model-variant','turbo',
    '--dataset-dir',"$taskData/tensors/train",'--output-dir',"$taskData/lora",
    '--device','cuda','--precision','bf16','--rank','8','--alpha','16',
    '--batch-size','1','--gradient-accumulation','4','--epochs',"$Epochs",
    '--lr','0.00005','--warmup-steps','2','--optimizer-type','adafactor',
    '--gradient-checkpointing','--offload-encoder','--num-workers','0',
    '--save-every','1','--log-every','1','--sample-every-n-epochs','0')
if ($Action -eq 'resume') {
    if (-not $ResumeFrom -or -not (Test-Path (Join-Path $ResumeFrom 'training_state.pt'))) {
        throw 'ResumeFrom must be a checkpoint folder containing training_state.pt.'
    }
    $taskArguments += @('--resume-from',$ResumeFrom)
}
Invoke-LocalPython $taskArguments ('train-' + (Get-Date -Format 'yyyyMMdd-HHmmss') + '.log')
