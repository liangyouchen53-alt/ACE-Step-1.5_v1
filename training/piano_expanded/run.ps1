param(
    [ValidateSet('prepare','preprocess','train','evaluate','compare')]
    [string]$Action = 'prepare'
)
$ErrorActionPreference = 'Stop'
$taskRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$taskPython = Join-Path $taskRoot '.venv\Scripts\python.exe'
Set-Location -LiteralPath $taskRoot
$env:PYTHONUTF8 = '1'
$env:HF_HOME = Join-Path $taskRoot '.hf-cache'
$env:HF_MODULES_CACHE = Join-Path $env:HF_HOME 'modules'
$env:TOKENIZERS_PARALLELISM = 'false'
$taskData = 'training/piano_expanded'
function Invoke-ExpandedPython([string[]]$TaskArguments, [string]$LogName) {
    $ErrorActionPreference = 'Continue'
    & $taskPython -X utf8 @TaskArguments 2>&1 | Tee-Object -FilePath (Join-Path $PSScriptRoot $LogName)
    $ErrorActionPreference = 'Stop'
    if ($LASTEXITCODE -ne 0) { throw "Failed: $LogName" }
}
if ($Action -ne 'prepare') {
    & (Join-Path $taskRoot 'training/piano_pilot/run.ps1') -Action stop-web
}
switch ($Action) {
    'prepare' {
        Invoke-ExpandedPython @("$taskData/download.py") 'download.log'
        Invoke-ExpandedPython @("$taskData/prepare.py") 'prepare.log'
    }
    'preprocess' {
        foreach ($taskSplit in @('train','heldout')) {
            Invoke-ExpandedPython @('train.py','--yes','--plain','fixed',
                '--checkpoint-dir','checkpoints','--model-variant','turbo',
                '--dataset-dir',"$taskData/tensors/$taskSplit",'--output-dir',"$taskData/lora",
                '--preprocess','--dataset-json',"$taskData/$taskSplit.json",
                '--audio-dir',"$taskData/clips/$taskSplit",'--tensor-output',"$taskData/tensors/$taskSplit",
                '--max-duration','20','--device','cuda','--precision','bf16') "preprocess-$taskSplit.log"
        }
    }
    'train' {
        if (Test-Path "$taskData/lora/final/adapter_model.safetensors") {
            throw 'Existing expanded adapter found; archive lora before starting another run.'
        }
        Invoke-ExpandedPython @('train.py','--yes','--plain','fixed',
            '--checkpoint-dir','checkpoints','--model-variant','turbo',
            '--dataset-dir',"$taskData/tensors/train",'--output-dir',"$taskData/lora",
            '--device','cuda','--precision','bf16','--rank','8','--alpha','16',
            '--batch-size','1','--gradient-accumulation','4','--epochs','6',
            '--lr','0.00005','--warmup-steps','6','--optimizer-type','adafactor',
            '--gradient-checkpointing','--offload-encoder','--num-workers','0',
            '--save-every','1','--log-every','1','--sample-every-n-epochs','0') 'train.log'
    }
    'evaluate' { Invoke-ExpandedPython @("$taskData/evaluate.py") 'evaluate.log' }
    'compare' {
        Invoke-ExpandedPython @("$taskData/compare.py") 'compare.log'
        Invoke-ExpandedPython @("$taskData/listening.py") 'listening.log'
    }
}
