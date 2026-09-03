param(
    [ValidateSet('android', 'christian')]
    [string]$Dataset = 'christian',
    [int]$Epochs = 50
)

$ErrorActionPreference = 'Stop'
$Python = 'D:\conda\envs\cgt_gpu128\python.exe'
$Repo = Split-Path -Parent $MyInvocation.MyCommand.Path

if (-not (Test-Path -LiteralPath $Python)) {
    throw "The verified Python environment was not found: $Python"
}

Push-Location $Repo
try {
    & $Python main.py `
        --dataset $Dataset `
        --max_epochs $Epochs `
        --batch_size 64 `
        --learning_rate 0.0005 `
        --n_heads 10 `
        --print_steps 10

    if ($LASTEXITCODE -ne 0) {
        throw "Training failed with exit code $LASTEXITCODE"
    }

    & $Python inference.py `
        --dataset $Dataset `
        --batch_size 64 `
        --learning_rate 0.0005 `
        --n_heads 10 `
        --warmup_batches 2

    if ($LASTEXITCODE -ne 0) {
        throw "Inference failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
