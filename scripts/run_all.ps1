$ErrorActionPreference = "Stop"
$project = Split-Path -Parent $PSScriptRoot
$python = "C:\Users\LENOVO\Desktop\cs231n\eurosat_landcover\.venv\Scripts\python.exe"

Push-Location $project
try {
    & $python -m pytest -q
    & $python -m seg_transformer.cli --config configs/base.json download
    & $python -m seg_transformer.cli --config configs/base.json prepare
    & $python -m seg_transformer.cli --config configs/base.json --override configs/smoke.json train --model resnet34_transformer --run-id smoke_resnet34_transformer
}
finally {
    Pop-Location
}
