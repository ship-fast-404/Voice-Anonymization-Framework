# Laptop (Windows) setup for both pipelines and the demo UI. Run from PowerShell in the project folder:
#   powershell -ExecutionPolicy Bypass -File env\local\setup_local.ps1
# Creates the conda env `seedvc` (python 3.10, torch 2.7.1 cu128), clones Seed-VC into <Root>\seed-vc (a short path:
# Windows trips over long paths in the model cache) and downloads the models of both pipelines (~8 GB):
# Seed-VC checkpoints into <Root>\seed-vc\checkpoints, the combined pipeline's DiT into <Root>\models\seedvc_stock.
# Two-speaker audio also needs the DiariZen env: bash env/local/setup_diarizen_local.sh
param(
  [string]$Root = "C:\anon-local",
  [string]$EnvName = "seedvc",
  [string]$TorchIndex = "https://download.pytorch.org/whl/cu128"
)
$ErrorActionPreference = "Stop"
$project = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$condaRoot = "$env:USERPROFILE\miniconda3"
$conda = "$condaRoot\Scripts\conda.exe"
$py = "$condaRoot\envs\$EnvName\python.exe"
New-Item -ItemType Directory -Force $Root | Out-Null

if (-not (Test-Path $py)) {
  Write-Host "== creating conda env $EnvName (python 3.10)"
  & $conda create -y -n $EnvName python=3.10
}
Write-Host "== torch $TorchIndex"
& $py -m pip install --upgrade pip
& $py -m pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url $TorchIndex

if (-not (Test-Path "$Root\seed-vc\inference.py")) {
  Write-Host "== cloning seed-vc"
  git clone https://github.com/Plachtaa/seed-vc.git "$Root\seed-vc"
}
Write-Host "== seed-vc requirements (torch pins and GUI-only packages dropped)"
Get-Content "$Root\seed-vc\requirements.txt" |
  Where-Object { $_ -notmatch '^(torch|torchvision|torchaudio|gradio|FreeSimpleGUI|sounddevice|resemblyzer|jiwer|--)' } |
  Set-Content "$Root\req_filtered.txt"
& $py -m pip install -r "$Root\req_filtered.txt"
& $py -m pip install soundfile scipy imageio-ffmpeg gradio hydra-core omegaconf
& $py -m pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url $TorchIndex   # re-assert
& $py -c "import torch, transformers; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'transformers', transformers.__version__)"

Write-Host "== models of both pipelines"
$env:HF_HOME = "$Root\hf_cache"
& $py "$project\env\stage_models_seedvc.py" --seedvc-root "$Root\seed-vc" --models-dir "$Root\models"
Write-Host "== done. Paths are the defaults of configs\run.env. Next: bash env/local/setup_diarizen_local.sh, then app\run_demo.ps1"
