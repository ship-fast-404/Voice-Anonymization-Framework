# Launch the local anonymization demo UI (opens http://127.0.0.1:7860 in the browser).
#   powershell -ExecutionPolicy Bypass -File app\run_demo.ps1
# Settings (paths, pipeline, language, speakers, diarization batch size, prompt cap, port) come from configs\run.env,
# overridden by configs\run.local.env (this machine only, not in the repository). The options below override both
# for one launch: -Pipeline L, -ModelsDir <folder holding seedvc_stock\DiT_uvit_tat_xlsr_ema.pth>, -PromptCap 10,
# -DiarizeBatch 8, -Port 7861. -EnvName picks the conda env that runs the UI (default seedvc).
param(
  [string]$EnvName = "seedvc",
  [string]$Pipeline,
  [string]$ModelsDir,
  [double]$PromptCap,
  [int]$DiarizeBatch,
  [int]$Port
)
$proj = Split-Path -Parent $PSScriptRoot
$py = "$env:USERPROFILE\miniconda3\envs\$EnvName\python.exe"
$env:PYTHONUNBUFFERED = "1"
$a = @("$proj\app\anon_demo.py")
if ($PSBoundParameters.ContainsKey("Pipeline"))     { $a += @("--default-pipeline", $Pipeline) }
if ($PSBoundParameters.ContainsKey("ModelsDir"))    { $a += @("--models-dir", $ModelsDir) }
if ($PSBoundParameters.ContainsKey("PromptCap"))    { $a += @("--prompt-cap", $PromptCap) }
if ($PSBoundParameters.ContainsKey("DiarizeBatch")) { $a += @("--diarize-batch", $DiarizeBatch) }
if ($PSBoundParameters.ContainsKey("Port"))         { $a += @("--port", $Port) }
& $py @a
