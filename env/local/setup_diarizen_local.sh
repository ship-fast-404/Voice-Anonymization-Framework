#!/bin/bash
# Local (Windows / Git Bash) DiariZen environment for the demo UI's diarization layer.
# Mirrors env/setup_envs.sh: fresh upstream clone + its vendored pyannote fork + torch cu128.
#   bash env/local/setup_diarizen_local.sh            (~10 min, ~3 GB of downloads)
# Then stage the models (gated wespeaker embedding needs HF_TOKEN in the shell):
#   HF_TOKEN=hf_... "$ROOT/envs/diarizen/python.exe" env/stage_models_diarizen.py --cache-dir "$ROOT/hf_cache/diarizen"
set -uo pipefail
ROOT="${ROOT:-/c/anon-local}"
CONDA="${CONDA:-$USERPROFILE/miniconda3/Scripts/conda.exe}"
ENV="$ROOT/envs/diarizen"
PYW="$(cygpath -w "$ENV")\\python.exe"
TORCH_SPEC="torch==2.7.1 torchaudio==2.7.1"
TORCH_INDEX="https://download.pytorch.org/whl/cu128"

mkdir -p "$ROOT/envs"
[ -d "$ROOT/DiariZen" ] || git clone --recurse-submodules https://github.com/BUTSpeechFIT/DiariZen.git "$ROOT/DiariZen"
( cd "$ROOT/DiariZen" && git submodule update --init --recursive || true )
[ -f "$ROOT/DiariZen/pyannote-audio/setup.py" ] || [ -f "$ROOT/DiariZen/pyannote-audio/pyproject.toml" ] || { echo "vendored pyannote fork missing"; exit 1; }
[ -d "$ENV" ] || "$CONDA" create -y -p "$(cygpath -w "$ENV")" python=3.10
"$PYW" -m pip install --upgrade pip
"$PYW" -m pip install $TORCH_SPEC --index-url "$TORCH_INDEX"
( cd "$ROOT/DiariZen" && "$PYW" -m pip install -r requirements.txt && "$PYW" -m pip install -e . )
( cd "$ROOT/DiariZen/pyannote-audio" && "$PYW" -m pip install -e . )
"$PYW" -m pip install toml soundfile
"$PYW" -m pip install $TORCH_SPEC --index-url "$TORCH_INDEX"     # re-pin in case a requirement downgraded torch
"$PYW" - <<'EOF'
import torch, torchaudio, pyannote.audio, diarizen
print("diarizen env OK: torch", torch.__version__, "cuda", torch.cuda.is_available(), "torchaudio", torchaudio.__version__)
EOF
echo "== done: $ENV"
