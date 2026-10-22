#!/bin/bash
# Download all model weights for both pipelines into the cluster root, so the (offline) compute nodes can load them.
# Run ON THE MASTER NODE (internet) after env/setup_envs.sh:
#   export HF_TOKEN=hf_xxxxx          # the pyannote embedding model used by DiariZen is gated: accept its terms first
#   bash env/stage_models.sh
# Lands under $BENCH_ROOT:
#   hf_cache/diarizen/                 DiariZen + wespeaker embedding
#   repos/seed-vc/checkpoints/         Seed-VC models of both pipelines (+ XLS-R, BigVGAN v2, whisper-small)
#   models/seedvc_stock/               the stock XLS-R DiT of the combined pipeline
set -euo pipefail
BENCH_ROOT="${BENCH_ROOT:-$HOME/anon-bench}"
DIARIZEN_MODEL="${DIARIZEN_MODEL:-BUT-FIT/diarizen-wavlm-large-s80-md-v2}"
cd "$BENCH_ROOT"
source "$BENCH_ROOT/env/conda_env.sh" || exit 1
export HF_HOME="$BENCH_ROOT/hf_cache"
export CUDA_VISIBLE_DEVICES=""            # the master node has no GPU
[ -n "${HF_TOKEN:-}" ] || echo "WARNING: HF_TOKEN not set - the gated pyannote embedding download will fail unless you ran huggingface-cli login"

echo "== DiariZen models"
(
  activate_env "$BENCH_ROOT/envs/diarizen"
  cd "$BENCH_ROOT/repos/DiariZen"
  python "$BENCH_ROOT/env/stage_models_diarizen.py" --model "$DIARIZEN_MODEL" --cache-dir "$BENCH_ROOT/hf_cache/diarizen"
)

echo "== Seed-VC models of both pipelines"
(
  activate_env "$BENCH_ROOT/envs/seedvc"
  python "$BENCH_ROOT/env/stage_models_seedvc.py" --seedvc-root "$BENCH_ROOT/repos/seed-vc" --models-dir "$BENCH_ROOT/models"
)

echo
echo "== all models staged. Next: put your data in $BENCH_ROOT/dataset/<sub-folder>/*.wav and sbatch jobs/anonymize_dataset.sbatch"
