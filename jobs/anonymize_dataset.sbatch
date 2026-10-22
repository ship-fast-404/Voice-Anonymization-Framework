#!/bin/bash
#SBATCH --job-name=anon_ds
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=normal
#SBATCH --time=06:00:00
#SBATCH --output=slurm_%x_%j.out
#
# ANONYMIZE A DATASET FOLDER TREE with a pipeline profile (default: combined):
#   $DATASET_DIR/<group>/<file>.wav  ->  $OUT_DIR/<group>/<file>.wav   (OUT_DIR default: <dataset>_anonymized)
# Every folder holding wavs is one group (the same people recorded several times): each person gets ONE target voice
# for all the group's files. Speakers per file: NUM_SPEAKERS (default 2), or per folder in $DATASET_DIR/speakers.csv
# (folder,num_speakers).
#   stage    src/anonymize_dataset.py stage: safe ids, <run>/wav links, <run>/manifest.csv
#   diar     DiariZen per speaker count                              -> <run>/rttm_hyp/<id>.rttm
#   vc       src/anonymize.py --groups <run>/manifest.csv             -> <run>/<OUT_TAG>/<id>.wav, logs/*group_refs.json
#   collect  back to the original names and folders                    -> $OUT_DIR (+ anonymization_targets.csv)
#
# Settings: configs/run.env, then configs/run.local.env (this machine), then exported variables (sbatch --export).
#   cd ~/anon-bench && sbatch jobs/anonymize_dataset.sbatch
#   cd ~/anon-bench && sbatch --export=ALL,DATASET_DIR=$HOME/anon-bench/dataset2 jobs/anonymize_dataset.sbatch
set -uo pipefail
export BENCH_ROOT="${BENCH_ROOT:-$HOME/anon-bench}"
PIPE="$BENCH_ROOT"
# run settings: a KEY=VALUE file never overrides a variable exported for this job; run.local.env beats run.env;
# a relative *_DIR / *_ROOT / HF_HOME path is under the cluster root
PRESET=" $(compgen -e | tr '\n' ' ') "
load_settings() {
  local f="$1" line k v
  [ -f "$f" ] || return 0
  while IFS= read -r line || [ -n "$line" ]; do
    line="${line%%#*}"
    [[ "$line" =~ ^[[:space:]]*([A-Z_][A-Z0-9_]*)=[[:space:]]*([^[:space:]]*) ]] || continue
    k="${BASH_REMATCH[1]}"; v="${BASH_REMATCH[2]}"
    [ -z "$v" ] && continue
    case "$PRESET" in *" $k "*) continue;; esac
    if [[ "$v" =~ ^[A-Za-z]:[\\/] ]]; then echo "NOTE: $f: $k=$v is a Windows path, ignored here"; continue; fi
    if [[ "$v" != /* && ( "$k" == *_DIR || "$k" == *_ROOT || "$k" == HF_HOME ) ]]; then v="$BENCH_ROOT/$v"; fi
    export "$k=$v"
  done < "$f"
}
load_settings "$PIPE/configs/run.env"
load_settings "$PIPE/configs/run.local.env"
export PIPE MODELS_DIR="${MODELS_DIR:-$BENCH_ROOT/models}"
DATASET_DIR="${DATASET_DIR:-$BENCH_ROOT/dataset}"; DATASET_DIR="${DATASET_DIR%/}"
OUT_DIR="${OUT_DIR:-${DATASET_DIR}_anonymized}"
PIPELINE="${PIPELINE:-combined}"
PROFILE="$PIPE/configs/pipelines/$PIPELINE.env"
[ -f "$PROFILE" ] || { echo "ERROR: no profile $PROFILE"; exit 1; }
set -a; source "$PROFILE"; set +a                     # OUT_TAG MODEL CKPT CFG_YML F0 REF_STRATEGY MAX_SPEAKERS STEPS CFG REF_SEC CTX
RUN_TAG="${RUN_TAG:-ds_$(basename "$DATASET_DIR")_$PIPELINE}"
RUN_DIR="$BENCH_ROOT/runs/$RUN_TAG"; LOG_DIR="$RUN_DIR/logs"; mkdir -p "$LOG_DIR"
STAGES="${STAGES:-stage diar vc collect}"
NUM_SPEAKERS="${NUM_SPEAKERS:-2}"
REF_POOL="${REF_POOL:-${REF_DIR:-$BENCH_ROOT/refs}}"
LANG_MODE="${LANG_MODE:-${LANGUAGE:-auto}}"
DIARIZE_BATCH="${DIARIZE_BATCH:-32}"
PROMPT_CAP="${PROMPT_CAP:-}"
DIARIZEN_MODEL="${DIARIZEN_MODEL:-BUT-FIT/diarizen-wavlm-large-s80-md-v2}"
ENV_DIAR="$BENCH_ROOT/envs/diarizen"; ENV_SVC="$BENCH_ROOT/envs/seedvc"; SEEDVC="${SEEDVC_ROOT:-$BENCH_ROOT/repos/seed-vc}"

source "$PIPE/env/conda_env.sh" || exit 1
export HF_HOME="${HF_HOME:-$BENCH_ROOT/hf_cache}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_CACHE="$SEEDVC/checkpoints/hf_cache"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
JID="${SLURM_JOB_ID:-local}"
echo "=== anonymize_dataset PIPELINE=$PIPELINE dataset=$DATASET_DIR out=$OUT_DIR run=$RUN_DIR stages=[$STAGES] host=$(hostname) job=$JID $(date -Is)"
echo "    profile: ckpt=${CKPT:-default} f0=$F0 ref=$REF_STRATEGY steps=$STEPS cfg=$CFG prompt=${PROMPT_CAP:-$REF_SEC}s ctx=${CTX}s"
echo "    run settings: language=$LANG_MODE speakers=$NUM_SPEAKERS diarize_batch=$DIARIZE_BATCH refs=$REF_POOL"
has() { case " $STAGES " in *" $1 "*) return 0;; *) return 1;; esac; }
tstage() { echo "$1,$2,$3" >> "$LOG_DIR/stage_times.csv"; }
cp "$PROFILE" "$RUN_DIR/pipeline_profile.env"

if has stage; then
  activate_env "$ENV_SVC"
  echo; echo "=== stage  $(date -Is)"
  python "$PIPE/src/anonymize_dataset.py" stage --dataset "$DATASET_DIR" --work "$RUN_DIR" --num-speakers "$NUM_SPEAKERS" || exit 1
fi

if has diar; then
  echo; echo "=== diar  $(date -Is)"; T0=$(date +%s)
  for L in "$RUN_DIR"/files_n*.txt; do
    N="$(basename "$L" .txt)"; N="${N#files_n}"
    echo "--- $(wc -l < "$L") files with $N speaker(s)"
    ( activate_env "$ENV_DIAR" && python "$PIPE/src/diarize.py" --wav-dir "$RUN_DIR/wav" --out-rttm-dir "$RUN_DIR/rttm_hyp" \
          --log-dir "$LOG_DIR" --model "$DIARIZEN_MODEL" --cache-dir "$HF_HOME/diarizen" \
          --files "$L" --max-speakers "$N" --num-speakers "$N" --batch-size "$DIARIZE_BATCH" --tag "$RUN_TAG" ) \
          2>&1 | tee -a "$LOG_DIR/stage_diar_$JID.log" | grep -E '^\[|Error|Traceback|speaker cap'
  done
  tstage diar 0 $(( $(date +%s) - T0 ))
fi

if has vc; then
  activate_env "$ENV_SVC"
  echo; echo "=== vc  $(date -Is)"; T0=$(date +%s)
  python "$PIPE/src/anonymize.py" --seedvc-root "$SEEDVC" --wav-dir "$RUN_DIR/wav" --log-dir "$LOG_DIR" \
      --rttm-dir "$RUN_DIR/rttm_hyp" --groups "$RUN_DIR/manifest.csv" --mode diarized --no-warmup \
      --out-dir "$RUN_DIR/$OUT_TAG" --tag "$RUN_TAG" \
      --profile "$PROFILE" --ref-dir "$REF_POOL" --lang "$LANG_MODE" --ref-exclude "" \
      --granularity segment --loudness match --non-speech attenuate --attenuate-db 20 ${PROMPT_CAP:+--ref-max-sec "$PROMPT_CAP"} \
      2>&1 | tee "$LOG_DIR/stage_vc.log" | grep -E '^\[|profile|pools|group |Error|Traceback'
  tstage vc "${PIPESTATUS[0]}" $(( $(date +%s) - T0 ))
fi

if has collect; then
  activate_env "$ENV_SVC"
  echo; echo "=== collect  $(date -Is)"
  python "$PIPE/src/anonymize_dataset.py" collect --work "$RUN_DIR" --anon-dir "$RUN_DIR/$OUT_TAG" --out "$OUT_DIR" \
      --group-refs "$LOG_DIR/anonymize_${RUN_TAG}_group_refs.json"
  echo; echo "=== done.  anonymized: $OUT_DIR   targets: $OUT_DIR/anonymization_targets.csv   logs: $LOG_DIR"
  echo "fetch (laptop PowerShell, project folder):  scp -r <user>@<master>:$OUT_DIR ."
fi
