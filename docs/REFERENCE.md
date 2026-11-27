# Voice Anonymization: full reference

(The short overview is in the repository [README](../README.md).)

Anonymizes recorded doctor-patient consultations before they leave the hospital: every speaker's voice is replaced by
an anonymous target voice, while **what** was said, the speaker's **gender** and the **language** are kept. Everything is
zero-shot: every model is used as released, nothing is trained on your data.

Two pipelines share all the code. A pipeline is one settings file in `configs/pipelines/`:

| Pipeline | Voice conversion | Target voice | Settings |
|---|---|---|---|
| **`combined`** (default) | Seed-VC V1, stock XLS-R "tiny" DiT, BigVGAN v2 22 kHz, no pitch branch | same pitch class, least similar timbre (CAM++); at most 2 voices per file | 30 steps, guidance 1.0, 10 s prompt, 3 s context |
| **`L`** | Seed-VC V1, 44.1 kHz DiT with F0 conditioning, Whisper-small content encoder | same gender, nearest pitch | 25 steps, guidance 0.7, 25 s prompt, 2 s context |

## How a recording is anonymized

1. **Diarization** (DiariZen): who speaks when, at most `NUM_SPEAKERS` (default 2) speakers.
2. **Speaker analysis**: per speaker, the median pitch gives the gender class (split at 165 Hz) and CAM++ gives the
   timbre; Whisper-small identifies the language, which picks the target-voice pool `refs/<lang>/`.
3. **Target voice** per speaker, by the pipeline's rule, always of the speaker's gender.
4. **Voice conversion** (Seed-VC): every turn is re-synthesised in its target voice with the same words.
5. **Reassembly** on the original timeline (loudness matched, background between turns kept at -20 dB): a 16 kHz wav.

## Layout

```
configs/run.env           THE settings file: pipeline, language, speakers, batch size, paths, port (edit this)
configs/pipelines/        combined.env, L.env: model, checkpoint and settings of each pipeline
configs/models/           the Seed-VC model config (yaml) of the combined pipeline
src/anonymize.py          diarized wav -> anonymized wav (one file, a folder, or --groups: one voice per person per group)
src/anonymize_dataset.py  dataset/<sub-folder>/<files> <-> the pipeline, and back to dataset_anonymized/
src/diarize.py            DiariZen -> <id>.rttm
src/build_ref_pool.py     cut a target-voice pool for a new language from a corpus
app/                      the demo UI (anon_demo.py) and its launcher (run_demo.ps1)
jobs/                     anonymize_dataset.sbatch: a dataset folder on a Slurm cluster
env/                      setup: cluster (setup_envs.sh, stage_models.sh), laptop (local/)
refs/hi/                  12 anonymous Hindi target voices (6 male, 6 female) + manifest.json with their gender
docs/                     the quality and time report (pdf) and the cluster guide (docs/cluster/)
```

## Setup

**Laptop (Windows, NVIDIA GPU; 8 GB is enough):**
```powershell
powershell -ExecutionPolicy Bypass -File env\local\setup_local.ps1      # conda env "seedvc", Seed-VC, models of both pipelines
bash env/local/setup_diarizen_local.sh                                   # DiariZen env (diarization)
HF_TOKEN=hf_... C:/anon-local/envs/diarizen/python.exe env/stage_models_diarizen.py --cache-dir C:/anon-local/hf_cache/diarizen
```
**Ubuntu (NVIDIA GPU, conda installed):** everything lives in the project folder (`envs/`, `repos/`, `hf_cache/`,
`models/` are git-ignored). Run from the project folder. The torch builds below are on CUDA 13.0: the CUDA 12.8 builds
pinned for the cluster cannot compile kernels for a Blackwell GB10 (compute capability 12.1); on an older GPU the cu128
builds work as well.
```bash
git clone --recurse-submodules https://github.com/BUTSpeechFIT/DiariZen.git repos/DiariZen
git clone https://github.com/Plachtaa/seed-vc.git repos/seed-vc
conda create -y -p envs/diarizen python=3.10 && conda create -y -p envs/seedvc python=3.10

# env 1/2: diarizen (onnxruntime-gpu has no arm64 wheel and the PyTorch embedding does not need it)
P=$PWD/envs/diarizen/bin/python; TORCH="torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0"; IDX=https://download.pytorch.org/whl/cu130
$P -m pip install $TORCH --index-url $IDX
grep -viE '^onnxruntime-gpu' repos/DiariZen/requirements.txt > /tmp/diarizen_req.txt
$P -m pip install onnxruntime -r /tmp/diarizen_req.txt
( cd repos/DiariZen && $P -m pip install -e . )
grep -viE '^(torch|torchaudio|torchvision)([=<>!~ ]|$)' repos/DiariZen/constraints.txt > /tmp/constraints_notorch.txt
( cd repos/DiariZen/pyannote-audio && $P -m pip install -e . -c /tmp/constraints_notorch.txt )
$P -m pip install toml soundfile && $P -m pip install $TORCH --index-url $IDX        # re-assert torch

# env 2/2: seedvc (gradio 6 pulls huggingface-hub 2.x, which transformers 4.46 rejects: keep gradio 5)
P=$PWD/envs/seedvc/bin/python; TORCH="torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1"
$P -m pip install $TORCH --index-url $IDX
grep -viE '^(torch|torchvision|torchaudio|gradio|FreeSimpleGUI|sounddevice|resemblyzer|jiwer|--)' repos/seed-vc/requirements.txt > /tmp/seedvc_req.txt
$P -m pip install -r /tmp/seedvc_req.txt
$P -m pip install soundfile scipy imageio-ffmpeg "gradio<6" "huggingface-hub<1.0" hydra-core omegaconf
$P -m pip install $TORCH --index-url $IDX                                            # re-assert torch

# models of both pipelines (~28 GB). The pyannote embedding is gated: accept the terms at
# https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM, then
export HF_TOKEN=hf_... HF_HOME=$PWD/hf_cache
envs/seedvc/bin/python env/stage_models_seedvc.py --seedvc-root repos/seed-vc --models-dir models
( cd repos/DiariZen && ../../envs/diarizen/bin/python ../../env/stage_models_diarizen.py --cache-dir ../../hf_cache/diarizen )
```
Then point the project at these paths (relative paths are under the project folder):
```
# configs/run.local.env
SEEDVC_ROOT=repos/seed-vc
MODELS_DIR=models
HF_HOME=hf_cache
DIARIZEN_PYTHON=envs/diarizen/bin/python
```
`env/setup_envs.sh` is for the cluster (it calls conda's activation, which needs the shell hook; the commands above call
each env's python by path instead).

**Run a dataset on Ubuntu** with the same job script as the cluster, from the project folder (`dataset/<sub-folder>/<files>`
-> `dataset_anonymized/`, settings from `configs/run.env`):
```bash
BENCH_ROOT=$PWD bash jobs/anonymize_dataset.sbatch                       # or: DATASET_DIR=/path/to/folder BENCH_ROOT=$PWD bash jobs/...
```

**Cluster (Slurm; the master node has internet, the compute nodes do not):** copy the repository to `~/anon-bench`, then on
the master node
```bash
bash env/setup_envs.sh                 # conda envs diarizen + seedvc, Seed-VC and DiariZen checkouts
export HF_TOKEN=hf_...                 # accept the terms of pyannote/wespeaker-voxceleb-resnet34-LM first
bash env/stage_models.sh               # models of both pipelines -> hf_cache/, repos/seed-vc/checkpoints/, models/
```

## Settings: `configs/run.env`

Every run setting is in `configs/run.env`, with comments: `PIPELINE` (`combined` or `L`), `LANGUAGE` (`auto` or a code
such as `hi`), `NUM_SPEAKERS`, `DIARIZE_BATCH`, `PROMPT_CAP`, `DATASET_DIR`, `OUT_DIR`, the model and environment paths and
the UI `PORT`. Edit it and run; an empty value means the default. Settings for one machine only go in
`configs/run.local.env` (not in the repository), which overrides `run.env`. For an 8 GB GPU:
```
# configs/run.local.env
DIARIZE_BATCH=8          # the default 32 needs ~11 GB; on 8 GB it spills into shared memory and runs ~20x slower
PROMPT_CAP=10            # lets pipeline L fit 8 GB
```
Order, last wins: default < `run.env` < `run.local.env` < exported variable (cluster job) < command-line flag.

## Run it: the UI (single file or a dataset .zip)

```powershell
powershell -ExecutionPolicy Bypass -File app\run_demo.ps1                # Windows: opens http://127.0.0.1:7860
```
```bash
envs/seedvc/bin/python app/anon_demo.py                                  # Ubuntu: then open http://127.0.0.1:7860; Ctrl+C stops it
```
The **Pipeline** dropdown switches between `combined` and `L` in both tabs.

**Single audio.** Upload or record one file in any format (wav, mp3, m4a, ...). Choose *Single speaker* or *Two
speakers*, the language (it chooses the target-voice pool) and press **Anonymize**: the anonymized audio plays on the right
and a text box lists each speaker's pitch, gender and target voice.

**Dataset folder.** Zip the dataset folder and drop the .zip in the tab (or type the folder's path on this computer):
```
dataset/                         dataset_anonymized/
  consultation_01/                 consultation_01/
    mic1.wav ... mic4.wav    ->      mic1.wav ... mic4.wav
  consultation_02/                 consultation_02/
    ...                              ...
                                   anonymization_targets.csv
```
Every sub-folder holds recordings of the **same people** (for example one consultation from 4 microphones). Each person
gets **one** target voice for all files of the sub-folder, always of their own gender, and the output keeps the folder
structure and the file names. The result is offered as `dataset_anonymized.zip`; a table shows every person's target voice
and the text box shows the progress while it runs.

## Run it: a dataset on the cluster

```bash
scp -r dataset <user>@<master>:~/anon-bench/                         # from the laptop
cd ~/anon-bench && sbatch jobs/anonymize_dataset.sbatch              # PIPELINE etc. from configs/run.env
sbatch --export=ALL,PIPELINE=L jobs/anonymize_dataset.sbatch         # or override a setting for one run
grep -E "group |CHECK" runs/ds_dataset_<pipeline>/logs/stage_vc.log  # people, target voices, flags
scp -r <user>@<master>:~/anon-bench/dataset_anonymized .             # back to the laptop
```
Stages: `stage` (safe file ids) -> `diar` -> `vc` -> `collect`; `--export=ALL,STAGES="vc collect"` re-converts with the
existing diarization.

## Run it: one file from the command line

```bash
python src/diarize.py --wav-dir in --out-rttm-dir out/rttm --log-dir out/logs --cache-dir <hf_cache>/diarizen \
    --max-speakers 2 --num-speakers 2                                    # in the diarizen env
python src/anonymize.py --profile configs/pipelines/combined.env --seedvc-root <seed-vc checkout> \
    --models-dir <models dir> --wav-dir in --rttm-dir out/rttm --mode diarized --out-dir out/anon --log-dir out/logs \
    --tag combined --ref-dir refs --lang auto --ref-exclude "" --granularity segment --loudness match \
    --non-speech attenuate --attenuate-db 20                             # in the seedvc env
```
Use `configs/pipelines/L.env` for pipeline L. For one speaker without diarization: `--mode whole` and no `--rttm-dir`.
On Ubuntu with the layout above: `envs/diarizen/bin/python src/diarize.py ... --cache-dir hf_cache/diarizen` and
`envs/seedvc/bin/python src/anonymize.py ... --seedvc-root repos/seed-vc --models-dir models`.

## Dataset rules

- **Gender is always kept.** A speaker's gender class comes from their pitch over all files of the sub-folder. Speakers
  clearly on opposite sides of 165 Hz (below 150 Hz and above 180 Hz) are never treated as one person, and a target of the
  wrong gender stops that sub-folder with an error before anything is converted.
- **Speakers per recording:** `NUM_SPEAKERS` in `configs/run.env`, or per sub-folder in `dataset/speakers.csv`
  (`folder,num_speakers`).
- **Check column:** `anonymization_targets.csv` lists file, person, gender and target voice; its `check` column flags
  links worth a listen (a speaker heard in one file only, low voice similarity).
- Non-wav audio in the dataset is converted to 16 kHz wav; the output keeps the original name with `.wav`.

## Another language

Add `refs/<lang>/` with anonymous target clips (both genders) and a `manifest.json` with their gender (`src/build_ref_pool.py`
cuts them from a corpus; listen and fix the labels). With `LANGUAGE=auto` the language ID picks the pool.

## Built on

[Seed-VC](https://github.com/Plachtaa/seed-vc) (voice conversion), [DiariZen](https://github.com/BUTSpeechFIT/DiariZen)
(diarization), CAM++ (speaker embedding), Whisper-small (language ID), BigVGAN (vocoder).
