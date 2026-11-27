# Voice Anonymization Framework

Anonymizes recorded doctor-patient consultations before they leave the hospital. Every speaker's voice is replaced by an
anonymous target voice, while **what** is said, the speaker's **gender** and the **language** are kept. Everything is
zero-shot: the models are used as released, nothing is trained on your data.

## How it works

1. **Diarization** (DiariZen): who speaks when.
2. **Speaker analysis**: pitch gives the gender, CAM++ the timbre, Whisper-small the language.
3. **Target voice**: one anonymous voice per speaker, always of the speaker's gender, from `refs/<language>/`.
4. **Voice conversion** (Seed-VC): every turn is re-synthesised in the target voice with the same words.
5. **Reassembly** on the original timeline: a 16 kHz wav with the same name.

Two pipelines, selected in `configs/run.env`: **`combined`** (default, fast, 22 kHz) and **`L`** (44.1 kHz with pitch
conditioning, slower). Both share all the code; each is one settings file in `configs/pipelines/`.

## Quick start (Ubuntu, NVIDIA GPU, conda)

Full setup commands for Ubuntu, Windows and a Slurm cluster are in [docs/REFERENCE.md](docs/REFERENCE.md). In short:

1. Clone Seed-VC and DiariZen into `repos/`, create the conda envs `envs/seedvc` and `envs/diarizen`.
2. Accept the terms of the gated model at https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM and set `HF_TOKEN`.
3. Download the models of both pipelines (~28 GB):
   ```bash
   envs/seedvc/bin/python env/stage_models_seedvc.py --seedvc-root repos/seed-vc --models-dir models
   ( cd repos/DiariZen && ../../envs/diarizen/bin/python ../../env/stage_models_diarizen.py --cache-dir ../../hf_cache/diarizen )
   ```
4. Point the project at these paths in `configs/run.local.env` (not in git):
   ```
   SEEDVC_ROOT=repos/seed-vc
   MODELS_DIR=models
   HF_HOME=hf_cache
   DIARIZEN_PYTHON=envs/diarizen/bin/python
   ```

## Run it

**A folder of recordings** (`dataset/<sub-folder>/<files>`, any audio format; every sub-folder holds recordings of the
same people, and each person keeps one target voice for the whole sub-folder):

```bash
NUM_SPEAKERS=2 BENCH_ROOT=$PWD bash jobs/anonymize_dataset.sbatch
```

The result is `dataset_anonymized/` with the same structure plus `anonymization_targets.csv` (file, person, gender,
target voice, and a `check` column that flags links worth a listen). Options, as variables in front of the command:
`NUM_SPEAKERS=1` for dictation, `DATASET_DIR=/path/to/folder` for another folder, `PIPELINE=L`, `LANGUAGE=hi`.
Per-folder speaker counts go in `dataset/speakers.csv` (`folder,num_speakers`). On a Slurm cluster the same script runs
with `sbatch`.

**The UI** (single file or a dataset .zip), at http://127.0.0.1:7860:

```bash
envs/seedvc/bin/python app/anon_demo.py          # Windows: powershell -File app\run_demo.ps1
```

## Settings

All run settings are in `configs/run.env` (pipeline, language, speakers, batch size, paths, port), with comments.
Machine-specific values go in `configs/run.local.env`, which overrides it. Order, last wins: `run.env` <
`run.local.env` < exported variable < command-line flag.

## Repository layout

```
app/        demo UI (anon_demo.py) and its Windows launcher
configs/    run.env (settings), pipelines/ (combined.env, L.env), models/ (Seed-VC model config)
docs/       REFERENCE.md (full guide), the quality and time report, the cluster guide
env/        environment setup and model download scripts
jobs/       anonymize_dataset.sbatch: a whole dataset folder, locally or on a Slurm cluster
refs/hi/    12 anonymous Hindi target voices (6 male, 6 female) with their gender in manifest.json
src/        diarize.py, anonymize.py, anonymize_dataset.py, build_ref_pool.py (a target-voice pool for a new language)
```

## Built on

[Seed-VC](https://github.com/Plachtaa/seed-vc) (voice conversion), [DiariZen](https://github.com/BUTSpeechFIT/DiariZen)
(diarization), CAM++ (speaker embedding), Whisper-small (language ID), BigVGAN (vocoder).
