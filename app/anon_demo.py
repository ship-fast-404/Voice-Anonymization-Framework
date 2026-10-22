#!/usr/bin/env python
"""
Local demo UI for the voice-anonymization pipelines (Gradio).

Two tabs:
  Single audio   : upload or record one file (below).
  Dataset folder : a .zip of a dataset folder, or its path on this computer: <dataset>/<sub-folder>/<files>, every
                   sub-folder the same people. Runs exactly the cluster code (src/anonymize_dataset.py + src/anonymize.py
                   --groups): one target voice per person per sub-folder, always of the speaker's gender, written to
                   <dataset>_anonymized/<sub-folder>/<file>.wav with anonymization_targets.csv (a "check" column flags
                   doubtful cross-file matches) and offered as a .zip.

Single audio: upload any audio file (wav, mp3, m4a, aac, ogg, flac, opus, webm, ...). It is converted to 16 kHz mono wav; for a
two-speaker recording DiariZen first finds who speaks when; each speaker gets a same-gender anonymous target voice from
refs/<lang>/ and Seed-VC re-synthesises every turn in that voice while keeping the words. A dropdown picks the pipeline:

  combined   (default) stock XLS-R generator + BigVGAN v2, timbre-distance target, speaker cap 2, 30 steps, cfg 1.0, 10 s prompt
  L          the previous production pipeline: 44 kHz F0-conditioned generator, nearest-pitch target, 25 steps, cfg 0.7

Both are read from configs/pipelines/<name>.env, the same files the cluster job uses, so the demo runs exactly what the
cluster runs. Only one generator is kept on the GPU at a time; switching pipelines reloads it (10-20 s).

  conda activate seedvc
  python app/anon_demo.py                 (or: powershell -ExecutionPolicy Bypass -File app\run_demo.ps1)

Settings come from configs/run.env (paths, pipeline, language, speakers, batch size, prompt cap, port), overridden by
configs/run.local.env when it exists (machine-specific, not in the repository), overridden by command-line flags.
On an 8 GB GPU set DIARIZE_BATCH=8 and PROMPT_CAP=10 there.
"""
import os
import sys
import time
import json
import argparse
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
sys.path.insert(0, str(SRC))
PROJECT = HERE.parent


# setting in configs/run.env -> (argparse destination, type, built-in default); relative paths are under the project
SETTINGS = {
    "SEEDVC_ROOT": ("seedvc_root", str, r"C:\anon-local\seed-vc"),
    "MODELS_DIR": ("models_dir", str, r"C:\anon-local\models"),
    "HF_HOME": ("hf_home", str, r"C:\anon-local\hf_cache"),
    "DIARIZEN_PYTHON": ("diarizen_python", str, r"C:\anon-local\envs\diarizen\python.exe"),
    "DIARIZEN_CACHE": ("diarizen_cache", str, None),                 # None: <HF_HOME>/diarizen
    "REF_DIR": ("ref_dir", str, "refs"),
    "PIPELINE": ("default_pipeline", str, "combined"),
    "LANGUAGE": ("language", str, "auto"),
    "NUM_SPEAKERS": ("num_speakers", int, 2),
    "DIARIZE_BATCH": ("diarize_batch", int, 32),
    "PROMPT_CAP": ("prompt_cap", float, 0.0),                      # 0: the pipeline's own prompt length
    "DEMO_OUT_DIR": ("out_dir", str, "results/demo"),
    "PORT": ("port", int, 7860),
}
PATH_KEYS = {"SEEDVC_ROOT", "MODELS_DIR", "HF_HOME", "DIARIZEN_PYTHON", "DIARIZEN_CACHE", "REF_DIR", "DEMO_OUT_DIR"}


def read_settings():
    """configs/run.env, then configs/run.local.env on top (both KEY=VALUE, # comments, ${VAR} expanded)."""
    from anonymize import read_profile
    out = {}
    for name in ("run.env", "run.local.env"):
        f = PROJECT / "configs" / name
        if f.exists():
            out.update({k: v for k, v in read_profile(f, project_root=PROJECT).items() if v != ""})
    return out


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seedvc-root", default=None)
    p.add_argument("--ref-dir", default=None)
    p.add_argument("--profiles-dir", default=str(PROJECT / "configs" / "pipelines"), help="one <name>.env per pipeline")
    p.add_argument("--models-dir", default=None,
                   help="${MODELS_DIR} of the profiles: holds seedvc_stock/DiT_uvit_tat_xlsr_ema.pth for the combined pipeline")
    p.add_argument("--default-pipeline", default=None)
    p.add_argument("--language", default=None, help="language preselected in the dropdowns (auto = Whisper language ID)")
    p.add_argument("--num-speakers", type=int, default=None, help="speakers per recording preselected in the dataset tab")
    p.add_argument("--hf-home", default=None)
    p.add_argument("--prompt-cap", type=float, default=None,
                   help="upper limit on the target prompt (s), 0 = none; 10 lets pipeline L fit an 8 GB GPU")
    p.add_argument("--chunk-sec", type=float, default=15.0,
                   help="single-speaker conversion chunk; prompt + chunk + 2x context must fit one 30 s diffusion window on a small GPU")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--diarizen-python", default=None,
                   help="python of the DiariZen env (stage 1 runs as a subprocess, its dependencies clash with Seed-VC's)")
    p.add_argument("--diarizen-cache", default=None)
    p.add_argument("--diarizen-model", default="BUT-FIT/diarizen-wavlm-large-s80-md-v2")
    p.add_argument("--diarize-batch", type=int, default=None,
                   help="DiariZen batch size (default 32, ~11 GB of GPU memory); 8 on an 8 GB GPU, which otherwise spills "
                        "into shared memory and runs ~20x slower")
    a = p.parse_args()
    conf = read_settings()
    for key, (dest, typ, default) in SETTINGS.items():
        if getattr(a, dest) is None:                      # a flag on the command line wins
            v = conf.get(key, default)
            if v is not None:
                v = typ(v)
                if key in PATH_KEYS and not Path(v).is_absolute():
                    v = str((PROJECT / v).resolve())
            setattr(a, dest, v)
    a.diarizen_cache = a.diarizen_cache or str(Path(a.hf_home) / "diarizen")
    a.prompt_cap = a.prompt_cap if a.prompt_cap and a.prompt_cap > 0 else float("inf")
    return a


ARGS = parse_args()
os.environ.setdefault("HF_HOME", ARGS.hf_home)
os.environ.setdefault("HF_HUB_OFFLINE", "0")

from anonymize import (bootstrap_seedvc, SeedVCEngine, LangID, load_ref_pools, pool_for, choose_refs,  # noqa: E402
                       estimate_f0_hz, match_loudness, place, fit_length, read_profile, speaker_embedding, PITCH_CLASS_HZ)
from common import read_rttm, merge_segments  # noqa: E402
DIARIZE_PY = str((SRC / "diarize.py").resolve())

ref_dir = Path(ARGS.ref_dir).resolve()
out_dir = Path(ARGS.out_dir).resolve()
out_dir.mkdir(parents=True, exist_ok=True)
PROFILES = {p.stem: read_profile(p, project_root=PROJECT, models_dir=ARGS.models_dir)
            for p in sorted(Path(ARGS.profiles_dir).resolve().glob("*.env"))}
if not PROFILES:
    sys.exit(f"no pipeline profiles in {ARGS.profiles_dir}")
svc = bootstrap_seedvc(ARGS.seedvc_root)          # chdir into seed-vc, so every path above was resolved first
import torch                                       # noqa: E402
import numpy as np                                 # noqa: E402
import soundfile as sf                             # noqa: E402
import torchaudio                                  # noqa: E402

POOLS = load_ref_pools(ref_dir, excl=[])
REFS = sorted({r for v in POOLS.values() for r in v})
if not REFS:
    sys.exit(f"no reference wavs under {ref_dir}")
REF_GENDER, REF_SOURCE = {}, {}
for r in REFS:
    man = Path(r).parent / "manifest.json"
    if man.exists():
        for m in json.loads(man.read_text(encoding="utf-8")):
            if m.get("file") == Path(r).name:
                REF_GENDER[r] = m.get("gender")
                REF_SOURCE[r] = m.get("source_file")
    if not REF_GENDER.get(r) and Path(r).stem[:1].lower() in "mf":
        REF_GENDER[r] = "male" if Path(r).stem[0].lower() == "m" else "female"

ENGINE = {"name": None, "eng": None, "prof": None}
LID = None


def truthy(v):
    return str(v).lower() in ("1", "true", "yes")


def get_engine(name):
    """The Seed-VC engine of pipeline `name`, loading it (and dropping the previous one) when it changes."""
    if ENGINE["name"] == name:
        return ENGINE["eng"], ENGINE["prof"]
    prof = PROFILES[name]
    if ENGINE["eng"] is not None:
        ENGINE["eng"] = None
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    prompt = min(float(prof.get("REF_SEC", 25)), ARGS.prompt_cap)
    ns = argparse.Namespace(f0_condition=truthy(prof.get("F0", "false")), checkpoint=prof.get("CKPT") or None,
                            config=prof.get("CFG_YML") or None, fp16=True, auto_f0_adjust=True, f0_shift=0,
                            diffusion_steps=int(prof.get("STEPS", 25)), cfg_rate=float(prof.get("CFG", 0.7)),
                            length_adjust=1.0, ref_max_sec=prompt)
    if ns.checkpoint and not Path(ns.checkpoint).exists():
        raise RuntimeError(f"pipeline '{name}': checkpoint not found: {ns.checkpoint} (pass --models-dir to where seedvc_stock/ lives)")
    print(f"loading pipeline '{name}': Seed-VC V1 (f0={ns.f0_condition}, ckpt={Path(ns.checkpoint).name if ns.checkpoint else 'default'}, "
          f"prompt {prompt:g}s, {ns.diffusion_steps} steps, cfg {ns.cfg_rate}) ...", flush=True)
    t0 = time.time()
    eng = SeedVCEngine(svc, ns)
    for r in REFS:
        eng.prepare_ref(r)
    print(f"  loaded in {time.time() - t0:.0f}s on {eng.device}; reference voices: "
          + ", ".join(f"{Path(r).stem}={REF_GENDER.get(r, '?')[:1]}/{(eng.refs[r]['f0_hz'] or 0):.0f}Hz" for r in REFS), flush=True)
    ENGINE.update(name=name, eng=eng, prof=prof)
    return eng, prof


def ffmpeg_bin():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def to_wav16k(path):
    """Any audio file -> (float32 mono 16 kHz array, path of the converted wav)."""
    src = Path(path)
    wav_path = out_dir / f"{src.stem}_16k.wav"
    try:
        x, sr = sf.read(str(src), dtype="float32", always_2d=True)   # wav, flac, ogg, mp3 (libsndfile >= 1.1)
    except Exception:
        cmd = [ffmpeg_bin(), "-y", "-loglevel", "error", "-i", str(src), "-ac", "1", "-ar", "16000", "-f", "wav", str(wav_path)]
        subprocess.run(cmd, check=True)
        x, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != 16000:
        x = torchaudio.functional.resample(torch.from_numpy(x)[None], sr, 16000)[0].numpy()
    x = x.astype(np.float32)
    sf.write(str(wav_path), x, 16000, subtype="PCM_16")
    return x, wav_path


def quiet_cut(x, nominal, window=2.0):
    """Sample index of the quietest 50 ms within +-window s of `nominal` seconds (avoids cutting inside a word)."""
    n = len(x)
    lo, hi = max(0, int((nominal - window) * 16000)), min(n, int((nominal + window) * 16000))
    if hi - lo < 1600:
        return min(n, int(nominal * 16000))
    frame = 800
    seg = x[lo:hi]
    m = len(seg) // frame
    e = (seg[: m * frame].reshape(m, frame) ** 2).mean(axis=1)
    return lo + int(e.argmin()) * frame


def diarize_file(wav_path, dur):
    """Stage 1 as a subprocess in the DiariZen env -> {spk: [(start, end)]} merged turns (gap 0.3 s, pad 0.1 s, min 0.5 s),
    speakers with < 2 s of speech merged into the main speaker (clustering blips)."""
    work = out_dir / "diar" / wav_path.stem
    work.mkdir(parents=True, exist_ok=True)
    (work / "files.txt").write_text(wav_path.stem + "\n", encoding="utf-8")
    cmd = [ARGS.diarizen_python, DIARIZE_PY, "--wav-dir", str(wav_path.parent), "--out-rttm-dir", str(work / "rttm"),
           "--log-dir", str(work / "logs"), "--cache-dir", ARGS.diarizen_cache, "--model", ARGS.diarizen_model,
           "--files", str(work / "files.txt"), "--max-speakers", "2", "--num-speakers", "2", "--no-warmup", "--tag", "demo",
           "--batch-size", str(ARGS.diarize_batch)]
    env = dict(os.environ, HF_HUB_OFFLINE="1", PYTHONIOENCODING="utf-8")
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(SRC))
    rttm = work / "rttm" / f"{wav_path.stem}.rttm"
    if r.returncode != 0 or not rttm.exists():
        tail = "\n".join((r.stdout + r.stderr).strip().splitlines()[-8:])
        raise RuntimeError(f"diarization failed (is the DiariZen env set up? env/local/setup_diarizen_local.sh)\n{tail}")
    raw = read_rttm(rttm)
    segs = merge_segments(raw, 0.3, 0.1, 0.5, dur)
    note = f"-> {len(segs)} speaker(s), {len(raw)} raw segments"
    if len(segs) > 1:
        tot = {k: sum(b - a for a, b in v) for k, v in segs.items()}
        main_spk = max(tot, key=tot.get)
        for k in [k for k in segs if k != main_spk and tot[k] < 2.0]:
            segs[main_spk] = sorted(segs[main_spk] + segs.pop(k))
            note += f", blip speaker {k} ({tot[k]:.1f} s) merged"
    return segs, note


def speaker_audio(xg, segs, max_s=60):
    pieces, tot = [], 0
    for a, b in segs:
        pieces.append(xg[:, int(a * 16000):int(b * 16000)])
        tot += int(b * 16000) - int(a * 16000)
        if tot > 16000 * max_s:
            break
    return torch.cat(pieces, dim=-1) if pieces else None


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def anonymize(file, pipeline, language, gender_override, audio_type, progress=None):
    if not file:
        return None, None, "Upload an audio file first."
    t_all = time.perf_counter()
    if progress is not None:
        progress((0, 1), desc=f"loading pipeline '{pipeline}'")
    eng, prof = get_engine(pipeline)
    strategy = prof.get("REF_STRATEGY", "pitch")
    cap = int(prof.get("MAX_SPEAKERS", 0) or 0)
    ctx_sec = float(prof.get("CTX", 2))
    ref_f0 = {r: eng.refs[r]["f0_hz"] for r in REFS}

    x, wav_path = to_wav16k(file)
    dur = len(x) / 16000
    if dur < 1.0:
        return None, str(wav_path), "The recording is shorter than one second."
    xg = torch.from_numpy(x)[None].to(eng.device)

    two = str(audio_type).lower().startswith("two")
    t_diar, diar_note = 0.0, ""
    if two:
        t0 = time.perf_counter()
        spk_segs, diar_note = diarize_file(wav_path, dur)
        t_diar = time.perf_counter() - t0
    else:
        step = ARGS.chunk_sec
        cuts = [0] + [quiet_cut(x, k * step) for k in range(1, int(np.ceil(dur / step)))] + [len(x)]
        cuts = sorted(set(c for c in cuts if 0 <= c <= len(x)))
        spk_segs = {"spk0": [(a / 16000, b / 16000) for a, b in zip(cuts[:-1], cuts[1:]) if b - a >= 1600]}
    speakers = sorted(spk_segs, key=lambda k: -sum(b - a for a, b in spk_segs[k]))

    # speaker analysis: pitch -> gender class (pitch rule), CAM++ embedding when the pipeline needs it
    need_emb = strategy == "timbre_far" or (cap > 0 and len(speakers) > cap)
    spk_f0, spk_gender, spk_emb = {}, {}, {}

    def analyse(spk):
        w = speaker_audio(xg, spk_segs[spk])
        spk_f0[spk] = estimate_f0_hz(w) if w is not None else None
        if gender_override in ("male", "female") and not two:
            spk_gender[spk] = gender_override
        elif spk_f0[spk] is not None:
            spk_gender[spk] = "male" if spk_f0[spk] < PITCH_CLASS_HZ else "female"
        if need_emb:
            spk_emb[spk] = speaker_embedding(eng, w)

    with torch.inference_mode():
        for spk in speakers:
            analyse(spk)
        # speaker cap: a surplus cluster joins the closest remaining voice
        cap_note = ""
        if cap > 0 and len(speakers) > cap:
            keep = speakers[:cap]
            for s_ in speakers[cap:]:
                def closeness(k):
                    if spk_emb.get(s_) is not None and spk_emb.get(k) is not None:
                        return cosine(spk_emb[s_], spk_emb[k])
                    return -abs((spk_f0.get(s_) or 0) - (spk_f0.get(k) or 0))
                target = max(keep, key=closeness)
                spk_segs[target] = sorted(spk_segs[target] + spk_segs.pop(s_))
                cap_note += f", cluster {s_} merged into {target}"
                for d in (spk_f0, spk_gender, spk_emb):
                    d.pop(s_, None)
            speakers = keep
            for spk in keep:
                analyse(spk)

    # language -> pool
    global LID
    lang, lang_prob = None, None
    if language == "auto":
        if LID is None:
            LID = LangID("openai/whisper-small", eng.device)
        lang, lang_prob = LID(x[: 16000 * 30])
    elif language != "none":
        lang = language
    pool, pool_name = pool_for(POOLS, lang) if language != "none" else (REFS, "all")

    # target voices
    ref_emb = {r: eng.refs[r]["style2"].detach().float().squeeze().cpu().numpy() for r in pool} if strategy == "timbre_far" else None
    ref_map = choose_refs(pool, speakers, Path(file).stem, strategy, spk_f0, {r: ref_f0[r] for r in pool},
                          spk_gender, REF_GENDER, REF_SOURCE, spk_emb=spk_emb if strategy == "timbre_far" else None, ref_emb=ref_emb)

    # conversion turn by turn (single speaker: chunks cut at quiet points) with context
    ctx = int(ctx_sec * 16000)
    if two:
        canvas = x * float(10 ** (-20 / 20))                 # room background kept at -20 dB between turns
        for ss in spk_segs.values():
            for a, b in ss:
                canvas[int(a * 16000):int(b * 16000)] = 0.0
    else:
        canvas = np.zeros(len(x), dtype=np.float32)
    jobs = [(spk, int(round(a * 16000)), int(round(b * 16000))) for spk in speakers for a, b in spk_segs[spk]]
    jobs = [(spk, a, b) for spk, a, b in jobs if b - a >= 800]
    t0 = time.perf_counter()
    with torch.inference_mode():
        for i, (spk, a, b) in enumerate(jobs):
            if progress is not None:
                progress((i, len(jobs)), desc=f"converting turn {i + 1}/{len(jobs)}")
            if torch.cuda.is_available() and i % 10 == 0:
                torch.cuda.empty_cache()
            a0, b0 = max(0, a - ctx), min(len(x), b + ctx)
            ts = {}
            y = eng.convert_to_16k(xg[:, a0:b0], eng.refs[ref_map[spk]], ts)
            y = fit_length(y[a - a0:a - a0 + (b - a)], b - a)
            y = match_loudness(y, x[a:b])
            place(canvas, y, a)
    t_vc = time.perf_counter() - t0
    peak = float(np.abs(canvas).max() + 1e-9)
    if peak > 0.99:
        canvas *= 0.99 / peak
    out_path = out_dir / f"{Path(file).stem}_{pipeline}_anonymized.wav"
    sf.write(str(out_path), canvas, 16000, subtype="PCM_16")
    wall = time.perf_counter() - t_all

    def target_desc(spk):
        r = ref_map[spk]
        sim = f", timbre similarity {cosine(spk_emb[spk], ref_emb[r]):.2f}" if ref_emb is not None and spk_emb.get(spk) is not None else ""
        return f"{Path(r).stem} ({REF_GENDER.get(r, '?')}, {ref_f0[r] or 0:.0f} Hz{sim})"
    spk_lines = "\n".join(
        f"  {spk}: {sum(b - a for a, b in spk_segs[spk]):.0f} s of speech, pitch {('%.0f Hz' % spk_f0[spk]) if spk_f0.get(spk) else 'n/a'} -> "
        f"{spk_gender.get(spk, 'unknown')} -> target {target_desc(spk)}" for spk in speakers)
    rule = {"pitch": "same gender, nearest pitch", "pitch_far": "same gender, farthest pitch",
            "timbre_far": "same pitch class, least similar timbre (CAM++)", "cross": "opposite gender, farthest pitch"}.get(strategy, strategy)
    info = (f"Pipeline: {pipeline}  ({prof.get('PIPELINE_NAME', pipeline)}; target rule: {rule}"
            f"{'; speaker cap ' + str(cap) if cap else ''})\n"
            f"Input: {Path(file).name} -> {wav_path.name} ({dur:.1f} s, 16 kHz mono)\n"
            f"Mode: {'two speakers (DiariZen, cap 2) ' + diar_note + cap_note if two else 'single speaker, ' + str(len(jobs)) + ' chunk(s) of ' + ('%.0f' % ARGS.chunk_sec) + ' s'}\n"
            f"Language: {lang or 'not used'}{f' (p={lang_prob:.2f})' if lang_prob else ''} -> pool '{pool_name}'"
            f"{'  (gender overridden)' if gender_override in ('male', 'female') and not two else '  (gender by pitch rule, 165 Hz split)'}\n"
            f"{spk_lines}\n"
            f"Model: Seed-VC V1 {'+ F0 conditioning (44.1 kHz)' if eng.f0_condition else '(22 kHz, no F0)'}, "
            f"{Path(prof.get('CKPT')).name if prof.get('CKPT') else 'default checkpoint'}, prompt {eng.ref_max_sec:g} s, "
            f"{eng.steps} steps, guidance {eng.cfg}, {len(jobs)} conversion call(s), {ctx_sec:g} s context\n"
            f"Time: {wall:.1f} s total = diarization {t_diar:.1f} s + conversion {t_vc:.1f} s  (real-time factor {wall / dur:.2f})")
    return str(out_path), str(wav_path), info


def release_engine():
    """Drop the in-process generator so a subprocess can have the GPU (one generator fits an 8 GB card)."""
    if ENGINE["eng"] is not None:
        ENGINE.update(name=None, eng=None, prof=None)
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_logged(cmd, log_path, on_line=None):
    """Run a pipeline step as a subprocess, its output into log_path (and to on_line). Returns the exit code."""
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    with open(log_path, "a", encoding="utf-8") as lf:
        proc = subprocess.Popen([str(c) for c in cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                encoding="utf-8", errors="replace", env=env, cwd=str(SRC))
        for line in proc.stdout:
            lf.write(line)
            if on_line:
                on_line(line)
        return proc.wait()


def anonymize_dataset(zip_file, folder, pipeline, language, num_speakers, progress=None, status=None):
    """A dataset folder (<dataset>/<folder>/<files>) through exactly the cluster code: src/anonymize_dataset.py stage,
    DiariZen per file, src/anonymize.py --groups (one target per person per folder, gender kept), collect.
    Source: a .zip of the dataset folder (a browser upload drops the sub-folders, a zip keeps them) or a folder path on
    this computer; the result goes to <dataset>_anonymized next to that folder, or under results/demo/datasets."""
    import re
    import csv
    import zipfile
    import shutil
    # status: a dict the UI polls while this runs ("step" = what is running now, "notes" = what has been decided)
    status = status if status is not None else {}
    status.update(step="starting", notes=[])

    def say(frac, desc):
        status["step"] = desc
        if progress is not None:
            progress(frac, desc=desc)
    base = out_dir / "datasets"
    if folder and str(folder).strip():
        root = Path(str(folder).strip().strip('"')).expanduser().resolve()
        if not root.is_dir():
            return None, [], f"No folder {root} on this computer."
        label, out_root = root.name, root.parent / f"{root.name}_anonymized"
    elif zip_file:
        zp = Path(zip_file)
        src = base / f"{zp.stem}_upload"
        if src.exists():
            shutil.rmtree(src)
        with zipfile.ZipFile(zp) as z:
            z.extractall(src)
        tops = [p for p in src.iterdir() if p.name != "__MACOSX"]
        # a zip of the folder itself holds one top folder; a zip of its contents holds the sub-folders directly
        root = tops[0] if len(tops) == 1 and tops[0].is_dir() else src
        label = root.name if root != src else zp.stem
        out_root = base / f"{label}_anonymized"
    else:
        return None, [], "Upload a .zip of the dataset folder, or type the path of the folder on this computer."
    name = re.sub(r"[^A-Za-z0-9_-]", "_", label) or "dataset"
    base.mkdir(parents=True, exist_ok=True)
    work = base / f"{name}_work"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    log_path = work / "ui_run.log"
    tag = f"ui_{name}"
    t_all = time.perf_counter()

    say((0, 4), "1/4 reading the folders")
    lines = []
    rc = run_logged([sys.executable, SRC / "anonymize_dataset.py", "stage", "--dataset", root, "--work", work,
                     "--num-speakers", int(num_speakers)], log_path, on_line=lines.append)
    if rc != 0:
        return None, [], "Reading the dataset failed:\n" + "".join(lines[-10:])
    with (work / "manifest.csv").open(newline="", encoding="utf-8") as fh:
        manifest = list(csv.DictReader(fh))
    n_files = len(manifest)
    stage_note = "".join(l for l in lines if l.startswith(("group", "skipped", "staged")))

    status["notes"] += [l.strip() for l in lines if l.startswith(("group", "skipped", "staged"))]
    # the in-process generator would share the 8 GB GPU with DiariZen and push both into slow shared memory
    release_engine()
    say((1, 4), f"2/4 diarization of {n_files} files")
    done = {"n": 0}

    def diar_line(line):
        if re.search(r"\] \[\d+/\d+\] ", line):
            done["n"] += 1
            say((done["n"], n_files), f"2/4 diarization: {done['n']}/{n_files} files")
    for lst in sorted(work.glob("files_n*.txt")):
        n = lst.stem[len("files_n"):]
        rc = run_logged([ARGS.diarizen_python, DIARIZE_PY, "--wav-dir", work / "wav", "--out-rttm-dir", work / "rttm",
                         "--log-dir", work / "logs", "--cache-dir", ARGS.diarizen_cache, "--model", ARGS.diarizen_model,
                         "--files", lst, "--max-speakers", n, "--num-speakers", n, "--no-warmup", "--tag", tag,
                         "--batch-size", ARGS.diarize_batch],
                        log_path, on_line=diar_line)
        if rc != 0:
            return None, [], f"Diarization failed (is the DiariZen env set up?). Log: {log_path}"

    say((2, 4), "3/4 loading the generator")
    prof = PROFILES[pipeline]
    prompt = min(float(prof.get("REF_SEC", 25)), ARGS.prompt_cap)
    conv = {"n": 0}
    msgs = []

    def vc_line(line):
        m = re.search(r"\] \[(\d+)/(\d+) ", line)
        if m:
            conv["n"] = int(m.group(1))
            say((conv["n"], n_files), f"3/4 converting: {conv['n']}/{n_files} files")
        if " group " in line or "CHECK" in line or "FAILED" in line or "gender check" in line:
            msgs.append(line.split("] ", 1)[-1].strip())
            status["notes"].append(msgs[-1])
    rc = run_logged([sys.executable, SRC / "anonymize.py", "--seedvc-root", ARGS.seedvc_root,
                     "--profile", Path(ARGS.profiles_dir).resolve() / f"{pipeline}.env", "--models-dir", ARGS.models_dir,
                     "--wav-dir", work / "wav", "--rttm-dir", work / "rttm", "--groups", work / "manifest.csv",
                     "--mode", "diarized", "--no-warmup", "--out-dir", work / "anon", "--log-dir", work / "logs", "--tag", tag,
                     "--ref-dir", ref_dir, "--lang", language, "--ref-exclude", "", "--ref-max-sec", prompt,
                     "--granularity", "segment", "--loudness", "match", "--non-speech", "attenuate", "--attenuate-db", "20"],
                    log_path, on_line=vc_line)
    if rc != 0:
        return None, [], f"Conversion failed. Log: {log_path}\n" + "\n".join(msgs[-10:])

    say((3, 4), "4/4 writing the anonymized folders")
    lines = []
    rc = run_logged([sys.executable, SRC / "anonymize_dataset.py", "collect", "--work", work, "--anon-dir", work / "anon",
                     "--out", out_root, "--group-refs", work / "logs" / f"anonymize_{tag}_group_refs.json"],
                    log_path, on_line=lines.append)
    table = []
    tcsv = out_root / "anonymization_targets.csv"
    if tcsv.exists():
        with tcsv.open(newline="", encoding="utf-8") as fh:
            table = [[r["file"], r["person"], r["gender_by_pitch"], r["target_voice"], r["check"]] for r in csv.DictReader(fh)]
    zip_out = shutil.make_archive(str(base / f"{name}_anonymized"), "zip", out_root) if out_root.exists() else None
    n_check = sum(1 for r in table if r[4])
    info = (f"Pipeline: {pipeline}; one target voice per person per folder, gender kept\n"
            f"Input: {root}\n{stage_note}"
            + "\n".join(m for m in msgs if not m.startswith("CHECK")) + "\n"
            + (f"{n_check} speaker(s) to check by ear (see the 'check' column):\n" + "\n".join(m for m in msgs if m.startswith("CHECK")) + "\n"
               if n_check else "No doubtful matches.\n")
            + "".join(lines[-3:])
            + f"Output: {out_root}\nTime: {time.perf_counter() - t_all:.0f} s. Full log: {log_path}")
    return zip_out, table, info


def build_ui():
    import gradio as gr
    langs = ["auto"] + sorted(k for k in POOLS if k != "neutral") + ["none"]
    names = list(PROFILES)
    default = ARGS.default_pipeline if ARGS.default_pipeline in names else names[0]
    desc = {n: f"{n}: " + ("XLS-R generator, timbre-distance target, speaker cap, 30 steps, cfg 1.0, 10 s prompt" if n == "combined"
                           else "44 kHz F0 generator, nearest-pitch target, 25 steps, cfg 0.7" if n == "L"
                           else f"{PROFILES[n].get('REF_STRATEGY')} target, {PROFILES[n].get('STEPS')} steps, cfg {PROFILES[n].get('CFG')}")
            for n in names}
    with gr.Blocks(title="Voice anonymization demo") as ui:
        gr.Markdown("## Voice anonymization demo\n"
                    "Anonymize **one audio file**, or a **whole dataset folder** whose sub-folders each hold recordings of the "
                    "same people. Each speaker gets a same-gender anonymous voice and Seed-VC re-synthesises every turn in that "
                    "voice while keeping the words. The pipeline dropdown runs the same settings files as the cluster "
                    "(`configs/pipelines/<name>.env`).\n\n" + "  \n".join(f"**{desc[n]}**" for n in names))
        with gr.Tabs():
            with gr.Tab("Single audio"):
                with gr.Row():
                    with gr.Column():
                        audio_in = gr.Audio(label="Original audio (any format)", type="filepath", sources=["upload", "microphone"])
                        pipeline = gr.Dropdown(names, value=default, label="Pipeline (model and settings)")
                        audio_type = gr.Dropdown(["Single speaker (dictation)", "Two speakers (doctor-patient)"],
                                                 value="Single speaker (dictation)", label="Type of recording")
                        language = gr.Dropdown(langs, value=ARGS.language if ARGS.language in langs else langs[0], label="Language of the speech (chooses the target-voice pool)")
                        gender = gr.Radio(["auto (pitch)", "male", "female"], value="auto (pitch)", label="Speaker gender (single-speaker mode only)")
                        btn = gr.Button("Anonymize", variant="primary")
                    with gr.Column():
                        audio_out = gr.Audio(label="Anonymized audio", type="filepath")
                        wav_out = gr.File(label="Converted input (16 kHz wav)")
                        info = gr.Textbox(label="What happened", lines=11)

            with gr.Tab("Dataset folder"):
                gr.Markdown("Every sub-folder is one group: the **same people** recorded in several files. Each person gets "
                            "**one** anonymous voice for all files of the sub-folder, always of their own gender. The output keeps "
                            "the folder structure and the file names (`<dataset>_anonymized/<sub-folder>/<file>.wav`).  \n"
                            "A browser cannot upload a folder with its sub-folders, so give it as a **.zip of the dataset folder**, or "
                            "type the **path of the folder** on this computer (the result is then written next to it).")
                with gr.Row():
                    with gr.Column():
                        ds_zip = gr.File(label="Dataset folder as .zip", file_types=[".zip"], type="filepath")
                        ds_path = gr.Textbox(label="...or the path of the dataset folder on this computer",
                                             placeholder=r"C:\Users\...\dataset")
                        ds_pipeline = gr.Dropdown(names, value=default, label="Pipeline (model and settings)")
                        ds_language = gr.Dropdown(langs, value=ARGS.language if ARGS.language in langs else langs[0], label="Language of the speech (chooses the target-voice pool)")
                        ds_speakers = gr.Number(value=ARGS.num_speakers, precision=0, minimum=1, maximum=6,
                                                label="Speakers per recording (a speakers.csv in the dataset folder overrides it per sub-folder)")
                        ds_btn = gr.Button("Anonymize the dataset", variant="primary")
                    with gr.Column():
                        ds_out = gr.File(label="Anonymized dataset (.zip)")
                        ds_table = gr.Dataframe(headers=["file", "person", "gender", "target voice", "check"],
                                                label="Target voice of every person (the same person keeps one voice per sub-folder)",
                                                interactive=False, wrap=True)
                        ds_info = gr.Textbox(label="What happened", lines=12)

        def run(f, pl, l, g, t, progress=gr.Progress()):
            return anonymize(f, pl, l, g.replace(" (pitch)", ""), t, progress)
        btn.click(run, [audio_in, pipeline, language, gender, audio_type], [audio_out, wav_out, info])

        def run_ds(z, p, pl, l, n, progress=gr.Progress()):
            # the run goes in a thread; this generator shows its live status in the text box every 2 s
            import threading
            status, box = {"step": "starting", "notes": []}, {}
            t0 = time.time()
            th = threading.Thread(target=lambda: box.update(r=anonymize_dataset(z, p, pl, l, n or ARGS.num_speakers, progress, status)), daemon=True)
            th.start()
            while th.is_alive():
                th.join(2.0)
                el = time.time() - t0
                yield (None, [], f"Running ({el // 60:.0f} min {el % 60:02.0f} s): {status['step']}\n\n"
                       + "\n".join(status["notes"][-30:])
                       + "\n\nThe anonymized dataset and the table appear here when all steps are done.")
            yield box.get("r") or (None, [], "The run stopped unexpectedly; see the console.")
        ds_btn.click(run_ds, [ds_zip, ds_path, ds_pipeline, ds_language, ds_speakers], [ds_out, ds_table, ds_info])
    return ui


if __name__ == "__main__":
    try:
        get_engine(ARGS.default_pipeline if ARGS.default_pipeline in PROFILES else next(iter(PROFILES)))
    except Exception as e:   # the UI still opens; the error shows when a conversion is attempted
        print(f"warning: {e}", flush=True)
    ui = build_ui()
    ui.launch(server_name="127.0.0.1", server_port=ARGS.port, share=False, inbrowser=True, allowed_paths=[str(out_dir)])
