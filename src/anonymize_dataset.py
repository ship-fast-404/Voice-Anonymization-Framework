#!/usr/bin/env python
"""
Anonymize a folder tree of recordings: dataset/<group>/<file>.wav -> dataset_anonymized/<group>/<file>.wav.

Every folder that holds wavs is one GROUP: the same people recorded several times. In a group each person gets ONE
target voice, kept for all the group's files (src/anonymize.py --groups). Speakers per file: --num-speakers
(default 2), or per folder from <dataset>/speakers.csv (columns folder,num_speakers; folder as written under the
dataset root, e.g. "6" or "site/visit1").

Gender preservation is compulsory (src/anonymize.py link_speakers + the gender check before conversion);
links that look doubtful are listed in the "check" column of anonymization_targets.csv.

Two steps around the normal pipeline (jobs/anonymize_dataset.sbatch runs all of them on the cluster):
  stage   : gives every file a safe id (<group>__<name>, no spaces or brackets: RTTM files split on whitespace),
            links it into <work>/wav/, writes <work>/manifest.csv (id, group, num_speakers, source) and one
            <work>/files_n<k>.txt list per speaker count for diarization
  collect : copies <anon-dir>/<id>.wav back to <out>/<group>/<original name>.wav and writes
            <out>/anonymization_targets.csv (file, person, target voice) from the group_refs log

  python src/anonymize_dataset.py stage   --dataset dataset --work runs/ds1
  python src/anonymize_dataset.py collect --work runs/ds1 --anon-dir runs/ds1/combined --out dataset_anonymized \
         --group-refs runs/ds1/logs/anonymize_ds1_group_refs.json
"""
import os
import re
import csv
import sys
import json
import shutil
import argparse
from pathlib import Path

AUDIO = {".wav"}
OTHER_AUDIO = {".mp3", ".flac", ".m4a", ".aac", ".ogg", ".opus", ".webm", ".wma", ".amr"}   # converted to 16 kHz wav


def to_wav(src, dst):
    """Any other audio format -> 16 kHz mono PCM16 wav (libsndfile first, then ffmpeg). False when neither can read it."""
    import subprocess
    try:
        import numpy as np
        import soundfile as sf
        x, sr = sf.read(str(src), dtype="float32", always_2d=True)
        x = x.mean(axis=1)
        if sr != 16000:
            from math import gcd
            from scipy.signal import resample_poly
            g = gcd(16000, sr)
            x = resample_poly(x, 16000 // g, sr // g).astype(np.float32)
        sf.write(str(dst), x, 16000, subtype="PCM_16")
        return True
    except Exception:
        pass
    try:
        import imageio_ffmpeg
        ff = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        ff = shutil.which("ffmpeg")
    if not ff:
        return False
    r = subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(src), "-ac", "1", "-ar", "16000", str(dst)])
    return r.returncode == 0 and dst.exists()


def is_junk(p, root):
    """Archive and OS clutter: __MACOSX folders, ._ resource forks, hidden folders."""
    parts = p.relative_to(root).parts
    return any(x == "__MACOSX" or x.startswith(".") for x in parts)


def safe(s):
    return re.sub(r"_+", "_", re.sub(r"[^A-Za-z0-9_-]", "_", s)).strip("_") or "x"


def speaker_counts(dataset, default):
    f = dataset / "speakers.csv"
    out = {}
    if f.exists():
        with f.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                k = (row.get("folder") or "").strip().strip("/\\").replace("\\", "/")
                if k and (row.get("num_speakers") or "").strip():
                    out[k] = int(row["num_speakers"])
    return lambda group: out.get(group, default)


def stage(a):
    dataset, work = Path(a.dataset).resolve(), Path(a.work).resolve()
    if not dataset.is_dir():
        sys.exit(f"no folder {dataset}")
    wav_dir = work / "wav"
    if wav_dir.exists():
        shutil.rmtree(wav_dir)
    wav_dir.mkdir(parents=True)
    n_of = speaker_counts(dataset, a.num_speakers)
    rows, used = [], set()
    files = sorted(p for p in dataset.rglob("*") if p.is_file() and not is_junk(p, dataset)
                   and p.suffix.lower() in AUDIO | OTHER_AUDIO)
    skipped = []
    for p in files:
        rel = p.relative_to(dataset)
        group = rel.parent.as_posix() if rel.parent != Path(".") else "."
        fid = f"{safe(group)}__{safe(p.stem)}"
        k = 2
        while fid in used:
            fid = f"{safe(group)}__{safe(p.stem)}_{k}"
            k += 1
        dst = wav_dir / f"{fid}.wav"
        if p.suffix.lower() in OTHER_AUDIO:
            if not to_wav(p, dst):
                skipped.append(p)
                continue
        else:
            try:
                os.symlink(p, dst)
            except OSError:
                shutil.copy2(p, dst)
        used.add(fid)
        rows.append({"id": fid, "group": group, "num_speakers": n_of(group), "source": rel.as_posix()})
    if not rows:
        sys.exit(f"no audio files under {dataset}")
    with (work / "manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "group", "num_speakers", "source"])
        w.writeheader()
        w.writerows(rows)
    (work / "dataset_root.txt").write_text(str(dataset), encoding="utf-8")
    for old in work.glob("files_n*.txt"):
        old.unlink()
    for n in sorted({r["num_speakers"] for r in rows}):
        (work / f"files_n{n}.txt").write_text("\n".join(r["id"] for r in rows if r["num_speakers"] == n) + "\n", encoding="utf-8")
    groups = {}
    for r in rows:
        groups.setdefault(r["group"], []).append(r)
    for g, rs in groups.items():
        print(f"group {g}: {len(rs)} files, {rs[0]['num_speakers']} speakers each")
    for p in skipped:
        print(f"skipped (could not be read or converted): {p.relative_to(dataset)}")
    print(f"staged {len(rows)} files in {len(groups)} groups -> {wav_dir}")


def collect(a):
    work = Path(a.work).resolve()
    anon = Path(a.anon_dir).resolve()
    with (work / "manifest.csv").open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if a.out:
        out = Path(a.out).resolve()
    else:
        root = Path((work / "dataset_root.txt").read_text(encoding="utf-8").strip())
        out = root.parent / f"{root.name}_anonymized"
    refs = json.loads(Path(a.group_refs).read_text(encoding="utf-8")) if a.group_refs and Path(a.group_refs).exists() else {}
    missing, table = [], []
    for r in rows:
        src = anon / f"{r['id']}.wav"
        dst = out / Path(r["source"]).with_suffix(".wav")
        if not src.exists():
            missing.append(r["source"])
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        g = refs.get(r["group"], {})
        checks = {fl["spk"]: fl["reason"] for fl in g.get("flags", []) if fl["file"] == r["id"]}
        for spk, gid in sorted(g.get("files", {}).get(r["id"], {}).items(), key=lambda kv: kv[1]):
            p = g.get("people", {}).get(gid, {})
            table.append({"file": r["source"], "group": r["group"], "person": gid, "diarization_label": spk,
                          "gender_by_pitch": p.get("gender"), "target_voice": p.get("target"),
                          "check": checks.get(spk, "")})
    if table:
        with (out / "anonymization_targets.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(table[0]))
            w.writeheader()
            w.writerows(table)
    print(f"{len(rows) - len(missing)} of {len(rows)} files -> {out}")
    for m in missing:
        print(f"MISSING (conversion failed or not run): {m}")
    if missing:
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stage")
    s.add_argument("--dataset", required=True)
    s.add_argument("--work", required=True)
    s.add_argument("--num-speakers", type=int, default=2, help="speakers per file unless speakers.csv says otherwise")
    c = sub.add_parser("collect")
    c.add_argument("--work", required=True)
    c.add_argument("--anon-dir", required=True)
    c.add_argument("--out", default=None, help="default: <dataset>_anonymized next to the dataset folder")
    c.add_argument("--group-refs", default=None)
    a = ap.parse_args()
    stage(a) if a.cmd == "stage" else collect(a)


if __name__ == "__main__":
    main()
