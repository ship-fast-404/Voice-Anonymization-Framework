#!/usr/bin/env python
"""
Build a pool of same-language pseudo-voice references from the diarized corpus itself.

Why: the target voice's accent and pitch class bleed into the converted speech, so anonymizing Hindi
consultations onto Hindi-speaking male/female targets preserves accent and gender without any training.
Every speaker of every file NOT in --exclude (the evaluation files) is a candidate; the cleanest few per
pitch class are cut into ~25 s single-speaker reference clips named m<k>_/f<k>_. The pitch class is only a
GUESS of the gender (low-voiced women exist): LISTEN to every clip and correct the "gender" field in
manifest.json (and the m/f prefix) before using the pool - anonymize.py trusts the manifest.
--low-max-f0 130 makes the "male" guesses safer by only taking clearly low voices.

Also writes speaker_table.json (per file/speaker: median F0, pitch class, speech seconds, overlap ratio)
which is used to pick evaluation files with both a male and a female speaker.

  python src/build_ref_pool.py --wav-dir data/wav --rttm-dir data/rttm --out-dir refs/hindi \
         --exclude "2051619 2190472 2006763 2011315" --per-class 4
NOTE: these clips are real (research-corpus) speakers; for deployment replace them with consented or
synthetic voices of the same language and pitch class (same folder layout, nothing else changes).
"""
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import read_rttm, find_rttm, log  # noqa: E402

PITCH_CLASS_HZ = 165.0  # same split as anonymize.py: below = low (typically male), above = high (typically female)


def f0_median(x16, energy_gate=0.02):
    import torch
    import torchaudio
    xt = torch.from_numpy(x16)[None].float()
    if xt.size(-1) < 16000:
        return None
    f0 = torchaudio.functional.detect_pitch_frequency(xt, 16000, frame_time=0.02, freq_low=60, freq_high=500)[0]
    n = f0.numel()
    if n == 0:
        return None
    frames = xt[0, : (xt.size(-1) // n) * n].reshape(n, -1)
    energy = frames.pow(2).mean(dim=1).sqrt()
    voiced = f0[(energy > energy_gate) & (f0 > 60) & (f0 < 500)]
    return float(voiced.median()) if voiced.numel() >= 10 else None


def load16(path):
    import numpy as np
    import soundfile as sf
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != 16000:
        import torch
        import torchaudio
        x = torchaudio.functional.resample(torch.from_numpy(x)[None], sr, 16000)[0].numpy()
    return x.astype(np.float32)


def overlap_free_turns(segs, spk, min_len):
    """Turns of `spk` that do not overlap any other speaker's segment, sorted by length (desc)."""
    others = [(s["start"], s["end"]) for s in segs if s["spk"] != spk]
    out = []
    for s in segs:
        if s["spk"] != spk or s["end"] - s["start"] < min_len:
            continue
        if any(a < s["end"] and b > s["start"] for a, b in others):
            continue
        out.append((s["start"], s["end"]))
    return sorted(out, key=lambda t: t[1] - t[0], reverse=True)


def main():
    import numpy as np
    import soundfile as sf
    p = argparse.ArgumentParser()
    p.add_argument("--wav-dir", required=True)
    p.add_argument("--rttm-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--exclude", default="", help="file ids (space/comma separated) never used as references")
    p.add_argument("--per-class", type=int, default=4, help="reference clips per pitch class")
    p.add_argument("--sec", type=float, default=25.0, help="target clip length")
    p.add_argument("--min-turn", type=float, default=1.5)
    p.add_argument("--min-speech", type=float, default=40.0, help="candidate speakers need at least this much speech")
    p.add_argument("--table-only", action="store_true", help="only write speaker_table.json")
    p.add_argument("--low-max-f0", type=float, default=None, help="only speakers with median F0 <= this count as low/male candidates")
    p.add_argument("--classes", default="low,high", help="which pitch classes to cut clips for (e.g. 'low' to add male clips only)")
    p.add_argument("--start-index", type=int, default=1, help="first k in m<k>_/f<k>_ (to extend an existing pool)")
    args = p.parse_args()

    excl = {s for s in args.exclude.replace(",", " ").split() if s}

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    wavs = sorted(Path(args.wav_dir).glob("*.wav"))
    table, cands = {}, []
    for w in wavs:
        fid = w.stem
        rp = find_rttm(args.rttm_dir, fid)
        if rp is None:
            continue
        segs = read_rttm(rp)
        x = load16(w)
        dur = len(x) / 16000
        # background level: RMS of everything outside any speech segment
        mask = np.zeros(len(x), dtype=bool)
        for s in segs:
            mask[int(s["start"] * 16000): int(s["end"] * 16000)] = True
        bg = float(np.sqrt(np.mean(x[~mask] ** 2) + 1e-12)) if (~mask).sum() > 16000 else None
        table[fid] = {"duration_s": round(dur, 1), "background_rms": bg, "speakers": {}}
        for spk in sorted({s["spk"] for s in segs}):
            mine = [s for s in segs if s["spk"] == spk]
            speech = sum(s["end"] - s["start"] for s in mine)
            free = overlap_free_turns(segs, spk, args.min_turn)
            free_s = sum(b - a for a, b in free)
            clip = np.concatenate([x[int(a * 16000):int(b * 16000)] for a, b in free[:40]]) if free else None
            f0 = f0_median(clip[: 16000 * 60]) if clip is not None and len(clip) >= 16000 else None
            spk_rms = float(np.sqrt(np.mean(clip ** 2) + 1e-12)) if clip is not None else None
            snr_db = 20 * np.log10(spk_rms / bg) if (spk_rms and bg) else None
            cls = None if f0 is None else ("high" if f0 >= PITCH_CLASS_HZ else "low")
            gender = None if f0 is None else ("male" if f0 < PITCH_CLASS_HZ else "female")   # a guess, see docstring
            table[fid]["speakers"][spk] = {"f0_hz": None if f0 is None else round(f0, 1), "pitch_class": cls,
                                           "gender_guess": gender,
                                           "speech_s": round(speech, 1), "overlap_free_s": round(free_s, 1),
                                           "overlap_ratio": round(1 - free_s / speech, 3) if speech else None,
                                           "snr_db": None if snr_db is None else round(float(snr_db), 1)}
            ok_low = args.low_max_f0 is None or cls != "low" or f0 <= args.low_max_f0
            if fid not in excl and cls and ok_low and free_s >= args.min_speech and snr_db is not None:
                cands.append({"fid": fid, "spk": spk, "cls": cls, "f0": f0, "snr": float(snr_db), "turns": free, "path": w})
        classes = sorted(v["pitch_class"] or "?" for v in table[fid]["speakers"].values())
        table[fid]["pitch_classes"] = "+".join(classes)
        log(f"{fid}: " + ", ".join(f"{k}={v['f0_hz']}Hz/{v['pitch_class']}/{v['speech_s']}s/snr{v['snr_db']}"
                                   for k, v in table[fid]["speakers"].items()))

    slim = {fid: {k: v for k, v in t.items()} for fid, t in table.items()}
    (out / "speaker_table.json").write_text(json.dumps(slim, indent=1), encoding="utf-8")
    mixed = sorted(fid for fid, t in table.items() if t["pitch_classes"] == "high+low")
    log(f"files with one low + one high pitch speaker: {len(mixed)} -> {' '.join(mixed)}")
    if args.table_only:
        return

    chosen = []
    for cls in [c.strip() for c in args.classes.split(",") if c.strip()]:
        pool = sorted([c for c in cands if c["cls"] == cls], key=lambda c: c["snr"], reverse=True)
        seen_files = set()
        for c in pool:                     # at most one speaker per file, cleanest first
            if c["fid"] in seen_files:
                continue
            seen_files.add(c["fid"])
            chosen.append(c)
            if len([d for d in chosen if d["cls"] == cls]) >= args.per_class:
                break
    manifest = []
    for k, c in enumerate(chosen):
        x = load16(c["path"])
        pieces, tot = [], 0.0
        fade = np.linspace(0, 1, 160, dtype=np.float32)   # 10 ms
        gap = np.zeros(int(0.15 * 16000), dtype=np.float32)
        for a, b in c["turns"]:
            a, b = a + 0.1, b - 0.1
            if b - a < args.min_turn - 0.2:
                continue
            seg = x[int(a * 16000):int(b * 16000)].copy()
            seg[:160] *= fade
            seg[-160:] *= fade[::-1]
            pieces += [seg, gap]
            tot += b - a
            if tot >= args.sec:
                break
        clip = np.concatenate(pieces)
        peak = float(np.abs(clip).max() + 1e-9)
        clip = clip * min(1.0, 0.9 / peak)
        tag = "m" if c["cls"] == "low" else "f"
        name = f"{tag}{len([m for m in manifest if m['pitch_class'] == c['cls']]) + args.start_index}_{c['fid']}_{c['spk']}.wav"
        sf.write(str(out / name), clip, 16000, subtype="PCM_16")
        manifest.append({"file": name, "source_file": c["fid"], "speaker": c["spk"], "pitch_class": c["cls"],
                         "gender": "male" if c["cls"] == "low" else "female", "gender_source": "pitch class, unverified - listen!",
                         "f0_hz": round(c["f0"], 1), "snr_db": round(c["snr"], 1), "length_s": round(len(clip) / 16000, 1)})
        log(f"ref {name}: f0={c['f0']:.0f}Hz snr={c['snr']:.1f}dB len={len(clip) / 16000:.1f}s")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    log(f"wrote {len(manifest)} reference clips to {out}")


if __name__ == "__main__":
    main()
