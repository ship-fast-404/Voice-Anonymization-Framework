#!/usr/bin/env python
"""
Stage 1 - Speaker diarization with DiariZen (BUT-FIT), timed per file.

Runs inside the `diarizen` conda env. For every wav it writes <id>.rttm and
appends one JSON record (durations, sub-stage timings, GPU memory, optional
DER against reference RTTMs) to <log-dir>/diarization_records.jsonl.

Offline use (compute nodes have no internet): pass --cache-dir pointing to the
HF cache populated by env/stage_models_diarizen.py on the master node.
"""
import os
import sys
import json
import time
import argparse
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (Timer, JsonlWriter, env_info, gpu_peak_mem_mb, gpu_reset_peak,
                    list_wavs, wav_duration_s, log, write_json, now_iso, cuda_sync)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wav-dir", required=True)
    p.add_argument("--out-rttm-dir", required=True, help="where <id>.rttm hypotheses are written")
    p.add_argument("--log-dir", required=True)
    p.add_argument("--model", default="BUT-FIT/diarizen-wavlm-large-s80-md-v2",
                   help="HF repo id (also: BUT-FIT/diarizen-wavlm-large-s80-md, BUT-FIT/diarizen-wavlm-base-s80-md)")
    p.add_argument("--cache-dir", default=None,
                   help="HF cache dir populated on the master node. Setting it makes DiariZen run fully offline.")
    p.add_argument("--max-speakers", type=int, default=None,
                   help="override clustering max_speakers (default: model config, 20)")
    p.add_argument("--min-speakers", type=int, default=None)
    p.add_argument("--num-speakers", type=int, default=None,
                   help="hard cap on the number of distinct speakers. DiariZen's VBx clustering ignores max_speakers "
                        "(it only caps simultaneous speakers per frame); the cap is enforced by initialising AHC with "
                        "criterion=maxclust and threshold=N, so VBx starts from at most N clusters (it may still merge to fewer)")
    p.add_argument("--batch-size", type=int, default=None, help="override segmentation/embedding batch size")
    p.add_argument("--ref-rttm-dir", default=None, help="reference RTTMs (<id>_SPEAKER.rttm) to compute DER")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--shard", default=None, help="i/N - process every N-th file starting at i (parallel workers)")
    p.add_argument("--files", default=None, help="text file with wav names to process")
    p.add_argument("--no-warmup", action="store_true", help="skip the warm-up pass on the first file")
    p.add_argument("--tag", default="", help="free-text tag stored in every record")
    return p.parse_args()


def _patch_torch_load():
    """DiariZen / pyannote checkpoints are pickled Lightning checkpoints; torch>=2.6
    defaults to weights_only=True which rejects them. These checkpoints are trusted."""
    import torch
    _orig = torch.load

    def _load(*a, **k):
        k["weights_only"] = False
        return _orig(*a, **k)

    torch.load = _load


def _patch_torchaudio_load():
    """torchaudio>=2.9 routes torchaudio.load()/info() through TorchCodec (which has no build matching torch on every
    platform, e.g. aarch64) and dropped torchaudio.AudioMetaData (imported by pyannote's training code at module level).
    When TorchCodec is missing, read wavs with soundfile instead (same (waveform[channels, samples], sample_rate)
    result) and provide the metadata class."""
    import torchaudio
    try:
        import torchcodec  # noqa: F401
        has_codec = True
    except Exception:
        has_codec = False
    if not hasattr(torchaudio, "AudioMetaData"):
        class AudioMetaData:
            def __init__(self, sample_rate, num_frames, num_channels, bits_per_sample=16, encoding="PCM_S"):
                self.sample_rate, self.num_frames, self.num_channels = sample_rate, num_frames, num_channels
                self.bits_per_sample, self.encoding = bits_per_sample, encoding
        torchaudio.AudioMetaData = AudioMetaData
    if has_codec:
        return
    import torch
    import soundfile as sf

    def _load(path, *a, **k):
        x, sr = sf.read(path, dtype="float32", always_2d=True)
        return torch.from_numpy(x.T.copy()), sr

    def _info(path, *a, **k):
        i = sf.info(path)
        return torchaudio.AudioMetaData(i.samplerate, i.frames, i.channels)

    torchaudio.load = _load
    torchaudio.info = _info


def build_pipeline(args, rttm_out_dir):
    from diarizen.pipelines.inference import DiariZenPipeline
    overrides = any(v is not None for v in (args.max_speakers, args.min_speakers, args.num_speakers, args.batch_size))
    if not overrides:
        return DiariZenPipeline.from_pretrained(args.model, cache_dir=args.cache_dir, rttm_out_dir=rttm_out_dir)

    # Reproduce from_pretrained() but with a parsed config override.
    import toml
    from huggingface_hub import snapshot_download, hf_hub_download
    offline = args.cache_dir is not None
    hub = Path(snapshot_download(repo_id=args.model, cache_dir=args.cache_dir, local_files_only=offline))
    emb = hf_hub_download(repo_id="pyannote/wespeaker-voxceleb-resnet34-LM", filename="pytorch_model.bin",
                          cache_dir=args.cache_dir, local_files_only=offline)
    cfg = toml.load((hub / "config.toml").as_posix())
    if args.max_speakers is not None:
        cfg["clustering"]["args"]["max_speakers"] = args.max_speakers
    if args.min_speakers is not None:
        cfg["clustering"]["args"]["min_speakers"] = args.min_speakers
    if args.num_speakers is not None:
        ca = cfg["clustering"]["args"]
        ca["ahc_criterion"], ca["ahc_threshold"] = "maxclust", int(args.num_speakers)   # AHC init -> at most N clusters
        ca["max_speakers"] = min(ca.get("max_speakers", 20), int(args.num_speakers))     # per-frame simultaneous cap
        log(f"speaker cap {args.num_speakers}: clustering args -> {ca}")
    if args.batch_size is not None:
        cfg["inference"]["args"]["batch_size"] = args.batch_size
    config_parse = {"inference": {"args": cfg["inference"]["args"]},
                    "clustering": {"args": cfg["clustering"]["args"]}}
    return DiariZenPipeline(diarizen_hub=hub.expanduser().absolute(), embedding_model=emb,
                            config_parse=config_parse, rttm_out_dir=rttm_out_dir)


def patch_substage_timers(pipeline, store):
    """Wrap the pyannote sub-steps used by DiariZenPipeline.__call__ so each file
    gets segmentation / embedding / clustering / reconstruction timings. Best effort."""
    def timed(fn, key):
        def w(*a, **k):
            cuda_sync()
            t0 = time.perf_counter()
            r = fn(*a, **k)
            cuda_sync()
            store[key] = store.get(key, 0.0) + (time.perf_counter() - t0)
            return r
        return w

    patched = []
    for name, key in (("get_segmentations", "segmentation_s"), ("get_embeddings", "embedding_s"),
                      ("reconstruct", "reconstruct_s"), ("speaker_count", "speaker_count_s")):
        try:
            object.__setattr__(pipeline, name, timed(getattr(pipeline, name), key))
            patched.append(name)
        except Exception as e:
            log(f"could not patch {name}: {e!r}")
    try:
        cls = type(pipeline.clustering)
        if not getattr(cls, "_anon_timed", False):
            cls.__call__ = timed(cls.__call__, "clustering_s")
            cls._anon_timed = True
        patched.append("clustering")
    except Exception as e:
        log(f"could not patch clustering: {e!r}")
    return patched


def annotation_stats(ann):
    spk_secs = {}
    n_seg = 0
    for turn, _, spk in ann.itertracks(yield_label=True):
        spk_secs[spk] = spk_secs.get(spk, 0.0) + (turn.end - turn.start)
        n_seg += 1
    return {"n_speakers": len(spk_secs), "n_segments": n_seg,
            "speech_s_by_speaker": {k: round(v, 2) for k, v in spk_secs.items()},
            "speech_s_total": round(sum(spk_secs.values()), 2),
            "speech_s_union": round(ann.get_timeline().support().duration(), 2)}


def compute_der(ref_rttm, hyp_ann, uri):
    try:
        from pyannote.database.util import load_rttm
        from pyannote.metrics.diarization import DiarizationErrorRate
        refs = load_rttm(str(ref_rttm))
        ref = refs.get(uri) or next(iter(refs.values()))
        out = {}
        for collar, skip in ((0.0, False), (0.25, True)):
            m = DiarizationErrorRate(collar=collar, skip_overlap=skip)
            det = m(ref, hyp_ann, detailed=True)
            key = f"der_collar{collar}_{'skipovl' if skip else 'full'}"
            out[key] = round(float(det["diarization error rate"]), 4)
            out[key + "_detail"] = {k: round(float(v), 3) for k, v in det.items() if k != "diarization error rate"}
        out["ref_n_speakers"] = len(ref.labels())
        return out
    except Exception as e:
        return {"der_error": repr(e)}


def main():
    args = parse_args()
    _patch_torch_load()
    _patch_torchaudio_load()
    import torch

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    out_rttm = Path(args.out_rttm_dir)
    out_rttm.mkdir(parents=True, exist_ok=True)
    rec_path = log_dir / "diarization_records.jsonl"
    writer = JsonlWriter(rec_path)

    info = env_info()
    info["stage"] = "diarization"
    info["args"] = vars(args)
    log(f"env: {json.dumps(info)}")

    t_load = {}
    with Timer(t_load, "model_load_s"):
        pipeline = build_pipeline(args, str(out_rttm))
    log(f"pipeline loaded in {t_load['model_load_s']:.1f}s on {info.get('gpu_name', 'cpu')}")
    load_mem = gpu_peak_mem_mb()

    substage = {}
    patched = patch_substage_timers(pipeline, substage)
    log(f"patched sub-stage timers: {patched}")

    wavs = list_wavs(args.wav_dir, args.limit, args.shard, args.files)
    if not wavs:
        sys.exit(f"no wav files found in {args.wav_dir}")
    log(f"{len(wavs)} files to diarize")

    jobs = [(wavs[0], True)] if not args.no_warmup else []
    jobs += [(w, False) for w in wavs]

    stage_start = now_iso()
    total_audio, total_wall = 0.0, 0.0
    n_ok, n_fail = 0, 0
    for idx, (wav, warm) in enumerate(jobs):
        file_id = wav.stem
        dur, sr, ch = wav_duration_s(wav)
        substage.clear()
        gpu_reset_peak()
        rec = {"stage": "diarization", "tag": args.tag, "file": file_id, "path": str(wav),
               "duration_s": round(dur, 3), "sample_rate": sr, "channels": ch, "warmup": warm,
               "model": args.model, "started": now_iso()}
        try:
            cuda_sync()
            t0 = time.perf_counter()
            ann = pipeline(str(wav), sess_name=file_id)
            cuda_sync()
            wall = time.perf_counter() - t0
            rec.update({"ok": True, "wall_s": round(wall, 3), "rtf": round(wall / dur, 5),
                        "gpu_peak_mem_mb": gpu_peak_mem_mb(),
                        "substage_s": {k: round(v, 3) for k, v in substage.items()}})
            rec.update(annotation_stats(ann))
            if args.ref_rttm_dir:
                from common import find_rttm
                ref = find_rttm(args.ref_rttm_dir, file_id)
                if ref is not None:
                    rec.update(compute_der(ref, ann, file_id))
            if not warm:
                total_audio += dur
                total_wall += wall
                n_ok += 1
            log(f"[{idx + 1}/{len(jobs)}] {file_id} dur={dur:.0f}s wall={wall:.1f}s rtf={wall / dur:.3f} "
                f"spk={rec.get('n_speakers')} {'(warm-up)' if warm else ''}"
                + (f" DER={rec.get('der_collar0.0_full')}" if 'der_collar0.0_full' in rec else ""))
        except Exception as e:
            import traceback
            rec.update({"ok": False, "error": repr(e), "traceback": traceback.format_exc()})
            n_fail += 1
            log(f"[{idx + 1}/{len(jobs)}] {file_id} FAILED: {e!r}")
        rec["finished"] = now_iso()
        writer.write(rec)

    summary = {
        "stage": "diarization", "tag": args.tag, "env": info, "model": args.model,
        "model_load_s": round(t_load["model_load_s"], 3), "gpu_mem_after_load_mb": load_mem,
        "started": stage_start, "finished": now_iso(),
        "n_files_ok": n_ok, "n_files_failed": n_fail,
        "audio_hours": round(total_audio / 3600, 4), "wall_hours": round(total_wall / 3600, 4),
        "aggregate_rtf": round(total_wall / total_audio, 5) if total_audio else None,
        "audio_hours_per_wall_hour": round(total_audio / total_wall, 3) if total_wall else None,
        "records": str(rec_path),
    }
    write_json(log_dir / "diarization_summary.json", summary)
    log(f"SUMMARY {json.dumps(summary)}")
    writer.close()


if __name__ == "__main__":
    main()
