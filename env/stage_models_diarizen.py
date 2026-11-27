#!/usr/bin/env python
"""
Download everything DiariZen needs into a shared cache (run on the MASTER node,
inside the `diarizen` env), then instantiate the pipeline offline and run a short
smoke test so any lazily-downloaded asset lands in the shared cache too.

The embedding model pyannote/wespeaker-voxceleb-resnet34-LM is GATED on Hugging Face:
  1. accept the conditions at https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM
  2. export HF_TOKEN=hf_xxx   (or `huggingface-cli login`)
"""
import os
import sys
import argparse
import tempfile
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="BUT-FIT/diarizen-wavlm-large-s80-md-v2")
    ap.add_argument("--cache-dir", required=True, help="e.g. $BENCH_ROOT/hf_cache/diarizen")
    ap.add_argument("--smoke-wav", default=None, help="a wav to run a 20 s CPU smoke test on")
    ap.add_argument("--smoke-seconds", type=float, default=20.0)
    args = ap.parse_args()

    import torch
    _orig = torch.load
    torch.load = lambda *a, **k: _orig(*a, **{**k, "weights_only": False})
    import torchaudio  # torchaudio>=2.9: TorchCodec-based load()/info() and no AudioMetaData -> soundfile fallback
    if not hasattr(torchaudio, "AudioMetaData"):
        class AudioMetaData:
            def __init__(self, sample_rate, num_frames, num_channels, bits_per_sample=16, encoding="PCM_S"):
                self.sample_rate, self.num_frames, self.num_channels = sample_rate, num_frames, num_channels
                self.bits_per_sample, self.encoding = bits_per_sample, encoding
        torchaudio.AudioMetaData = AudioMetaData
    try:
        import torchcodec  # noqa: F401
    except Exception:
        import soundfile as sf
        torchaudio.load = lambda path, *a, **k: (torch.from_numpy(sf.read(path, dtype="float32", always_2d=True)[0].T.copy()),
                                                 sf.info(path).samplerate)
        torchaudio.info = lambda path, *a, **k: (lambda i: torchaudio.AudioMetaData(i.samplerate, i.frames, i.channels))(sf.info(path))

    from huggingface_hub import snapshot_download, hf_hub_download
    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)

    print(f"== snapshot_download {args.model} -> {args.cache_dir}")
    hub = snapshot_download(repo_id=args.model, cache_dir=args.cache_dir)
    print("   ", hub)

    print("== hf_hub_download pyannote/wespeaker-voxceleb-resnet34-LM (gated)")
    try:
        emb = hf_hub_download(repo_id="pyannote/wespeaker-voxceleb-resnet34-LM", filename="pytorch_model.bin",
                              cache_dir=args.cache_dir)
        print("   ", emb)
    except Exception as e:
        print(f"FAILED: {e!r}\n-> accept the model conditions on huggingface.co and set HF_TOKEN, then re-run.")
        sys.exit(2)

    print("== instantiating pipeline OFFLINE from the cache (CPU)")
    from diarizen.pipelines.inference import DiariZenPipeline
    pipe = DiariZenPipeline.from_pretrained(args.model, cache_dir=args.cache_dir)
    print("   pipeline OK")

    if args.smoke_wav:
        import soundfile as sf
        x, sr = sf.read(args.smoke_wav, dtype="float32", frames=int(args.smoke_seconds * 16000), always_2d=True)
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "smoke.wav")
            sf.write(p, x[:, 0], sr)
            ann = pipe(p, sess_name="smoke")
        spk = set(l for _, _, l in ann.itertracks(yield_label=True))
        print(f"   smoke test OK: {len(spk)} speakers found in {args.smoke_seconds:.0f}s clip")
    print("== staging complete. Compute nodes can now run with HF_HUB_OFFLINE=1.")


if __name__ == "__main__":
    main()
