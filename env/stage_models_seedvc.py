#!/usr/bin/env python
"""
Download every model the two pipelines need, once, on a machine with internet (run inside the `seedvc` env):

  both    CAM++ speaker embedding and openai/whisper-small (language ID), via Seed-VC's own load_models()
  L       the F0-conditioned 44.1 kHz DiT (seed-uvit-whisper-base-f0-44k), RMVPE, BigVGAN 44 kHz
  combined  facebook/wav2vec2-xls-r-300m, nvidia/bigvgan_v2_22khz_80band_fmax8k_256x and the stock XLS-R DiT
          DiT_uvit_tat_xlsr_ema.pth -> <models-dir>/seedvc_stock/  (MODELS_DIR in configs/run.env)

Seed-VC's weights land in <seedvc-root>/checkpoints (its hf_cache), so the machines that run the pipelines can stay
offline (HF_HUB_OFFLINE=1). Loads on CPU; nothing heavy is computed.

  python env/stage_models_seedvc.py --seedvc-root <seed-vc checkout> --models-dir <models dir>
"""
import os
import sys
import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seedvc-root", required=True)
    ap.add_argument("--models-dir", required=True, help="gets seedvc_stock/DiT_uvit_tat_xlsr_ema.pth (combined pipeline)")
    args = ap.parse_args()

    root = Path(args.seedvc_root).resolve()
    models = Path(args.models_dir).resolve()
    os.chdir(root)
    sys.path.insert(0, str(root))
    os.environ["HF_HUB_CACHE"] = str(root / "checkpoints" / "hf_cache")

    import torch
    _orig = torch.load
    torch.load = lambda *a, **k: _orig(*a, **{**k, "weights_only": False})
    import inference as svc

    print("== both pipelines: CAM++, whisper-small (language ID), Seed-VC V1 base")
    svc.load_models(argparse.Namespace(f0_condition=False, checkpoint=None, config=None, fp16=True))
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    WhisperProcessor.from_pretrained("openai/whisper-small")
    WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

    print("== pipeline L: 44.1 kHz F0 DiT, RMVPE, BigVGAN 44 kHz")
    svc.load_models(argparse.Namespace(f0_condition=True, checkpoint=None, config=None, fp16=True))

    print("== pipeline combined: XLS-R encoder, BigVGAN v2 fmax-8k, stock XLS-R DiT")
    from huggingface_hub import snapshot_download, hf_hub_download
    snapshot_download("facebook/wav2vec2-xls-r-300m", allow_patterns=["*.json", "*.bin", "*.safetensors"])
    snapshot_download("nvidia/bigvgan_v2_22khz_80band_fmax8k_256x")
    (models / "seedvc_stock").mkdir(parents=True, exist_ok=True)
    p = hf_hub_download("Plachta/Seed-VC", "DiT_uvit_tat_xlsr_ema.pth", local_dir=str(models / "seedvc_stock"))
    print("   stock XLS-R DiT ->", p)

    total = sum(f.stat().st_size for f in (root / "checkpoints").rglob("*") if f.is_file())
    print(f"== done: {root / 'checkpoints'} holds {total / 2**30:.1f} GB; {models / 'seedvc_stock'} the combined DiT")


if __name__ == "__main__":
    main()
