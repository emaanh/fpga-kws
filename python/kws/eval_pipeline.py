"""Test-set accuracy of each stage of the hardware pipeline, all integer after the mic:

  float    float log-mel frontend (what the model was trained on) + int8 model
  fixed    fixed-point frontend (fixed_frontend.py) on the dataset's int16 PCM
  mic      dataset audio -> PDM at `--level` of full scale -> mic path (mic_model.py) at
           `--gain` -> fixed-point frontend

    uv run python -m kws.eval_pipeline                  # mic at -36 dBFS, gain +36 dB
    uv run python -m kws.eval_pipeline --level-db -48 --gain 8
"""

import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

from .config import CKPT_DIR, CLASSES
from .data import Split
from .features import LogMel
from .fixed_frontend import features_fixed
from .mic_model import mic_pcm, to_pdm
from .quant import int_forward, quantize_input
from .train import print_confusion


def _mic_features(job):
    audio, level, gain, mean, f_in = job
    pcm = np.stack([mic_pcm(to_pdm(a, level), gain) for a in audio])
    return features_fixed(pcm, mean, f_in)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--level-db", type=float, default=-36.0, help="speech level at the mic, dBFS")
    p.add_argument("--gain", type=int, default=6, help="mic path gain step (6 dB each)")
    p.add_argument("--n", type=int, help="limit to n clips")
    p.add_argument("--int8", default="dscnn_int8.pt")
    p.add_argument("--float-ckpt", default="dscnn_float.pt", help="for the float frontend's stats")
    p.add_argument("--mic-only", action="store_true")
    args = p.parse_args()

    params = torch.load(CKPT_DIR / args.int8)
    frontend = LogMel()
    frontend.load_state_dict(torch.load(CKPT_DIR / args.float_ckpt)["frontend"])
    test = Split("test", torch.device("cpu"))
    audio = torch.cat([x for x, _ in test.batches(512)]).numpy()
    labels = torch.cat([y for _, y in test.batches(512)]).numpy()
    if args.n:
        idx = np.linspace(0, len(audio) - 1, args.n).astype(int)
        audio, labels = audio[idx], labels[idx]
    pcm16 = np.clip(np.round(audio * 32768), -32768, 32767).astype(np.int16)
    mean, f_in = params["mean"], params["f_in"]

    feats = {}
    if not args.mic_only:
        with torch.no_grad():
            feats["float"] = quantize_input(
                (frontend.raw(torch.from_numpy(audio)) - mean).unsqueeze(1), f_in).squeeze(1).numpy()
        feats["fixed"] = np.concatenate([features_fixed(pcm16[i : i + 256], mean, f_in)
                                         for i in range(0, len(pcm16), 256)])
    level = 10 ** (args.level_db / 20)
    jobs = [(audio[i : i + 64], level, args.gain, mean, f_in) for i in range(0, len(audio), 64)]
    with ProcessPoolExecutor() as pool:
        feats["mic"] = np.concatenate(list(pool.map(_mic_features, jobs)))

    for name, f in feats.items():
        x = torch.from_numpy(f.astype(np.int64)).unsqueeze(1)
        preds = torch.cat([int_forward(params, b) for b in x.split(512)]).argmax(1).numpy()
        print(f"{name:>6}: accuracy {np.mean(preds == labels):.4f}  "
              f"(features clipped: {np.mean((f == 127) | (f == -128)):.4f})")
        if name == "mic" and not args.mic_only:
            confusion = torch.zeros(len(CLASSES), len(CLASSES), dtype=torch.long)
            confusion.index_put_((torch.from_numpy(labels), torch.from_numpy(preds)),
                                 torch.ones(len(labels), dtype=torch.long), accumulate=True)
            print_confusion(confusion)


if __name__ == "__main__":
    main()
