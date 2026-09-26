"""Evaluate a model on board-mic recordings (recordings/names_test/, made with kws.record_ui).

Runs the board's bit-exact pipeline (fixed-point frontend, integer model, live decision rule)
over each take and reports what would have been detected.

    uv run python -m kws.eval_recordings                       # the default (config) model
    uv run python -m kws.eval_recordings --model dscnn_int8_all.pt
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile

from .config import CKPT_DIR, CLASSES, HOP, MARGIN, MODEL_INT8, N_CONSEC, ROOT, WIN
from .eval_stream import detect
from .fixed_frontend import features_fixed
from .quant import logits_over_time

REC_DIR = ROOT / "recordings" / "names_test"
MARGINS = {"x0.75 (SW4)": MARGIN - MARGIN // 4, "default": MARGIN, "x1.25 (SW5)": MARGIN + MARGIN // 4}


def run(params, pcm, every):
    """Integer logits for every result, as the board computes them (see logits_over_time)."""
    q = features_fixed(pcm, params["mean"], params["f_in"])
    ends, logits = logits_over_time(params, q, every)
    if len(ends) == 0:
        return np.zeros(0), [], []
    top2 = logits.topk(2, dim=1).values
    t = (ends * HOP + WIN) / 16000
    return t, logits.argmax(1).tolist(), (top2[:, 0] - top2[:, 1]).tolist()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=MODEL_INT8)
    p.add_argument("--n-consec", type=int, default=N_CONSEC, help="wins in a row for a detection")
    p.add_argument("--every", type=int, default=5, help="frames per inference (full-window models)")
    p.add_argument("--gain-db", type=float, default=0.0, help="digital gain applied to the takes")
    args = p.parse_args()

    params = torch.load(CKPT_DIR / args.model)
    files = sorted(REC_DIR.glob("*.wav"))
    if not files:
        raise SystemExit(f"no recordings in {REC_DIR}")
    print(f"model {args.model}, classes {[c.strip('_') for c in CLASSES]}, gain {args.gain_db:+.0f} dB, "
          f"{args.n_consec} wins in a row\n")
    summary = {m: {"hit": 0, "want": 0, "false": 0} for m in MARGINS}
    for f in files:
        sr, pcm = wavfile.read(f)
        pcm = np.clip(np.round(pcm.astype(np.float64) * 10 ** (args.gain_db / 20)), -32768, 32767).astype(np.int16)
        label = f.name.rsplit("_", 1)[0]
        t, cls, mar = run(params, pcm, args.every)
        tops = {}
        for c in cls:
            tops[CLASSES[c].strip("_")] = tops.get(CLASSES[c].strip("_"), 0) + 1
        print(f"{f.name}  ({len(pcm) / sr:.0f} s)  top class counts: "
              + ", ".join(f"{k} {v}" for k, v in sorted(tops.items(), key=lambda kv: -kv[1])))
        for name, m in MARGINS.items():
            dets = [(round(float(t[i]), 1), CLASSES[c].strip("_")) for i, c in detect(cls, mar, args.n_consec, m)]
            right = [d for d in dets if d[1] == label]
            wrong = [d for d in dets if d[1] != label]
            prompts = int((len(pcm) / sr - 0.8 - 1) // 2.5) + 1 if label in CLASSES else 0
            summary[name]["hit"] += len(right)
            summary[name]["want"] += prompts
            summary[name]["false"] += len(wrong)
            print(f"   margin {name:<15} detected {len(right):2d}/{prompts:<2d} {label if prompts else '':<8} "
                  f"false {len(wrong)}  {[d for d in dets][:14]}")
    print()
    for name, s in summary.items():
        print(f"margin {name:<15} names detected {s['hit']}/{s['want']}, false detections {s['false']}")


if __name__ == "__main__":
    main()
