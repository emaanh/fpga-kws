"""Evaluate a model on board-mic recordings (recordings/names_test/, made with kws.record_ui).

Runs the board's bit-exact pipeline (fixed-point frontend, integer model, live decision rule)
over each take and reports what would have been detected.

    KWS_KEYWORDS=emaan,heidari uv run python -m kws.eval_recordings --model dscnn_int8_names.pt
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile

from .config import CKPT_DIR, CLASSES, HOP, ROOT
from .eval_stream import detect
from .fixed_frontend import features_fixed
from .quant import int_forward

REC_DIR = ROOT / "recordings" / "names_test"
MARGINS = {"2^17 (SW4)": 1 << 17, "2^18 (default)": 1 << 18, "2^19 (SW5)": 1 << 19}


def run(params, pcm):
    """Integer logits for every inference (every 5 frames, once 49 frames exist)."""
    q = features_fixed(pcm, params["mean"], params["f_in"])
    ends = np.arange(50, len(q) + 1, 5)
    if len(ends) == 0:
        return np.zeros(0), [], []
    x = torch.from_numpy(np.stack([q[k - 49 : k] for k in ends]).astype(np.int64)).unsqueeze(1)
    logits = torch.cat([int_forward(params, b) for b in x.split(256)])
    top2 = logits.topk(2, dim=1).values
    t = (ends - 1) * HOP / 16000 + 512 / 16000
    return t, logits.argmax(1).tolist(), (top2[:, 0] - top2[:, 1]).tolist()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="dscnn_int8.pt")
    p.add_argument("--gain-db", type=float, default=0.0, help="digital gain applied to the takes")
    args = p.parse_args()

    params = torch.load(CKPT_DIR / args.model)
    files = sorted(REC_DIR.glob("*.wav"))
    if not files:
        raise SystemExit(f"no recordings in {REC_DIR}")
    print(f"model {args.model}, classes {[c.strip('_') for c in CLASSES]}, gain {args.gain_db:+.0f} dB\n")
    summary = {m: {"hit": 0, "want": 0, "false": 0} for m in MARGINS}
    for f in files:
        sr, pcm = wavfile.read(f)
        pcm = np.clip(np.round(pcm.astype(np.float64) * 10 ** (args.gain_db / 20)), -32768, 32767).astype(np.int16)
        label = f.name.rsplit("_", 1)[0]
        t, cls, mar = run(params, pcm)
        tops = {}
        for c in cls:
            tops[CLASSES[c].strip("_")] = tops.get(CLASSES[c].strip("_"), 0) + 1
        print(f"{f.name}  ({len(pcm) / sr:.0f} s)  top class counts: "
              + ", ".join(f"{k} {v}" for k, v in sorted(tops.items(), key=lambda kv: -kv[1])))
        for name, m in MARGINS.items():
            dets = [(round(float(t[i]), 1), CLASSES[c].strip("_")) for i, c in detect(cls, mar, 3, m)]
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
