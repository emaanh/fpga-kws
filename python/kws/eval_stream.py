"""Streaming evaluation of the live decision rule, on the bit-exact pipeline (without PDM).

Builds a long stream of test-set clips (keywords, non-keyword words, silence) over light
background noise, runs the fixed-point frontend and the integer model every INFER_EVERY
frames like kws_live.sv, and applies the rule "a keyword that wins N inferences in a row is
detected" for several N. Reports hit rate, wrong-word rate and false alarms per hour.

    uv run python -m kws.eval_stream
"""

import argparse

import numpy as np
import torch

from .config import CKPT_DIR, CLASSES, HOP
from .data import CACHE_DIR
from .fixed_frontend import features_fixed
from .quant import int_forward

INFER_EVERY, IN_H = 5, 49
SR = 16_000


def detect(classes, margins, n_consec, min_margin=0):
    """Mirror of kws_live.sv: a keyword counts as a win when its logit beats the runner-up by
    at least min_margin; N wins in a row is a detection. Returns (inference index, class)."""
    out, hits, prev = [], 0, 0
    for i, (c, m) in enumerate(zip(classes, margins)):
        if m < min_margin:
            c = 0  # not confident: treated like silence
        hits = hits + 1 if c >= 2 and c == prev else (1 if c >= 2 else 0)
        if hits == n_consec:
            out.append((i, c))
        prev = c
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-keywords", type=int, default=300)
    p.add_argument("--n-unknown", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split", default="val", help="tune on val; report on test")
    p.add_argument("--n-consec", type=int, nargs="*", default=[3, 4])
    p.add_argument("--margins", type=float, nargs="*", default=[0, 0.1, 0.2, 0.3, 0.4],
                   help="as fractions of the logit scale (median top-1 logit)")
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)

    params = torch.load(CKPT_DIR / "dscnn_int8.pt")
    d = np.load(CACHE_DIR / f"{args.split}.npz")
    audio, labels = d["audio"].astype(np.float64) / 32768, d["labels"]
    noise = np.load(CACHE_DIR / "noise.npy")
    kw = rng.choice(np.nonzero(labels >= 2)[0], args.n_keywords, replace=False)
    unk = rng.choice(np.nonzero(labels == 1)[0], args.n_unknown, replace=False)
    order = rng.permutation(np.concatenate([kw, unk]))

    # Each clip is followed by 0.5-1.5 s of gap, all over light background noise.
    parts, events, t = [], [], 0
    for i in order:
        parts.append(audio[i])
        events.append((t, t + SR, int(labels[i])))
        gap = int(rng.uniform(0.5, 1.5) * SR)
        parts.append(np.zeros(gap))
        t += SR + gap
    stream = np.concatenate(parts)
    start = rng.integers(0, len(noise) - len(stream)) if len(noise) > len(stream) else 0
    bg = np.resize(noise[start:], len(stream)) * 0.02
    pcm = np.clip(np.round((stream + bg) * 32768), -32768, 32767).astype(np.int16)

    q = np.concatenate([features_fixed(pcm[i : i + SR * 60 + 512], params["mean"], params["f_in"])
                        [: (SR * 60) // HOP] for i in range(0, len(pcm) - 512, SR * 60)])
    ends = np.arange(INFER_EVERY * 10, len(q) + 1, INFER_EVERY)  # inference after frame k
    x = torch.from_numpy(np.stack([q[k - IN_H : k] for k in ends]).astype(np.int64)).unsqueeze(1)
    logits = torch.cat([int_forward(params, b) for b in x.split(512)])
    top2 = logits.topk(2, dim=1).values
    classes = logits.argmax(1).tolist()
    margins = (top2[:, 0] - top2[:, 1]).tolist()
    scale = float(top2[:, 0].abs().median())
    t_inf = ends * HOP + 512 - HOP  # sample index at which each inference's window ends

    hours = len(stream) / SR / 3600
    print(f"{len(order)} clips ({args.n_keywords} keywords), {len(stream) / SR / 60:.1f} min, "
          f"{len(classes)} inferences")
    print(f"logit scale (median top-1): {scale:.0f}")
    for n in args.n_consec:
        for frac in args.margins:
            dets = [(t_inf[i], c) for i, c in detect(classes, margins, n, frac * scale)]
            hit = wrong = false = 0
            matched = set()
            for td, c in dets:
                # A detection belongs to a clip if it fires while the clip is in the window.
                ev = next((j for j, (s, e, _) in enumerate(events) if s <= td <= e + SR), None)
                if ev is None or events[ev][2] < 2:
                    false += 1
                elif events[ev][2] == c and ev not in matched:
                    hit += 1
                    matched.add(ev)
                elif events[ev][2] != c:
                    wrong += 1
            print(f"N={n} margin {frac:.2f} ({int(frac * scale)}): hit {hit / args.n_keywords:.3f}, "
                  f"wrong word {wrong / args.n_keywords:.3f}, false alarms {false / hours:.0f}/hour")


if __name__ == "__main__":
    main()
