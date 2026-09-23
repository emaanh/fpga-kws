"""Bit-exact fixed-point model of the on-chip frontend: int16 PCM at 16 kHz -> int8 features.

This is the spec the frontend RTL must match. Per 512-sample frame (hop 320):

  window   y[n] = (x[n] * HANN_Q15[n] + 2^14) >> 15                          |y| < 2^15
  FFT      512-pt radix-2 DIT, bit-reversed input, no scaling. Butterfly with
           twiddle W = (wr, wi) in Q16 (18-bit signed):
             tr = (wr*br - wi*bi + 2^15) >> 16,  ti = (wr*bi + wi*br + 2^15) >> 16
             a' = a + t,  b' = a - t                                          |X| < 2^24
  power    P[k] = re^2 + im^2, k = 0..256                                     < 2^49
  mel      M[m] = sum_k P[k] * MEL_Q8[k][m] + EPS                             < 2^63
  log2     L = 64*msb(M) + LOG2_LUT[next 8 bits after the leading one]        Q6
  feature  q = clamp((L - OFFSET + 8) >> 4, -128, 127)                        int8, Q2

Scaling: x = 2^15 * audio and the window/FFT keep that scale, so P = 2^30 * float power and
M = 2^38 * float mel energy. EPS and OFFSET (38 + training mean, in Q6) make q match the
float frontend in features.py followed by quant.quantize_input.
"""

import math

import numpy as np
import torch

from .config import HOP, LOG_EPS, N_FFT, N_MELS, WIN
from .features import mel_filterbank

TW_BITS = 16      # twiddle fraction bits
MEL_BITS = 8      # mel weight fraction bits
MANT_BITS = 8     # mantissa bits used by the log2 LUT
LOG_FRAC = 6      # log2 fraction bits
SCALE_LOG2 = 30 + MEL_BITS  # M = 2^SCALE_LOG2 * float mel energy

HANN_Q15 = np.round(torch.hann_window(WIN, periodic=True).double().numpy() * 32767).astype(np.int64)
_k = np.arange(N_FFT // 2)
TW_RE = np.round(np.cos(2 * np.pi * _k / N_FFT) * 2**TW_BITS).astype(np.int64)
TW_IM = np.round(-np.sin(2 * np.pi * _k / N_FFT) * 2**TW_BITS).astype(np.int64)
MEL_Q8 = np.round(mel_filterbank().double().numpy() * 2**MEL_BITS).astype(np.int64)  # (257, 40)
EPS = int(round(LOG_EPS * 2**SCALE_LOG2))
LOG2_LUT = np.round(np.log2(1 + np.arange(2**MANT_BITS) / 2**MANT_BITS) * 2**LOG_FRAC).astype(np.int64)
BITREV = np.array([int(f"{i:09b}"[::-1], 2) for i in range(N_FFT)])


def offset_q6(mean: float, f_in: int) -> int:
    """Constant subtracted from the Q6 log2 so that (L - OFFSET) >> (LOG_FRAC - f_in) gives q."""
    assert LOG_FRAC - f_in == 4, "rounding below assumes f_in = 2"
    return int(round((SCALE_LOG2 + mean) * 2**LOG_FRAC))


def frames(pcm: np.ndarray) -> np.ndarray:
    """(..., n_samples) int16 -> (..., n_frames, WIN) int64."""
    n = (pcm.shape[-1] - WIN) // HOP + 1
    idx = np.arange(n)[:, None] * HOP + np.arange(WIN)
    return pcm[..., idx].astype(np.int64)


def fft_fixed(y: np.ndarray):
    """Radix-2 DIT FFT of real int frames (..., 512), exactly as the RTL computes it."""
    re = y[..., BITREV].copy()
    im = np.zeros_like(re)
    half = 1
    while half < N_FFT:
        k = np.arange(half) * (N_FFT // (2 * half))
        wr, wi = TW_RE[k], TW_IM[k]
        re = re.reshape(*re.shape[:-1], -1, 2, half)
        im = im.reshape(*im.shape[:-1], -1, 2, half)
        ar, ai, br, bi = re[..., 0, :], im[..., 0, :], re[..., 1, :], im[..., 1, :]
        tr = (wr * br - wi * bi + (1 << (TW_BITS - 1))) >> TW_BITS
        ti = (wr * bi + wi * br + (1 << (TW_BITS - 1))) >> TW_BITS
        re = np.stack([ar + tr, ar - tr], -2).reshape(*y.shape)
        im = np.stack([ai + ti, ai - ti], -2).reshape(*y.shape)
        half *= 2
    return re, im


def log2_q6(m: np.ndarray) -> np.ndarray:
    """Q6 log2 of positive int64 values via leading-one position + mantissa LUT."""
    msb = np.floor(np.log2(m.astype(np.float64))).astype(np.int64)
    # np.log2 in float64 can be off by one right at powers of two for large values; fix up.
    msb = np.where((np.int64(1) << msb) > m, msb - 1, msb)
    msb = np.where((np.int64(1) << (msb + 1)) <= m, msb + 1, msb)
    shift = msb - MANT_BITS
    mant = np.where(shift >= 0, m >> np.maximum(shift, 0), m << np.maximum(-shift, 0))
    return msb * 2**LOG_FRAC + LOG2_LUT[mant & (2**MANT_BITS - 1)]


def features_fixed(pcm: np.ndarray, mean: float, f_in: int, return_stages=False):
    """(..., 16000) int16 PCM -> (..., 49, 40) int8-range int64 features."""
    y = (frames(pcm) * HANN_Q15 + (1 << 14)) >> 15
    re, im = fft_fixed(y)
    assert np.abs(re).max() < 2**24 and np.abs(im).max() < 2**24, "FFT exceeds 25-bit range"
    power = re[..., : N_FFT // 2 + 1] ** 2 + im[..., : N_FFT // 2 + 1] ** 2
    mel = np.zeros((*power.shape[:-1], N_MELS), np.int64)
    for m in range(N_MELS):  # sparse: each band touches a few bins
        nz = np.nonzero(MEL_Q8[:, m])[0]
        mel[..., m] = (power[..., nz] * MEL_Q8[nz, m]).sum(-1)
    mel += EPS
    log = log2_q6(mel)
    q = np.clip((log - offset_q6(mean, f_in) + 8) >> 4, -128, 127)
    if return_stages:
        return q, {"window": y, "re": re, "im": im, "power": power, "mel": mel, "log": log}
    return q


if __name__ == "__main__":
    # Compare against the float frontend + input quantization on the test set.
    from .config import CKPT_DIR
    from .data import CACHE_DIR
    from .features import LogMel
    from .quant import int_forward, quantize_input

    params = torch.load(CKPT_DIR / "dscnn_int8.pt")
    frontend = LogMel()
    frontend.load_state_dict(torch.load(CKPT_DIR / "dscnn_float.pt")["frontend"])
    d = np.load(CACHE_DIR / "test.npz")
    audio, labels = d["audio"], d["labels"]
    keep = labels != 1  # keywords only: a fair spot check that needs no silence synthesis
    audio, labels = audio[keep], labels[keep]

    qf, qx, preds_f, preds_x = [], [], [], []
    for i in range(0, len(audio), 256):
        a = audio[i : i + 256]
        with torch.no_grad():
            f = quantize_input((frontend.raw(torch.from_numpy(a).float() / 32768) - params["mean"]).unsqueeze(1),
                               params["f_in"])
        x = torch.from_numpy(features_fixed(a, params["mean"], params["f_in"])).unsqueeze(1)
        qf.append(f), qx.append(x)
        preds_f.append(int_forward(params, f).argmax(1)), preds_x.append(int_forward(params, x).argmax(1))
    qf, qx = torch.cat(qf), torch.cat(qx)
    preds_f, preds_x = torch.cat(preds_f).numpy(), torch.cat(preds_x).numpy()
    diff = (qf - qx).abs()
    print(f"features: {(diff == 0).float().mean():.4f} identical, {(diff <= 1).float().mean():.4f} within 1, "
          f"max diff {int(diff.max())}")
    print(f"keyword accuracy  float frontend {np.mean(preds_f == labels):.4f}   "
          f"fixed frontend {np.mean(preds_x == labels):.4f}   "
          f"prediction agreement {np.mean(preds_f == preds_x):.4f}")
    print(f"EPS={EPS} OFFSET={offset_q6(params['mean'], params['f_in'])}  "
          f"{math.log2(EPS):.1f} bits")
