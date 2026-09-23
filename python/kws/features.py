"""Log-mel frontend. Kept deliberately simple so it maps directly onto hardware:

    frame (512, hop 320) -> Hann window (ROM) -> 512-pt real FFT -> |X|^2
    -> 40 triangular mel filters (sparse ROM, weights in [0, 1]) -> log2

A bit-exact fixed-point version of this will be written later for RTL verification.
"""

import math

import torch
from torch import nn

from .config import F_MAX, F_MIN, HOP, LOG_EPS, N_FFT, N_MELS, SAMPLE_RATE, WIN


def _hz_to_mel(f):
    return 2595.0 * math.log10(1.0 + f / 700.0)


def mel_filterbank() -> torch.Tensor:
    """(N_FFT//2 + 1, N_MELS) triangular HTK-mel filters with peak weight 1."""
    n_bins = N_FFT // 2 + 1
    mels = torch.linspace(_hz_to_mel(F_MIN), _hz_to_mel(F_MAX), N_MELS + 2)
    hz = 700.0 * (10 ** (mels / 2595.0) - 1.0)
    bin_hz = torch.arange(n_bins) * SAMPLE_RATE / N_FFT
    lo, mid, hi = hz[:-2, None], hz[1:-1, None], hz[2:, None]
    up = (bin_hz - lo) / (mid - lo)
    down = (hi - bin_hz) / (hi - mid)
    return torch.clamp(torch.minimum(up, down), min=0).T.contiguous()


class LogMel(nn.Module):
    """(B, 16000) float audio in [-1, 1] -> (B, 1, 49, 40) normalized log2 mel."""

    def __init__(self):
        super().__init__()
        self.register_buffer("window", torch.hann_window(WIN, periodic=True))
        self.register_buffer("fbank", mel_filterbank())
        # Global normalization, set from training data by `fit_norm`.
        self.register_buffer("mean", torch.tensor(0.0))
        self.register_buffer("std", torch.tensor(1.0))

    def raw(self, x):
        frames = x.unfold(1, WIN, HOP) * self.window
        spec = torch.fft.rfft(frames, n=N_FFT)
        power = spec.real**2 + spec.imag**2
        return torch.log2(power @ self.fbank + LOG_EPS)

    @torch.no_grad()
    def fit_norm(self, batches):
        feats = torch.cat([self.raw(x).flatten() for x in batches])
        self.mean.fill_(feats.mean())
        self.std.fill_(feats.std())

    def forward(self, x):
        return ((self.raw(x) - self.mean) / self.std).unsqueeze(1)
