"""Bit-exact model of the PDM microphone path: 2 MHz PDM bits -> int16 PCM at 16 kHz.

The Nexys A7's ADMP421 mic is clocked at 100 MHz / 50 = 2 MHz. Spec (what the RTL matches):

  CIC      4 stages, decimate by 25 -> 80 kHz. Input +1/-1. Output c, |c| <= 25^4 = 390625
           (20-bit registers suffice; the RTL may use wider ones, wraparound is harmless).
           Output k is taken after input sample 25k + 24.
  FIR      FIR_TAPS taps, Q17, decimate by 5 -> 16 kHz: acc[m] = sum_t h[t] * c[5m + 4 - t].
           Lowpass to 7 kHz with CIC droop compensation; DC gain 16 * 32768 / 25^4, so with
           the shifts below full-scale PDM maps to full-scale int16 at gain 0.
  scale    d = (acc + 2^7) >> 8                                        |d| < 2^29
  DC block y[n] = d[n] - d[n-1] + y[n-1] - (y[n-1] >> 8)          (~10 Hz high-pass)
  gain     pcm = sat16((y + 2^(s-1)) >> s),  s = 13 - gain,  gain in 0..12 (6 dB steps)

Also here: a 2nd-order sigma-delta modulator to turn dataset audio into PDM for testing.
"""

import numpy as np
from numba import njit
from scipy import signal

PDM_FS = 2_000_000
CIC_R, CIC_N = 25, 4
FIR_D = 5
FIR_TAPS = 191
FIR_BITS = 17
MAX_GAIN = 12
PCM_FS = PDM_FS // (CIC_R * FIR_D)  # 16 kHz
CIC_GAIN = CIC_R**CIC_N


def cic_response(f):
    """Magnitude of the CIC at frequency f (Hz), normalized to 1 at DC."""
    x = np.pi * np.asarray(f, dtype=np.float64) / PDM_FS
    with np.errstate(invalid="ignore", divide="ignore"):
        h = np.abs(np.sin(CIC_R * x) / (CIC_R * np.sin(x)))
    return np.where(x == 0, 1.0, h) ** CIC_N


def design_fir():
    """Q17 taps: lowpass (7 kHz pass, 8.5 kHz stop at 80 kHz) with CIC compensation."""
    fs = PDM_FS / CIC_R
    f_pass, f_stop = 7000, 8500
    freqs = np.concatenate([np.linspace(0, f_pass, 64), [f_stop, fs / 2]])
    gains = np.concatenate([1 / cic_response(np.linspace(0, f_pass, 64)), [0, 0]])
    h = signal.firwin2(FIR_TAPS, freqs, gains, fs=fs, window=("kaiser", 8.0))
    h *= 16 * 32768 / CIC_GAIN / h.sum()
    q = np.round(h * 2**FIR_BITS).astype(np.int64)
    assert np.abs(q).max() < 2**FIR_BITS, "taps must fit 18-bit signed"
    return q


FIR_Q17 = design_fir()


def cic_decimate(bits: np.ndarray) -> np.ndarray:
    """0/1 PDM bits -> CIC output at 80 kHz (int64, exact)."""
    x = bits.astype(np.int64) * 2 - 1
    for _ in range(CIC_N):
        x = np.cumsum(x)  # int64 wraparound is fine: the combs undo it exactly
    x = x[CIC_R - 1 :: CIC_R]
    for _ in range(CIC_N):
        x = x - np.concatenate([[0], x[:-1]])
    return x


def fir_decimate(c: np.ndarray) -> np.ndarray:
    return np.convolve(c, FIR_Q17)[: len(c)][FIR_D - 1 :: FIR_D]


@njit(cache=True)
def _dc_block_and_gain(d, s):
    out = np.empty(len(d), np.int16)
    y = np.int64(0)
    d_prev = np.int64(0)
    for n in range(len(d)):
        y = d[n] - d_prev + y - (y >> 8)
        d_prev = d[n]
        v = (y + (np.int64(1) << (s - 1))) >> s
        out[n] = min(max(v, -32768), 32767)
    return out


def mic_pcm(bits: np.ndarray, gain: int, return_stages=False):
    """PDM bits at 2 MHz -> int16 PCM at 16 kHz, exactly as the RTL computes it."""
    assert 0 <= gain <= MAX_GAIN
    c = cic_decimate(bits)
    acc = fir_decimate(c)
    d = (acc + (1 << 7)) >> 8
    pcm = _dc_block_and_gain(d, 13 - gain)
    if return_stages:
        return pcm, {"cic": c, "fir": acc, "scaled": d}
    return pcm


@njit(cache=True)
def _sigma_delta(u):
    """2nd-order sigma-delta modulator (CIFB). u in [-0.5, 0.5] -> 0/1 bits."""
    bits = np.empty(len(u), np.uint8)
    v1 = 0.0
    v2 = 0.0
    y = 0.0
    for n in range(len(u)):
        v1 += u[n] - y
        v2 += v1 - y
        y = 1.0 if v2 >= 0 else -1.0
        bits[n] = 1 if y > 0 else 0
    return bits


def to_pdm(audio: np.ndarray, level: float) -> np.ndarray:
    """Float audio at 16 kHz in [-1, 1] -> PDM bits at 2 MHz, scaled by `level` of full scale."""
    up = signal.resample_poly(audio.astype(np.float64), PDM_FS // PCM_FS, 1)
    return _sigma_delta(np.clip(up * level, -0.5, 0.5))


if __name__ == "__main__":
    w, h = signal.freqz(FIR_Q17 / FIR_Q17.sum(), worN=8192, fs=PDM_FS / CIC_R)
    total = np.abs(h) * cic_response(w)
    passband = total[w <= 7000]
    print(f"FIR: {FIR_TAPS} taps, max |tap| {np.abs(FIR_Q17).max()}")
    print(f"CIC+FIR passband 0-7 kHz: {20 * np.log10(passband.min()):.2f} .. "
          f"{20 * np.log10(passband.max()):.2f} dB")
    print(f"aliasing into 0-7 kHz from >= 9 kHz: {20 * np.log10(total[w >= 9000].max()):.1f} dB")

    # A 1 kHz tone at -42 dBFS through the modulator and back at gain 6 (+36 dB).
    t = np.arange(PCM_FS) / PCM_FS
    tone = 0.5 * np.sin(2 * np.pi * 1000 * t)
    pcm = mic_pcm(to_pdm(tone, 2**-6), gain=6)
    steady = pcm[4000:12192].astype(np.float64)
    spec = np.abs(np.fft.rfft(steady * np.blackman(len(steady)))) ** 2
    f = np.fft.rfftfreq(len(steady), 1 / PCM_FS)
    sig = (np.abs(f - 1000) < 30)
    band = (f > 30) & (f < 7600) & ~sig
    print(f"1 kHz tone: amplitude {steady.max():.0f} (expected ~{0.5 * 32768:.0f}), "
          f"in-band SNR {10 * np.log10(spec[sig].sum() / spec[band].sum()):.1f} dB")
