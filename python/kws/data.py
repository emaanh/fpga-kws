"""Speech Commands v2: download, cache as int16 arrays, and build batches with augmentation.

Run `uv run python -m kws.data` once to download (~2.3 GB) and build the cache.
"""

import tarfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile
from tqdm import tqdm

from .config import (CLASSES, CLIP_SAMPLES, DATA_DIR, DEFAULT_CLASSES, KEYWORDS, SAMPLE_RATE,
                     UNKNOWN)

URL = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
RAW_DIR = DATA_DIR / "speech_commands_v2"
CACHE_DIR = DATA_DIR / "cache"

SILENCE_IDX = CLASSES.index("_silence_")
UNKNOWN_IDX = CLASSES.index("_unknown_")


def download():
    if (RAW_DIR / "testing_list.txt").exists():
        return
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    tar_path = DATA_DIR / "speech_commands_v0.02.tar.gz"
    if not tar_path.exists():
        with tqdm(unit="B", unit_scale=True, desc="download") as bar:
            def hook(blocks, bs, total):
                bar.total = total
                bar.update(blocks * bs - bar.n)
            urllib.request.urlretrieve(URL, tar_path, hook)
    with tarfile.open(tar_path) as tf:
        tf.extractall(RAW_DIR, filter="data")
    tar_path.unlink()


def _read_clip(path: Path) -> np.ndarray:
    sr, x = wavfile.read(path)
    assert sr == SAMPLE_RATE and x.dtype == np.int16, path
    out = np.zeros(CLIP_SAMPLES, np.int16)
    out[: min(len(x), CLIP_SAMPLES)] = x[:CLIP_SAMPLES]
    return out


def build_cache():
    """Write {train,val,test}.npz with int16 audio (N, 16000) and int64 labels."""
    if (CACHE_DIR / "test.npz").exists():
        return
    download()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    val = set((RAW_DIR / "validation_list.txt").read_text().split())
    test = set((RAW_DIR / "testing_list.txt").read_text().split())
    splits = {"train": [], "val": [], "test": []}
    for wav in sorted(RAW_DIR.glob("*/*.wav")):
        word = wav.parent.name
        if word.startswith("_"):
            continue
        rel = f"{word}/{wav.name}"
        split = "val" if rel in val else "test" if rel in test else "train"
        label = CLASSES.index(word) if word in KEYWORDS else UNKNOWN_IDX
        splits[split].append((wav, label))

    for name, items in splits.items():
        with ThreadPoolExecutor(16) as pool:
            audio = list(tqdm(pool.map(_read_clip, [p for p, _ in items]),
                              total=len(items), desc=name))
        np.savez(CACHE_DIR / f"{name}.npz",
                 audio=np.stack(audio),
                 labels=np.array([l for _, l in items], np.int64))

    noise = [wavfile.read(p)[1] for p in sorted((RAW_DIR / "_background_noise_").glob("*.wav"))]
    noise = np.concatenate([n.astype(np.float32) / 32768 for n in noise])
    np.save(CACHE_DIR / "noise.npy", noise)


class Split:
    """One split held in memory. Each epoch uses every keyword clip plus a
    `extra_frac`-sized share of unknown and of silence clips (as in the TF recipe)."""

    def __init__(self, name: str, device, extra_frac=0.1, seed=0, gain_db=0.0, arrays=None,
                 realism=False):
        """`arrays` = (int16 audio (N, 16000), labels) replaces the cached split `name`.
        `realism` adds pitch/tempo, microphone EQ and room reverb augmentation (training)."""
        build_cache()
        self.gain_db = gain_db
        self.realism = realism
        if arrays is None:
            d = np.load(CACHE_DIR / f"{name}.npz")
            labels = d["labels"]
            if CLASSES != DEFAULT_CLASSES:
                # The cache is labelled for the default words; others become "unknown".
                lut = np.array([CLASSES.index(c) if c in CLASSES else UNKNOWN_IDX
                                for c in DEFAULT_CLASSES])
                labels = lut[labels]
            arrays = d["audio"], labels
        self.audio = torch.from_numpy(np.asarray(arrays[0]))
        self.labels = torch.from_numpy(np.asarray(arrays[1]).astype(np.int64))
        self.noise = torch.from_numpy(np.load(CACHE_DIR / "noise.npy")).to(device)
        self.device = device
        self.train = name.startswith("train")
        self.kw_idx = torch.nonzero(self.labels != UNKNOWN_IDX).squeeze(1)
        self.unk_idx = torch.nonzero(self.labels == UNKNOWN_IDX).squeeze(1)
        self.n_extra = int(len(self.kw_idx) * extra_frac)
        if not self.train and len(self.kw_idx) == 0:
            # No keywords in this split (a custom vocabulary on Speech Commands): evaluate on
            # a fixed set of real non-keywords and silences, i.e. a false-alarm test.
            self.n_extra = min(len(self.unk_idx), 1000)
        self.gen = torch.Generator().manual_seed(seed)
        if not self.train:
            # Fixed eval set: same unknowns and same silence noise every time.
            self.fixed_order = self._epoch_order()
            self.fixed_silence = self._silence(self.n_extra, self.gen)

    def _epoch_order(self):
        unk = self.unk_idx[torch.randperm(len(self.unk_idx), generator=self.gen)[: self.n_extra]]
        sil = torch.full((self.n_extra,), -1)  # -1 marks a silence clip
        order = torch.cat([self.kw_idx, unk, sil])
        return order[torch.randperm(len(order), generator=self.gen)] if self.train else order

    def _noise_slices(self, n, gen):
        starts = torch.randint(0, len(self.noise) - CLIP_SAMPLES, (n,), generator=gen)
        idx = starts[:, None] + torch.arange(CLIP_SAMPLES)
        return self.noise[idx.to(self.device)]

    def _silence(self, n, gen):
        vol = torch.rand(n, 1, generator=gen).to(self.device)
        return self._noise_slices(n, gen) * vol

    def __len__(self):
        return len(self.kw_idx) + 2 * self.n_extra

    def batches(self, batch_size):
        order = self._epoch_order() if self.train else self.fixed_order
        for i in range(0, len(order), batch_size):
            yield self._make_batch(order[i : i + batch_size], i)

    def _make_batch(self, idx, offset):
        is_sil = idx < 0
        sil_d = is_sil.to(self.device)
        x = torch.zeros(len(idx), CLIP_SAMPLES, device=self.device)
        y = torch.full((len(idx),), SILENCE_IDX, device=self.device)
        speech = idx[~is_sil]
        x[~sil_d] = self.audio[speech].to(self.device).float() / 32768
        y[~sil_d] = self.labels[speech].to(self.device)

        if not self.train:
            n_before = int((self.fixed_order[:offset] < 0).sum())
            x[sil_d] = self.fixed_silence[n_before : n_before + int(is_sil.sum())]
            return x, y

        # Random time shift of +-100 ms, zero-filled.
        shift = torch.randint(-1600, 1601, (len(idx),), generator=self.gen).to(self.device)
        padded = torch.nn.functional.pad(x, (1600, 1600))
        pos = torch.arange(CLIP_SAMPLES, device=self.device)[None, :] + 1600 - shift[:, None]
        x = padded.gather(1, pos)

        if self.realism:
            x = self._realism(x)

        # Random speech level, +-10 dB: the board mic's level depends on the speaker's distance.
        if self.gain_db:
            db = (torch.rand(len(idx), 1, generator=self.gen) * 2 - 1) * self.gain_db
            x = x * (10 ** (db / 20)).to(self.device)

        # Background noise on 80% of clips at volume up to 0.1; silence clips get up to 1.0.
        n = len(idx)
        vol = torch.rand(n, 1, generator=self.gen).to(self.device) * 0.1
        vol *= (torch.rand(n, 1, generator=self.gen) < 0.8).to(self.device)
        vol[sil_d] = torch.rand(int(is_sil.sum()), 1, generator=self.gen).to(self.device)
        x = (x + vol * self._noise_slices(n, self.gen)).clamp(-1, 1)
        return x, y


    def _rand(self, *shape):
        return torch.rand(*shape, generator=self.gen).to(self.device)

    def _realism(self, x):
        """Make clean audio sound recorded: pitch/tempo, microphone EQ, room reverb."""
        n = len(x)
        peak = x.abs().amax(1, keepdim=True)

        # Pitch and tempo together: resample the whole batch by one factor.
        if float(self._rand(1)) < 0.5:
            f = float(0.9 + 0.22 * self._rand(1))
            y = torch.nn.functional.interpolate(x[:, None], size=int(CLIP_SAMPLES / f),
                                                mode="linear", align_corners=False)[:, 0]
            if y.shape[1] >= CLIP_SAMPLES:
                o = (y.shape[1] - CLIP_SAMPLES) // 2
                x = y[:, o : o + CLIP_SAMPLES]
            else:
                o = (CLIP_SAMPLES - y.shape[1]) // 2
                x = torch.nn.functional.pad(y, (o, CLIP_SAMPLES - y.shape[1] - o))

        # Microphone: random tilt, high-pass, low-pass and resonances, per clip (80%).
        spec = torch.fft.rfft(x, n=CLIP_SAMPLES)
        freqs = torch.linspace(1, SAMPLE_RATE / 2, spec.shape[1], device=self.device)[None]
        tilt = (self._rand(n, 1) * 8 - 4) * torch.log2(freqs / 1000)            # dB
        bumps = sum((self._rand(n, 1) * 12 - 6) *
                    torch.exp(-0.5 * ((torch.log2(freqs) - (6 + 7 * self._rand(n, 1))) /
                                      (0.2 + 0.6 * self._rand(n, 1))) ** 2) for _ in range(3))
        hp = 1 / (1 + ((50 + 250 * self._rand(n, 1)) / freqs) ** 4)
        lp = 1 / (1 + (freqs / (3000 + 5000 * self._rand(n, 1))) ** 4)
        g = 10 ** ((tilt + bumps) / 20) * hp * lp
        use = (self._rand(n, 1) < 0.8).float()
        spec = spec * (use * g + (1 - use))

        # Room: a decaying noise tail with a direct path, per clip (50%).
        L = 8000
        t = torch.arange(L, device=self.device)[None] / SAMPLE_RATE
        rt60 = 0.15 + 0.55 * self._rand(n, 1)
        rir = torch.randn(n, L, generator=self.gen).to(self.device) * torch.exp(-6.9 * t / rt60)
        rir = rir / rir.norm(dim=1, keepdim=True) * (0.2 + 1.3 * self._rand(n, 1))
        rir[:, 0] = 1.0
        use = (self._rand(n, 1) < 0.5).float()
        nfft = 32768
        wet = torch.fft.irfft(torch.fft.rfft(torch.fft.irfft(spec, n=CLIP_SAMPLES), n=nfft) *
                              torch.fft.rfft(rir, n=nfft), n=nfft)[:, :CLIP_SAMPLES]
        dry = torch.fft.irfft(spec, n=CLIP_SAMPLES)
        x = use * wet + (1 - use) * dry
        return x / (x.abs().amax(1, keepdim=True) + 1e-9) * peak


def spec_augment(f, gen, n_freq=2, max_f=5, n_time=2, max_t=8):
    """Blank random mel bands and frames of (B, 1, T, F) features (set to the mean, 0)."""
    B, _, T, F = f.shape
    f = f.clone()
    for _ in range(n_freq):
        w = torch.randint(0, max_f + 1, (B,), generator=gen)
        s0 = (torch.rand(B, generator=gen) * (F - w)).long()
        idx = torch.arange(F)[None]
        m = ((idx >= s0[:, None]) & (idx < (s0 + w)[:, None])).to(f.device)
        f = f.masked_fill(m[:, None, None, :], 0.0)
    for _ in range(n_time):
        w = torch.randint(0, max_t + 1, (B,), generator=gen)
        s0 = (torch.rand(B, generator=gen) * (T - w)).long()
        idx = torch.arange(T)[None]
        m = ((idx >= s0[:, None]) & (idx < (s0 + w)[:, None])).to(f.device)
        f = f.masked_fill(m[:, None, :, None], 0.0)
    return f


# ------------------------------------------------------------------------------------------
# Synthetic (TTS) data, see tts_data.py
# ------------------------------------------------------------------------------------------
def _real_peaks():
    """Peak levels of real speech clips, to give synthetic clips the same loudness spread."""
    d = np.load(CACHE_DIR / "train.npz")
    return np.abs(d["audio"][:: 20].astype(np.float32)).max(1) / 32768


def tts_arrays(split: str, seed=0, clips=None):
    """TTS utterances for `split` ("train"/"val") as 1 s int16 clips and class labels.

    Each utterance goes at a random position in a 1 s window (like the real clips, whose
    words are not centred either) and is scaled to a peak drawn from the real clips' peaks.
    """
    import os

    from .tts_data import TTS_DIR

    d = np.load(TTS_DIR / (clips or os.environ.get("KWS_TTS_CLIPS", "clips.npz")))
    rng = np.random.default_rng(seed + (split == "val"))
    peaks = _real_peaks()
    keep = np.nonzero(d["splits"] == split)[0]
    audio = np.zeros((len(keep), CLIP_SAMPLES), np.int16)
    labels = np.empty(len(keep), np.int64)
    for j, i in enumerate(keep):
        x = d["audio"][d["offsets"][i] : d["offsets"][i] + d["lengths"][i]].astype(np.float32)
        x *= rng.choice(peaks) / (np.abs(x).max() / 32768 + 1e-9) / 32768
        start = rng.integers(0, CLIP_SAMPLES - len(x) + 1)
        audio[j, start : start + len(x)] = np.clip(np.round(x * 32768), -32768, 32767)
        word = str(d["words"][i])
        labels[j] = CLASSES.index(word) if word in KEYWORDS else UNKNOWN_IDX
    return audio, labels


def make_splits(kind: str, device, gain_db=0.0, seed=0, realism=False, extra_frac=0.2):
    """Train / model-selection / real-test splits for a data recipe.

      real      Speech Commands (the original recipe)
      tts       keywords and negatives all synthetic; silence from real background noise
      tts+real  tts, plus real non-keyword speech in "unknown" (as many clips as synthetic)

    Model selection uses held-out synthetic speakers for the TTS recipes, so real speech is
    only ever used for the final test.
    """
    test = Split("test", device)
    if kind == "real":
        return (Split("train", device, gain_db=gain_db, seed=seed, realism=realism),
                Split("val", device), test)
    audio, labels = tts_arrays("train", seed)
    if kind == "tts+real":
        d = np.load(CACHE_DIR / "train.npz")
        real_unk = np.nonzero(d["labels"] == DEFAULT_CLASSES.index(UNKNOWN))[0]
        n = int((labels == UNKNOWN_IDX).sum())
        pick = np.random.default_rng(seed).choice(real_unk, min(n, len(real_unk)), replace=False)
        audio = np.concatenate([audio, d["audio"][pick]])
        labels = np.concatenate([labels, d["labels"][pick]])
    elif kind != "tts":
        raise ValueError(kind)
    # Each epoch: every keyword clip, plus unknowns and silences at `extra_frac` of that count
    # each (20% by default; more for a small vocabulary, to see enough non-keywords).
    train = Split("train_tts", device, gain_db=gain_db, seed=seed, arrays=(audio, labels),
                  extra_frac=extra_frac, realism=realism)
    val = Split("val_tts", device, arrays=tts_arrays("val", seed))
    return train, val, test


if __name__ == "__main__":
    build_cache()
    for s in ["train", "val", "test"]:
        d = np.load(CACHE_DIR / f"{s}.npz")
        counts = np.bincount(d["labels"], minlength=len(CLASSES))
        print(s, len(d["labels"]), dict(zip(CLASSES, counts.tolist())))
