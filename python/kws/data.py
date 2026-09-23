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

from .config import CLASSES, CLIP_SAMPLES, DATA_DIR, KEYWORDS, SAMPLE_RATE

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

    def __init__(self, name: str, device, extra_frac=0.1, seed=0, gain_db=0.0):
        build_cache()
        self.gain_db = gain_db
        d = np.load(CACHE_DIR / f"{name}.npz")
        self.audio = torch.from_numpy(d["audio"])
        self.labels = torch.from_numpy(d["labels"])
        self.noise = torch.from_numpy(np.load(CACHE_DIR / "noise.npy")).to(device)
        self.device = device
        self.train = name == "train"
        self.kw_idx = torch.nonzero(self.labels != UNKNOWN_IDX).squeeze(1)
        self.unk_idx = torch.nonzero(self.labels == UNKNOWN_IDX).squeeze(1)
        self.n_extra = int(len(self.kw_idx) * extra_frac)
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


if __name__ == "__main__":
    build_cache()
    for s in ["train", "val", "test"]:
        d = np.load(CACHE_DIR / f"{s}.npz")
        counts = np.bincount(d["labels"], minlength=len(CLASSES))
        print(s, len(d["labels"]), dict(zip(CLASSES, counts.tolist())))
