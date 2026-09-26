"""Turn prompted recordings of your own voice (kws.record_ui) into 1 s training clips.

  <name>_<n>.wav   prompted every 2.5 s: for each prompt window, the loudest 1 s is one clip
  talk_<n>.wav     overlapping 1 s windows (hop 0.5 s), all "unknown"
  silence_<n>.wav  overlapping 1 s windows, all "silence"
"""

from pathlib import Path

import numpy as np
from scipy.io import wavfile

from .config import CLASSES, CLIP_SAMPLES, SILENCE, UNKNOWN

PROMPT_START, PROMPT_EVERY = 0.8, 2.5  # must match record_ui.html


def loudest_window(x: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """The 1 s window with the most energy whose centre lies in [lo, hi)."""
    e = np.convolve(x.astype(np.float64) ** 2, np.ones(CLIP_SAMPLES // 2), "same")
    c = lo + int(np.argmax(e[lo:hi]))
    s = int(np.clip(c - CLIP_SAMPLES // 2, 0, max(0, len(x) - CLIP_SAMPLES)))
    out = np.zeros(CLIP_SAMPLES, np.int16)
    seg = x[s : s + CLIP_SAMPLES]
    out[: len(seg)] = seg
    return out


def clips_from_folder(folder: Path):
    """(audio int16 (N, 16000), labels, source file per clip) for every take in folder."""
    audio, labels, src = [], [], []
    for f in sorted(folder.glob("*.wav")):
        sr, x = wavfile.read(f)
        assert sr == 16000, f
        label = f.name.rsplit("_", 1)[0]
        if label in CLASSES and label not in (SILENCE, UNKNOWN):
            t = PROMPT_START
            while t + 0.5 < len(x) / sr:
                # The word follows the prompt within about 2 s.
                lo, hi = int((t + 0.1) * sr), int(min(len(x), (t + PROMPT_EVERY) * sr))
                if hi > lo:
                    audio.append(loudest_window(x, lo, hi))
                    labels.append(CLASSES.index(label))
                    src.append(f.name)
                t += PROMPT_EVERY
        elif label in ("talk", "silence"):
            cls = CLASSES.index(UNKNOWN if label == "talk" else SILENCE)
            for s in range(0, len(x) - CLIP_SAMPLES + 1, CLIP_SAMPLES // 2):
                audio.append(x[s : s + CLIP_SAMPLES].astype(np.int16))
                labels.append(cls)
                src.append(f.name)
    if not audio:
        return np.zeros((0, CLIP_SAMPLES), np.int16), np.zeros(0, np.int64), []
    return np.stack(audio), np.array(labels, np.int64), src
