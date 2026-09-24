"""Shared constants. Everything here is something the RTL will have to match."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
CKPT_DIR = ROOT / "checkpoints"

# Audio
SAMPLE_RATE = 16_000
CLIP_SAMPLES = SAMPLE_RATE  # 1 s clips

# Frontend: 512-sample Hann window (32 ms), 320-sample hop (20 ms), 512-pt FFT,
# 40 triangular mel bands, log2. Gives 49 frames x 40 bands per clip.
WIN = 512
HOP = 320
N_FFT = 512
N_MELS = 40
F_MIN = 20.0
F_MAX = 7600.0
N_FRAMES = (CLIP_SAMPLES - WIN) // HOP + 1  # 49
LOG_EPS = 1e-6

# Labels
# Set KWS_KEYWORDS=emaan,heidari (for example) to train and export a custom vocabulary.
DEFAULT_KEYWORDS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
KEYWORDS = (os.environ["KWS_KEYWORDS"].split(",") if os.environ.get("KWS_KEYWORDS")
            else DEFAULT_KEYWORDS)
SILENCE, UNKNOWN = "_silence_", "_unknown_"
CLASSES = [SILENCE, UNKNOWN, *KEYWORDS]
DEFAULT_CLASSES = [SILENCE, UNKNOWN, *DEFAULT_KEYWORDS]  # what the Speech Commands cache uses
