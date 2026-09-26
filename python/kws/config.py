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
# The default vocabulary is the 10 Speech Commands words plus two names trained from
# synthetic speech. Set KWS_KEYWORDS=yes,no,... to train and export another one.
DEFAULT_KEYWORDS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
KEYWORDS = (os.environ["KWS_KEYWORDS"].split(",") if os.environ.get("KWS_KEYWORDS")
            else [*DEFAULT_KEYWORDS, "emaan", "heidari"])
SILENCE, UNKNOWN = "_silence_", "_unknown_"
CLASSES = [SILENCE, UNKNOWN, *KEYWORDS]
DEFAULT_CLASSES = [SILENCE, UNKNOWN, *DEFAULT_KEYWORDS]  # what the Speech Commands cache uses

# Which model to export/test and where the generated RTL files go. Defaults: the 12-word
# streaming model into rtl/gen/.
MODEL_INT8 = os.environ.get("KWS_MODEL", "dscnn_int8_stream.pt")
MODEL_FLOAT = os.environ.get("KWS_FLOAT", "dscnn_float_stream.pt")
GEN_DIR = ROOT / os.environ.get("KWS_GEN", "rtl/gen")
VEC_DIR = ROOT / os.environ.get("KWS_VEC", "build/vectors")
# Live rule (25 results/s): a keyword must win N_CONSEC results in a row, each by at least
# MARGIN integer logit units over the runner-up. Tuned with kws.eval_stream/eval_recordings.
N_CONSEC = int(os.environ.get("KWS_N_CONSEC", "6"))
MARGIN = int(os.environ.get("KWS_MARGIN", str(3 << 16)))
