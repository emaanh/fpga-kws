"""Watch the board's live-mode decisions over the USB-UART (SW15 on, SW14 off).

After every engine result the board sends: 'K', class, margin (int32 LE),
{detected, 3'b0, mic level}, then the frontend's last frame: leading-one position of the
largest windowed sample and of the largest power value, the number of features it produced
(should be 40), the OR of the features the engine read, and the 40 features. This prints
one line per result (25 per second).

    uv run python -m kws.debug_live            # until Ctrl+C
    uv run python -m kws.debug_live --seconds 15
"""

import argparse
import struct
import time

import serial
import torch

from .config import CKPT_DIR, CLASSES, MARGIN, MODEL_INT8
from .host import BAUD, find_port

MARGIN_DEFAULT = MARGIN  # SW5:4 = 00
N_MELS = 40
PACKET = 20 + N_MELS


def parse_packets(buf: bytearray):
    """Pop complete debug packets off the front of buf; returns a list of dicts."""
    out = []
    while len(buf) >= PACKET:
        if buf[0] != ord("K") or buf[1] >= len(CLASSES) or buf[18] > N_MELS:
            del buf[0]  # resync
            continue
        (margin,) = struct.unpack("<i", bytes(buf[2:6]))
        out.append({
            "cls": buf[1], "margin": margin, "det": buf[6] >> 7, "level": buf[6] & 0xF,
            "win": int.from_bytes(bytes(buf[7:11]), "little").bit_length(),
            "pow": int.from_bytes(bytes(buf[11:18]), "little").bit_length(),
            "nfeat": buf[18], "feat_or": buf[19],
            "row": struct.unpack(f"{N_MELS}b", bytes(buf[20:PACKET])),
        })
        del buf[:PACKET]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seconds", type=float, default=0, help="0 = until Ctrl+C")
    p.add_argument("--port")
    args = p.parse_args()

    # Integer logits are float logits times a fixed scale; show margins in float units too.
    params = torch.load(CKPT_DIR / MODEL_INT8)
    qat = torch.load(CKPT_DIR / MODEL_INT8.replace("int8", "qat"), map_location="cpu")
    scale = float((params["fc_b"].double() / qat["fc_bias"].double()).median())

    port = args.port or find_port()
    print(f"{port}: board in live mode, record off (SW15 on, SW14 off). Ctrl+C to stop.")
    print(f"{'time':>6}  {'class':<8} {'margin':>7}  {'level':<15}  win pow feats  "
          f"features min/mean/max  eng_or")
    buf = bytearray()
    t0 = time.time()
    counts, n = {}, 0
    with serial.Serial(port, BAUD, timeout=0.2) as ser:
        ser.reset_input_buffer()
        try:
            while not args.seconds or time.time() - t0 < args.seconds:
                buf += ser.read(256)
                for pk in parse_packets(buf):
                    cls, margin, row = pk["cls"], pk["margin"], pk["row"]
                    n += 1
                    counts[CLASSES[cls]] = counts.get(CLASSES[cls], 0) + 1
                    conf = "*" if margin >= MARGIN_DEFAULT else " "
                    line = (f"{time.time() - t0:6.1f}  {CLASSES[cls].strip('_'):<8} "
                            f"{margin / scale:6.2f}{conf}  {'#' * pk['level']:<15}  "
                            f"{pk['win']:3d} {pk['pow']:3d} {pk['nfeat']:5d}  "
                            f"{min(row):4d} {sum(row) / len(row):6.1f} {max(row):4d}  "
                            f"  0x{pk['feat_or']:02x}")
                    if pk["det"]:
                        line += f"  <== DETECTED {CLASSES[cls].strip('_').upper()}"
                    print(line, flush=True)
                if not buf and n == 0 and time.time() - t0 > 3:
                    print("no data yet: is SW15 up and SW14 down?", flush=True)
                    t0 = time.time()
        except KeyboardInterrupt:
            pass
    print(f"\n{n} results: " + ", ".join(f"{k.strip('_')} {v}" for k, v in
                                           sorted(counts.items(), key=lambda kv: -kv[1])))
    print("margin: float logit gap between the top two classes; * = above the default "
          "detection margin (SW5:4 = 00)")


if __name__ == "__main__":
    main()
