"""Run the test set through the board over the Nexys A7's USB-UART and check it against
the integer reference, logits and all.

    uv run python -m kws.host                 # whole test set, auto-detect the port
    uv run python -m kws.host --n 200 --port /dev/cu.usbserial-XXXXXXXX1

Protocol: see rtl/kws_top.sv.
"""

import argparse
import glob
import struct
import time

import numpy as np
import serial
import torch

from .config import CKPT_DIR, CLASSES, ROOT
from .quant import int_forward

VEC_DIR = ROOT / "build" / "vectors"
BAUD = 1_000_000
REPLY_LEN = 2 + 4 * len(CLASSES)


def find_port():
    # The FT2232 shows up twice: interface A is JTAG, interface B (the higher one) is the UART.
    ports = sorted(glob.glob("/dev/cu.usbserial-*"))
    if not ports:
        raise SystemExit("no /dev/cu.usbserial-* found: is the board plugged in and powered?")
    return ports[-1]


def infer(ser, feat: np.ndarray):
    ser.write(b"I" + feat.astype(np.int8).tobytes())
    reply = ser.read(REPLY_LEN)
    if len(reply) != REPLY_LEN or reply[:1] != b"R":
        # Let the board's frame timeout (50 ms) expire so the next request starts clean.
        time.sleep(0.1)
        ser.reset_input_buffer()
        raise IOError(f"bad reply ({len(reply)} bytes): {reply[:8]!r}")
    return reply[1], list(struct.unpack(f"<{len(CLASSES)}i", reply[2:]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port")
    p.add_argument("--n", type=int, help="number of test clips (default: all)")
    args = p.parse_args()

    feats = np.fromfile(VEC_DIR / "test_features.bin", np.int8).reshape(-1, 49, 40)
    labels = np.fromfile(VEC_DIR / "test_labels_preds.bin", np.uint8).reshape(-1, 2)[:, 0]
    idx = np.arange(len(feats)) if args.n is None else np.linspace(0, len(feats) - 1, args.n).astype(int)

    params = torch.load(CKPT_DIR / "dscnn_int8.pt")
    x = torch.from_numpy(feats[idx].astype(np.int64)).unsqueeze(1)
    ref_logits = torch.cat([int_forward(params, b) for b in x.split(512)]).tolist()

    port = args.port or find_port()
    print(f"{port} @ {BAUD} baud, {len(idx)} clips")
    correct = mismatches = errors = 0
    t0 = time.time()
    with serial.Serial(port, BAUD, timeout=1) as ser:
        ser.reset_input_buffer()
        for n, (i, ref) in enumerate(zip(idx, ref_logits)):
            try:
                cls, logits = infer(ser, feats[i])
            except IOError as e:
                errors += 1
                print(f"clip {i}: {e}")
                continue
            correct += cls == labels[i]
            if logits != ref:
                mismatches += 1
                print(f"clip {i}: logits differ from the reference")
            if n % 500 == 0:
                print(f"  {n}/{len(idx)}  clip {i}: {CLASSES[cls]} (label {CLASSES[labels[i]]})")
    elapsed = time.time() - t0
    done = len(idx) - errors
    print(f"\naccuracy {correct / max(done, 1):.4f} on {done} clips; "
          f"{mismatches} differ from the integer reference; {errors} UART errors; "
          f"{elapsed / max(done, 1) * 1000:.1f} ms/clip")


if __name__ == "__main__":
    main()
