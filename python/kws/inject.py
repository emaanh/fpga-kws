"""Test the board's live path with audio sent over USB instead of the mic.

Board: SW15 (live) and SW12 (audio from host) up, SW14 down. This streams test-set keywords
at 16 kHz in real time, reads the board's debug packets, and checks:
  - every feature row the board reports against the bit-exact Python frontend
  - which words the board detects

    uv run python -m kws.inject                       # yes, stop, go, left
    uv run python -m kws.inject --words up down on off
"""

import argparse
import threading
import time

import numpy as np
import serial
import torch

from .config import CKPT_DIR, CLASSES, HOP
from .data import CACHE_DIR
from .debug_live import parse_packets
from .fixed_frontend import features_fixed
from .host import BAUD, find_port

SR = 16_000


def encode(pcm: np.ndarray) -> bytes:
    x = pcm.astype(np.int32) & 0xFFFF
    b = np.stack([0x80 | (x >> 14), (x >> 7) & 0x7F, x & 0x7F], 1).astype(np.uint8)
    return b.tobytes()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--words", nargs="*", default=["yes", "stop", "go", "left"])
    p.add_argument("--port")
    args = p.parse_args()

    d = np.load(CACHE_DIR / "test.npz")
    parts = [np.zeros(SR, np.int16)]
    for w in args.words:
        i = int(np.nonzero(d["labels"] == CLASSES.index(w))[0][0])
        parts += [d["audio"][i], np.zeros(SR // 2, np.int16)]
    parts.append(np.zeros(SR, np.int16))
    stream = np.concatenate(parts)

    port = args.port or find_port()
    packets, buf, stop = [], bytearray(), threading.Event()
    with serial.Serial(port, BAUD, timeout=0.05) as ser:
        ser.reset_input_buffer()

        def reader():
            while not stop.is_set():
                buf.extend(ser.read(4096))
                packets.extend(parse_packets(buf))

        th = threading.Thread(target=reader, daemon=True)
        th.start()
        print(f"{port}: sending {len(stream) / SR:.1f} s of audio: {args.words}")
        t0 = time.time()
        for i in range(0, len(stream), HOP):
            ser.write(encode(stream[i : i + HOP]))
            time.sleep(max(0.0, t0 + (i + HOP) / SR - time.time()))
        time.sleep(0.5)
        stop.set()
        th.join()

    if not packets:
        raise SystemExit("no debug packets: is the board in live mode (SW15 up, SW14 down)?")
    # The board's frames started before the stream, so its frame grid is offset: try all.
    params = torch.load(CKPT_DIR / "dscnn_int8.pt")
    rows = [tuple(pk["row"]) for pk in packets]
    best = (-1, 0, None)
    for k in range(HOP):
        r = features_fixed(np.concatenate([np.zeros(k, np.int16), stream]), params["mean"], params["f_in"])
        n = sum(row in {tuple(x) for x in r.tolist()} for row in rows)
        if n > best[0]:
            best = (n, k, r)
    exact, offset, ref = best
    ref_rows = {tuple(r) for r in ref.tolist()}
    names = [CLASSES[pk["cls"]].strip("_")[:5] for pk in packets]
    dets = [CLASSES[pk["cls"]].strip("_") for pk in packets if pk["det"]]
    print(f"{len(packets)} inferences; copied bytes {sorted({pk['copy_n'] for pk in packets})}, "
          f"copy OR {sorted({hex(pk['copy_or']) for pk in packets})}")
    print(f"board feature rows matching a Python frame exactly: {exact}/{len(packets)} "
          f"(frame offset {offset})")
    if exact < len(packets):
        pk = next(pk for pk in packets if tuple(pk["row"]) not in ref_rows)
        best = ref[np.abs(ref - np.array(pk["row"])).sum(1).argmin()]
        print(f"  e.g. board {list(pk['row'][:10])}...\n       python {list(best[:10])}...")
    print("classes:", " ".join(names))
    print("detections:", dets, " played:", args.words)


if __name__ == "__main__":
    main()
