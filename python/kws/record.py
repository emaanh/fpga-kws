"""Record the board's microphone over the USB-UART (live mode with record on: SW15 and SW14).

The board streams 16 kHz PCM as 3 bytes per sample: {1, 5'b0, x[15:14]}, {0, x[13:7]},
{0, x[6:0]}. Saves a WAV file and prints the level once a second, which is handy for setting
the gain switches (SW3:0): speech should peak around -20 to -10 dBFS.

    uv run python -m kws.record out.wav --seconds 10
"""

import argparse
import time

import numpy as np
import serial
from scipy.io import wavfile

from .host import BAUD, find_port

SR = 16_000


def decode(buf: bytearray) -> tuple[np.ndarray, bytearray]:
    """Decode complete 3-byte samples from buf; returns (samples, leftover bytes)."""
    out = []
    i = 0
    while i + 3 <= len(buf):
        b0, b1, b2 = buf[i], buf[i + 1], buf[i + 2]
        if not b0 & 0x80 or b1 & 0x80 or b2 & 0x80:
            i += 1  # out of sync: skip to the next start byte
            continue
        x = ((b0 & 3) << 14) | (b1 << 7) | b2
        out.append(x - 65536 if x & 0x8000 else x)
        i += 3
    return np.array(out, np.int16), buf[i:]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out", help="WAV file to write")
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--port")
    args = p.parse_args()

    port = args.port or find_port()
    print(f"{port}: recording {args.seconds:.0f} s (board needs SW15 and SW14 on)")
    chunks, buf = [], bytearray()
    n_total, t_last = 0, time.time()
    with serial.Serial(port, BAUD, timeout=0.2) as ser:
        ser.reset_input_buffer()
        while n_total < args.seconds * SR:
            buf += ser.read(4096)
            samples, buf = decode(buf)
            if len(samples):
                chunks.append(samples)
                n_total += len(samples)
            if time.time() - t_last >= 1.0:
                t_last = time.time()
                recent = np.concatenate(chunks[-8:]) if chunks else np.zeros(1, np.int16)
                peak = max(int(np.abs(recent.astype(np.int32)).max()), 1)
                print(f"  {n_total / SR:5.1f} s  peak {20 * np.log10(peak / 32768):6.1f} dBFS")
            if not samples.size and not buf and time.time() - t_last > 0.9:
                print("  no data: is the board in live + record mode?")
    pcm = np.concatenate(chunks)[: int(args.seconds * SR)]
    wavfile.write(args.out, SR, pcm)
    print(f"wrote {args.out}: {len(pcm) / SR:.1f} s")


if __name__ == "__main__":
    main()
