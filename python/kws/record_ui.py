"""Browser UI for recording test clips from the board's microphone.

Board: SW15 (live) and SW14 (record) up, gain on SW3:0. Then:

    uv run python -m kws.record_ui        # http://127.0.0.1:8778

Shows a live level meter, prompts you ("Say: Emaan") at a steady pace while recording, and
saves each take as recordings/names_test/<label>_<n>.wav (16 kHz int16) for evaluation.
"""

import argparse
import json
import queue
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import serial
from scipy.io import wavfile

from .config import ROOT
from .host import BAUD, find_port
from .record import decode

SR = 16_000
OUT_DIR = ROOT / "recordings" / "names_test"
HTML = Path(__file__).with_name("record_ui.html")
LABELS = {
    "emaan": {"title": "Emaan", "prompt": "Emaan", "seconds": 30, "every": 2.5},
    "heidari": {"title": "Heidari", "prompt": "Heidari", "seconds": 30, "every": 2.5},
    "talk": {"title": "Normal talking", "prompt": "Talk normally, no names", "seconds": 30, "every": 0},
    "silence": {"title": "Silence", "prompt": "Stay quiet", "seconds": 15, "every": 0},
}


class Recorder:
    def __init__(self, port):
        self.port = port
        self.clients: list[queue.Queue] = []
        self.lock = threading.Lock()
        self.take: list[np.ndarray] | None = None  # samples collected for the current take
        self.take_info = None
        self.last_rx = 0.0

    def broadcast(self, event):
        msg = json.dumps(event)
        with self.lock:
            for q in self.clients:
                if q.qsize() < 200:
                    q.put(msg)

    def files(self):
        out = []
        for f in sorted(OUT_DIR.glob("*.wav")):
            sr, x = wavfile.read(f)
            out.append({"name": f.name, "label": f.name.rsplit("_", 1)[0],
                        "seconds": round(len(x) / sr, 1)})
        return out

    def state(self):
        return {"type": "state", "files": self.files(), "labels": LABELS,
                "recording": self.take_info}

    def run_serial(self):
        buf = bytearray()
        pending = np.zeros(0, np.int16)
        with serial.Serial(self.port, BAUD, timeout=0.05) as ser:
            ser.reset_input_buffer()
            while True:
                buf += ser.read(4096)
                samples, buf = decode(buf)
                if len(samples):
                    self.last_rx = time.time()
                    if self.take is not None:
                        self.take.append(samples)
                    pending = np.concatenate([pending, samples])
                if len(pending) >= SR // 10:  # a level update every 100 ms
                    peak = int(np.abs(pending.astype(np.int32)).max())
                    self.broadcast({"type": "level",
                                    "dbfs": round(20 * np.log10(max(peak, 1) / 32768), 1)})
                    pending = np.zeros(0, np.int16)
                if time.time() - self.last_rx > 1.0:
                    self.broadcast({"type": "level", "dbfs": None})
                    self.last_rx = time.time()

    def record(self, label):
        spec = LABELS[label]
        self.take = []
        self.take_info = {"label": label, "start": time.time(), **spec}
        self.broadcast(self.state())
        time.sleep(spec["seconds"])
        chunks, self.take = self.take, None
        self.take_info = None
        pcm = np.concatenate(chunks) if chunks else np.zeros(0, np.int16)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        n = 1 + max([int(m.group(1)) for f in OUT_DIR.glob(f"{label}_*.wav")
                     if (m := re.search(r"_(\d+)\.wav$", f.name))], default=0)
        name = f"{label}_{n}.wav"
        if len(pcm):
            wavfile.write(OUT_DIR / name, SR, pcm)
        self.broadcast({**self.state(), "saved": name if len(pcm) else None,
                        "error": None if len(pcm) else "No audio arrived: is SW14 up?"})


def make_handler(rec: Recorder):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send_body(self, body: bytes, ctype: str):
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/":
                return self.send_body(HTML.read_bytes(), "text/html; charset=utf-8")
            if self.path.startswith("/audio/"):
                f = OUT_DIR / Path(self.path[7:]).name
                if f.exists():
                    return self.send_body(f.read_bytes(), "audio/wav")
                return self.send_error(404)
            if self.path == "/events":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                q: queue.Queue = queue.Queue()
                q.put(json.dumps(rec.state()))
                with rec.lock:
                    rec.clients.append(q)
                try:
                    while True:
                        try:
                            msg = q.get(timeout=5)
                        except queue.Empty:
                            msg = None
                        self.wfile.write(f"data: {msg}\n\n".encode() if msg else b": ping\n\n")
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
                finally:
                    with rec.lock:
                        rec.clients.remove(q)
                return
            self.send_error(404)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])) or b"{}")
            if self.path == "/record" and body.get("label") in LABELS and rec.take_info is None:
                threading.Thread(target=rec.record, args=(body["label"],), daemon=True).start()
            elif self.path == "/delete":
                f = OUT_DIR / Path(body.get("name", "")).name
                if f.exists() and f.suffix == ".wav":
                    f.unlink()
                rec.broadcast(rec.state())
            self.send_response(204)
            self.end_headers()

    return Handler


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8778)
    p.add_argument("--serial")
    args = p.parse_args()
    rec = Recorder(args.serial or find_port())
    threading.Thread(target=rec.run_serial, daemon=True).start()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(rec))
    print(f"Recorder on {rec.port}. Open http://127.0.0.1:{args.port}  (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
