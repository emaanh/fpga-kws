"""Live demo on the Mac: your microphone through the FPGA's bit-exact pipeline, in a browser.

    uv run python -m kws.demo            # then open http://127.0.0.1:8777

Mac mic (16 kHz) -> gain in 6 dB steps (like SW3:0) -> fixed-point frontend (bit-exact to
audio_frontend.sv) -> int8 engine (bit-exact to kws_engine.sv) every 5 frames on the last 49
-> the decision rule of kws_live.sv (margin + 3 wins in a row, 1 s hold). The board's PDM mic
path is not modelled here (in simulation it costs ~0.4% accuracy).
"""

import argparse
import json
import queue
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import sounddevice as sd
import torch

from .config import CKPT_DIR, CLASSES, HOP
from .fixed_frontend import features_fixed
from .quant import int_forward

SR = 16_000
WIN = 512
IN_H = 49
INFER_EVERY = 5
N_CONSEC = 3
HOLD_S = 1.0
MARGINS = {"00": 1 << 18, "01": 1 << 17, "10": 1 << 19, "11": 0}  # SW5:4 on the board
HTML = Path(__file__).with_name("demo.html")


class Pipeline:
    def __init__(self, model="dscnn_int8.pt"):
        self.params = torch.load(CKPT_DIR / model)
        qat = torch.load(CKPT_DIR / model.replace("int8", "qat"), map_location="cpu")
        # Integer logits are float logits times a fixed scale (500 * 2^k).
        self.logit_scale = float(np.median(self.params["fc_b"].numpy() / qat["fc_bias"].numpy()))
        self.gain_db = 0
        self.margin_sel = "00"
        self.audio = queue.Queue()
        self.clients: list[queue.Queue] = []
        self.lock = threading.Lock()

    # -------------------------------------------------------------------------------------
    def broadcast(self, event: dict):
        msg = json.dumps(event)
        with self.lock:
            for q in self.clients:
                if q.qsize() < 500:
                    q.put(msg)

    def state(self):
        return {"type": "state", "gain_db": self.gain_db, "margin_sel": self.margin_sel,
                "classes": [c.strip("_") for c in CLASSES]}

    # -------------------------------------------------------------------------------------
    def run(self):
        buf = np.zeros(0, np.int16)
        total = 0           # samples seen
        next_end = WIN      # sample count at which the next frame is complete
        frames = deque(maxlen=64)
        n_frames = 0
        prev_class, hits = 0, 0
        show_class, show_until = 0, 0.0
        last_rms_warn = 0.0

        while True:
            x = self.audio.get()
            gain = 10 ** (self.gain_db / 20)
            pcm = np.clip(np.round(x * gain * 32768), -32768, 32767).astype(np.int16)
            buf = np.concatenate([buf, pcm])[-(SR + WIN):]
            total += len(pcm)

            while total >= next_end:
                end = len(buf) - (total - next_end)
                q = features_fixed(buf[end - WIN : end], self.params["mean"], self.params["f_in"])[0]
                frames.append(q)
                n_frames += 1
                next_end += HOP

                recent = buf[-1024:].astype(np.int32)
                peak = int(np.abs(recent).max()) if len(recent) else 0
                level = peak.bit_length()  # kws_live.sv: index of the leading one, plus one
                self.broadcast({"type": "frame", "q": q.tolist(), "level": min(level, 15),
                                "dbfs": round(20 * np.log10(max(peak, 1) / 32768), 1)})
                if peak == 0 and time.time() - last_rms_warn > 3:
                    last_rms_warn = time.time()
                    self.broadcast({"type": "warn", "msg": "The mic is silent. Check that this "
                                    "terminal app has microphone access (System Settings > "
                                    "Privacy & Security > Microphone)."})

                if n_frames % INFER_EVERY or n_frames < IN_H:
                    continue
                t0 = time.perf_counter()
                window = np.stack(list(frames)[-IN_H:])
                x_in = torch.from_numpy(window.astype(np.int64)).view(1, 1, IN_H, -1)
                logits = int_forward(self.params, x_in)[0]
                top2 = logits.topk(2).values
                cls = int(logits.argmax())
                margin = int(top2[0] - top2[1])
                infer_ms = (time.perf_counter() - t0) * 1000

                # Decision, as in kws_live.sv
                now = time.time()
                win = cls if margin >= MARGINS[self.margin_sel] else 0
                hits = (hits + 1 if win == prev_class else 1) if win >= 2 else 0
                prev_class = win
                showing = now < show_until
                if hits == N_CONSEC and not (showing and show_class == win):
                    show_class, show_until = win, now + HOLD_S
                    self.broadcast({"type": "detect", "cls": win, "word": CLASSES[win].strip("_"),
                                    "t": now})
                probs = torch.softmax(logits.double() / self.logit_scale, 0).tolist()
                self.broadcast({
                    "type": "infer", "cls": cls, "win": win, "hits": hits,
                    "margin": margin, "margin_f": margin / self.logit_scale,
                    "threshold_f": MARGINS[self.margin_sel] / self.logit_scale,
                    "probs": probs, "logits": logits.tolist(), "infer_ms": round(infer_ms, 1),
                    "showing": now < show_until, "show_class": show_class,
                })


def make_handler(pipe: Pipeline):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            if self.path == "/":
                body = HTML.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/events":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                q: queue.Queue = queue.Queue()
                q.put(json.dumps(pipe.state()))
                with pipe.lock:
                    pipe.clients.append(q)
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
                    with pipe.lock:
                        pipe.clients.remove(q)
            else:
                self.send_error(404)

        def do_POST(self):
            if self.path != "/control":
                return self.send_error(404)
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if "gain_db" in body:
                pipe.gain_db = int(np.clip(int(body["gain_db"]), -24, 48))
            if body.get("margin_sel") in MARGINS:
                pipe.margin_sel = body["margin_sel"]
            pipe.broadcast(pipe.state())
            self.send_response(204)
            self.end_headers()

    return Handler


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8777)
    p.add_argument("--device", help="input device name or index (default: system default)")
    p.add_argument("--model", default="dscnn_int8.pt", help="integer model in checkpoints/")
    p.add_argument("--file", help="play a 16 kHz WAV (e.g. a board recording) instead of the mic")
    p.add_argument("--loop", action="store_true", help="with --file: repeat forever")
    args = p.parse_args()

    pipe = Pipeline(args.model)
    threading.Thread(target=pipe.run, daemon=True).start()

    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(pipe))
    url = f"http://127.0.0.1:{args.port}"

    if args.file:
        from scipy.io import wavfile

        sr, pcm = wavfile.read(args.file)
        assert sr == SR and pcm.dtype == np.int16, "expects 16 kHz int16 WAV"
        x = pcm.astype(np.float32) / 32768

        def play():
            time.sleep(1.0)  # give the browser a moment to connect
            while True:
                sd.play(x, SR)  # through the speakers, in step with the pipeline
                t0 = time.time()
                for i in range(0, len(x), HOP):
                    pipe.audio.put(x[i : i + HOP])
                    time.sleep(max(0.0, t0 + (i + HOP) / SR - time.time()))
                sd.wait()
                if not args.loop:
                    break
                pipe.audio.put(np.zeros(SR, np.float32))  # a second of silence between loops

        threading.Thread(target=play, daemon=True).start()
        print(f"Model {args.model}. Playing {args.file}. Dashboard: {url}  (Ctrl+C to stop)", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        return

    def callback(indata, n, t, status):
        pipe.audio.put(indata[:, 0].copy())

    device = int(args.device) if args.device and args.device.isdigit() else args.device
    stream = sd.InputStream(samplerate=SR, channels=1, dtype="float32", blocksize=HOP,
                            device=device, callback=callback)
    name = sd.query_devices(stream.device, "input")["name"]
    with stream:
        print(f"Model {args.model}. Listening on '{name}'. Dashboard: {url}  (Ctrl+C to stop)",
              flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
