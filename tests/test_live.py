"""cocotb end-to-end test of live mode: PDM bits -> kws_top -> classes and detections.

Plays a few seconds of audio (quiet noise, then test-set keywords) as PDM bits into the
mic pins and checks every inference's class and every detection against the bit-exact
Python model: mic_model -> fixed_frontend -> int_forward -> decision rule.

    uv run pytest tests/test_live.py
"""

from pathlib import Path

import numpy as np

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, RisingEdge

from test_top import TOP_SOURCES

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build" / "sim_live"
CLK_DIV = 4          # PDM clock = clk / 8 here (clk / 50 on the board)
GAIN = 6
LEVEL = 2**-6        # speech level at the mic, relative to full scale
WORDS = ["yes", "stop", "go", "left"]
INFER_EVERY, IN_H = 5, 49
N_CONSEC, MIN_MARGIN = 3, 262_144  # kws_live.sv defaults


def make_stream():
    import torch  # noqa: F401  (kws imports need it)

    from kws.config import CLASSES
    from kws.data import CACHE_DIR

    d = np.load(CACHE_DIR / "test.npz")
    noise = np.load(CACHE_DIR / "noise.npy")
    quiet = noise[:8000] * 0.01
    parts = [quiet]
    for w in WORDS:
        i = int(np.nonzero(d["labels"] == CLASSES.index(w))[0][0])
        parts.append(d["audio"][i].astype(np.float64) / 32768)
    parts.append(quiet)
    return np.concatenate(parts)


def expected_results(bits):
    import torch

    from kws.config import CKPT_DIR
    from kws.fixed_frontend import features_fixed
    from kws.mic_model import mic_pcm
    from kws.quant import int_forward

    params = torch.load(CKPT_DIR / "dscnn_int8.pt")
    q = features_fixed(mic_pcm(bits, GAIN), params["mean"], params["f_in"])
    windows = [q[k - IN_H : k] for k in range(INFER_EVERY * ((IN_H + INFER_EVERY - 1) // INFER_EVERY),
                                            len(q) + 1, INFER_EVERY)]
    x = torch.from_numpy(np.stack(windows).astype(np.int64)).unsqueeze(1)
    from kws.eval_stream import detect

    logits = int_forward(params, x)
    top2 = logits.topk(2, dim=1).values
    classes = logits.argmax(1).tolist()
    margins = (top2[:, 0] - top2[:, 1]).tolist()
    detections = [c for _, c in detect(classes, margins, N_CONSEC, MIN_MARGIN)]
    return classes, margins, detections


def write_bits(bits):
    padded = np.concatenate([bits, np.zeros(-len(bits) % 32, np.uint8)])
    words = np.packbits(padded.reshape(-1, 32)[:, ::-1], axis=1).view(">u4").ravel()
    BUILD.mkdir(parents=True, exist_ok=True)
    (BUILD / "pdm_bits.hex").write_text("\n".join(f"{w:08x}" for w in words) + "\n")
    return len(words)


@cocotb.test()
async def test_live_mode(dut):
    from kws.config import CLASSES
    from kws.mic_model import to_pdm

    bits = to_pdm(make_stream(), LEVEL)
    classes_ref, margins_ref, detections_ref = expected_results(bits)

    Clock(dut.clk, 10, unit="ns").start()
    dut.sw.value = (1 << 15) | GAIN  # live mode, gain
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 10)
    dut.rst_n.value = 1

    classes, margins, detections = [], [], []

    async def watch_engine():
        while True:
            await RisingEdge(dut.u_top.u_engine.done)
            classes.append(int(dut.u_top.u_engine.class_idx.value))
            margins.append(dut.u_top.u_engine.margin.value.to_signed())

    async def watch_detect():
        while True:
            await RisingEdge(dut.u_top.u_live.detect)
            await RisingEdge(dut.clk)
            detections.append(int(dut.u_top.u_live.show_class.value))

    cocotb.start_soon(watch_engine())
    cocotb.start_soon(watch_detect())
    await RisingEdge(dut.done_playing)
    await ClockCycles(dut.clk, 800_000)  # let the last inference finish

    names = lambda cs: [CLASSES[c].strip("_") for c in cs]  # noqa: E731
    dut._log.info(f"inferences: {names(classes)}")
    dut._log.info(f"detections: {names(detections)} (played {WORDS})")
    n = min(len(classes), len(classes_ref))
    assert n >= len(classes_ref) - 1, f"only {len(classes)} inferences, expected {len(classes_ref)}"
    assert classes[:n] == classes_ref[:n], f"classes differ:\n{classes}\n{classes_ref}"
    assert margins[:n] == margins_ref[:n], f"margins differ:\n{margins}\n{margins_ref}"
    assert detections == detections_ref[: len(detections)] and \
        len(detections) >= len(detections_ref) - 1, f"{detections} vs {detections_ref}"


def test_live():
    from cocotb_tools.runner import get_runner

    from kws.mic_model import to_pdm

    n_words = write_bits(to_pdm(make_stream(), LEVEL))
    runner = get_runner("verilator")
    runner.build(
        sources=TOP_SOURCES + [ROOT / "tests/hdl/tb_live.sv"],
        hdl_toplevel="tb_live",
        build_dir=BUILD,
        always=True,
        parameters={"BITS_FILE": f'"{BUILD / "pdm_bits.hex"}"', "N_WORDS": n_words,
                    "CLK_DIV": CLK_DIV},
        build_args=["--public-flat-rw", "-Wno-fatal", "-Wno-WIDTHEXPAND", "-Wno-UNUSEDSIGNAL",
                    "-O3", f'-DKWS_MEM_DIR="{ROOT}/rtl/gen/"'],
    )
    runner.test(hdl_toplevel="tb_live", test_module="test_live",
                test_dir=Path(__file__).parent, build_dir=BUILD)
