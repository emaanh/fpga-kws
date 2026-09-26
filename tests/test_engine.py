"""cocotb tests for rtl/kws_engine.sv against the streaming integer reference
(python/kws/quant.py stream_forward).

Feeds test-set feature frames as a continuous stream and checks every result that covers a
whole window, logits and all, then restarts and checks a second stream.

    uv run pytest tests/test_engine.py
    KWS_N_CLIPS=20 uv run pytest tests/test_engine.py     # a longer stream

Needs `uv run python -m kws.export` first (ROMs + test features).
"""

import os
from pathlib import Path

import numpy as np
import torch

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, RisingEdge

ROOT = Path(__file__).resolve().parents[1]


def load_reference():
    """(test features (N, 49, 40), labels, window reference feat -> logits)."""
    from kws.config import CKPT_DIR, MODEL_INT8, VEC_DIR
    from kws.quant import int_forward

    params = torch.load(CKPT_DIR / MODEL_INT8)
    feats = np.fromfile(VEC_DIR / "test_features.bin", np.int8).reshape(-1, 49, 40)
    labels = np.fromfile(VEC_DIR / "test_labels_preds.bin", np.uint8).reshape(-1, 2)[:, 0]

    def reference(feat):
        x = torch.from_numpy(feat.astype(np.int64)).view(1, 1, 49, 40)
        return int_forward(params, x)[0].tolist()

    return feats, labels, reference


def make_stream(feats, first, n_clips):
    """Back-to-back test clips as one feature stream, and its streaming reference logits."""
    from kws.config import CKPT_DIR, MODEL_INT8
    from kws.quant import stream_forward

    params = torch.load(CKPT_DIR / MODEL_INT8)
    idx = np.linspace(first, len(feats) - 1, n_clips).astype(int)
    q = np.concatenate([feats[i] for i in idx]).astype(np.int64)
    return q, stream_forward(params, q).tolist()


async def feed_frame(dut, frame):
    for band, v in enumerate(frame):
        dut.feat_we.value = 1
        dut.feat_band.value = band
        dut.feat_wdata.value = int(v) & 0xFF
        await RisingEdge(dut.clk)
    dut.feat_we.value = 0
    dut.frame_end.value = 1
    await RisingEdge(dut.clk)
    dut.frame_end.value = 0


async def run_stream(dut, q, expected, label):
    """Restart, feed q frame by frame like live mode, check every full result."""
    results, partial, cycles = [], 0, []

    async def watch():
        nonlocal partial
        while True:
            await RisingEdge(dut.busy)
            t0 = cocotb.utils.get_sim_time("ns")
            await RisingEdge(dut.done)
            cycles.append(int((cocotb.utils.get_sim_time("ns") - t0) / 10))
            await RisingEdge(dut.clk)  # outputs are registered with done
            if int(dut.full.value):
                results.append(([dut.logits[k].value.to_signed() for k in range(len(expected[0]))],
                                int(dut.class_idx.value), dut.margin.value.to_signed()))
            else:
                partial += 1

    dut.restart.value = 1
    await RisingEdge(dut.clk)
    dut.restart.value = 0
    watcher = cocotb.start_soon(watch())
    for frame in q:
        await feed_frame(dut, frame)
        # Like live mode: the next frame only comes once the engine has caught up.
        await ClockCycles(dut.clk, 2)
        while int(dut.pending.value) or int(dut.busy.value):
            await RisingEdge(dut.clk)
    await ClockCycles(dut.clk, 10)
    watcher.cancel()

    assert partial == 19, f"{label}: {partial} results before the first full window, expected 19"
    assert len(results) == len(expected), f"{label}: {len(results)} results, expected {len(expected)}"
    for k, ((logits, cls, margin), ref) in enumerate(zip(results, expected)):
        assert logits == ref, f"{label} result {k}: logits {logits} != {ref}"
        top = sorted(ref, reverse=True)
        assert cls == int(np.argmax(ref)), f"{label} result {k}: class {cls}"
        assert margin == top[0] - top[1], f"{label} result {k}: margin {margin}"
    dut._log.info(f"{label}: {len(results)} results bit-exact, {min(cycles)}..{max(cycles)} "
                  f"cycles per step")


@cocotb.test()
async def test_stream(dut):
    feats, _, _ = load_reference()
    Clock(dut.clk, 10, unit="ns").start()
    dut.rst.value = 1
    dut.restart.value = 0
    dut.feat_we.value = 0
    dut.frame_end.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst.value = 0
    await RisingEdge(dut.clk)

    n = int(os.environ.get("KWS_N_CLIPS", "3"))
    q, expected = make_stream(feats, 0, n)
    await run_stream(dut, q, expected, f"stream of {n} clips")
    # A restart must forget the first stream completely.
    q, expected = make_stream(feats, 7, 2)
    await run_stream(dut, q, expected, "second stream after a restart")


def test_engine():
    from cocotb_tools.runner import get_runner

    from kws.config import GEN_DIR

    runner = get_runner("verilator")
    runner.build(
        sources=[GEN_DIR / "kws_pkg.sv", ROOT / "rtl/sdp_ram.sv", ROOT / "rtl/kws_engine.sv"],
        hdl_toplevel="kws_engine",
        build_dir=ROOT / "build" / "sim_engine",
        always=True,
        build_args=["--public-flat-rw", "-Wno-fatal", "-Wno-WIDTHEXPAND", "-Wno-UNUSEDSIGNAL",
                    f'-DKWS_MEM_DIR="{GEN_DIR}/"'],
    )
    runner.test(hdl_toplevel="kws_engine", test_module="test_engine",
                test_dir=Path(__file__).parent, build_dir=ROOT / "build" / "sim_engine")
