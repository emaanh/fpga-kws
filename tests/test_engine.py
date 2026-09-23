"""cocotb tests for rtl/kws_engine.sv against the integer reference in python/kws/quant.py.

    uv run pytest tests/test_engine.py                     # per-layer check on a few clips
    KWS_N_CLIPS=200 uv run pytest tests/test_engine.py     # more clips (logits/class only)
    KWS_N_CLIPS=all uv run pytest tests/test_engine.py     # whole test set (~1 h)

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
N_LAYER_CHECK_CLIPS = 4  # clips that also get every layer's output compared


def load_reference():
    from kws.export import pack_acts
    from kws.quant import int_forward

    params = torch.load(ROOT / "checkpoints" / "dscnn_int8.pt")
    feats = np.fromfile(ROOT / "build" / "vectors" / "test_features.bin", np.int8).reshape(-1, 49, 40)
    labels = np.fromfile(ROOT / "build" / "vectors" / "test_labels_preds.bin", np.uint8).reshape(-1, 2)[:, 0]

    def reference(feat):
        x = torch.from_numpy(feat.astype(np.int64)).view(1, 1, 49, 40)
        logits, acts, pooled = int_forward(params, x, return_acts=True)
        words = [[int.from_bytes(row.astype(np.uint8).tobytes(), "little") for row in pack_acts(a[0])]
                 for a in acts]
        return logits[0].tolist(), words, pooled[0].tolist()

    return feats, labels, reference


def pick_clips(n_total):
    n = os.environ.get("KWS_N_CLIPS", "8")
    if n == "all":
        return list(range(n_total))
    return np.linspace(0, n_total - 1, int(n)).astype(int).tolist()


async def run_clip(dut, feat):
    for addr, v in enumerate(feat.flatten()):
        dut.feat_we.value = 1
        dut.feat_waddr.value = addr
        dut.feat_wdata.value = int(v) & 0xFF
        await RisingEdge(dut.clk)
    dut.feat_we.value = 0
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0


def read_buffer(mem, n_words):
    return [int(mem[i].value) for i in range(n_words)]


@cocotb.test()
async def test_clips(dut):
    feats, labels, reference = load_reference()
    Clock(dut.clk, 10, unit="ns").start()
    dut.rst.value = 1
    dut.feat_we.value = 0
    dut.start.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst.value = 0
    await RisingEdge(dut.clk)

    clips = pick_clips(len(feats))
    correct = 0
    for n, idx in enumerate(clips):
        logits_ref, words_ref, pooled_ref = reference(feats[idx])
        await run_clip(dut, feats[idx])
        start_time = cocotb.utils.get_sim_time("ns")

        if n < N_LAYER_CHECK_CLIPS:
            for layer, expected in enumerate(words_ref):
                await RisingEdge(dut.layer_done)
                mem = (dut.u_act_a if layer % 2 == 0 else dut.u_act_b).g_auto.mem
                got = read_buffer(mem, len(expected))
                bad = [i for i, (g, e) in enumerate(zip(got, expected)) if g != e]
                assert not bad, (f"clip {idx} layer {layer}: {len(bad)} words differ, first at "
                                 f"{bad[0]}: got {got[bad[0]]:032x} expected {expected[bad[0]]:032x}")

        await RisingEdge(dut.done)
        cycles = int((cocotb.utils.get_sim_time("ns") - start_time) / 10)
        logits = [dut.logits[k].value.to_signed() for k in range(len(logits_ref))]
        pooled = [int(dut.pooled[i // 16][i % 16].value) for i in range(64)]
        assert pooled == pooled_ref, f"clip {idx}: pooled sums differ"
        assert logits == logits_ref, f"clip {idx}: logits {logits} != {logits_ref}"
        pred = int(dut.class_idx.value)
        assert pred == int(np.argmax(logits_ref)), f"clip {idx}: class {pred}"
        correct += pred == labels[idx]
        if n < N_LAYER_CHECK_CLIPS or n % 50 == 0:
            dut._log.info(f"clip {idx}: class {pred} (label {labels[idx]}), {cycles} cycles")

    dut._log.info(f"{len(clips)} clips bit-exact; accuracy {correct / len(clips):.4f}")


def test_engine():
    from cocotb_tools.runner import get_runner

    runner = get_runner("verilator")
    runner.build(
        sources=[ROOT / "rtl/gen/kws_pkg.sv", ROOT / "rtl/sdp_ram.sv", ROOT / "rtl/kws_engine.sv"],
        hdl_toplevel="kws_engine",
        build_dir=ROOT / "build" / "sim_engine",
        always=True,
        build_args=["--public-flat-rw", "-Wno-fatal", "-Wno-WIDTHEXPAND", "-Wno-UNUSEDSIGNAL",
                    f'-DKWS_MEM_DIR="{ROOT}/rtl/gen/"'],
    )
    runner.test(hdl_toplevel="kws_engine", test_module="test_engine",
                test_dir=Path(__file__).parent, build_dir=ROOT / "build" / "sim_engine")
