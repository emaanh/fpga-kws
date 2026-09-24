"""cocotb test: rtl/audio_frontend.sv vs python/kws/fixed_frontend.py, bit-exact.

    uv run pytest tests/test_frontend.py
"""

import os
from pathlib import Path

import numpy as np

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, RisingEdge

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PERIOD = 64  # clocks per PCM sample here (6250 on the board); a frame needs ~18k
N_CLIPS = int(os.environ.get("KWS_FE_CLIPS", "3"))


@cocotb.test()
async def test_features(dut):
    import torch

    from kws.config import CKPT_DIR
    from kws.data import CACHE_DIR
    from kws.fixed_frontend import features_fixed

    params = torch.load(CKPT_DIR / "dscnn_int8.pt")
    d = np.load(CACHE_DIR / "test.npz")
    clips = d["audio"][np.linspace(0, len(d["audio"]) - 1, N_CLIPS).astype(int)]
    # One continuous stream: frames that straddle two clips are checked too.
    stream = np.concatenate(clips)
    expected = features_fixed(stream, params["mean"], params["f_in"])

    Clock(dut.clk, 10, unit="ns").start()
    dut.rst.value = 1
    dut.pcm_valid.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst.value = 0
    await ClockCycles(dut.clk, 5)

    rows, row = [], {}

    async def collect():
        while True:
            await RisingEdge(dut.clk)
            if dut.feat_valid.value:
                row[int(dut.feat_band.value)] = dut.feat_q.value.to_signed()
            if dut.frame_done.value:
                rows.append([row.get(m) for m in range(expected.shape[1])])
                row.clear()

    cocotb.start_soon(collect())
    for x in stream:
        dut.pcm_valid.value = 1
        dut.pcm.value = int(x)
        await RisingEdge(dut.clk)
        dut.pcm_valid.value = 0
        await ClockCycles(dut.clk, SAMPLE_PERIOD - 1)
    await ClockCycles(dut.clk, 30_000)

    got = np.array(rows, dtype=object)
    assert got.shape == expected.shape, f"{got.shape} frames, expected {expected.shape}"
    bad = np.argwhere(got != expected)
    assert len(bad) == 0, (f"{len(bad)} features differ, first frame {bad[0][0]} band {bad[0][1]}: "
                           f"{got[tuple(bad[0])]} vs {expected[tuple(bad[0])]}")
    dut._log.info(f"{len(rows)} frames x {expected.shape[1]} bands bit-exact "
                  f"(range {expected.min()}..{expected.max()})")


def test_frontend():
    from cocotb_tools.runner import get_runner

    runner = get_runner("verilator")
    build_dir = ROOT / "build" / "sim_frontend"
    runner.build(
        sources=[ROOT / f for f in ["rtl/gen/kws_pkg.sv", "rtl/sdp_ram.sv", "rtl/audio_frontend.sv"]],
        hdl_toplevel="audio_frontend",
        build_dir=build_dir,
        always=True,
        build_args=["-Wno-fatal", "-Wno-WIDTHEXPAND", "-Wno-UNUSEDSIGNAL",
                    f'-DKWS_MEM_DIR="{ROOT}/rtl/gen/"'],
    )
    runner.test(hdl_toplevel="audio_frontend", test_module="test_frontend",
                test_dir=Path(__file__).parent, build_dir=build_dir)


def test_frontend_gates():
    """The same test on the yosys gate-level netlist (build/postsynth/fe_netlist.v), with
    yosys' Xilinx cell models."""
    import subprocess

    from cocotb_tools.runner import get_runner

    datdir = subprocess.check_output(["yosys-config", "--datdir"], text=True).strip()
    # Verilator, like the hardware, starts every register and memory bit at 0.
    runner = get_runner("verilator")
    build_dir = ROOT / "build" / "sim_frontend_gates"
    runner.build(
        sources=[f"{datdir}/xilinx/cells_sim.v", ROOT / "build/postsynth/fe_netlist.v"],
        hdl_toplevel="audio_frontend",
        build_dir=build_dir,
        always=True,
        build_args=["-Wno-fatal", "-Wno-lint", "-Wno-style", "--x-assign", "0", "--x-initial", "0",
                    "-Wno-MULTIDRIVEN", "-Wno-COMBDLY", "--timing"],
        timescale=("1ns", "1ps"),
    )
    runner.test(hdl_toplevel="audio_frontend", test_module="test_frontend",
                test_dir=Path(__file__).parent, build_dir=build_dir,
                extra_env={"KWS_FE_CLIPS": os.environ.get("KWS_FE_CLIPS", "1")})
