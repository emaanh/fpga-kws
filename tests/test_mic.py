"""cocotb test: RTL mic path (cic_decim + mic_fir) vs python/kws/mic_model.py, bit-exact.

    uv run pytest tests/test_mic.py
"""

from pathlib import Path

import numpy as np

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, RisingEdge

ROOT = Path(__file__).resolve().parents[1]
BIT_PERIOD = 8   # clocks per PDM bit in this test (50 on the board)
GAIN = 6
N_PCM = 3000     # PCM samples to compare (~0.19 s of audio)


@cocotb.test()
async def test_mic_chain(dut):
    from kws.data import CACHE_DIR
    from kws.mic_model import CIC_R, FIR_D, mic_pcm, to_pdm

    audio = np.load(CACHE_DIR / "test.npz")["audio"][0].astype(np.float64) / 32768
    loudest = int(np.argmax(np.convolve(audio**2, np.ones(1600), "same")))
    audio = audio[max(0, loudest - N_PCM // 2):]  # cover the word, not the leading silence
    bits = to_pdm(audio, 2**-6)[: N_PCM * CIC_R * FIR_D]
    expected = mic_pcm(bits, GAIN)

    Clock(dut.clk, 10, unit="ns").start()
    dut.rst.value = 1
    dut.bit_valid.value = 0
    dut.gain.value = GAIN
    await ClockCycles(dut.clk, 5)
    dut.rst.value = 0
    await ClockCycles(dut.clk, 5)

    got = []

    async def collect():
        while True:
            await RisingEdge(dut.clk)
            if dut.pcm_valid.value:
                got.append(dut.pcm.value.to_signed())

    cocotb.start_soon(collect())
    for b in bits:
        dut.bit_valid.value = 1
        dut.bit_data.value = int(b)
        await RisingEdge(dut.clk)
        dut.bit_valid.value = 0
        await ClockCycles(dut.clk, BIT_PERIOD - 1)
    await ClockCycles(dut.clk, 400)

    assert len(got) == len(expected), f"{len(got)} PCM samples, expected {len(expected)}"
    bad = np.nonzero(np.array(got) != expected)[0]
    assert len(bad) == 0, f"{len(bad)} samples differ, first at {bad[0]}: {got[bad[0]]} vs {expected[bad[0]]}"
    dut._log.info(f"{len(got)} PCM samples bit-exact (peak {np.abs(expected).max()})")


def test_mic():
    from cocotb_tools.runner import get_runner

    runner = get_runner("verilator")
    build_dir = ROOT / "build" / "sim_mic"
    runner.build(
        sources=[ROOT / f for f in ["rtl/gen/kws_pkg.sv", "rtl/sdp_ram.sv", "rtl/cic_decim.sv",
                                    "rtl/mic_fir.sv", "tests/hdl/mic_chain.sv"]],
        hdl_toplevel="mic_chain",
        build_dir=build_dir,
        always=True,
        build_args=["-Wno-fatal", "-Wno-WIDTHEXPAND", "-Wno-UNUSEDSIGNAL",
                    f'-DKWS_MEM_DIR="{ROOT}/rtl/gen/"'],
    )
    runner.test(hdl_toplevel="mic_chain", test_module="test_mic",
                test_dir=Path(__file__).parent, build_dir=build_dir)
