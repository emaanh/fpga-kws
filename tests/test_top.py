"""End-to-end cocotb test of rtl/kws_top.sv over its UART, with a fast baud rate for simulation.

    uv run pytest tests/test_top.py
"""

import struct
from pathlib import Path

import numpy as np

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, FallingEdge, Timer, with_timeout

from test_engine import ROOT, load_reference

CLK_NS = 10
CLKS_PER_BIT = 8
BIT_NS = CLK_NS * CLKS_PER_BIT
N_CLIPS = 2
RX_TIMEOUT = 5 * 10 * CLKS_PER_BIT  # 5 byte times


async def uart_send(line, data: bytes):
    for byte in data:
        for bit in [0, *((byte >> i) & 1 for i in range(8)), 1]:
            line.value = bit
            await Timer(BIT_NS, unit="ns")


async def uart_recv(line, n: int) -> bytes:
    out = bytearray()
    for _ in range(n):
        await FallingEdge(line)
        await Timer(BIT_NS * 3 // 2, unit="ns")  # middle of bit 0
        byte = 0
        for i in range(8):
            byte |= int(line.value) << i
            await Timer(BIT_NS, unit="ns")
        assert int(line.value) == 1, "missing stop bit"
        out.append(byte)
    return bytes(out)


@cocotb.test()
async def test_uart_inference(dut):
    feats, labels, reference = load_reference()
    Clock(dut.clk, CLK_NS, unit="ns").start()
    dut.uart_rx.value = 1
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 5)

    # Junk and a truncated frame first: the board must ignore/drop them and resync.
    await uart_send(dut.uart_rx, b"xyz" + b"I" + bytes(100))
    await ClockCycles(dut.clk, 2 * RX_TIMEOUT)

    for idx in np.linspace(0, len(feats) - 1, N_CLIPS).astype(int):
        logits_ref, _, _ = reference(feats[idx])
        recv = cocotb.start_soon(uart_recv(dut.uart_tx, 2 + 4 * len(logits_ref)))
        await uart_send(dut.uart_rx, b"I" + feats[idx].astype(np.int8).tobytes())
        try:
            reply = await with_timeout(recv, 10, "ms")  # inference alone is 6.65 ms
        except Exception:
            dut._log.error(f"state={dut.state.value} rx_count={int(dut.rx_count.value)} "
                           f"engine busy={dut.eng_busy.value}")
            raise
        assert reply[0:1] == b"R", reply[:2]
        logits = list(struct.unpack(f"<{len(logits_ref)}i", reply[2:]))
        assert logits == logits_ref, f"clip {idx}: {logits} != {logits_ref}"
        assert reply[1] == int(np.argmax(logits_ref))
        assert int(dut.led.value) & 0xFFF == 1 << reply[1]
        dut._log.info(f"clip {idx}: class {reply[1]} (label {labels[idx]}) over UART, bit-exact")


def test_top():
    from cocotb_tools.runner import get_runner

    runner = get_runner("verilator")
    build_dir = ROOT / "build" / "sim_top"
    runner.build(
        sources=[ROOT / f for f in ["rtl/gen/kws_pkg.sv", "rtl/sdp_ram.sv", "rtl/kws_engine.sv",
                                    "rtl/uart_rx.sv", "rtl/uart_tx.sv", "rtl/seg7_word.sv",
                                    "rtl/kws_top.sv"]],
        hdl_toplevel="kws_top",
        build_dir=build_dir,
        always=True,
        parameters={"CLKS_PER_BIT": CLKS_PER_BIT, "RX_TIMEOUT": RX_TIMEOUT},
        build_args=["--public-flat-rw", "-Wno-fatal", "-Wno-WIDTHEXPAND", "-Wno-UNUSEDSIGNAL",
                    f'-DKWS_MEM_DIR="{ROOT}/rtl/gen/"'],
    )
    runner.test(hdl_toplevel="kws_top", test_module="test_top",
                test_dir=Path(__file__).parent, build_dir=build_dir)
