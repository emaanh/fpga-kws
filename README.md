# Keyword Spotting on FPGA

## motivation
A year ago I implemented inference of a 4-layer MLP on an FPGA for doing MNIST digit classification. I wanted to see what other ML models could be run this hardware. In this project, I'm doing keyword spotting. 

## background
Keyword spotting is a pretty well suited task for hardware since it's typically "always-on" and thus needs to be low-power, esp on edge, battery-powered devices. The most obvious example is Apple's low-power always-on processors that support features like  "Hey Siri." 

## overview
I did keyword spotting on a Digilent Nexys A7-100T, designing a pipeline to 
### SW (Python)
- Generated synthetic training data using open weight text-to-speech models + data augmentation
- Trained model and quantized to int8 using PyTorch
### HW (SystemVerilog):
- Ingest PDM micrphone with audio filters & FFT & sliding-window
- Feed to CNN inference engine 
- Display detected word/phrase on 7-segment display

Categories/Words: Emaan, Heidari, yes, no, up, down, left, right, on, off, stop, go, *silence, *unknown
\* = not displayed

## Demo
coming soon I need to borrow Josh's FPGA again lmao

## Results

| Pipeline | Test accuracy |
|---|---|
| Float frontend, float model | 95.81% |
| Float frontend, int8 model  | 95.83% |

Live detection on a 20 minute stream of held-out keywords, other words and silence
(`kws.eval_stream`): about 91% of keywords detected, 5% reported as the wrong word, and
about 100 false alarms per hour from non-keyword speech.

Resources used on FPGA: about 10k LUTs (8%), 32 DSP48 (13%) and 20% of block RAM. One
inference takes 665k cycles (6.65 ms at 100 MHz) and runs every 100 ms.

## How it works

```
PDM mic (2 MHz, 1 bit)
  -> CIC decimator /25 -> FIR /5 with CIC compensation -> DC blocker -> gain   16 kHz PCM
  -> every 20 ms: Hann window, 512-point FFT, power, 40 mel bands, log2         40 int8 features
  -> 64-frame ring buffer
  -> every 100 ms: the last 49 frames (~1 s) -> DS-CNN engine                   12 logits
  -> decision: a keyword that wins 3 inferences in a row by a margin             display
```

- **Model:** DS-CNN-S, 23k parameters: a 10x4 conv,
  four depthwise-separable blocks with 64 channels, global average pool and a 12-way FC.
  Trained with time shift, background noise and +-10 dB level
  augmentation, then quantization-aware trained to int8.
- **Engine** (`rtl/kws_engine.sv`): 16 parallel MACs, one per output channel, running the
  layers one after another with ping-pong activation buffers in block RAM. BatchNorm,
  input normalization and rounding are folded into the weights and biases ahead of time.
- **Frontend** (`rtl/audio_frontend.sv`): a sequential radix-2 FFT and a sparse mel filter
  bank, about 18k cycles per 20 ms frame.

## Getting started

Requirements: [uv](https://docs.astral.sh/uv/), and from Homebrew `verilator`, `yosys`,
`sv2v` and `openfpgaloader`.

```sh
uv sync
uv run python -m kws.data      # download Speech Commands v2 (~2.3 GB) and build the cache
uv run python -m kws.train     # float training (~8 min on an M-series Mac)
uv run python -m kws.qat       # int8 QAT and integer-only evaluation
uv run python -m kws.export    # ROMs, SV package and test vectors for the RTL
uv run pytest tests/           # all RTL tests (~3 min)
```

`rtl/gen/` is committed, so building a bitstream does not need the dataset or training.

Useful extras:

```sh
uv run python -m kws.eval_pipeline                  # accuracy through the mic and frontend models
uv run python -m kws.eval_stream                    # tune the live decision rule
KWS_N_CLIPS=all uv run pytest tests/test_engine.py  # engine on the whole test set (~1 h)
```

## Building the bitstream

```sh
scripts/build_bitstream.sh           # build/bit/kws_top.bit
scripts/build_bitstream.sh program   # build and load it over JTAG
```

The flow is sv2v, yosys, nextpnr-xilinx and prjxray. The script tries 8 placement seeds in
parallel and keeps the fastest. The best seeds reach about 104 MHz (nextpnr's estimate)
against the 100 MHz target, so the margin is thin; `scripts/vivado_synth.tcl` builds the
same design in Vivado for a proper timing sign-off.

nextpnr-xilinx and prjxray are not in Homebrew. The script expects them under `$OPENXC7`
(default `~/tools/openxc7`):

```sh
brew install boost eigen pkgconf
mkdir -p ~/tools/openxc7 && cd ~/tools/openxc7
git clone --recursive https://github.com/openXC7/nextpnr-xilinx.git
git clone --recursive https://github.com/f4pga/prjxray.git

# nextpnr-xilinx (Apple clang has no OpenMP)
cmake -S nextpnr-xilinx -B nextpnr-xilinx/build -G Ninja -DARCH=xilinx -DBUILD_GUI=OFF \
  -DBUILD_PYTHON=OFF -DUSE_OPENMP=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build nextpnr-xilinx/build

# Python tools (fasm2frames) and the xc7a100t chip database
uv venv --python 3.12 venv
(cd prjxray && VIRTUAL_ENV=../venv uv pip install -r requirements.txt)
mkdir chipdb
venv/bin/python nextpnr-xilinx/xilinx/python/bbaexport.py --device xc7a100tcsg324-1 --bba chipdb/xc7a100t.bba
nextpnr-xilinx/build/bbasm -l chipdb/xc7a100t.bba chipdb/xc7a100t.bin && rm chipdb/xc7a100t.bba

# xc7frames2bit
cmake -S prjxray -B prjxray/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build prjxray/build --target xc7frames2bit
```

## Board controls

| Switch | Function |
|---|---|
| SW15 | 0: UART feature mode, 1: live microphone mode |
| SW14 | live mode: stream the mic's PCM over UART for `kws.record` |
| SW13 | sample the mic on the falling clock edge instead of the rising edge |
| SW5:4 | detection margin: 00 = 2^18 (default), 01 = 2^17 (more sensitive), 10 = 2^19, 11 = off |
| SW3:0 | mic gain in 6 dB steps (0 to 12); 6 (+36 dB) is the starting point |

| LED | Meaning |
|---|---|
| 11:0 | detected class (UART mode) or mic level meter (live mode) |
| 12 | a word is on the display (live mode) |
| 13 | live mode |
| 14 | engine busy |
| 15 | heartbeat |
