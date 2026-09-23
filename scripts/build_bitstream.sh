#!/usr/bin/env bash
# Build build/bit/kws_top.bit for the Nexys A7-100T with the open-source openXC7 flow:
#   sv2v -> yosys (synth_xilinx) -> nextpnr-xilinx -> fasm2frames -> xc7frames2bit
#
#   scripts/build_bitstream.sh            # build
#   scripts/build_bitstream.sh program    # build, then load onto the board over JTAG
#
# OPENXC7 points at the toolchain directory (nextpnr-xilinx, prjxray, chipdb, venv).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OPENXC7="${OPENXC7:-$HOME/tools/openxc7}"
PART=xc7a100tcsg324-1
DB="$OPENXC7/nextpnr-xilinx/xilinx/external/prjxray-db/artix7"
OUT="$ROOT/build/bit"
mkdir -p "$OUT"

SOURCES=(
  "$ROOT/rtl/gen/kws_pkg.sv"
  "$ROOT/rtl/sdp_ram.sv"
  "$ROOT/rtl/kws_engine.sv"
  "$ROOT/rtl/uart_rx.sv"
  "$ROOT/rtl/uart_tx.sv"
  "$ROOT/rtl/seg7_word.sv"
  "$ROOT/rtl/kws_top.sv"
)

step() { echo "==> $1"; }

step "sv2v"
sv2v -DKWS_MEM_DIR="\"$ROOT/rtl/gen/\"" "${SOURCES[@]}" > "$OUT/kws_top.v"

step "yosys (log: build/bit/yosys.log)"
yosys -q -l "$OUT/yosys.log" -p "
  read_verilog $OUT/kws_top.v
  synth_xilinx -flatten -abc9 -arch xc7 -top kws_top
  write_json $OUT/kws_top.json
"

step "nextpnr-xilinx (log: build/bit/nextpnr.log)"
"$OPENXC7/nextpnr-xilinx/build/nextpnr-xilinx" \
  --chipdb "$OPENXC7/chipdb/xc7a100t.bin" \
  --xdc "$ROOT/constraints/nexys_a7_100t.xdc" \
  --json "$OUT/kws_top.json" \
  --fasm "$OUT/kws_top.fasm" \
  --report "$OUT/report.json" \
  --freq 100 \
  --log "$OUT/nextpnr.log" -q

step "fasm2frames"
"$OPENXC7/venv/bin/python" "$OPENXC7/prjxray/utils/fasm2frames.py" \
  --part "$PART" --db-root "$DB" "$OUT/kws_top.fasm" > "$OUT/kws_top.frames" 2> "$OUT/fasm2frames.log"

step "xc7frames2bit"
"$OPENXC7/prjxray/build/tools/xc7frames2bit" \
  --part_file "$DB/$PART/part.yaml" --part_name "$PART" \
  --frm_file "$OUT/kws_top.frames" --output_file "$OUT/kws_top.bit"

grep -E "Max frequency for clock" "$OUT/nextpnr.log" | tail -1 || true
echo "Bitstream: build/bit/kws_top.bit"

if [[ "${1:-}" == "program" ]]; then
  step "openFPGALoader"
  openFPGALoader -b nexys_a7_100 "$OUT/kws_top.bit"
fi
