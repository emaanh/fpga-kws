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
  "$ROOT/rtl/pdm_mic.sv"
  "$ROOT/rtl/cic_decim.sv"
  "$ROOT/rtl/mic_fir.sv"
  "$ROOT/rtl/audio_frontend.sv"
  "$ROOT/rtl/kws_live.sv"
  "$ROOT/rtl/kws_top.sv"
)

step() { echo "==> $1"; }

step "sv2v"
sv2v -DSYNTHESIS -DKWS_MEM_DIR="\"$ROOT/rtl/gen/\"" "${SOURCES[@]}" > "$OUT/kws_top.v"

step "yosys (log: build/bit/yosys.log)"
yosys -q -l "$OUT/yosys.log" -p "
  read_verilog $OUT/kws_top.v
  synth_xilinx -flatten -abc9 -arch xc7 -top kws_top
  write_json $OUT/kws_top.json
"

# Placement quality varies with the seed; try several in parallel and keep the fastest.
SEEDS="${SEEDS:-1 2 3 4 5 6 7 8}"
step "nextpnr-xilinx, seeds: $SEEDS (logs: build/bit/seed*/nextpnr.log)"
for seed in $SEEDS; do
  mkdir -p "$OUT/seed$seed"
  "$OPENXC7/nextpnr-xilinx/build/nextpnr-xilinx" \
    --chipdb "$OPENXC7/chipdb/xc7a100t.bin" \
    --xdc "$ROOT/constraints/nexys_a7_100t.xdc" \
    --json "$OUT/kws_top.json" \
    --fasm "$OUT/seed$seed/kws_top.fasm" \
    --report "$OUT/seed$seed/report.json" \
    --freq 50 --seed "$seed" \
    --log "$OUT/seed$seed/nextpnr.log" -q > /dev/null 2>&1 &
done
wait
# Each seed's margin: the worst ratio of achieved to required frequency over all clocks.
best=""; best_margin=0
for seed in $SEEDS; do
  # nextpnr prints each clock before and after routing; keep the last line per clock.
  margin=$(grep "Max frequency for clock" "$OUT/seed$seed/nextpnr.log" | \
    sed -E "s/.*clock +'([^']*)': ([0-9.]+) MHz \\((PASS|FAIL) at ([0-9.]+) MHz.*/\\1 \\2 \\4/" | \
    awk 'NF == 3 {f[$1] = $2; t[$1] = $3} END {m = 1e9; for (c in f) if (f[c] / t[c] < m) {m = f[c] / t[c]; w = c}
         if (m < 1e9) printf "%.3f %s", m, w}')
  echo "    seed $seed: ${margin:-failed} (achieved/required, worst clock)"
  m=${margin%% *}
  if [[ -n "$m" ]] && awk "BEGIN{exit !($m > $best_margin)}"; then best=$seed; best_margin=$m; fi
done
[[ -n "$best" ]] || { echo "nextpnr failed for every seed"; exit 1; }
cp "$OUT/seed$best/kws_top.fasm" "$OUT/kws_top.fasm"
cp "$OUT/seed$best/nextpnr.log" "$OUT/nextpnr.log"
cp "$OUT/seed$best/report.json" "$OUT/report.json"
grep "Max frequency for clock" "$OUT/nextpnr.log" | tail -2 | sed 's/^Info: */      /'
echo "    using seed $best: margin $best_margin $(awk "BEGIN{print ($best_margin >= 1 ? \"(meets timing)\" : \"(DOES NOT MEET TIMING)\")}")"

step "fasm2frames"
"$OPENXC7/venv/bin/python" "$OPENXC7/prjxray/utils/fasm2frames.py" \
  --part "$PART" --db-root "$DB" "$OUT/kws_top.fasm" > "$OUT/kws_top.frames" 2> "$OUT/fasm2frames.log"

step "xc7frames2bit"
"$OPENXC7/prjxray/build/tools/xc7frames2bit" \
  --part_file "$DB/$PART/part.yaml" --part_name "$PART" \
  --frm_file "$OUT/kws_top.frames" --output_file "$OUT/kws_top.bit"

echo "Bitstream: build/bit/kws_top.bit (seed $best, timing margin $best_margin)"

if [[ "${1:-}" == "program" ]]; then
  step "openFPGALoader"
  openFPGALoader -b nexys_a7_100 "$OUT/kws_top.bit"
fi
