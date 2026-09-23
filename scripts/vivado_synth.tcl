# Vivado build of the full design for the Nexys A7-100T: synthesis, place & route, timing
# sign-off at 100 MHz and a bitstream. (The day-to-day flow is scripts/build_bitstream.sh.)
#
#   vivado -mode batch -source scripts/vivado_synth.tcl
#
# Reports and build/vivado/kws_top.bit go to build/vivado/. Needs rtl/gen/ (committed).

set root    [file normalize [file join [file dirname [info script]] ..]]
set gen_dir [file join $root rtl gen]
set out_dir [file join $root build vivado]
file mkdir $out_dir

set_part xc7a100tcsg324-1

set sources {
  gen/kws_pkg.sv sdp_ram.sv kws_engine.sv uart_rx.sv uart_tx.sv seg7_word.sv
  pdm_mic.sv cic_decim.sv mic_fir.sv audio_frontend.sv kws_live.sv kws_top.sv
}
foreach f $sources { read_verilog -sv [file join $root rtl $f] }
read_xdc [file join $root constraints nexys_a7_100t.xdc]

synth_design -top kws_top -verilog_define KWS_MEM_DIR=\"$gen_dir/\"
opt_design
place_design
route_design

report_utilization    -file [file join $out_dir utilization.rpt]
report_timing_summary -file [file join $out_dir timing_summary.rpt]
report_timing -max_paths 10 -file [file join $out_dir timing_paths.rpt]
write_bitstream -force [file join $out_dir kws_top.bit]

set wns [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -setup]]
puts "\n=== kws_top @ 100 MHz: worst setup slack $wns ns ([expr {$wns >= 0 ? "MET" : "FAILED"}]) ==="
puts "Reports and bitstream in $out_dir"
