# Out-of-context synthesis + place & route of kws_engine for the Nexys A7-100T,
# to check resource use and 100 MHz timing before the board wrapper exists.
#
#   vivado -mode batch -source scripts/vivado_synth.tcl
#
# Reports go to build/vivado/. Needs the generated files in rtl/gen/ (committed).

set root    [file normalize [file join [file dirname [info script]] ..]]
set gen_dir [file join $root rtl gen]
set out_dir [file join $root build vivado]
file mkdir $out_dir

set_part xc7a100tcsg324-1

read_verilog -sv [list \
  [file join $gen_dir kws_pkg.sv] \
  [file join $root rtl sdp_ram.sv] \
  [file join $root rtl kws_engine.sv] \
]

# Out-of-context: the engine's ports are not pins, so no I/O placement is needed.
synth_design -top kws_engine -mode out_of_context \
  -verilog_define KWS_MEM_DIR=\"$gen_dir/\"

create_clock -name clk -period 10.000 [get_ports clk]

opt_design
place_design
route_design

report_utilization    -file [file join $out_dir utilization.rpt]
report_timing_summary -file [file join $out_dir timing_summary.rpt]
report_timing -max_paths 10 -file [file join $out_dir timing_paths.rpt]

set wns [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -setup]]
puts "\n=== kws_engine @ 100 MHz: worst setup slack $wns ns ([expr {$wns >= 0 ? "MET" : "FAILED"}]) ==="
puts "Reports in $out_dir"
