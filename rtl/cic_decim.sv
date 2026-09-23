// 4-stage CIC decimator by CIC_R for +-1 PDM bits (python/kws/mic_model.py: cic_decimate).
//
// Integrators and combs are updated one stage per cycle, so each stage sees the previous
// stage's new value, exactly like the cumulative sums in the model. Registers wrap mod 2^W,
// which the combs undo; the output magnitude is at most CIC_R^4 < 2^(W-1).
// Needs at least CIC_N + 1 clocks between input bits.
module cic_decim
  import kws_pkg::*;
#(
  parameter int W = 24
) (
  input  logic                clk,
  input  logic                rst,
  input  logic                in_valid,
  input  logic                in_bit,
  output logic                out_valid,
  output logic signed [W-1:0] out_data
);
  logic signed [W-1:0] integ [CIC_N];
  logic signed [W-1:0] comb_prev [CIC_N];
  logic signed [W-1:0] comb;
  logic [$clog2(CIC_R)-1:0] phase;
  logic [CIC_N:0]      int_step;   // one-hot: which integrator updates this cycle
  logic [CIC_N:0]      comb_step;  // one-hot: which comb updates this cycle
  logic                in_value;

  always_ff @(posedge clk) begin
    out_valid <= 1'b0;
    int_step  <= int_step << 1;
    comb_step <= comb_step << 1;

    if (in_valid) begin
      int_step <= 1;
      in_value <= in_bit;
    end

    if (int_step[0]) integ[0] <= integ[0] + (in_value ? W'(1) : -W'(1));
    for (int s = 1; s < CIC_N; s++) begin
      if (int_step[s]) integ[s] <= integ[s] + integ[s - 1];
    end

    // After the last integrator has updated, every CIC_R-th input starts the combs.
    if (int_step[CIC_N]) begin
      if (phase == ($clog2(CIC_R))'(CIC_R - 1)) begin
        phase     <= '0;
        comb      <= integ[CIC_N - 1];
        comb_step <= 1;
      end else begin
        phase <= phase + 1'b1;
      end
    end

    for (int s = 0; s < CIC_N; s++) begin
      if (comb_step[s]) begin
        comb         <= comb - comb_prev[s];
        comb_prev[s] <= comb;
      end
    end

    if (comb_step[CIC_N]) begin
      out_valid <= 1'b1;
      out_data  <= comb;
    end

    if (rst) begin
      for (int s = 0; s < CIC_N; s++) begin
        integ[s]     <= '0;
        comb_prev[s] <= '0;
      end
      phase     <= '0;
      int_step  <= '0;
      comb_step <= '0;
    end
  end
endmodule
