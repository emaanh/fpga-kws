// Test wrapper: PDM bits -> CIC -> FIR/DC block/gain -> PCM.
module mic_chain (
  input  logic               clk,
  input  logic               rst,
  input  logic               bit_valid,
  input  logic               bit_data,
  input  logic [3:0]         gain,
  output logic               pcm_valid,
  output logic signed [15:0] pcm
);
  logic               c_valid;
  logic signed [23:0] c_data;

  cic_decim u_cic (.clk, .rst, .in_valid(bit_valid), .in_bit(bit_data),
                   .out_valid(c_valid), .out_data(c_data));
  mic_fir u_fir (.clk, .rst, .in_valid(c_valid), .in_data(c_data), .gain,
                 .pcm_valid, .pcm);
endmodule
