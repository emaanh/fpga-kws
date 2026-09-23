// FIR decimator by FIR_D, DC blocker and gain: CIC output (80 kHz) -> int16 PCM (16 kHz).
// Bit-exact to python/kws/mic_model.py (fir_decimate, _dc_block_and_gain):
//
//   acc = sum_t h[t] * c[newest - t]          after every FIR_D-th input, one MAC per cycle
//   d   = (acc + 2^7) >>> 8
//   y   = d - d_prev + y - (y >>> 8)
//   pcm = sat16((y + 2^(s-1)) >>> s),  s = 13 - min(gain, MAX_GAIN)

`ifndef KWS_MEM_DIR
`define KWS_MEM_DIR ""
`endif

module mic_fir
  import kws_pkg::*;
#(
  parameter string MEM_DIR = `KWS_MEM_DIR
) (
  input  logic               clk,
  input  logic               rst,
  input  logic               in_valid,
  input  logic signed [23:0] in_data,
  input  logic [3:0]         gain,
  output logic               pcm_valid,
  output logic signed [15:0] pcm
);
  // Sample history (the newest FIR_TAPS CIC outputs) and taps
  logic [7:0]         wptr, newest, rd_idx;
  logic [23:0]        sample;
  logic [17:0]        coef;
  logic [$clog2(FIR_D)-1:0] phase;

  sdp_ram #(.WIDTH(24), .DEPTH(256)) u_hist (
    .clk, .we(in_valid), .waddr(wptr), .wdata(in_data), .raddr(rd_idx), .rdata(sample));

  // MAC sequencer
  logic [7:0]  tap;
  logic        busy;
  logic        m1_valid, m1_first, m1_last;
  logic        m2_valid, m2_first, m2_last;
  logic signed [41:0] m2_prod;
  logic signed [39:0] acc;

  sdp_ram #(.WIDTH(18), .DEPTH(FIR_TAPS), .INIT({MEM_DIR, "fe_fir.hex"}), .STYLE("block")) u_taps (
    .clk, .we(1'b0), .waddr('0), .wdata('0), .raddr(tap), .rdata(coef));

  assign rd_idx = newest - tap;

  always_ff @(posedge clk) begin
    if (in_valid) begin
      wptr <= wptr + 1'b1;
      if (phase == ($clog2(FIR_D))'(FIR_D - 1)) begin
        phase  <= '0;
        newest <= wptr;
        tap    <= '0;
        busy   <= 1'b1;
      end else begin
        phase <= phase + 1'b1;
      end
    end

    // M0: addresses (tap, newest - tap) go to the memories
    m1_valid <= busy;
    m1_first <= tap == 0;
    m1_last  <= tap == 8'(FIR_TAPS - 1);
    if (busy) begin
      if (tap == 8'(FIR_TAPS - 1)) busy <= 1'b0;
      else tap <= tap + 1'b1;
    end

    // M1: multiply
    {m2_valid, m2_first, m2_last} <= {m1_valid, m1_first, m1_last};
    m2_prod <= $signed(coef) * $signed(sample);

    // M2: accumulate
    if (m2_valid) acc <= (m2_first ? 40'sd0 : acc) + 40'(m2_prod);

    if (rst) begin
      wptr  <= '0;
      phase <= '0;
      busy  <= 1'b0;
      {m1_valid, m2_valid} <= '0;
    end
  end

  // Post-processing: scale, DC block, round, shift, saturate. One step per cycle.
  logic               p0_valid, p1_valid, p2_valid, p3_valid, p4_valid;
  logic signed [31:0] d, d_prev;
  logic signed [35:0] y, t, v;
  logic [3:0]         shift;
  logic signed [35:0] rnd;

  always_ff @(posedge clk) begin
    p0_valid  <= m2_valid && m2_last;
    p1_valid  <= p0_valid;
    p2_valid  <= p1_valid;
    p3_valid  <= p2_valid;
    p4_valid  <= p3_valid;
    pcm_valid <= p4_valid;
    // The gain only changes when a switch moves, so these are effectively constants.
    shift <= 4'd13 - (gain > 4'(MAX_GAIN) ? 4'(MAX_GAIN) : gain);
    rnd   <= 36'sd1 <<< (shift - 4'd1);

    // The last product is accumulated in M2, so acc is complete one cycle later.
    if (p0_valid) d <= 32'((acc + 40'sd128) >>> 8);
    if (p1_valid) begin
      y      <= 36'(d) - 36'(d_prev) + y - (y >>> 8);
      d_prev <= d;
    end
    if (p2_valid) t <= y + rnd;
    if (p3_valid) v <= t >>> shift;
    if (p4_valid) pcm <= v > 32767 ? 16'sd32767 : v < -32768 ? -16'sd32768 : 16'(v);

    if (rst) begin
      {p0_valid, p1_valid, p2_valid, p3_valid, p4_valid, pcm_valid} <= '0;
      y      <= '0;
      d_prev <= '0;
    end
  end
endmodule
