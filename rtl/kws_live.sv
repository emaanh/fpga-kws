// Live keyword spotting: PDM mic -> PCM -> features -> engine, with a simple decision rule.
//
//   mic      pdm_mic -> cic_decim -> mic_fir (DC block, gain)       16 kHz int16 PCM
//   features audio_frontend, one row of N_MELS every HOP samples     into a 64-frame ring
//   infer    every INFER_EVERY frames (once IN_H frames exist), copy the newest IN_H rows
//            into the engine's feature RAM and start it
//   decide   an inference is a "win" for keyword c when c is the argmax and its logit
//            beats the runner-up by at least min_margin; N_CONSEC wins in a row for the
//            same keyword is a detection. It is shown for HOLD_CYCLES and not re-reported
//            while shown. (Tuned with python -m kws.eval_stream; retune after retraining.)
//
// The engine is shared with the UART feature mode, so this module drives its feature write
// port and start through the top level's mux, and only while `enable` is set.

`ifndef KWS_MEM_DIR
`define KWS_MEM_DIR ""
`endif

module kws_live
  import kws_pkg::*;
#(
  parameter string MEM_DIR     = `KWS_MEM_DIR,
  parameter int    CLK_DIV     = 25,           // PDM clock = clk / (2 * CLK_DIV)
  parameter int    INFER_EVERY = 5,            // frames between inferences (5 = 100 ms)
  parameter int    HOLD_CYCLES = 100_000_000,  // how long a detection stays shown
  parameter int    N_CONSEC    = 3
) (
  input  logic               clk,
  input  logic               rst,
  input  logic               enable,
  input  logic [3:0]         gain,
  input  logic               sample_fall,
  input  logic signed [31:0] min_margin,    // 2^18 by default (see kws_top)
  // mic pins
  output logic               m_clk,
  output logic               m_lrsel,
  input  logic               m_data,
  // PCM out (for recording) and a peak level for a meter
  output logic               pcm_valid,
  output logic signed [15:0] pcm,
  output logic [3:0]         level,         // position of the leading one of the recent peak
  // engine
  output logic               feat_we,
  output logic [10:0]        feat_waddr,
  output logic [7:0]         feat_wdata,
  output logic               eng_start,
  input  logic               eng_busy,
  input  logic               eng_done,
  input  logic [3:0]         eng_class,
  input  logic signed [31:0] eng_margin,
  // result
  output logic               detect,        // one-cycle pulse per detection
  output logic               showing,
  output logic [3:0]         show_class
);
  // ---------------------------------------------------------------------------------------
  // Mic path and frontend
  // ---------------------------------------------------------------------------------------
  logic               bit_valid, bit_data, c_valid;
  logic signed [23:0] c_data;
  logic               f_valid, frame_done;
  logic [5:0]         f_band;
  logic signed [7:0]  f_q;

  pdm_mic #(.CLK_DIV(CLK_DIV)) u_pdm (
    .clk, .rst, .sample_fall, .m_clk, .m_lrsel, .m_data, .bit_valid, .bit_data);
  cic_decim u_cic (
    .clk, .rst, .in_valid(bit_valid), .in_bit(bit_data), .out_valid(c_valid), .out_data(c_data));
  mic_fir #(.MEM_DIR(MEM_DIR)) u_fir (
    .clk, .rst, .in_valid(c_valid), .in_data(c_data), .gain, .pcm_valid, .pcm);
  audio_frontend #(.MEM_DIR(MEM_DIR)) u_fe (
    .clk, .rst, .pcm_valid, .pcm,
    .feat_valid(f_valid), .feat_band(f_band), .feat_q(f_q), .frame_done);

  // Peak level over the last 1024 samples (64 ms), as the index of its leading one.
  logic [14:0] peak, peak_run;
  logic [9:0]  peak_cnt;
  always_ff @(posedge clk) begin
    if (pcm_valid) begin
      logic [14:0] mag;
      mag = pcm[15] ? 15'(-pcm) : 15'(pcm);
      peak_cnt <= peak_cnt + 1'b1;
      if (peak_cnt == '1) begin
        peak     <= peak_run > mag ? peak_run : mag;
        peak_run <= '0;
      end else if (mag > peak_run) begin
        peak_run <= mag;
      end
    end
    level <= '0;
    for (int i = 0; i < 15; i++) if (peak[i]) level <= 4'(i + 1);
    if (rst) begin
      peak     <= '0;
      peak_run <= '0;
      peak_cnt <= '0;
    end
  end

  // ---------------------------------------------------------------------------------------
  // Feature ring: 64 frames x N_MELS
  // ---------------------------------------------------------------------------------------
  localparam int RING = 64;

  logic [5:0]  wr_slot, newest;
  logic [11:0] ring_raddr;
  logic [7:0]  ring_rdata;
  logic [15:0] frames_total;
  logic [3:0]  since_infer;
  logic        want_infer;

  sdp_ram #(.WIDTH(8), .DEPTH(RING * N_MELS)) u_ring (
    .clk, .we(f_valid), .waddr(12'(wr_slot * N_MELS + f_band)), .wdata(f_q),
    .raddr(ring_raddr), .rdata(ring_rdata));

  always_ff @(posedge clk) begin
    want_infer <= 1'b0;
    if (frame_done) begin
      newest  <= wr_slot;
      wr_slot <= wr_slot + 1'b1;
      if (frames_total != '1) frames_total <= frames_total + 1'b1;
      if (since_infer == 4'(INFER_EVERY - 1)) begin
        since_infer <= '0;
        want_infer  <= frames_total + 1 >= IN_H;
      end else begin
        since_infer <= since_infer + 1'b1;
      end
    end
    if (rst) begin
      wr_slot      <= '0;
      frames_total <= '0;
      since_infer  <= '0;
    end
  end

  // ---------------------------------------------------------------------------------------
  // Copy the newest IN_H frames (oldest first) into the engine, then start it
  // ---------------------------------------------------------------------------------------
  logic        copying, c1_valid;
  logic [5:0]  ch;          // row 0..IN_H-1
  logic [5:0]  cw;          // band 0..N_MELS-1
  logic [10:0] dst, c1_dst;
  logic [5:0]  src_slot;

  assign src_slot   = newest - 6'(IN_H - 1) + ch;
  assign ring_raddr = 12'(src_slot * N_MELS + cw);

  always_ff @(posedge clk) begin
    eng_start <= 1'b0;
    c1_valid  <= copying;
    c1_dst    <= dst;

    if (want_infer && enable && !eng_busy && !copying) begin
      copying <= 1'b1;
      {ch, cw, dst} <= '0;
    end else if (copying) begin
      dst <= dst + 1'b1;
      if (cw == 6'(N_MELS - 1)) begin
        cw <= '0;
        if (ch == 6'(IN_H - 1)) copying <= 1'b0;
        else ch <= ch + 1'b1;
      end else cw <= cw + 1'b1;
    end

    if (c1_valid && !copying) eng_start <= 1'b1;  // the last word is written this cycle

    if (rst) begin
      copying  <= 1'b0;
      c1_valid <= 1'b0;
    end
  end

  assign feat_we    = c1_valid;
  assign feat_waddr = c1_dst;
  assign feat_wdata = ring_rdata;

  // ---------------------------------------------------------------------------------------
  // Decision
  // ---------------------------------------------------------------------------------------
  logic [3:0]  prev_class, win_class;
  logic [3:0]  hits, hits_next;
  logic        decide;
  logic [$clog2(HOLD_CYCLES + 1)-1:0] hold;
  logic signed [31:0] margin_q;

  assign showing   = hold != 0;
  assign hits_next = win_class < 2 ? 4'd0 : win_class == prev_class ? (hits == '1 ? hits : hits + 1'b1)
                                                                    : 4'd1;

  always_ff @(posedge clk) begin
    margin_q <= min_margin;  // a switch setting: keep it off the paths
    // Step 1, the cycle after an inference: is it a confident keyword?
    decide    <= eng_done && enable;
    win_class <= eng_margin >= margin_q ? eng_class : 4'd0;  // not confident: silence

    // Step 2: count wins in a row, detect
    detect <= 1'b0;
    if (hold != 0) hold <= hold - 1'b1;

    if (decide) begin
      prev_class <= win_class;
      hits       <= hits_next;
      if (hits_next == 4'(N_CONSEC) && !(showing && show_class == win_class)) begin
        detect     <= 1'b1;
        show_class <= win_class;
        hold       <= ($bits(hold))'(HOLD_CYCLES);
      end
    end

    if (rst) begin
      hold       <= '0;
      hits       <= '0;
      prev_class <= '0;
      show_class <= '0;
    end
  end
endmodule
