// Live keyword spotting: PDM mic -> PCM -> features -> streaming engine -> decision.
//
//   mic      pdm_mic -> cic_decim -> mic_fir (DC block, gain)       16 kHz int16 PCM
//   features audio_frontend, one row of N_MELS every HOP samples, straight to the engine,
//            which computes a new result every second frame (40 ms)
//   decide   a result is a "win" for keyword c when c is the argmax and its logit beats the
//            runner-up by at least min_margin; N_CONSEC wins in a row for the same keyword
//            is a detection. It is shown for HOLD_CYCLES and not re-reported while shown.
//            (Tuned with python -m kws.eval_stream; retune after retraining.)
//
// The engine is shared with the UART feature mode, so this module drives its feature input
// through the top level's mux, and only uses its results while `enable` is set.

`ifndef KWS_MEM_DIR
`define KWS_MEM_DIR ""
`endif

module kws_live
  import kws_pkg::*;
#(
  parameter string MEM_DIR     = `KWS_MEM_DIR,
  parameter int    PDM_PERIOD  = 25,           // PDM clock = clk / PDM_PERIOD
  parameter int    PDM_DELAY   = 0,            // see pdm_mic
  parameter int    HOLD_CYCLES = 100_000_000,  // how long a detection stays shown
  parameter int    N_CONSEC    = LIVE_N_CONSEC  // from the exported model (kws_pkg)
) (
  input  logic               clk,
  input  logic               rst,
  input  logic               enable,
  input  logic [3:0]         gain,
  input  logic               sample_fall,
  input  logic signed [31:0] min_margin,    // LIVE_MARGIN by default (see kws_top)
  // PCM from the host instead of the mic (testing)
  input  logic               ext_en,
  input  logic               ext_valid,
  input  logic signed [15:0] ext_pcm,
  // mic pins
  output logic               m_clk,
  output logic               m_lrsel,
  input  logic               m_data,
  // PCM out (for recording) and a peak level for a meter
  output logic               pcm_valid,
  output logic signed [15:0] pcm,
  output logic [3:0]         level,         // position of the leading one of the recent peak
  // features for the engine: N_MELS (band, value) pairs per frame, then frame_end
  output logic               feat_we,
  output logic [5:0]         feat_band,
  output logic [7:0]         feat_wdata,
  output logic               frame_end,
  // engine results
  input  logic               eng_done,
  input  logic               eng_full,
  input  logic [3:0]         eng_class,
  input  logic signed [31:0] eng_margin,
  // result
  // debug: last frame's frontend summary and feature row
  output logic [31:0]        dbg_win_or,
  output logic [48:0]        dbg_pow_or,
  output logic [5:0]         dbg_nfeat,
  output logic [7:0]         dbg_row [N_MELS],
  output logic               detect,        // one-cycle pulse per detection
  output logic               showing,
  output logic [3:0]         show_class
);
  // ---------------------------------------------------------------------------------------
  // Mic path and frontend
  // ---------------------------------------------------------------------------------------
  logic               bit_valid, bit_data, c_valid, mic_valid;
  logic signed [15:0] mic_pcm;
  logic signed [23:0] c_data;
  logic               f_valid, frame_done;
  logic [5:0]         f_band;
  logic signed [7:0]  f_q;

  pdm_mic #(.PERIOD(PDM_PERIOD), .DELAY(PDM_DELAY)) u_pdm (
    .clk, .rst, .sample_fall, .m_clk, .m_lrsel, .m_data, .bit_valid, .bit_data);
  cic_decim u_cic (
    .clk, .rst, .in_valid(bit_valid), .in_bit(bit_data), .out_valid(c_valid), .out_data(c_data));
  mic_fir #(.MEM_DIR(MEM_DIR)) u_fir (
    .clk, .rst, .in_valid(c_valid), .in_data(c_data), .gain, .pcm_valid(mic_valid), .pcm(mic_pcm));

  // The frontend (and the level meter and recording) take the mic or the host's PCM.
  always_ff @(posedge clk) begin
    pcm_valid <= ext_en ? ext_valid : mic_valid;
    pcm       <= ext_en ? ext_pcm : mic_pcm;
  end
  audio_frontend #(.MEM_DIR(MEM_DIR)) u_fe (
    .clk, .rst, .pcm_valid, .pcm,
    .feat_valid(f_valid), .feat_band(f_band), .feat_q(f_q), .frame_done,
    .dbg_win_or, .dbg_pow_or, .dbg_nfeat);

  logic [7:0] row_buf [N_MELS];
  always_ff @(posedge clk) begin
    if (f_valid) row_buf[f_band] <= f_q;
    if (frame_done) dbg_row <= row_buf;
  end

  // Level meter: the leading-one position of the peak over each 1024 samples (64 ms). It
  // jumps up at once and falls back one step per window, so the LEDs do not flicker.
  logic [14:0] peak_run;
  logic [9:0]  peak_cnt;
  logic        window_end;
  logic [3:0]  new_level;
  always_ff @(posedge clk) begin
    window_end <= 1'b0;
    if (pcm_valid) begin
      logic [14:0] mag;
      mag = pcm[15] ? 15'(-pcm) : 15'(pcm);
      peak_cnt <= peak_cnt + 1'b1;
      if (peak_cnt == '1) begin
        new_level  <= '0;
        for (int i = 0; i < 15; i++)
          if ((peak_run > mag ? peak_run : mag) >> i != 0) new_level <= 4'(i + 1);
        window_end <= 1'b1;
        peak_run   <= '0;
      end else if (mag > peak_run) begin
        peak_run <= mag;
      end
    end
    if (window_end) level <= new_level > level ? new_level : level != 0 ? level - 1'b1 : '0;
    if (rst) begin
      peak_run <= '0;
      peak_cnt <= '0;
      level    <= '0;
    end
  end

  assign feat_we    = f_valid;
  assign feat_band  = f_band;
  assign feat_wdata = f_q;
  assign frame_end  = frame_done;

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
    // Step 1, the cycle after a result: is it a confident keyword? (Results before the
    // engine has a whole window, just after a restart, are ignored.)
    decide    <= eng_done && eng_full && enable;
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
