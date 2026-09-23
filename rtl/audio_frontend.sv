// Audio frontend: int16 PCM at 16 kHz -> one row of N_MELS int8 features per HOP samples.
// Bit-exact to python/kws/fixed_frontend.py (features_fixed). Frame j covers PCM samples
// [HOP*j, HOP*j + FFT_N) counted from reset and is processed once its last sample arrives.
//
// Per frame, sequentially (~18k cycles, far below the 320-sample frame period):
//   WIN   window the last FFT_N samples (Hann, Q15) into the FFT RAM in bit-reversed order
//   FFT   in-place radix-2 DIT, one butterfly every 7 cycles
//   POW   power of bins 0..FFT_N/2 into the power RAM
//   MEL   walk the sparse mel ROM {last, weight, bin}, accumulate, then log2 -> int8
// Outputs one (band, value) pair per band, then frame_done.

`ifndef KWS_MEM_DIR
`define KWS_MEM_DIR ""
`endif

module audio_frontend
  import kws_pkg::*;
#(
  parameter string MEM_DIR = `KWS_MEM_DIR
) (
  input  logic              clk,
  input  logic              rst,
  input  logic              pcm_valid,
  input  logic signed [15:0] pcm,
  output logic              feat_valid,
  output logic [5:0]        feat_band,
  output logic signed [7:0] feat_q,
  output logic              frame_done
);
  localparam int NBINS = FFT_N / 2 + 1;

  typedef enum logic [2:0] {F_IDLE, F_WIN, F_FFT, F_POW, F_MEL, F_DONE} fstate_t;
  fstate_t state;

  // ---------------------------------------------------------------------------------------
  // PCM ring buffer and frame trigger
  // ---------------------------------------------------------------------------------------
  logic [9:0]  pcm_wptr, frame_start;
  logic [9:0]  until_frame;  // samples still needed for the next frame, minus one
  logic        trigger;
  logic [9:0]  pcm_raddr;
  logic [15:0] pcm_rdata;

  sdp_ram #(.WIDTH(16), .DEPTH(1024)) u_pcm (
    .clk, .we(pcm_valid), .waddr(pcm_wptr), .wdata(pcm), .raddr(pcm_raddr), .rdata(pcm_rdata));

  always_ff @(posedge clk) begin
    trigger <= 1'b0;
    if (pcm_valid) begin
      pcm_wptr <= pcm_wptr + 1'b1;
      if (until_frame == 0) begin
        trigger     <= 1'b1;
        frame_start <= pcm_wptr - 10'(FFT_N - 1);
        until_frame <= 10'(HOP - 1);
      end else begin
        until_frame <= until_frame - 1'b1;
      end
    end
    if (rst) begin
      pcm_wptr    <= '0;
      until_frame <= 10'(FFT_N - 1);
    end
  end

  // ---------------------------------------------------------------------------------------
  // Memories
  // ---------------------------------------------------------------------------------------
  logic [8:0]  hann_raddr, fft_raddr, fft_waddr;
  logic [15:0] hann_rdata;
  logic [49:0] fft_rdata, fft_wdata;   // {im[24:0], re[24:0]}
  logic        fft_we;
  logic [7:0]  tw_raddr;
  logic [35:0] tw_rdata;               // {wi[17:0], wr[17:0]}
  logic [8:0]  pow_raddr, pow_waddr;
  logic [48:0] pow_rdata, pow_wdata;
  logic        pow_we;
  logic [8:0]  mel_raddr;
  logic [18:0] mel_rdata;              // {last, weight[8:0], bin[8:0]}
  logic [7:0]  lut_raddr;
  logic [6:0]  lut_rdata;

  sdp_ram #(.WIDTH(16), .DEPTH(FFT_N), .INIT({MEM_DIR, "fe_hann.hex"}), .STYLE("block")) u_hann (
    .clk, .we(1'b0), .waddr('0), .wdata('0), .raddr(hann_raddr), .rdata(hann_rdata));
  sdp_ram #(.WIDTH(50), .DEPTH(FFT_N)) u_fft (
    .clk, .we(fft_we), .waddr(fft_waddr), .wdata(fft_wdata), .raddr(fft_raddr), .rdata(fft_rdata));
  sdp_ram #(.WIDTH(36), .DEPTH(FFT_N / 2), .INIT({MEM_DIR, "fe_twiddle.hex"}), .STYLE("block")) u_tw (
    .clk, .we(1'b0), .waddr('0), .wdata('0), .raddr(tw_raddr), .rdata(tw_rdata));
  sdp_ram #(.WIDTH(49), .DEPTH(FFT_N)) u_pow (
    .clk, .we(pow_we), .waddr(pow_waddr), .wdata(pow_wdata), .raddr(pow_raddr), .rdata(pow_rdata));
  sdp_ram #(.WIDTH(19), .DEPTH(MEL_ENTRIES), .INIT({MEM_DIR, "fe_mel.hex"}), .STYLE("block")) u_mel (
    .clk, .we(1'b0), .waddr('0), .wdata('0), .raddr(mel_raddr), .rdata(mel_rdata));
  sdp_ram #(.WIDTH(7), .DEPTH(256), .INIT({MEM_DIR, "fe_log2.hex"}), .STYLE("block")) u_log2 (
    .clk, .we(1'b0), .waddr('0), .wdata('0), .raddr(lut_raddr), .rdata(lut_rdata));

  function automatic logic [8:0] bitrev9(input logic [8:0] x);
    for (int i = 0; i < 9; i++) bitrev9[i] = x[8 - i];
  endfunction

  // ---------------------------------------------------------------------------------------
  // Sequencer counters
  // ---------------------------------------------------------------------------------------
  logic [9:0] n;        // WIN: sample index; POW: bin; MEL: ROM entry
  logic [3:0] stage;    // FFT stage 0..8
  logic [7:0] bfly;     // FFT butterfly index within the stage
  logic [2:0] bstep;    // FFT butterfly sub-step 0..6
  logic       issuing;  // WIN/POW/MEL: still issuing indices

  // Butterfly addresses for (stage, bfly)
  logic [8:0] ia, ib, jmask;
  logic [7:0] tw_k;
  always_comb begin
    jmask = (9'd1 << stage) - 1'b1;
    ia    = ((9'(bfly) >> stage) << (stage + 1)) | (9'(bfly) & jmask);
    ib    = ia | (9'd1 << stage);
    tw_k  = 8'((9'(bfly) & jmask) << (4'd8 - stage));
  end

  // ---------------------------------------------------------------------------------------
  // WIN pipeline: issue n -> (pcm, hann) data -> product -> write fft[bitrev(n)]
  // ---------------------------------------------------------------------------------------
  logic               w1_valid, w2_valid;
  logic [8:0]         w1_n, w2_n;
  logic signed [31:0] w2_prod;

  // ---------------------------------------------------------------------------------------
  // FFT butterfly registers
  // ---------------------------------------------------------------------------------------
  logic signed [24:0] ar, ai, br, bi, tr, ti;
  logic signed [17:0] wr, wi;
  logic signed [42:0] p_rr, p_ii, p_ri, p_ir;

  // ---------------------------------------------------------------------------------------
  // POW pipeline: issue k -> data -> squares -> write
  // ---------------------------------------------------------------------------------------
  logic               q1_valid, q2_valid;
  logic [8:0]         q1_k, q2_k;
  logic [49:0]        q2_re2, q2_im2;

  // ---------------------------------------------------------------------------------------
  // MEL pipeline: issue e -> entry -> power data -> product -> accumulate -> log2 -> q
  // ---------------------------------------------------------------------------------------
  logic        e1_valid, e2_valid, e3_valid;
  logic        e2_last, e3_last, band_first;
  logic [8:0]  e2_w;
  logic [57:0] e3_prod;
  logic [62:0] mel_acc;
  logic        l0_valid, la_valid, l1_valid, l2_valid, l3_valid;
  logic [62:0] l0_v, la_v, l1_v;
  logic [7:0]  la_any;           // per 8-bit group of v: any bit set
  logic [2:0]  la_loc [8];       // per group: position of its leading one
  logic [5:0]  l1_msb, l2_msb, l3_msb;
  logic [5:0]  band;

  assign pow_raddr = 9'(mel_rdata[8:0]);  // E1: the entry's bin addresses the power RAM

  always_ff @(posedge clk) begin
    fft_we     <= 1'b0;
    pow_we     <= 1'b0;
    feat_valid <= 1'b0;
    frame_done <= 1'b0;

    // ---------------- sequencer ----------------
    unique case (state)
      F_IDLE: if (trigger) begin
        n       <= '0;
        issuing <= 1'b1;
        state   <= F_WIN;
      end

      F_WIN: begin
        if (issuing) begin
          if (n == 10'(FFT_N - 1)) issuing <= 1'b0;
          n <= n + 1'b1;
        end else if (!w1_valid && !w2_valid && !fft_we) begin
          {stage, bfly, bstep} <= '0;
          state <= F_FFT;
        end
      end

      F_FFT: begin
        bstep <= bstep + 1'b1;
        unique case (bstep)
          3'd0: ;                                         // read a, twiddle
          3'd1: begin                                     // read b; a and w arrive
            ar <= $signed(fft_rdata[24:0]);
            ai <= $signed(fft_rdata[49:25]);
            wr <= $signed(tw_rdata[17:0]);
            wi <= $signed(tw_rdata[35:18]);
          end
          3'd2: begin                                     // b arrives
            br <= $signed(fft_rdata[24:0]);
            bi <= $signed(fft_rdata[49:25]);
          end
          3'd3: begin
            p_rr <= wr * br;
            p_ii <= wi * bi;
            p_ri <= wr * bi;
            p_ir <= wi * br;
          end
          3'd4: begin
            tr <= 25'((p_rr - p_ii + 43'sd32768) >>> 16);
            ti <= 25'((p_ri + p_ir + 43'sd32768) >>> 16);
          end
          3'd5: begin
            fft_we    <= 1'b1;
            fft_waddr <= ia;
            fft_wdata <= {25'(ai + ti), 25'(ar + tr)};
          end
          default: begin                                  // 3'd6
            fft_we    <= 1'b1;
            fft_waddr <= ib;
            fft_wdata <= {25'(ai - ti), 25'(ar - tr)};
            bstep     <= '0;
            if (bfly == 8'(FFT_N / 2 - 1)) begin
              bfly <= '0;
              if (stage == 4'd8) begin
                n       <= '0;
                issuing <= 1'b1;
                state   <= F_POW;
              end else stage <= stage + 1'b1;
            end else bfly <= bfly + 1'b1;
          end
        endcase
      end

      F_POW: begin
        if (issuing) begin
          if (n == 10'(NBINS - 1)) issuing <= 1'b0;
          n <= n + 1'b1;
        end else if (!q1_valid && !q2_valid && !pow_we) begin
          n       <= '0;
          issuing <= 1'b1;
          state   <= F_MEL;
        end
      end

      F_MEL: begin
        if (issuing) begin
          if (n == 10'(MEL_ENTRIES - 1)) issuing <= 1'b0;
          n <= n + 1'b1;
        end else if (!(e1_valid || e2_valid || e3_valid || l0_valid || la_valid || l1_valid ||
                       l2_valid || l3_valid)) begin
          state <= F_DONE;
        end
      end

      F_DONE: begin
        frame_done <= 1'b1;
        state      <= F_IDLE;
      end

      default: state <= F_IDLE;
    endcase

    // ---------------- WIN datapath ----------------
    w1_valid <= state == F_WIN && issuing;
    w1_n     <= n[8:0];
    w2_valid <= w1_valid;
    w2_n     <= w1_n;
    w2_prod  <= $signed(pcm_rdata) * $signed({1'b0, hann_rdata});
    if (w2_valid) begin
      fft_we    <= 1'b1;
      fft_waddr <= bitrev9(w2_n);
      fft_wdata <= {25'sd0, 25'((w2_prod + 32'sd16384) >>> 15)};
    end

    // ---------------- POW datapath ----------------
    q1_valid <= state == F_POW && issuing;
    q1_k     <= n[8:0];
    q2_valid <= q1_valid;
    q2_k     <= q1_k;
    q2_re2   <= 50'($signed(fft_rdata[24:0]) * $signed(fft_rdata[24:0]));
    q2_im2   <= 50'($signed(fft_rdata[49:25]) * $signed(fft_rdata[49:25]));
    if (q2_valid) begin
      pow_we    <= 1'b1;
      pow_waddr <= q2_k;
      pow_wdata <= 49'(q2_re2 + q2_im2);
    end

    // ---------------- MEL datapath ----------------
    e1_valid <= state == F_MEL && issuing;
    e2_valid <= e1_valid;
    e2_last  <= mel_rdata[18];
    e2_w     <= mel_rdata[17:9];
    e3_valid <= e2_valid;
    e3_last  <= e2_last;
    e3_prod  <= 58'(pow_rdata) * 58'(e2_w);
    if (e3_valid) begin
      mel_acc    <= e3_last ? '0 : (band_first ? 63'(e3_prod) : mel_acc + 63'(e3_prod));
      band_first <= e3_last;
    end

    // log2: leading one (per 8-bit group, then across groups), mantissa, LUT, offset
    l0_valid <= e3_valid && e3_last;
    l0_v     <= (band_first ? 63'(e3_prod) : mel_acc + 63'(e3_prod)) + 63'(FE_EPS);
    la_valid <= l0_valid;
    la_v     <= l0_v;
    for (int g = 0; g < 8; g++) begin
      logic [7:0] grp_bits;
      grp_bits  = 8'({1'b0, l0_v} >> (8 * g));
      la_any[g] <= |grp_bits;
      la_loc[g] <= '0;
      for (int i = 0; i < 8; i++) if (grp_bits[i]) la_loc[g] <= 3'(i);
    end
    l1_valid <= la_valid;
    l1_v     <= la_v;
    for (int g = 0; g < 8; g++) if (la_any[g]) l1_msb <= {3'(g), la_loc[g]};
    l2_valid  <= l1_valid;
    l2_msb    <= l1_msb;
    lut_raddr <= 8'(l1_v >> (l1_msb - 6'd8));  // mantissa: the 8 bits below the leading one
    l3_valid  <= l2_valid;
    l3_msb    <= l2_msb;
    if (l3_valid) begin
      logic signed [15:0] lg, qv;
      lg = 16'({l3_msb, 6'd0}) + 16'(lut_rdata);
      qv = (lg - 16'(FE_OFFSET) + 16'sd8) >>> 4;
      feat_q     <= qv > 127 ? 8'sd127 : qv < -128 ? -8'sd128 : 8'(qv);
      feat_band  <= band;
      feat_valid <= 1'b1;
      band       <= band + 1'b1;
    end

    if (state == F_IDLE) begin
      band       <= '0;
      band_first <= 1'b1;
    end

    if (rst) begin
      state <= F_IDLE;
      {w1_valid, w2_valid, q1_valid, q2_valid, e1_valid, e2_valid, e3_valid} <= '0;
      {l0_valid, la_valid, l1_valid, l2_valid, l3_valid} <= '0;
    end
  end


  // Read addresses for the memories, from the sequencer state
  always_comb begin
    pcm_raddr  = frame_start + n;
    hann_raddr = n[8:0];
    tw_raddr   = tw_k;
    mel_raddr  = n[8:0];
    fft_raddr  = state == F_FFT ? (bstep == 3'd0 ? ia : ib) : n[8:0];
  end
endmodule
