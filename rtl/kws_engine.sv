// DS-CNN-S keyword-spotting inference engine.
//
// Runs the 9 conv layers, global average pool and FC on an int8 feature map written into
// the feature RAM, then reports the 12 integer logits and the argmax class. Number format:
// python/kws/quant.py. Memory layouts and constants: python/kws/export.py -> rtl/gen/.
//
// Conv datapath: LANES parallel MACs, each producing one output channel of the current
// channel group. Per (layer, group) it walks every output pixel and every tap:
//   STEM  40 taps (10x4, stride 2), one int8 input value broadcast to all lanes
//   DW     9 taps (3x3), lane i reads channel i of the input word
//   PW    64 taps (one per input channel), that channel's value broadcast to all lanes
// The pipeline drains between groups so bias/shift can change safely.
//
// Pipeline:  S0 issue -> C coordinates -> S1 mem addr -> S2 mem data, operand select
//            -> S3 multiply -> S4 accumulate -> S5 shift -> S5b clamp -> S6 write (+ pool)

`ifndef KWS_MEM_DIR
`define KWS_MEM_DIR ""
`endif

module kws_engine
  import kws_pkg::*;
#(
  parameter string MEM_DIR = `KWS_MEM_DIR
) (
  input  logic               clk,
  input  logic               rst,
  // Feature RAM write port: int8 at address h*IN_W + w. Write only while !busy.
  input  logic               feat_we,
  input  logic [10:0]        feat_waddr,
  input  logic [7:0]         feat_wdata,
  input  logic               start,
  output logic               busy,
  output logic               done,        // one-cycle pulse; outputs below are then valid
  output logic [3:0]         class_idx,
  output logic signed [31:0] margin,      // top logit minus the runner-up
  output logic [7:0]         dbg_feat_or, // debug: OR of all feature bytes the stem layer read
  output logic signed [31:0] logits [N_CLASSES]
);
  localparam int ACT_AW  = $clog2(ACT_WORDS);
  localparam int WROM_AW = $clog2(WROM_WORDS);
  localparam int FEAT_AW = 11;
  localparam int BS_AW   = $clog2(N_LAYERS * GROUPS);
  localparam int FCW_AW  = $clog2(N_CLASSES * CHANNELS);
  localparam int LAST    = N_LAYERS - 1;

  typedef enum logic [2:0] {
    S_IDLE, S_SETUP, S_SETUP2, S_RUN, S_DRAIN, S_FC, S_FC_DRAIN, S_DONE
  } state_t;

  state_t state;

  // ---------------------------------------------------------------------------------------
  // Loop counters
  // ---------------------------------------------------------------------------------------
  logic [3:0] layer;
  logic [1:0] grp;
  logic [4:0] oh, ow;   // output pixel
  logic [8:0] pix;      // oh * OUT_W + ow
  logic [5:0] tap;     // weight index within (layer, group): kh*KW + kw, or input channel
  logic [3:0] kh;
  logic [1:0] kw;
  logic       layer_done;  // pulse after each layer fully written (testbench hook)
  logic [$clog2(WROM_WORDS)-1:0]      w_base;   // weight ROM word of tap 0 for (layer, group)
  logic [$clog2(N_LAYERS*GROUPS)-1:0] bs_addr;  // bias/shift ROM word for (layer, group)

  // Per-layer constants, registered in S_SETUP to keep the table lookup off the counter paths.
  layer_kind_t kind;
  logic [5:0]  last_tap;
  logic [1:0]  last_kw;

  always_ff @(posedge clk) begin
    if (state == S_SETUP) begin
      kind <= LAYER_KIND[layer];
      unique case (LAYER_KIND[layer])
        L_STEM:  begin last_tap <= 6'd39; last_kw <= 2'd3; end
        L_DW:    begin last_tap <= 6'd8;  last_kw <= 2'd2; end
        default: begin last_tap <= 6'd63; last_kw <= 2'd0; end
      endcase
    end
  end

  // ---------------------------------------------------------------------------------------
  // S0: input coordinates for the current tap (padding and addresses are formed in S1)
  // ---------------------------------------------------------------------------------------
  logic signed [7:0] oh_s, ow_s, kh_s, kw_s, ih, iw;
  logic [1:0]        in_grp;

  assign oh_s = 8'(oh);
  assign ow_s = 8'(ow);
  assign kh_s = 8'(kh);
  assign kw_s = 8'(kw);

  always_comb begin
    unique case (kind)
      L_STEM: begin  // 10x4 kernel, stride 2, padding (5, 1)
        ih = 2 * oh_s + kh_s - 8'sd5;
        iw = 2 * ow_s + kw_s - 8'sd1;
      end
      L_DW: begin    // 3x3 kernel, stride 1, padding 1
        ih = oh_s + kh_s - 8'sd1;
        iw = ow_s + kw_s - 8'sd1;
      end
      default: begin
        ih = oh_s;
        iw = ow_s;
      end
    endcase
    in_grp = kind == L_PW ? tap[5:4] : grp;
  end

  // ---------------------------------------------------------------------------------------
  // Memories
  // ---------------------------------------------------------------------------------------
  logic [FEAT_AW-1:0]       feat_raddr;
  logic [7:0]               feat_rdata;
  logic [ACT_AW-1:0]        act_raddr, act_waddr;
  logic [8*LANES-1:0]       act_a_rdata, act_b_rdata, act_in_rdata, act_wdata;
  logic                     act_a_we, act_b_we;
  logic [WROM_AW-1:0]       wrom_raddr;
  logic [8*LANES-1:0]       wrom_rdata;
  logic [32*LANES-1:0]      bias_rdata;
  logic [4*LANES-1:0]       shift_rdata;
  logic [FCW_AW-1:0]        fcw_raddr;
  logic [7:0]               fcw_rdata;
  logic [3:0]               fcb_raddr;
  logic [31:0]              fcb_rdata;

  sdp_ram #(.WIDTH(8), .DEPTH(IN_H * IN_W)) u_feat (
    .clk, .we(feat_we), .waddr(feat_waddr), .wdata(feat_wdata),
    .raddr(feat_raddr), .rdata(feat_rdata));

  // Ping-pong activation buffers: even layers write A, odd layers write B.
  sdp_ram #(.WIDTH(8 * LANES), .DEPTH(ACT_WORDS)) u_act_a (
    .clk, .we(act_a_we), .waddr(act_waddr), .wdata(act_wdata),
    .raddr(act_raddr), .rdata(act_a_rdata));
  sdp_ram #(.WIDTH(8 * LANES), .DEPTH(ACT_WORDS)) u_act_b (
    .clk, .we(act_b_we), .waddr(act_waddr), .wdata(act_wdata),
    .raddr(act_raddr), .rdata(act_b_rdata));
  assign act_in_rdata = layer[0] ? act_a_rdata : act_b_rdata;

  sdp_ram #(.WIDTH(8 * LANES), .DEPTH(WROM_WORDS), .INIT({MEM_DIR, "weights.hex"})) u_wrom (
    .clk, .we(1'b0), .waddr('0), .wdata('0), .raddr(wrom_raddr), .rdata(wrom_rdata));
  // Bias and shift ROMs: one word per (layer, group), read during S_SETUP.
  sdp_ram #(.WIDTH(32 * LANES), .DEPTH(N_LAYERS * GROUPS), .INIT({MEM_DIR, "bias.hex"}), .STYLE("block")) u_bias (
    .clk, .we(1'b0), .waddr('0), .wdata('0),
    .raddr(bs_addr), .rdata(bias_rdata));
  sdp_ram #(.WIDTH(4 * LANES), .DEPTH(N_LAYERS * GROUPS), .INIT({MEM_DIR, "shift.hex"}), .STYLE("block")) u_shift (
    .clk, .we(1'b0), .waddr('0), .wdata('0),
    .raddr(bs_addr), .rdata(shift_rdata));
  sdp_ram #(.WIDTH(8), .DEPTH(N_CLASSES * CHANNELS), .INIT({MEM_DIR, "fc_w.hex"}), .STYLE("block")) u_fcw (
    .clk, .we(1'b0), .waddr('0), .wdata('0), .raddr(fcw_raddr), .rdata(fcw_rdata));
  sdp_ram #(.WIDTH(32), .DEPTH(N_CLASSES), .INIT({MEM_DIR, "fc_b.hex"})) u_fcb (
    .clk, .we(1'b0), .waddr('0), .wdata('0), .raddr(fcb_raddr), .rdata(fcb_rdata));

  // ---------------------------------------------------------------------------------------
  // Per-(layer, group) constants, latched in S_SETUP2. The weight ROM is laid out layer by
  // layer, group by group, so both ROM bases just advance by one block per group.
  // ---------------------------------------------------------------------------------------
  logic signed [31:0] bias_rnd [LANES];  // bias + rounding constant 2^(shift-1), from bias.hex
  logic [3:0]         shift    [LANES];

  always_ff @(posedge clk) begin
    if (state == S_SETUP2) begin
      for (int i = 0; i < LANES; i++) begin
        shift[i]    <= shift_rdata[4*i +: 4];
        bias_rnd[i] <= $signed(bias_rdata[32*i +: 32]);
      end
    end
  end

  // ---------------------------------------------------------------------------------------
  // Conv pipeline
  // ---------------------------------------------------------------------------------------
  logic              c_valid, c_first, c_last;
  logic signed [7:0] c_h_lim, c_w_lim;  // input height/width for the padding check
  logic [3:0]        c_lane;
  logic signed [7:0] c_ih, c_iw;
  logic [1:0]        c_in_grp;
  logic [5:0]        c_tap;
  logic [ACT_AW-1:0] c_addr;
  logic              s1_valid, s1_first, s1_last, s1_pad;
  logic [3:0]        s1_lane;
  logic [ACT_AW-1:0] s1_addr;
  logic              s2_valid, s2_first, s2_last, s2_pad;
  logic [3:0]        s2_lane;
  logic [ACT_AW-1:0] s2_addr;
  logic              s3_valid, s3_first, s3_last;
  logic [ACT_AW-1:0] s3_addr;
  logic              s4_valid, s4_first, s4_last;
  logic [ACT_AW-1:0] s4_addr;
  logic              s5_valid, s5b_valid;
  logic [ACT_AW-1:0] s5_addr, s5b_addr;
  logic              s6_valid;
  logic [ACT_AW-1:0] s6_addr;

  logic signed [8:0]  s3_a    [LANES];  // activation: int8 (stem input) or uint8
  logic signed [7:0]  s3_w    [LANES];
  logic signed [16:0] s4_prod [LANES];
  logic signed [31:0] acc     [LANES];
  logic signed [31:0] s5_sum  [LANES];  // accumulator, rounding constant included
  logic signed [31:0] s5b_y   [LANES];  // after the shift
  logic [7:0]         s6_y    [LANES];
  logic signed [31:0] acc_next[LANES];

  always_comb begin
    for (int i = 0; i < LANES; i++)
      acc_next[i] = (s4_first ? bias_rnd[i] : acc[i]) + 32'(s4_prod[i]);
  end

  logic [7:0] feat_or;
  always_ff @(posedge clk) begin
    if (state == S_IDLE && start) feat_or <= '0;
    else if (s2_valid && kind == L_STEM) feat_or <= feat_or | feat_rdata;
    if (state == S_DONE) dbg_feat_or <= feat_or;
  end

  always_ff @(posedge clk) begin
    // S0 -> C: register coordinates
    c_valid  <= state == S_RUN;
    c_first  <= tap == 0;
    c_last   <= tap == last_tap;
    c_h_lim  <= kind == L_STEM ? 8'(IN_H) : 8'(OUT_H);
    c_w_lim  <= kind == L_STEM ? 8'(IN_W) : 8'(OUT_W);
    c_lane   <= tap[3:0];
    c_ih     <= ih;
    c_iw     <= iw;
    c_in_grp <= in_grp;
    c_tap    <= tap;
    c_addr   <= ACT_AW'(pix * GROUPS + grp);

    // C -> S1: padding and memory addresses (PW coordinates are always in range)
    {s1_valid, s1_first, s1_last, s1_lane, s1_addr} <= {c_valid, c_first, c_last, c_lane, c_addr};
    s1_pad <= c_ih < 0 || c_ih >= c_h_lim || c_iw < 0 || c_iw >= c_w_lim;
    feat_raddr <= FEAT_AW'(c_ih * IN_W + c_iw);
    act_raddr  <= ACT_AW'((c_ih * OUT_W + c_iw) * GROUPS + c_in_grp);
    wrom_raddr <= w_base + WROM_AW'(c_tap);

    // S1 -> S2 (memories register their outputs)
    {s2_valid, s2_first, s2_last, s2_pad, s2_lane, s2_addr} <=
        {s1_valid, s1_first, s1_last, s1_pad, s1_lane, s1_addr};

    // S2 -> S3: select operands
    {s3_valid, s3_first, s3_last, s3_addr} <= {s2_valid, s2_first, s2_last, s2_addr};
    for (int i = 0; i < LANES; i++) begin
      logic signed [8:0] a;
      unique case (kind)
        L_STEM:  a = 9'($signed(feat_rdata));
        L_DW:    a = {1'b0, act_in_rdata[8*i +: 8]};
        default: a = {1'b0, act_in_rdata[8*s2_lane +: 8]};
      endcase
      s3_a[i] <= s2_pad ? '0 : a;
      s3_w[i] <= $signed(wrom_rdata[8*i +: 8]);
    end

    // S3 -> S4: multiply
    {s4_valid, s4_first, s4_last, s4_addr} <= {s3_valid, s3_first, s3_last, s3_addr};
    for (int i = 0; i < LANES; i++) s4_prod[i] <= s3_a[i] * s3_w[i];

    // S4 -> S5: accumulate; on the last tap hand the sum to requant
    s5_valid <= s4_valid && s4_last;
    s5_addr  <= s4_addr;
    for (int i = 0; i < LANES; i++) begin
      if (s4_valid) acc[i] <= acc_next[i];
      s5_sum[i] <= acc_next[i];
    end

    // S5 -> S5b: shift
    s5b_valid <= s5_valid;
    s5b_addr  <= s5_addr;
    for (int i = 0; i < LANES; i++) s5b_y[i] <= s5_sum[i] >>> shift[i];

    // S5b -> S6: ReLU, saturate to uint8
    s6_valid <= s5b_valid;
    s6_addr  <= s5b_addr;
    for (int i = 0; i < LANES; i++)
      s6_y[i] <= s5b_y[i] < 0 ? 8'd0 : s5b_y[i] > 255 ? 8'd255 : s5b_y[i][7:0];

    if (rst) {c_valid, s1_valid, s2_valid, s3_valid, s4_valid, s5_valid, s5b_valid, s6_valid} <= '0;
  end

  // S6: write the output word
  always_comb begin
    for (int i = 0; i < LANES; i++) act_wdata[8*i +: 8] = s6_y[i];
  end
  assign act_waddr = s6_addr;
  assign act_a_we  = s6_valid && !layer[0];
  assign act_b_we  = s6_valid &&  layer[0];

  wire conv_pipe_empty = !(c_valid || s1_valid || s2_valid || s3_valid || s4_valid || s5_valid ||
                             s5b_valid || s6_valid);

  // Global average pool, as a plain sum of the last layer's outputs (max 500 * 255 < 2^17).
  // Each lane sums its channel for the current group; the sums are stored per group once
  // the group has drained, which keeps the group select out of the adder path.
  logic [16:0] pool_acc [LANES];
  logic [16:0] pooled   [GROUPS][LANES];
  logic [GROUPS-1:0] pool_we;

  always_ff @(posedge clk) begin
    if (state == S_SETUP) begin
      for (int i = 0; i < LANES; i++) pool_acc[i] <= '0;
    end else if (s6_valid) begin
      for (int i = 0; i < LANES; i++) pool_acc[i] <= pool_acc[i] + 17'(s6_y[i]);
    end
    // Store one cycle after the group drains (a registered, one-hot enable: it fans out to
    // every pooled register). pool_acc is only cleared in the S_SETUP that follows.
    for (int g = 0; g < GROUPS; g++)
      pool_we[g] <= state == S_DRAIN && conv_pipe_empty && layer == LAST && grp == 2'(g);
    for (int g = 0; g < GROUPS; g++)
      if (pool_we[g]) for (int i = 0; i < LANES; i++) pooled[g][i] <= pool_acc[i];
  end

  // ---------------------------------------------------------------------------------------
  // FC: logits[k] = fc_b[k] + sum_c fc_w[k][c] * pooled[c], then argmax.
  // F0 issue -> F1 ROM data, pick the group of 16 pooled sums -> F2 pick the lane
  // -> F3 multiply -> F4 accumulate. (The 64:1 pooled mux in one cycle was too slow.)
  // ---------------------------------------------------------------------------------------
  logic [3:0]         fk;
  logic [5:0]         fc;
  logic [FCW_AW-1:0]  fcw_addr;  // fk * CHANNELS + fc
  logic               f1_valid, f1_first, f1_last;
  logic [3:0]         f1_k, f1_lane;
  logic [16:0]        f1_group [LANES];
  logic               f2_valid, f2_first, f2_last;
  logic [3:0]         f2_k;
  logic [16:0]        f2_pool;
  logic signed [7:0]  f2_w;
  logic signed [31:0] f2_bias;
  logic               f3_valid, f3_first, f3_last;
  logic [3:0]         f3_k;
  logic signed [31:0] f3_bias;
  logic signed [25:0] f3_prod;
  logic signed [31:0] facc, facc_next, best, second;

  assign fcw_raddr = fcw_addr;
  assign fcb_raddr = fk;
  assign facc_next = (f3_first ? f3_bias : facc) + 32'(f3_prod);

  always_ff @(posedge clk) begin
    f1_valid <= state == S_FC;
    f1_first <= fc == 0;
    f1_last  <= fc == CHANNELS - 1;
    f1_k     <= fk;
    f1_lane  <= fc[3:0];
    for (int i = 0; i < LANES; i++) f1_group[i] <= pooled[fc[5:4]][i];

    {f2_valid, f2_first, f2_last, f2_k} <= {f1_valid, f1_first, f1_last, f1_k};
    f2_pool <= f1_group[f1_lane];
    f2_w    <= $signed(fcw_rdata);
    f2_bias <= $signed(fcb_rdata);

    {f3_valid, f3_first, f3_last, f3_k} <= {f2_valid, f2_first, f2_last, f2_k};
    f3_bias <= f2_bias;
    f3_prod <= f2_w * $signed({1'b0, f2_pool});

    if (f3_valid) begin
      facc <= facc_next;
      if (f3_last) begin
        logits[f3_k] <= facc_next;
        // Strict > keeps the first max, like argmax. Also track the runner-up for `margin`.
        if (f3_k == 0) begin
          best      <= facc_next;
          second    <= 32'sh8000_0000;
          class_idx <= f3_k;
        end else if (facc_next > best) begin
          best      <= facc_next;
          second    <= best;
          class_idx <= f3_k;
        end else if (facc_next > second) begin
          second <= facc_next;
        end
      end
    end

    if (rst) {f1_valid, f2_valid, f3_valid} <= '0;
  end

  wire fc_pipe_empty = !(f1_valid || f2_valid || f3_valid);

  // ---------------------------------------------------------------------------------------
  // Control
  // ---------------------------------------------------------------------------------------
  always_ff @(posedge clk) begin
    layer_done <= 1'b0;
    done       <= 1'b0;

    unique case (state)
      S_IDLE: if (start) begin
        {layer, grp, oh, ow, pix, tap, kh, kw} <= '0;
        w_base  <= '0;
        bs_addr <= '0;
        state   <= S_SETUP;
      end

      S_SETUP:  state <= S_SETUP2;  // bias/shift ROMs read the new (layer, group)
      S_SETUP2: state <= S_RUN;     // latch bias/shift

      S_RUN: begin
        if (tap == last_tap) begin
          {tap, kh, kw} <= '0;
          if (ow == 5'(OUT_W - 1)) begin
            ow <= '0;
            if (oh == 5'(OUT_H - 1)) begin
              oh    <= '0;
              pix   <= '0;
              state <= S_DRAIN;
            end else begin
              oh  <= oh + 1'b1;
              pix <= pix + 1'b1;
            end
          end else begin
            ow  <= ow + 1'b1;
            pix <= pix + 1'b1;
          end
        end else begin
          tap <= tap + 1'b1;
          if (kw == last_kw) begin
            kw <= '0;
            kh <= kh + 1'b1;
          end else kw <= kw + 1'b1;
        end
      end

      S_DRAIN: if (conv_pipe_empty) begin
        w_base  <= w_base + WROM_AW'(last_tap + 1);
        bs_addr <= bs_addr + 1'b1;
        if (grp == 2'(GROUPS - 1)) begin
          layer_done <= 1'b1;
          grp        <= '0;
          if (layer == 4'(LAST)) begin
            {fk, fc, fcw_addr} <= '0;
            state              <= S_FC;
          end else begin
            layer <= layer + 1'b1;
            state <= S_SETUP;
          end
        end else begin
          grp   <= grp + 1'b1;
          state <= S_SETUP;
        end
      end

      S_FC: begin
        fcw_addr <= fcw_addr + 1'b1;
        if (fc == 6'(CHANNELS - 1)) begin
          fc <= '0;
          if (fk == 4'(N_CLASSES - 1)) state <= S_FC_DRAIN;
          else fk <= fk + 1'b1;
        end else fc <= fc + 1'b1;
      end

      S_FC_DRAIN: if (fc_pipe_empty) state <= S_DONE;

      S_DONE: begin
        done   <= 1'b1;
        margin <= best - second;
        state <= S_IDLE;
      end

      default: state <= S_IDLE;
    endcase

    if (rst) state <= S_IDLE;
  end

  assign busy = state != S_IDLE;
endmodule
