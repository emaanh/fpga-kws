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
// Pipeline:  S0 issue -> S1 mem addr -> S2 mem data, operand select -> S3 multiply
//            -> S4 accumulate (+ rounding) -> S5 shift, clamp -> S6 write (+ pool on last layer)

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
  output logic signed [31:0] logits [N_CLASSES]
);
  localparam int ACT_AW  = $clog2(ACT_WORDS);
  localparam int WROM_AW = $clog2(WROM_WORDS);
  localparam int FEAT_AW = 11;
  localparam int BS_AW   = $clog2(N_LAYERS * GROUPS);
  localparam int FCW_AW  = $clog2(N_CLASSES * CHANNELS);
  localparam int LAST    = N_LAYERS - 1;

  typedef enum logic [2:0] {
    S_IDLE, S_SETUP, S_RUN, S_DRAIN, S_FC, S_FC_DRAIN, S_DONE
  } state_t;

  state_t state;

  // ---------------------------------------------------------------------------------------
  // Loop counters
  // ---------------------------------------------------------------------------------------
  logic [3:0] layer;
  logic [1:0] grp;
  logic [4:0] oh, ow;   // output pixel
  logic [5:0] tap;      // weight index within (layer, group): kh*KW + kw, or input channel
  logic [3:0] kh;
  logic [1:0] kw;
  logic       layer_done;  // pulse after each layer fully written (testbench hook)

  layer_kind_t kind;
  logic [5:0]  last_tap;
  logic [1:0]  last_kw;
  assign kind = LAYER_KIND[layer];

  always_comb begin
    unique case (kind)
      L_STEM:  begin last_tap = 6'd39; last_kw = 2'd3; end
      L_DW:    begin last_tap = 6'd8;  last_kw = 2'd2; end
      default: begin last_tap = 6'd63; last_kw = 2'd0; end
    endcase
  end

  // ---------------------------------------------------------------------------------------
  // S0: input coordinates, padding and addresses for the current tap
  // ---------------------------------------------------------------------------------------
  logic signed [7:0] oh_s, ow_s, kh_s, kw_s, ih, iw;
  logic              pad;
  logic [15:0]       feat_addr, act_addr, w_addr, out_addr;
  logic [1:0]        in_grp;

  assign oh_s = 8'(oh);
  assign ow_s = 8'(ow);
  assign kh_s = 8'(kh);
  assign kw_s = 8'(kw);

  always_comb begin
    unique case (kind)
      L_STEM: begin  // 10x4 kernel, stride 2, padding (5, 1)
        ih  = 2 * oh_s + kh_s - 8'sd5;
        iw  = 2 * ow_s + kw_s - 8'sd1;
        pad = ih < 0 || ih >= IN_H || iw < 0 || iw >= IN_W;
      end
      L_DW: begin    // 3x3 kernel, stride 1, padding 1
        ih  = oh_s + kh_s - 8'sd1;
        iw  = ow_s + kw_s - 8'sd1;
        pad = ih < 0 || ih >= OUT_H || iw < 0 || iw >= OUT_W;
      end
      default: begin
        ih  = oh_s;
        iw  = ow_s;
        pad = 1'b0;
      end
    endcase
    in_grp    = kind == L_PW ? tap[5:4] : grp;
    feat_addr = 16'(ih * IN_W + iw);
    act_addr  = 16'((ih * OUT_W + iw) * GROUPS + in_grp);
    w_addr    = 16'(LAYER_WBASE[layer] + grp * (last_tap + 1) + tap);
    out_addr  = 16'((oh * OUT_W + ow) * GROUPS + grp);
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
  // Bias and shift ROMs are addressed by (layer, group) and stay stable while a group runs.
  sdp_ram #(.WIDTH(32 * LANES), .DEPTH(N_LAYERS * GROUPS), .INIT({MEM_DIR, "bias.hex"})) u_bias (
    .clk, .we(1'b0), .waddr('0), .wdata('0),
    .raddr(BS_AW'(layer * GROUPS + grp)), .rdata(bias_rdata));
  sdp_ram #(.WIDTH(4 * LANES), .DEPTH(N_LAYERS * GROUPS), .INIT({MEM_DIR, "shift.hex"})) u_shift (
    .clk, .we(1'b0), .waddr('0), .wdata('0),
    .raddr(BS_AW'(layer * GROUPS + grp)), .rdata(shift_rdata));
  sdp_ram #(.WIDTH(8), .DEPTH(N_CLASSES * CHANNELS), .INIT({MEM_DIR, "fc_w.hex"})) u_fcw (
    .clk, .we(1'b0), .waddr('0), .wdata('0), .raddr(fcw_raddr), .rdata(fcw_rdata));
  sdp_ram #(.WIDTH(32), .DEPTH(N_CLASSES), .INIT({MEM_DIR, "fc_b.hex"})) u_fcb (
    .clk, .we(1'b0), .waddr('0), .wdata('0), .raddr(fcb_raddr), .rdata(fcb_rdata));

  // ---------------------------------------------------------------------------------------
  // Conv pipeline
  // ---------------------------------------------------------------------------------------
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
  logic              s5_valid;
  logic [ACT_AW-1:0] s5_addr;
  logic              s6_valid;
  logic [ACT_AW-1:0] s6_addr;

  logic signed [8:0]  s3_a    [LANES];  // activation: int8 (stem input) or uint8
  logic signed [7:0]  s3_w    [LANES];
  logic signed [16:0] s4_prod [LANES];
  logic signed [31:0] acc     [LANES];
  logic signed [31:0] s5_sum  [LANES];  // accumulator + rounding constant
  logic [7:0]         s6_y    [LANES];

  logic signed [31:0] bias    [LANES];
  logic [3:0]         shift   [LANES];
  logic signed [31:0] acc_next[LANES];

  always_comb begin
    for (int i = 0; i < LANES; i++) begin
      bias[i]     = $signed(bias_rdata[32*i +: 32]);
      shift[i]    = shift_rdata[4*i +: 4];
      acc_next[i] = (s4_first ? bias[i] : acc[i]) + 32'(s4_prod[i]);
    end
  end

  always_ff @(posedge clk) begin
    // S0 -> S1
    s1_valid   <= state == S_RUN;
    s1_first   <= tap == 0;
    s1_last    <= tap == last_tap;
    s1_pad     <= pad;
    s1_lane    <= tap[3:0];
    s1_addr    <= ACT_AW'(out_addr);
    feat_raddr <= FEAT_AW'(feat_addr);
    act_raddr  <= ACT_AW'(act_addr);
    wrom_raddr <= WROM_AW'(w_addr);

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

    // S4 -> S5: accumulate; on the last tap hand the sum (+ rounding constant) to requant
    s5_valid <= s4_valid && s4_last;
    s5_addr  <= s4_addr;
    for (int i = 0; i < LANES; i++) begin
      if (s4_valid) acc[i] <= acc_next[i];
      s5_sum[i] <= acc_next[i] + (32'sd1 <<< (shift[i] - 4'd1));
    end

    // S5 -> S6: shift, ReLU, saturate to uint8
    s6_valid <= s5_valid;
    s6_addr  <= s5_addr;
    for (int i = 0; i < LANES; i++) begin
      logic signed [31:0] y;
      y = s5_sum[i] >>> shift[i];
      s6_y[i] <= y < 0 ? 8'd0 : y > 255 ? 8'd255 : y[7:0];
    end

    if (rst) {s1_valid, s2_valid, s3_valid, s4_valid, s5_valid, s6_valid} <= '0;
  end

  // S6: write the output word
  always_comb begin
    for (int i = 0; i < LANES; i++) act_wdata[8*i +: 8] = s6_y[i];
  end
  assign act_waddr = s6_addr;
  assign act_a_we  = s6_valid && !layer[0];
  assign act_b_we  = s6_valid &&  layer[0];

  // Global average pool, as a plain sum of the last layer's outputs (max 500 * 255 < 2^17).
  logic [16:0] pooled [GROUPS][LANES];

  always_ff @(posedge clk) begin
    if (state == S_IDLE && start) begin
      for (int g = 0; g < GROUPS; g++)
        for (int i = 0; i < LANES; i++) pooled[g][i] <= '0;
    end else if (s6_valid && layer == LAST) begin
      for (int i = 0; i < LANES; i++) pooled[grp][i] <= pooled[grp][i] + 17'(s6_y[i]);
    end
  end

  wire conv_pipe_empty = !(s1_valid || s2_valid || s3_valid || s4_valid || s5_valid || s6_valid);

  // ---------------------------------------------------------------------------------------
  // FC: logits[k] = fc_b[k] + sum_c fc_w[k][c] * pooled[c], then argmax.
  // F0 issue -> F1 ROM data, multiply -> F2 accumulate
  // ---------------------------------------------------------------------------------------
  logic [3:0]         fk;
  logic [5:0]         fc;
  logic               f1_valid, f1_first, f1_last;
  logic [3:0]         f1_k;
  logic [16:0]        f1_pool;
  logic               f2_valid, f2_first, f2_last;
  logic [3:0]         f2_k;
  logic signed [31:0] f2_bias;
  logic signed [25:0] f2_prod;
  logic signed [31:0] facc, facc_next, best;

  assign fcw_raddr = FCW_AW'(fk * CHANNELS + fc);
  assign fcb_raddr = fk;
  assign facc_next = (f2_first ? f2_bias : facc) + 32'(f2_prod);

  always_ff @(posedge clk) begin
    f1_valid <= state == S_FC;
    f1_first <= fc == 0;
    f1_last  <= fc == CHANNELS - 1;
    f1_k     <= fk;
    f1_pool  <= pooled[fc[5:4]][fc[3:0]];

    {f2_valid, f2_first, f2_last, f2_k} <= {f1_valid, f1_first, f1_last, f1_k};
    f2_bias <= $signed(fcb_rdata);
    f2_prod <= $signed(fcw_rdata) * $signed({1'b0, f1_pool});

    if (f2_valid) begin
      facc <= facc_next;
      if (f2_last) begin
        logits[f2_k] <= facc_next;
        if (f2_k == 0 || facc_next > best) begin  // strict > keeps the first max, like argmax
          best      <= facc_next;
          class_idx <= f2_k;
        end
      end
    end

    if (rst) {f1_valid, f2_valid} <= '0;
  end

  wire fc_pipe_empty = !(f1_valid || f2_valid);

  // ---------------------------------------------------------------------------------------
  // Control
  // ---------------------------------------------------------------------------------------
  always_ff @(posedge clk) begin
    layer_done <= 1'b0;
    done       <= 1'b0;

    unique case (state)
      S_IDLE: if (start) begin
        {layer, grp, oh, ow, tap, kh, kw} <= '0;
        state <= S_SETUP;
      end

      S_SETUP: state <= S_RUN;  // bias/shift ROM outputs update for the new (layer, group)

      S_RUN: begin
        if (tap == last_tap) begin
          {tap, kh, kw} <= '0;
          if (ow == 5'(OUT_W - 1)) begin
            ow <= '0;
            if (oh == 5'(OUT_H - 1)) begin
              oh    <= '0;
              state <= S_DRAIN;
            end else oh <= oh + 1'b1;
          end else ow <= ow + 1'b1;
        end else begin
          tap <= tap + 1'b1;
          if (kw == last_kw) begin
            kw <= '0;
            kh <= kh + 1'b1;
          end else kw <= kw + 1'b1;
        end
      end

      S_DRAIN: if (conv_pipe_empty) begin
        if (grp == 2'(GROUPS - 1)) begin
          layer_done <= 1'b1;
          grp        <= '0;
          if (layer == 4'(LAST)) begin
            {fk, fc} <= '0;
            state    <= S_FC;
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
        if (fc == 6'(CHANNELS - 1)) begin
          fc <= '0;
          if (fk == 4'(N_CLASSES - 1)) state <= S_FC_DRAIN;
          else fk <= fk + 1'b1;
        end else fc <= fc + 1'b1;
      end

      S_FC_DRAIN: if (fc_pipe_empty) state <= S_DONE;

      S_DONE: begin
        done  <= 1'b1;
        state <= S_IDLE;
      end

      default: state <= S_IDLE;
    endcase

    if (rst) state <= S_IDLE;
  end

  assign busy = state != S_IDLE;
endmodule
