// DS-CNN-S streaming keyword-spotting engine.
//
// Takes feature frames (N_MELS int8 values each, one every 20 ms) and, every second frame,
// computes one new row of every layer instead of re-running the whole 1 s window: the model
// has no padding along time (python/kws/model.py), so the rows it already computed stay
// valid. Each such step then updates the global average pool and the logits. Bit-exact to
// python/kws/quant.py stream_forward; about 25x fewer MACs per result than the full window.
//
// Per step j (stem row j, from frames 2j .. 2j+9):
//   STEM   row j of layer 0 from the frame ring                     -> act ring 0
//   block b = 1..4:
//     DW   row j-2b from rows j-2b .. j-2b+2 of act ring b-1        -> dw row buffer
//     PW   the dw row                                               -> act ring b (b < 4)
//   POOL   channel sums of the last layer's row, then FC on them: this row's partial logits
//   SUM    logits = fc_b + the partial logits of the last POOL_ROWS rows, argmax, margin
// Act rings keep 4 rows (the depthwise layers need 3). After FULL_STEPS steps from a restart
// the logits cover exactly one IN_H-frame window; before that they are not meaningful.
//
// Conv datapath: LANES parallel MACs, each producing one output channel of the current
// channel group. Per (layer, group) it walks the OUT_W pixels of the row and every tap:
//   STEM  40 taps (10x4, stride 2), one int8 input value broadcast to all lanes
//   DW     9 taps (3x3), lane i reads channel i of the input word
//   PW    64 taps (one per input channel), that channel's value broadcast to all lanes
// The pipeline drains between groups so bias/shift can change safely.
//
// Pipeline:  S0 issue -> C coordinates -> S1 mem addr -> S2 mem data, operand select
//            -> S3 multiply -> S4 accumulate -> S5 shift -> S5b clamp -> S6 write (+ pool)
//
// Power: every memory reads only when its data is used, so an idle engine (about 93% of
// the time in live mode at 10 MHz) does not toggle.

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
  input  logic               restart,     // start a new stream: forget earlier frames and steps
  // Feature frames: N_MELS writes (band 0..N_MELS-1), then a frame_end pulse.
  input  logic               feat_we,
  input  logic [5:0]         feat_band,
  input  logic [7:0]         feat_wdata,
  input  logic               frame_end,
  output logic               busy,
  output logic               done,        // one-cycle pulse per step; outputs below are then valid
  output logic               full,        // with done: the logits cover a whole IN_H-frame window
  output logic [3:0]         class_idx,
  output logic signed [31:0] margin,      // top logit minus the runner-up
  output logic [7:0]         dbg_feat_or, // debug: OR of all feature bytes the stem layer read
  output logic signed [31:0] logits [N_CLASSES]
);
  localparam int WROM_AW = $clog2(WROM_WORDS);
  localparam int RING_AW = 12;             // frame ring: 64 frames x N_MELS
  localparam int ACT_AW  = 9;              // act ring: {row slot (4), pixel (32), group (4)}
  localparam int DW_AW   = 7;              // dw row: {pixel (32), group (4)}
  localparam int FCW_AW  = $clog2(N_CLASSES * CHANNELS);
  localparam int LAST    = N_LAYERS - 1;

  typedef enum logic [3:0] {
    S_IDLE, S_SETUP, S_SETUP2, S_RUN, S_DRAIN, S_FC, S_FC_DRAIN, S_SUM, S_SUM_DRAIN, S_DONE
  } state_t;

  state_t state;

  // ---------------------------------------------------------------------------------------
  // Frame ring and step scheduling. A step can run once its 10 stem frames are in, i.e.
  // after frame 10, 12, 14, ... of the stream. Steps queue up while the engine is busy
  // (UART mode sends a whole window at once; live mode never waits).
  // ---------------------------------------------------------------------------------------
  logic [5:0] wslot;       // ring slot of the frame being written
  logic [3:0] nframes;     // frames since restart, saturating at STEM_KH
  logic       odd;         // frames since restart is odd
  logic [5:0] pending;     // steps ready to run
  logic [5:0] next_base;   // ring slot of the next step's first stem frame
  logic [5:0] base;        // ... of the running step
  logic [4:0] nsteps;      // steps started since restart, saturating at FULL_STEPS
  logic       step_full;   // the running step completes a window
  logic [3:0] row;         // step counter, mod 16: picks act ring and partial logit slots
  logic       add_step, take_step;

  // A frame completes a step's input when the count after it is even and >= STEM_KH.
  assign add_step  = frame_end && odd && nframes >= 4'(STEM_KH - 1);
  assign take_step = state == S_IDLE && pending != 0 && !restart;

  always_ff @(posedge clk) begin
    if (frame_end) begin
      wslot <= wslot + 1'b1;
      odd   <= !odd;
      if (nframes != 4'(STEM_KH)) nframes <= nframes + 1'b1;
    end
    pending <= pending + 6'(add_step) - 6'(take_step);
    if (take_step) begin
      base      <= next_base;
      next_base <= next_base + 6'd2;
      step_full <= nsteps >= 5'(FULL_STEPS - 1);
      if (nsteps != 5'(FULL_STEPS)) nsteps <= nsteps + 1'b1;
    end
    if (rst || restart) begin
      wslot     <= '0;
      odd       <= 1'b0;
      nframes   <= '0;
      pending   <= '0;
      next_base <= '0;
      nsteps    <= '0;
    end
  end

  // Ring address slot * 40 + band, written as shifts: a multiply here would go onto a
  // combinational DSP48, which computed wrong addresses on the board with the open flow.
  function automatic logic [RING_AW-1:0] ring_addr(input logic [5:0] slot, input logic [5:0] band);
    return {1'b0, slot, 5'b0} + {3'b0, slot, 3'b0} + RING_AW'(band);
  endfunction

  // ---------------------------------------------------------------------------------------
  // Loop counters
  // ---------------------------------------------------------------------------------------
  logic [3:0] layer;
  logic [1:0] grp;
  logic [4:0] ow;       // output pixel within the row
  logic [5:0] tap;      // weight index within (layer, group): kh*KW + kw, or input channel
  logic [3:0] kh;
  logic [1:0] kw;
  logic       layer_done;  // pulse after each layer fully written (testbench hook)
  logic [WROM_AW-1:0]                 w_base;   // weight ROM word of tap 0 for (layer, group)
  logic [$clog2(N_LAYERS*GROUPS)-1:0] bs_addr;  // bias/shift ROM word for (layer, group)

  // Per-layer constants, registered in S_SETUP to keep the table lookup off the counter paths.
  //   ring   act ring the layer reads (DW) or writes (STEM, PW)
  //   rslot  ring slot of the row it writes (STEM, PW) or of its first input row (DW):
  //          row j - 2*ceil(layer/2), mod 4
  layer_kind_t kind;
  logic [5:0]  last_tap;
  logic [1:0]  last_kw;
  logic [2:0]  ring;
  logic [1:0]  rslot;

  always_ff @(posedge clk) begin
    if (state == S_SETUP) begin
      logic [3:0] half;
      half  = (layer + 4'd1) >> 1;
      kind  <= LAYER_KIND[layer];
      ring  <= 3'(layer >> 1);
      rslot <= row[1:0] ^ {half[0], 1'b0};
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
  logic signed [7:0] iw;
  logic [5:0]        islot;   // frame ring slot (STEM) or act ring slot (DW) of the input row
  logic [1:0]        in_grp;

  always_comb begin
    unique case (kind)
      L_STEM: begin  // 10x4 kernel, stride 2, padding 1 along frequency only
        iw    = 8'(2 * ow) + 8'(kw) - 8'sd1;
        islot = base + 6'(kh);
      end
      L_DW: begin    // 3x3 kernel, stride 1, padding 1 along frequency only
        iw    = 8'(ow) + 8'(kw) - 8'sd1;
        islot = 6'(rslot + kh[1:0]);
      end
      default: begin
        iw    = 8'(ow);
        islot = '0;
      end
    endcase
    in_grp = kind == L_PW ? tap[5:4] : grp;
  end

  // ---------------------------------------------------------------------------------------
  // Memories
  // ---------------------------------------------------------------------------------------
  logic [RING_AW-1:0]  feat_raddr;
  logic [7:0]          feat_rdata;
  logic [ACT_AW-1:0]   act_raddr, act_waddr;
  logic [8*LANES-1:0]  act_rdata [4];
  logic [3:0]          act_re, act_we;
  logic [DW_AW-1:0]    dw_raddr, dw_waddr;
  logic [8*LANES-1:0]  dw_rdata, wdata;
  logic                dw_re, dw_we, feat_re;
  logic [WROM_AW-1:0]  wrom_raddr;
  logic [8*LANES-1:0]  wrom_rdata;
  logic [32*LANES-1:0] bias_rdata;
  logic [4*LANES-1:0]  shift_rdata;
  logic [FCW_AW-1:0]   fcw_raddr;
  logic [7:0]          fcw_rdata;
  logic [3:0]          fcb_raddr;
  logic [31:0]         fcb_rdata;
  logic [7:0]          part_raddr, part_waddr;
  logic signed [31:0]  part_rdata, part_wdata;
  logic                part_we;
  logic                s1_valid;

  // LUT RAM: as a written RAMB36 it read back zeros on the board with the open flow.
  sdp_ram #(.WIDTH(8), .DEPTH(64 * N_MELS), .STYLE("distributed")) u_ring (
    .clk, .we(feat_we), .waddr(ring_addr(wslot, feat_band)), .wdata(feat_wdata),
    .re(feat_re), .raddr(feat_raddr), .rdata(feat_rdata));

  // Four rings of 4 rows, one memory each (512 x 128 bits, as 32-bit slices: RAMB18s).
  sdp_ram_sliced #(.WIDTH(8 * LANES), .DEPTH(512)) u_act0 (
    .clk, .we(act_we[0]), .waddr(act_waddr), .wdata(wdata),
    .re(act_re[0]), .raddr(act_raddr), .rdata(act_rdata[0]));
  sdp_ram_sliced #(.WIDTH(8 * LANES), .DEPTH(512)) u_act1 (
    .clk, .we(act_we[1]), .waddr(act_waddr), .wdata(wdata),
    .re(act_re[1]), .raddr(act_raddr), .rdata(act_rdata[1]));
  sdp_ram_sliced #(.WIDTH(8 * LANES), .DEPTH(512)) u_act2 (
    .clk, .we(act_we[2]), .waddr(act_waddr), .wdata(wdata),
    .re(act_re[2]), .raddr(act_raddr), .rdata(act_rdata[2]));
  sdp_ram_sliced #(.WIDTH(8 * LANES), .DEPTH(512)) u_act3 (
    .clk, .we(act_we[3]), .waddr(act_waddr), .wdata(wdata),
    .re(act_re[3]), .raddr(act_raddr), .rdata(act_rdata[3]));
  sdp_ram_sliced #(.WIDTH(8 * LANES), .DEPTH(128)) u_dw (
    .clk, .we(dw_we), .waddr(dw_waddr), .wdata(wdata),
    .re(dw_re), .raddr(dw_raddr), .rdata(dw_rdata));

  sdp_ram #(.WIDTH(8 * LANES), .DEPTH(WROM_WORDS), .INIT({MEM_DIR, "weights.hex"})) u_wrom (
    .clk, .we(1'b0), .waddr('0), .wdata('0),
    .re(s1_valid), .raddr(wrom_raddr), .rdata(wrom_rdata));
  // Bias and shift ROMs: one word per (layer, group), read during S_SETUP.
  sdp_ram #(.WIDTH(32 * LANES), .DEPTH(N_LAYERS * GROUPS), .INIT({MEM_DIR, "bias.hex"}), .STYLE("block")) u_bias (
    .clk, .we(1'b0), .waddr('0), .wdata('0),
    .re(state == S_SETUP), .raddr(bs_addr), .rdata(bias_rdata));
  sdp_ram #(.WIDTH(4 * LANES), .DEPTH(N_LAYERS * GROUPS), .INIT({MEM_DIR, "shift.hex"}), .STYLE("block")) u_shift (
    .clk, .we(1'b0), .waddr('0), .wdata('0),
    .re(state == S_SETUP), .raddr(bs_addr), .rdata(shift_rdata));
  sdp_ram #(.WIDTH(8), .DEPTH(N_CLASSES * CHANNELS), .INIT({MEM_DIR, "fc_w.hex"}), .STYLE("block")) u_fcw (
    .clk, .we(1'b0), .waddr('0), .wdata('0),
    .re(state == S_FC), .raddr(fcw_raddr), .rdata(fcw_rdata));
  sdp_ram #(.WIDTH(32), .DEPTH(N_CLASSES), .INIT({MEM_DIR, "fc_b.hex"})) u_fcb (
    .clk, .we(1'b0), .waddr('0), .wdata('0),
    .re(state == S_SUM), .raddr(fcb_raddr), .rdata(fcb_rdata));
  // Partial logits of the last 16 rows: {row mod 16, class}.
  sdp_ram #(.WIDTH(32), .DEPTH(256)) u_part (
    .clk, .we(part_we), .waddr(part_waddr), .wdata(part_wdata),
    .re(state == S_SUM), .raddr(part_raddr), .rdata(part_rdata));

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
  logic signed [7:0] c_w_lim;  // input width for the padding check
  logic [3:0]        c_lane;
  logic signed [7:0] c_iw;
  logic [5:0]        c_slot;
  logic [1:0]        c_in_grp;
  logic [5:0]        c_tap;
  logic [4:0]        c_ow;
  logic              s1_first, s1_last, s1_pad;
  logic [3:0]        s1_lane;
  logic [4:0]        s1_ow;
  logic              s2_valid, s2_first, s2_last, s2_pad;
  logic [3:0]        s2_lane;
  logic [4:0]        s2_ow;
  logic              s3_valid, s3_first, s3_last;
  logic [4:0]        s3_ow;
  logic              s4_valid, s4_first, s4_last;
  logic [4:0]        s4_ow;
  logic              s5_valid, s5b_valid;
  logic [4:0]        s5_ow, s5b_ow;
  logic              s6_valid;
  logic [4:0]        s6_ow;

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

  // Reads (S1 -> S2) only from the memory the layer uses.
  assign feat_re = s1_valid && kind == L_STEM;
  assign dw_re   = s1_valid && kind == L_PW;
  always_comb for (int r = 0; r < 4; r++) act_re[r] = s1_valid && kind == L_DW && ring == 3'(r);

  logic [7:0] feat_or;
  always_ff @(posedge clk) begin
    if (take_step) feat_or <= '0;
    else if (s2_valid && kind == L_STEM) feat_or <= feat_or | feat_rdata;
    if (state == S_DONE) dbg_feat_or <= feat_or;
  end

  always_ff @(posedge clk) begin
    // S0 -> C: register coordinates
    c_valid  <= state == S_RUN;
    c_first  <= tap == 0;
    c_last   <= tap == last_tap;
    c_w_lim  <= kind == L_STEM ? 8'(N_MELS) : 8'(OUT_W);
    c_lane   <= tap[3:0];
    c_iw     <= iw;
    c_slot   <= islot;
    c_in_grp <= in_grp;
    c_tap    <= tap;
    c_ow     <= ow;

    // C -> S1: padding and memory addresses (PW coordinates are always in range)
    {s1_valid, s1_first, s1_last, s1_lane, s1_ow} <= {c_valid, c_first, c_last, c_lane, c_ow};
    s1_pad     <= c_iw < 0 || c_iw >= c_w_lim;
    feat_raddr <= ring_addr(c_slot, c_iw[5:0]);
    act_raddr  <= {c_slot[1:0], c_iw[4:0], c_in_grp};
    dw_raddr   <= {c_iw[4:0], c_in_grp};
    wrom_raddr <= w_base + WROM_AW'(c_tap);

    // S1 -> S2 (memories register their outputs)
    {s2_valid, s2_first, s2_last, s2_pad, s2_lane, s2_ow} <=
        {s1_valid, s1_first, s1_last, s1_pad, s1_lane, s1_ow};

    // S2 -> S3: select operands
    {s3_valid, s3_first, s3_last, s3_ow} <= {s2_valid, s2_first, s2_last, s2_ow};
    for (int i = 0; i < LANES; i++) begin
      logic signed [8:0] a;
      unique case (kind)
        L_STEM:  a = 9'($signed(feat_rdata));
        L_DW:    a = {1'b0, act_rdata[ring[1:0]][8*i +: 8]};
        default: a = {1'b0, dw_rdata[8*s2_lane +: 8]};
      endcase
      if (s2_valid) begin
        s3_a[i] <= s2_pad ? '0 : a;
        s3_w[i] <= $signed(wrom_rdata[8*i +: 8]);
      end
    end

    // S3 -> S4: multiply
    {s4_valid, s4_first, s4_last, s4_ow} <= {s3_valid, s3_first, s3_last, s3_ow};
    for (int i = 0; i < LANES; i++) if (s3_valid) s4_prod[i] <= s3_a[i] * s3_w[i];

    // S4 -> S5: accumulate; on the last tap hand the sum to requant
    s5_valid <= s4_valid && s4_last;
    s5_ow    <= s4_ow;
    for (int i = 0; i < LANES; i++) begin
      if (s4_valid) acc[i] <= acc_next[i];
      if (s4_valid && s4_last) s5_sum[i] <= acc_next[i];
    end

    // S5 -> S5b: shift
    s5b_valid <= s5_valid;
    s5b_ow    <= s5_ow;
    for (int i = 0; i < LANES; i++) if (s5_valid) s5b_y[i] <= s5_sum[i] >>> shift[i];

    // S5b -> S6: ReLU, saturate to uint8
    s6_valid <= s5b_valid;
    s6_ow    <= s5b_ow;
    for (int i = 0; i < LANES; i++)
      if (s5b_valid) s6_y[i] <= s5b_y[i] < 0 ? 8'd0 : s5b_y[i] > 255 ? 8'd255 : s5b_y[i][7:0];

    if (rst) {c_valid, s1_valid, s2_valid, s3_valid, s4_valid, s5_valid, s5b_valid, s6_valid} <= '0;
  end

  // S6: write the output word. STEM and PW write a row of their act ring, DW the dw row.
  always_comb begin
    for (int i = 0; i < LANES; i++) wdata[8*i +: 8] = s6_y[i];
  end
  assign act_waddr = {rslot, s6_ow, grp};
  assign dw_waddr  = {s6_ow, grp};
  assign dw_we     = s6_valid && kind == L_DW;
  always_comb for (int r = 0; r < 4; r++) act_we[r] = s6_valid && kind != L_DW && ring == 3'(r);

  wire conv_pipe_empty = !(c_valid || s1_valid || s2_valid || s3_valid || s4_valid || s5_valid ||
                             s5b_valid || s6_valid);

  // Pool: the channel sums of the last layer's new row (max 20 * 255 < 2^13). Each lane sums
  // its channel for the current group; the sums are stored per group once the group has
  // drained, which keeps the group select out of the adder path.
  logic [12:0]       pool_acc [LANES];
  logic [12:0]       pooled   [GROUPS][LANES];
  logic [GROUPS-1:0] pool_we;

  always_ff @(posedge clk) begin
    if (state == S_SETUP) begin
      for (int i = 0; i < LANES; i++) pool_acc[i] <= '0;
    end else if (s6_valid) begin
      for (int i = 0; i < LANES; i++) pool_acc[i] <= pool_acc[i] + 13'(s6_y[i]);
    end
    // Store one cycle after the group drains (a registered, one-hot enable: it fans out to
    // every pooled register). pool_acc is only cleared in the S_SETUP that follows.
    for (int g = 0; g < GROUPS; g++)
      pool_we[g] <= state == S_DRAIN && conv_pipe_empty && layer == LAST && grp == 2'(g);
    for (int g = 0; g < GROUPS; g++)
      if (pool_we[g]) for (int i = 0; i < LANES; i++) pooled[g][i] <= pool_acc[i];
  end

  // ---------------------------------------------------------------------------------------
  // FC on the row: part[k] = sum_c fc_w[k][c] * pooled[c], stored for this row.
  // F0 issue -> F1 ROM data, pick the group of 16 pooled sums -> F2 pick the lane
  // -> F3 multiply -> F4 accumulate. (The 64:1 pooled mux in one cycle was too slow.)
  // ---------------------------------------------------------------------------------------
  logic [3:0]         fk;
  logic [5:0]         fc;
  logic [FCW_AW-1:0]  fcw_addr;  // fk * CHANNELS + fc
  logic               f1_valid, f1_first, f1_last;
  logic [3:0]         f1_k, f1_lane;
  logic [12:0]        f1_group [LANES];
  logic               f2_valid, f2_first, f2_last;
  logic [3:0]         f2_k;
  logic [12:0]        f2_pool;
  logic signed [7:0]  f2_w;
  logic               f3_valid, f3_first, f3_last;
  logic [3:0]         f3_k;
  logic signed [21:0] f3_prod;
  logic signed [31:0] facc, facc_next;

  assign fcw_raddr  = fcw_addr;
  assign facc_next  = (f3_first ? 32'sd0 : facc) + 32'(f3_prod);
  assign part_we    = f3_valid && f3_last;
  assign part_waddr = {row, f3_k};
  assign part_wdata = facc_next;

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

    {f3_valid, f3_first, f3_last, f3_k} <= {f2_valid, f2_first, f2_last, f2_k};
    f3_prod <= f2_w * $signed({1'b0, f2_pool});

    if (f3_valid) facc <= facc_next;

    if (rst) {f1_valid, f2_valid, f3_valid} <= '0;
  end

  wire fc_pipe_empty = !(f1_valid || f2_valid || f3_valid);

  // ---------------------------------------------------------------------------------------
  // SUM: logits[k] = fc_b[k] + the last POOL_ROWS rows' part[k], then argmax.
  // U0 issue (ROM and partial RAM addresses) -> U1 data, accumulate.
  // ---------------------------------------------------------------------------------------
  logic [3:0]         sk, sr;
  logic               u1_valid, u1_first, u1_last;
  logic [3:0]         u1_k;
  logic signed [31:0] sacc, sacc_next, best, second;

  assign fcb_raddr  = sk;
  assign part_raddr = {row - sr, sk};
  assign sacc_next  = (u1_first ? $signed(fcb_rdata) : sacc) + part_rdata;

  always_ff @(posedge clk) begin
    u1_valid <= state == S_SUM;
    u1_first <= sr == 0;
    u1_last  <= sr == 4'(POOL_ROWS - 1);
    u1_k     <= sk;

    if (u1_valid) begin
      sacc <= sacc_next;
      if (u1_last) begin
        logits[u1_k] <= sacc_next;
        // Strict > keeps the first max, like argmax. Also track the runner-up for `margin`.
        if (u1_k == 0) begin
          best      <= sacc_next;
          second    <= 32'sh8000_0000;
          class_idx <= u1_k;
        end else if (sacc_next > best) begin
          best      <= sacc_next;
          second    <= best;
          class_idx <= u1_k;
        end else if (sacc_next > second) begin
          second <= sacc_next;
        end
      end
    end

    if (rst) u1_valid <= 1'b0;
  end

  // ---------------------------------------------------------------------------------------
  // Control
  // ---------------------------------------------------------------------------------------
  always_ff @(posedge clk) begin
    layer_done <= 1'b0;
    done       <= 1'b0;

    unique case (state)
      S_IDLE: if (take_step) begin
        {layer, grp, ow, tap, kh, kw} <= '0;
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
            ow    <= '0;
            state <= S_DRAIN;
          end else begin
            ow <= ow + 1'b1;
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

      S_FC_DRAIN: if (fc_pipe_empty) begin
        {sk, sr} <= '0;
        state    <= S_SUM;
      end

      S_SUM: begin
        if (sr == 4'(POOL_ROWS - 1)) begin
          sr <= '0;
          if (sk == 4'(N_CLASSES - 1)) state <= S_SUM_DRAIN;
          else sk <= sk + 1'b1;
        end else sr <= sr + 1'b1;
      end

      S_SUM_DRAIN: if (!u1_valid) state <= S_DONE;

      S_DONE: begin
        done   <= 1'b1;
        full   <= step_full;
        margin <= best - second;
        row    <= row + 1'b1;
        state  <= S_IDLE;
      end

      default: state <= S_IDLE;
    endcase

    if (rst) begin
      state <= S_IDLE;
      row   <= '0;
    end
  end

  assign busy = state != S_IDLE;
endmodule
