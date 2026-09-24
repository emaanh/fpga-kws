// Nexys A7-100T top level. Two modes, picked with SW15:
//
// SW15 = 0: UART feature mode (8N1 at 1 Mbaud)
//   host -> board  'I' followed by IN_H*IN_W int8 feature bytes (row-major, h*IN_W + w)
//   board -> host  'R', class index, then the N_CLASSES int32 logits, little-endian
//   A frame that stalls for RX_TIMEOUT cycles is dropped, so the host can always resync.
//   LEDs 11:0 one-hot class. The display shows the last keyword (silence/unknown show nothing).
//
// SW15 = 1: live mode, from the on-board PDM microphone (kws_live.sv)
//   SW3:0 mic gain in 6 dB steps (0..12), SW13 samples the mic on the falling clock edge.
//   SW5:4 detection margin: 00 = 2^18 (default), 01 = 2^17, 10 = 2^19, 11 = off.
//   The display shows a detected word for 2 s. LEDs 11:0 are a mic level meter.
//   SW12 takes the audio from the host instead of the mic, 3 bytes per sample as below
//   (python -m kws.inject), for testing the whole live path on the board.
//   SW14 streams the mic's 16 kHz PCM to the host, 3 bytes per sample:
//     {1, 5'b0, x[15:14]}, {0, x[13:7]}, {0, x[6:0]}   (bit 7 marks the first byte)
//   With SW14 off it sends a debug packet per inference instead (python -m kws.debug_live):
//     'K', class, margin (int32, little-endian), {detected, 3'b0, mic level},
//     frontend: OR of windowed magnitudes (4 B), OR of powers (7 B), features per frame,
//     copy into the engine: bytes written (2 B), their OR, OR of what the engine read,
//     then the last 40 features
//
// Both modes: LED 13 live mode, 14 engine busy, 15 heartbeat.

`ifndef KWS_MEM_DIR
`define KWS_MEM_DIR ""
`endif

module kws_top
  import kws_pkg::*;
#(
  parameter string MEM_DIR      = `KWS_MEM_DIR,
  // The design runs at the 100 MHz board clock divided by SYS_DIV (50 MHz): the open-source
  // flow could not place it reliably at 100 MHz. The constants below are in system clocks.
  parameter int    SYS_DIV      = 2,
  parameter int    CLKS_PER_BIT = 50,          // 1 Mbaud
  parameter int    RX_TIMEOUT   = 2_500_000,   // 50 ms
  parameter int    PDM_PERIOD   = 25,          // 2 MHz mic clock
  parameter int    HOLD_CYCLES  = 100_000_000  // 2 s
) (
  input  logic        clk,      // 100 MHz board clock
  input  logic        rst_n,    // CPU_RESETN button
  input  logic        uart_rx,  // from host (UART_TXD_IN)
  output logic        uart_tx,  // to host (UART_RXD_OUT)
  input  logic [15:0] sw,
  output logic        m_clk,
  output logic        m_lrsel,
  input  logic        m_data,
  output logic [15:0] led,
  output logic [6:0]  seg,
  output logic        dp,
  output logic [7:0]  an
);
  localparam int N_FEAT  = IN_H * IN_W;
  localparam int N_REPLY = 2 + 4 * N_CLASSES;

  // System clock: the board clock divided by SYS_DIV, through a global buffer on the FPGA.
  logic sys_clk;
  if (SYS_DIV == 1) begin : g_nodiv
    assign sys_clk = clk;
  end else begin : g_div
    logic [$clog2(SYS_DIV)-1:0] div_cnt = '0;
    logic                       div_clk = 1'b0;
    always_ff @(posedge clk) begin
      div_cnt <= div_cnt == ($clog2(SYS_DIV))'(SYS_DIV - 1) ? '0 : div_cnt + 1'b1;
      if (div_cnt == '0 || div_cnt == ($clog2(SYS_DIV))'(SYS_DIV / 2)) div_clk <= ~div_clk;
    end
`ifdef SYNTHESIS
    BUFG u_sys_bufg (.I(div_clk), .O(sys_clk));
`else
    assign sys_clk = div_clk;
`endif
  end

  // Reset: the CPU_RESET button, plus a power-on reset for the first 256 cycles after the
  // bitstream loads (configuration sets por_cnt to 0). Without it the logic starts in
  // whatever state the flip-flops power up in, which on the board needed a manual reset.
  logic [1:0] rst_sync;
  logic [7:0] por_cnt = '0;
  logic       rst;
  always_ff @(posedge sys_clk) begin
    rst_sync <= {rst_sync[0], ~rst_n};
    if (por_cnt != '1) por_cnt <= por_cnt + 1'b1;
  end
  assign rst = rst_sync[1] || por_cnt != '1;

  logic [15:0] sw_meta, sw_sync;
  always_ff @(posedge sys_clk) {sw_sync, sw_meta} <= {sw_meta, sw};

  logic live, record;
  assign live   = sw_sync[15];
  assign record = sw_sync[15] && sw_sync[14];

  // ---------------------------------------------------------------------------------------
  // UART
  // ---------------------------------------------------------------------------------------
  logic       rx_valid, tx_start, tx_busy;
  logic [7:0] rx_data, tx_data;

  uart_rx #(.CLKS_PER_BIT(CLKS_PER_BIT)) u_rx (
    .clk(sys_clk), .rst, .rx(uart_rx), .valid(rx_valid), .data(rx_data));
  uart_tx #(.CLKS_PER_BIT(CLKS_PER_BIT)) u_tx (
    .clk(sys_clk), .rst, .start(tx_start), .data(tx_data), .busy(tx_busy), .tx(uart_tx));

  // ---------------------------------------------------------------------------------------
  // Engine, shared by both modes
  // ---------------------------------------------------------------------------------------
  logic               eng_start, eng_busy, eng_done;
  logic               eng_we;
  logic [10:0]        eng_waddr;
  logic [7:0]         eng_wdata;
  logic [3:0]         class_idx;
  logic signed [31:0] margin;
  logic [7:0]         eng_feat_or;
  logic signed [31:0] logits [N_CLASSES];

  kws_engine #(.MEM_DIR(MEM_DIR)) u_engine (
    .clk(sys_clk), .rst,
    .feat_we(eng_we), .feat_waddr(eng_waddr), .feat_wdata(eng_wdata),
    .start(eng_start), .busy(eng_busy), .done(eng_done),
    .class_idx, .margin, .dbg_feat_or(eng_feat_or), .logits);

  // Debug: feature bytes written into the engine before each start, and their OR.
  logic [10:0] copy_n, copy_n_q;
  logic [7:0]  copy_or, copy_or_q;
  always_ff @(posedge sys_clk) begin
    if (eng_start) begin
      copy_n_q  <= copy_n;
      copy_or_q <= copy_or;
      copy_n    <= '0;
      copy_or   <= '0;
    end else if (eng_we) begin
      copy_n  <= copy_n + 1'b1;
      copy_or <= copy_or | eng_wdata;
    end
  end

  // ---------------------------------------------------------------------------------------
  // Live mode
  // ---------------------------------------------------------------------------------------
  logic               live_we, live_start, detect, showing;
  logic [10:0]        live_waddr;
  logic [7:0]         live_wdata;
  logic [3:0]         show_class, level;
  logic               pcm_valid;
  logic signed [15:0] pcm;
  logic [31:0]        fe_win_or;
  logic [48:0]        fe_pow_or;
  logic [5:0]         fe_nfeat;
  logic [7:0]         fe_row [N_MELS];

  logic signed [31:0] min_margin;
  always_comb begin
    unique case (sw_sync[5:4])
      2'b00: min_margin = 32'sd1 <<< 18;
      2'b01: min_margin = 32'sd1 <<< 17;
      2'b10: min_margin = 32'sd1 <<< 19;
      default: min_margin = '0;
    endcase
  end

  // Host PCM for SW12: {1, 5'b0, x[15:14]}, {0, x[13:7]}, {0, x[6:0]}
  logic               ext_valid;
  logic signed [15:0] ext_pcm;
  logic [1:0]         ext_have;   // bytes of the current sample received
  logic [8:0]         ext_hi;
  always_ff @(posedge sys_clk) begin
    ext_valid <= 1'b0;
    if (rx_valid && live) begin
      if (rx_data[7]) begin
        ext_hi   <= {rx_data[1:0], 7'b0};
        ext_have <= 2'd1;
      end else if (ext_have == 2'd1) begin
        ext_hi   <= {ext_hi[8:7], rx_data[6:0]};
        ext_have <= 2'd2;
      end else if (ext_have == 2'd2) begin
        ext_pcm   <= {ext_hi, rx_data[6:0]};
        ext_valid <= 1'b1;
        ext_have  <= 2'd0;
      end
    end
    if (rst) ext_have <= '0;
  end

  kws_live #(.MEM_DIR(MEM_DIR), .PDM_PERIOD(PDM_PERIOD), .HOLD_CYCLES(HOLD_CYCLES)) u_live (
    .clk(sys_clk), .rst, .enable(live), .gain(sw_sync[3:0]), .sample_fall(sw_sync[13]), .min_margin,
    .ext_en(sw_sync[12]), .ext_valid, .ext_pcm,
    .m_clk, .m_lrsel, .m_data,
    .pcm_valid, .pcm, .level,
    .feat_we(live_we), .feat_waddr(live_waddr), .feat_wdata(live_wdata),
    .eng_start(live_start), .eng_busy, .eng_done, .eng_class(class_idx), .eng_margin(margin),
    .dbg_win_or(fe_win_or), .dbg_pow_or(fe_pow_or), .dbg_nfeat(fe_nfeat), .dbg_row(fe_row),
    .detect, .showing, .show_class);

  // ---------------------------------------------------------------------------------------
  // UART feature mode
  // ---------------------------------------------------------------------------------------
  typedef enum logic [2:0] {P_IDLE, P_RECV, P_RUN, P_PREP, P_SEND} pstate_t;

  pstate_t                         state;
  logic                            uart_start;
  logic [10:0]                     rx_count;
  logic [$clog2(RX_TIMEOUT+1)-1:0] idle_cycles;
  logic [5:0]                      tx_idx;
  logic                            have_result;
  logic                            have_word;   // a keyword has been recognised (UART mode)
  logic [3:0]                      last_word;
  logic                            uart_rx_valid;

  assign uart_rx_valid = rx_valid && !live;

  // Registered mux into the engine: `live` fans out widely. Delaying start and the feature
  // writes by the same cycle keeps start after the last write.
  always_ff @(posedge sys_clk) begin
    eng_we    <= live ? live_we    : state == P_RECV && uart_rx_valid;
    eng_waddr <= live ? live_waddr : rx_count;
    eng_wdata <= live ? live_wdata : rx_data;
    eng_start <= live ? live_start : uart_start;
  end

  function automatic logic [7:0] reply_byte(input logic [5:0] i);
    logic [5:0] j;
    if (i == 0) return "R";
    if (i == 1) return 8'(class_idx);
    j = i - 6'd2;
    return logits[j[5:2]][8*j[1:0] +: 8];
  endfunction

  // The next reply byte, registered: a wide mux over the logits, kept off the UART's path.
  logic [7:0] reply_q;
  always_ff @(posedge sys_clk) reply_q <= reply_byte(tx_idx);

  // Record mode: 3 bytes per PCM sample
  logic [15:0] rec_sample;
  logic [1:0]  rec_left;  // bytes of rec_sample still to send

  // Live debug stream: one packet per inference (see the header), loaded whole and shifted
  // out a byte at a time (a byte mux here was the critical path).
  localparam int DBG_LEN = 23 + N_MELS;
  logic [8*DBG_LEN-1:0] dbg_pkt;
  logic [5:0]           dbg_left;
  logic [3:0]           dbg_wait;   // let the detection decision (a few cycles late) land
  logic [8*N_MELS-1:0]  row_bits;
  always_comb for (int i = 0; i < N_MELS; i++) row_bits[8*i +: 8] = fe_row[i];

  // tx_ok: registered "transmitter free". Every sender below checks it and clears it when
  // it starts a byte; tx_gap covers the two cycles before tx_busy rises.
  logic       tx_ok;
  logic [1:0] tx_gap;

  always_ff @(posedge sys_clk) begin
    uart_start <= 1'b0;
    tx_start   <= 1'b0;
    tx_ok      <= !tx_busy && tx_gap == 0;
    if (tx_gap != 0) tx_gap <= tx_gap - 1'b1;

    unique case (state)
      P_IDLE: if (uart_rx_valid && rx_data == "I") begin
        rx_count    <= '0;
        idle_cycles <= '0;
        state       <= P_RECV;
      end

      P_RECV: begin
        if (uart_rx_valid) begin
          rx_count    <= rx_count + 1'b1;
          idle_cycles <= '0;
          if (rx_count == 11'(N_FEAT - 1)) begin
            uart_start <= 1'b1;
            state      <= P_RUN;
          end
        end else if (idle_cycles == ($bits(idle_cycles))'(RX_TIMEOUT) || live) begin
          state <= P_IDLE;
        end else begin
          idle_cycles <= idle_cycles + 1'b1;
        end
      end

      P_RUN: if (eng_done) begin
        have_result <= 1'b1;
        if (class_idx >= 4'd2) begin  // hold the last keyword; silence/unknown change nothing
          have_word <= 1'b1;
          last_word <= class_idx;
        end
        tx_idx      <= '0;
        state       <= P_PREP;
      end

      P_PREP: state <= P_SEND;  // reply_q catches up with tx_idx

      P_SEND: if (tx_ok) begin
        tx_ok    <= 1'b0;
        tx_gap   <= 2'd3;
        tx_start <= 1'b1;
        tx_data  <= reply_q;
        tx_idx   <= tx_idx + 1'b1;
        if (tx_idx == 6'(N_REPLY - 1)) state <= P_IDLE;
      end
    endcase

    if (record && state == P_IDLE) begin
      if (pcm_valid) begin
        rec_sample <= pcm;
        rec_left   <= 2'd3;
      end else if (rec_left != 0 && tx_ok) begin
        tx_ok    <= 1'b0;
        tx_gap   <= 2'd3;
        tx_start <= 1'b1;
        unique case (rec_left)
          2'd3:    tx_data <= {1'b1, 5'b0, rec_sample[15:14]};
          2'd2:    tx_data <= {1'b0, rec_sample[13:7]};
          default: tx_data <= {1'b0, rec_sample[6:0]};
        endcase
        rec_left <= rec_left - 1'b1;
      end
    end else begin
      rec_left <= '0;
    end

    if (live && !record && state == P_IDLE) begin
      if (eng_done) begin
        dbg_pkt  <= {row_bits, eng_feat_or, copy_or_q, 16'(copy_n_q), 8'(fe_nfeat), 56'(fe_pow_or),
                     fe_win_or, {4'b0, level}, margin, 8'(class_idx), 8'("K")};
        dbg_left <= 6'(DBG_LEN);
        dbg_wait <= '1;
      end else if (dbg_wait != 0) begin
        dbg_wait <= dbg_wait - 1'b1;
        if (detect) dbg_pkt[8*6+7] <= 1'b1;  // flags byte: detected
      end else if (dbg_left != 0 && tx_ok) begin
        tx_ok    <= 1'b0;
        tx_gap   <= 2'd3;
        tx_start <= 1'b1;
        tx_data  <= dbg_pkt[7:0];
        dbg_pkt  <= dbg_pkt >> 8;
        dbg_left <= dbg_left - 1'b1;
      end
    end else begin
      dbg_left <= '0;
    end

    if (rst) begin
      state       <= P_IDLE;
      have_result <= 1'b0;
      have_word   <= 1'b0;
      rec_left    <= '0;
      dbg_left    <= '0;
      tx_gap      <= '0;
    end
  end

  // ---------------------------------------------------------------------------------------
  // Display
  // ---------------------------------------------------------------------------------------
  logic [26:0] heartbeat;
  always_ff @(posedge sys_clk) heartbeat <= heartbeat + 1'b1;

  logic [11:0] meter;
  always_comb begin
    for (int i = 0; i < 12; i++) meter[i] = level > 4'(i + 3);  // bar grows with the peak
  end

  assign led[11:0] = live ? meter : have_result ? 12'(1) << class_idx : '0;
  assign led[12]   = live && showing;
  assign led[13]   = live;
  assign led[14]   = eng_busy;
  assign led[15]   = heartbeat[26];

  seg7_word u_seg (
    .clk(sys_clk),
    .show(live ? showing : have_word),
    .class_idx(live ? show_class : last_word),
    .seg, .dp, .an);
endmodule
