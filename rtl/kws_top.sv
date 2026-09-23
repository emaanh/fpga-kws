// Nexys A7-100T top level. Two modes, picked with SW15:
//
// SW15 = 0: UART feature mode (8N1 at 100 MHz / CLKS_PER_BIT baud, 1 Mbaud by default)
//   host -> board  'I' followed by IN_H*IN_W int8 feature bytes (row-major, h*IN_W + w)
//   board -> host  'R', class index, then the N_CLASSES int32 logits, little-endian
//   A frame that stalls for RX_TIMEOUT cycles is dropped, so the host can always resync.
//   LEDs 11:0 one-hot class. The display shows the word.
//
// SW15 = 1: live mode, from the on-board PDM microphone (kws_live.sv)
//   SW3:0 mic gain in 6 dB steps (0..12), SW13 samples the mic on the falling clock edge.
//   SW5:4 detection margin: 00 = 2^18 (default), 01 = 2^17, 10 = 2^19, 11 = off.
//   The display shows a detected word for a second. LEDs 11:0 are a mic level meter.
//   SW14 streams the mic's 16 kHz PCM to the host, 3 bytes per sample:
//     {1, 5'b0, x[15:14]}, {0, x[13:7]}, {0, x[6:0]}   (bit 7 marks the first byte)
//
// Both modes: LED 13 live mode, 14 engine busy, 15 heartbeat.

`ifndef KWS_MEM_DIR
`define KWS_MEM_DIR ""
`endif

module kws_top
  import kws_pkg::*;
#(
  parameter string MEM_DIR      = `KWS_MEM_DIR,
  parameter int    CLKS_PER_BIT = 100,
  parameter int    RX_TIMEOUT   = 5_000_000,   // 50 ms
  parameter int    CLK_DIV      = 25,          // PDM clock divider (2 MHz)
  parameter int    HOLD_CYCLES  = 100_000_000  // 1 s
) (
  input  logic        clk,      // 100 MHz
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

  logic [1:0] rst_sync;
  logic       rst;
  always_ff @(posedge clk) rst_sync <= {rst_sync[0], ~rst_n};
  assign rst = rst_sync[1];

  logic [15:0] sw_meta, sw_sync;
  always_ff @(posedge clk) {sw_sync, sw_meta} <= {sw_meta, sw};

  logic live, record;
  assign live   = sw_sync[15];
  assign record = sw_sync[15] && sw_sync[14];

  // ---------------------------------------------------------------------------------------
  // UART
  // ---------------------------------------------------------------------------------------
  logic       rx_valid, tx_start, tx_busy;
  logic [7:0] rx_data, tx_data;

  uart_rx #(.CLKS_PER_BIT(CLKS_PER_BIT)) u_rx (
    .clk, .rst, .rx(uart_rx), .valid(rx_valid), .data(rx_data));
  uart_tx #(.CLKS_PER_BIT(CLKS_PER_BIT)) u_tx (
    .clk, .rst, .start(tx_start), .data(tx_data), .busy(tx_busy), .tx(uart_tx));

  // ---------------------------------------------------------------------------------------
  // Engine, shared by both modes
  // ---------------------------------------------------------------------------------------
  logic               eng_start, eng_busy, eng_done;
  logic               eng_we;
  logic [10:0]        eng_waddr;
  logic [7:0]         eng_wdata;
  logic [3:0]         class_idx;
  logic signed [31:0] margin;
  logic signed [31:0] logits [N_CLASSES];

  kws_engine #(.MEM_DIR(MEM_DIR)) u_engine (
    .clk, .rst,
    .feat_we(eng_we), .feat_waddr(eng_waddr), .feat_wdata(eng_wdata),
    .start(eng_start), .busy(eng_busy), .done(eng_done),
    .class_idx, .margin, .logits);

  // ---------------------------------------------------------------------------------------
  // Live mode
  // ---------------------------------------------------------------------------------------
  logic               live_we, live_start, detect, showing;
  logic [10:0]        live_waddr;
  logic [7:0]         live_wdata;
  logic [3:0]         show_class, level;
  logic               pcm_valid;
  logic signed [15:0] pcm;

  logic signed [31:0] min_margin;
  always_comb begin
    unique case (sw_sync[5:4])
      2'b00: min_margin = 32'sd1 <<< 18;
      2'b01: min_margin = 32'sd1 <<< 17;
      2'b10: min_margin = 32'sd1 <<< 19;
      default: min_margin = '0;
    endcase
  end

  kws_live #(.MEM_DIR(MEM_DIR), .CLK_DIV(CLK_DIV), .HOLD_CYCLES(HOLD_CYCLES)) u_live (
    .clk, .rst, .enable(live), .gain(sw_sync[3:0]), .sample_fall(sw_sync[13]), .min_margin,
    .m_clk, .m_lrsel, .m_data,
    .pcm_valid, .pcm, .level,
    .feat_we(live_we), .feat_waddr(live_waddr), .feat_wdata(live_wdata),
    .eng_start(live_start), .eng_busy, .eng_done, .eng_class(class_idx), .eng_margin(margin),
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
  logic                            uart_rx_valid;

  assign uart_rx_valid = rx_valid && !live;

  // Registered mux into the engine: `live` fans out widely. Delaying start and the feature
  // writes by the same cycle keeps start after the last write.
  always_ff @(posedge clk) begin
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
  always_ff @(posedge clk) reply_q <= reply_byte(tx_idx);

  // Record mode: 3 bytes per PCM sample
  logic [15:0] rec_sample;
  logic [1:0]  rec_left;  // bytes of rec_sample still to send

  always_ff @(posedge clk) begin
    uart_start <= 1'b0;
    tx_start   <= 1'b0;

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
        tx_idx      <= '0;
        state       <= P_PREP;
      end

      P_PREP: state <= P_SEND;  // reply_q catches up with tx_idx

      // tx_busy rises the cycle after tx_start; by the next send reply_q has caught up
      P_SEND: if (!tx_busy && !tx_start) begin
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
      end else if (rec_left != 0 && !tx_busy && !tx_start) begin
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

    if (rst) begin
      state       <= P_IDLE;
      have_result <= 1'b0;
      rec_left    <= '0;
    end
  end

  // ---------------------------------------------------------------------------------------
  // Display
  // ---------------------------------------------------------------------------------------
  logic [26:0] heartbeat;
  always_ff @(posedge clk) heartbeat <= heartbeat + 1'b1;

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
    .clk,
    .show(live ? showing : have_result),
    .class_idx(live ? show_class : class_idx),
    .seg, .dp, .an);
endmodule
