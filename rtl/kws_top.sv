// Nexys A7-100T top level: features in over the USB-UART, inference, result out + display.
//
// Protocol (8N1 at 100 MHz / CLKS_PER_BIT baud, 1 Mbaud by default):
//   host -> board  'I' followed by IN_H*IN_W int8 feature bytes (row-major, h*IN_W + w)
//   board -> host  'R', class index, then the N_CLASSES int32 logits, little-endian
// A frame that stalls for RX_TIMEOUT cycles is dropped, so the host can always resync.
// Unexpected bytes while idle are ignored.
//
// LEDs 11:0 one-hot class, 14 busy, 15 heartbeat. The 7-segment display shows the word.

module kws_top
  import kws_pkg::*;
#(
  parameter int CLKS_PER_BIT = 100,
  parameter int RX_TIMEOUT   = 5_000_000  // 50 ms
) (
  input  logic        clk,      // 100 MHz
  input  logic        rst_n,    // CPU_RESETN button
  input  logic        uart_rx,  // from host (UART_TXD_IN)
  output logic        uart_tx,  // to host (UART_RXD_OUT)
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

  // UART
  logic       rx_valid, tx_start, tx_busy;
  logic [7:0] rx_data, tx_data;

  uart_rx #(.CLKS_PER_BIT(CLKS_PER_BIT)) u_rx (
    .clk, .rst, .rx(uart_rx), .valid(rx_valid), .data(rx_data));
  uart_tx #(.CLKS_PER_BIT(CLKS_PER_BIT)) u_tx (
    .clk, .rst, .start(tx_start), .data(tx_data), .busy(tx_busy), .tx(uart_tx));

  // Engine
  typedef enum logic [1:0] {P_IDLE, P_RECV, P_RUN, P_SEND} pstate_t;

  pstate_t                   state;
  logic                      eng_start, eng_busy, eng_done;
  logic [3:0]                class_idx;
  logic signed [31:0]        logits [N_CLASSES];
  logic [10:0]               rx_count;
  logic [$clog2(RX_TIMEOUT+1)-1:0] idle_cycles;
  logic [5:0]                tx_idx;
  logic                      have_result;

  kws_engine u_engine (
    .clk, .rst,
    .feat_we(state == P_RECV && rx_valid), .feat_waddr(rx_count), .feat_wdata(rx_data),
    .start(eng_start), .busy(eng_busy), .done(eng_done),
    .class_idx, .logits);

  function automatic logic [7:0] reply_byte(input logic [5:0] i);
    logic [5:0] j;
    if (i == 0) return "R";
    if (i == 1) return 8'(class_idx);
    j = i - 6'd2;
    return logits[j[5:2]][8*j[1:0] +: 8];
  endfunction

  always_ff @(posedge clk) begin
    eng_start <= 1'b0;
    tx_start  <= 1'b0;

    unique case (state)
      P_IDLE: if (rx_valid && rx_data == "I") begin
        rx_count    <= '0;
        idle_cycles <= '0;
        state       <= P_RECV;
      end

      P_RECV: begin
        if (rx_valid) begin
          rx_count    <= rx_count + 1'b1;
          idle_cycles <= '0;
          if (rx_count == 11'(N_FEAT - 1)) begin
            eng_start <= 1'b1;
            state     <= P_RUN;
          end
        end else if (idle_cycles == ($bits(idle_cycles))'(RX_TIMEOUT)) begin
          state <= P_IDLE;
        end else begin
          idle_cycles <= idle_cycles + 1'b1;
        end
      end

      P_RUN: if (eng_done) begin
        have_result <= 1'b1;
        tx_idx      <= '0;
        state       <= P_SEND;
      end

      P_SEND: if (!tx_busy && !tx_start) begin  // tx_busy rises the cycle after tx_start
        tx_start <= 1'b1;
        tx_data  <= reply_byte(tx_idx);
        tx_idx   <= tx_idx + 1'b1;
        if (tx_idx == 6'(N_REPLY - 1)) state <= P_IDLE;
      end
    endcase

    if (rst) begin
      state       <= P_IDLE;
      have_result <= 1'b0;
    end
  end

  // Display
  logic [26:0] heartbeat;
  always_ff @(posedge clk) heartbeat <= heartbeat + 1'b1;

  assign led[11:0]  = have_result ? 12'(1) << class_idx : '0;
  assign led[13:12] = '0;
  assign led[14]    = eng_busy;
  assign led[15]    = heartbeat[26];

  seg7_word u_seg (.clk, .show(have_result), .class_idx, .seg, .dp, .an);
endmodule
