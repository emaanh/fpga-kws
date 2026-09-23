// 8N1 UART receiver. `valid` pulses for one cycle with each received byte.
module uart_rx #(
  parameter int CLKS_PER_BIT = 100
) (
  input  logic       clk,
  input  logic       rst,
  input  logic       rx,
  output logic       valid,
  output logic [7:0] data
);
  typedef enum logic [1:0] {IDLE, START, DATA, STOP} state_t;

  state_t                          state;
  logic                            rx_meta, rx_sync;
  logic [$clog2(CLKS_PER_BIT)-1:0] cnt;
  logic [2:0]                      bit_idx;
  logic [7:0]                      shreg;

  always_ff @(posedge clk) begin
    rx_meta <= rx;
    rx_sync <= rx_meta;
    valid   <= 1'b0;

    if (state != IDLE) cnt <= cnt - 1'b1;

    unique case (state)
      IDLE: if (!rx_sync) begin
        cnt   <= ($clog2(CLKS_PER_BIT))'(CLKS_PER_BIT / 2 - 1);  // to the middle of the start bit
        state <= START;
      end
      START: if (cnt == 0) begin
        cnt     <= ($clog2(CLKS_PER_BIT))'(CLKS_PER_BIT - 1);
        bit_idx <= '0;
        state   <= rx_sync ? IDLE : DATA;  // a glitch, not a start bit
      end
      DATA: if (cnt == 0) begin
        cnt     <= ($clog2(CLKS_PER_BIT))'(CLKS_PER_BIT - 1);
        shreg   <= {rx_sync, shreg[7:1]};
        bit_idx <= bit_idx + 1'b1;
        if (bit_idx == 3'd7) state <= STOP;
      end
      STOP: if (cnt == 0) begin
        if (rx_sync) begin  // drop the byte on a framing error
          valid <= 1'b1;
          data  <= shreg;
        end
        state <= IDLE;
      end
    endcase

    if (rst) begin
      state   <= IDLE;
      rx_meta <= 1'b1;
      rx_sync <= 1'b1;
    end
  end
endmodule
