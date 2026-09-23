// 8N1 UART transmitter. Pulse `start` with `data` while !busy.
module uart_tx #(
  parameter int CLKS_PER_BIT = 100
) (
  input  logic       clk,
  input  logic       rst,
  input  logic       start,
  input  logic [7:0] data,
  output logic       busy,
  output logic       tx
);
  logic [9:0]                      shreg;  // {stop, data, start}, sent LSB first
  logic [3:0]                      bits_left;
  logic [$clog2(CLKS_PER_BIT)-1:0] cnt;

  assign busy = bits_left != 0;
  assign tx   = busy ? shreg[0] : 1'b1;

  always_ff @(posedge clk) begin
    if (!busy) begin
      if (start) begin
        shreg     <= {1'b1, data, 1'b0};
        bits_left <= 4'd10;
        cnt       <= ($clog2(CLKS_PER_BIT))'(CLKS_PER_BIT - 1);
      end
    end else if (cnt == 0) begin
      shreg     <= {1'b1, shreg[9:1]};
      bits_left <= bits_left - 1'b1;
      cnt       <= ($clog2(CLKS_PER_BIT))'(CLKS_PER_BIT - 1);
    end else begin
      cnt <= cnt - 1'b1;
    end

    if (rst) bits_left <= '0;
  end
endmodule
