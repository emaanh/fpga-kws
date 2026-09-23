// Test wrapper for live mode: kws_top plus a PDM "microphone" that plays bits from a file.
// The player changes its data on the falling edge of m_clk; kws_top samples just before the
// rising edge (SW13 = 0).
module tb_live #(
  parameter string BITS_FILE   = "",
  parameter int    N_WORDS     = 1,      // 32 bits per word, LSB first
  parameter int    CLK_DIV     = 4,
  parameter int    HOLD_CYCLES = 1
) (
  input  logic        clk,
  input  logic        rst_n,
  input  logic [15:0] sw,
  output logic [15:0] led,
  output logic        done_playing
);
  logic        m_clk, m_lrsel, m_data, uart_tx;
  logic [6:0]  seg;
  logic        dp;
  logic [7:0]  an;
  logic [31:0] bits [N_WORDS];
  int unsigned idx;

  initial $readmemh(BITS_FILE, bits);

  assign m_data       = bits[idx / 32][idx % 32];
  assign done_playing = idx >= 32 * N_WORDS - 1;

  always_ff @(negedge m_clk or negedge rst_n) begin
    if (!rst_n) idx <= 0;
    else if (!done_playing) idx <= idx + 1;
  end

  kws_top #(.CLKS_PER_BIT(8), .CLK_DIV(CLK_DIV), .HOLD_CYCLES(HOLD_CYCLES)) u_top (
    .clk, .rst_n, .uart_rx(1'b1), .uart_tx, .sw, .m_clk, .m_lrsel, .m_data,
    .led, .seg, .dp, .an);
endmodule
