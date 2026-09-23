// PDM microphone interface: drives the mic clock (clk / (2*CLK_DIV), 2 MHz by default) and
// samples its data once per mic clock period.
//
// The ADMP421 with L/R select low drives data for the rising clock edge; `sample_fall`
// switches to sampling at the falling edge instead, in case the board needs it.
module pdm_mic #(
  parameter int CLK_DIV = 25
) (
  input  logic clk,
  input  logic rst,
  input  logic sample_fall,
  output logic m_clk,
  output logic m_lrsel,
  input  logic m_data,
  output logic bit_valid,
  output logic bit_data   // 1 = +1, 0 = -1
);
  logic [$clog2(CLK_DIV)-1:0] cnt;
  logic [1:0]                 data_sync;

  assign m_lrsel = 1'b0;

  always_ff @(posedge clk) begin
    data_sync <= {data_sync[0], m_data};
    bit_valid <= 1'b0;
    if (cnt == ($clog2(CLK_DIV))'(CLK_DIV - 1)) begin
      cnt   <= '0;
      m_clk <= ~m_clk;
      // Sample just before the edge we are about to make.
      if (m_clk == sample_fall) begin
        bit_valid <= 1'b1;
        bit_data  <= data_sync[1];
      end
    end else begin
      cnt <= cnt + 1'b1;
    end

    if (rst) begin
      cnt   <= '0;
      m_clk <= 1'b0;
    end
  end
endmodule
