// PDM microphone interface: drives the mic clock (clk / PERIOD, 2 MHz by default at 50 MHz)
// and samples its data once per mic clock period.
//
// The ADMP421 with L/R select low drives data for the rising clock edge; `sample_fall`
// switches to sampling at the falling edge instead, in case the board needs it.
module pdm_mic #(
  parameter int PERIOD = 25   // clocks per mic clock period (low for PERIOD/2, then high)
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
  localparam int LOW = PERIOD / 2;

  logic [$clog2(PERIOD)-1:0] cnt;
  logic [1:0]                data_sync;

  assign m_lrsel = 1'b0;

  always_ff @(posedge clk) begin
    data_sync <= {data_sync[0], m_data};
    bit_valid <= 1'b0;
    cnt       <= cnt == ($clog2(PERIOD))'(PERIOD - 1) ? '0 : cnt + 1'b1;
    m_clk     <= cnt >= ($clog2(PERIOD))'(LOW - 1) && cnt != ($clog2(PERIOD))'(PERIOD - 1);
    // Sample just before the edge that is about to be made.
    if (cnt == (sample_fall ? ($clog2(PERIOD))'(PERIOD - 1) : ($clog2(PERIOD))'(LOW - 1))) begin
      bit_valid <= 1'b1;
      bit_data  <= data_sync[1];
    end
    if (rst) begin
      cnt   <= '0;
      m_clk <= 1'b0;
    end
  end
endmodule
