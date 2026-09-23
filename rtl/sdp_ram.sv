// Simple dual-port RAM with synchronous read (1-cycle latency). Infers block RAM.
// With we tied low and INIT set, it is a ROM.
module sdp_ram #(
  parameter int    WIDTH = 8,
  parameter int    DEPTH = 1024,
  parameter string INIT  = ""
) (
  input  logic                     clk,
  input  logic                     we,
  input  logic [$clog2(DEPTH)-1:0] waddr,
  input  logic [WIDTH-1:0]         wdata,
  input  logic [$clog2(DEPTH)-1:0] raddr,
  output logic [WIDTH-1:0]         rdata
);
  logic [WIDTH-1:0] mem [DEPTH];

  initial if (INIT != "") $readmemh(INIT, mem);

  always_ff @(posedge clk) begin
    if (we) mem[waddr] <= wdata;
    rdata <= mem[raddr];
  end
endmodule
