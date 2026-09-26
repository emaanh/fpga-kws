// Simple dual-port RAM with synchronous read (1-cycle latency). `re` enables the read: tie
// it to "the data is needed" so an idle memory does not toggle (block RAM read power).
// With we tied low and INIT set, it is a ROM. STYLE "block" forces block RAM, "distributed"
// forces LUT RAM, "auto" lets synthesis choose.
module sdp_ram #(
  parameter int    WIDTH = 8,
  parameter int    DEPTH = 1024,
  parameter string INIT  = "",
  parameter string STYLE = "auto"
) (
  input  logic                     clk,
  input  logic                     we,
  input  logic [$clog2(DEPTH)-1:0] waddr,
  input  logic [WIDTH-1:0]         wdata,
  input  logic                     re,
  input  logic [$clog2(DEPTH)-1:0] raddr,
  output logic [WIDTH-1:0]         rdata
);
  if (STYLE == "block") begin : g_block
    (* ram_style = "block", rom_style = "block" *) logic [WIDTH-1:0] mem [DEPTH];

    initial if (INIT != "") $readmemh(INIT, mem);

    always_ff @(posedge clk) begin
      if (we) mem[waddr] <= wdata;
      if (re) rdata <= mem[raddr];
    end
  end else if (STYLE == "distributed") begin : g_dist
    (* ram_style = "distributed" *) logic [WIDTH-1:0] mem [DEPTH];

    initial if (INIT != "") $readmemh(INIT, mem);

    always_ff @(posedge clk) begin
      if (we) mem[waddr] <= wdata;
      if (re) rdata <= mem[raddr];
    end
  end else begin : g_auto
    logic [WIDTH-1:0] mem [DEPTH];

    initial if (INIT != "") $readmemh(INIT, mem);

    always_ff @(posedge clk) begin
      if (we) mem[waddr] <= wdata;
      if (re) rdata <= mem[raddr];
    end
  end
endmodule

// A wide RAM built from SLICE-bit sdp_rams side by side. For runtime-written block RAMs:
// yosys maps a 512 x 128 memory onto RAMB36s (512 x 72), and written RAMB36s misbehaved on
// the board with the open flow; 512 x 32 slices map onto RAMB18s (512 x 36), which work.
module sdp_ram_sliced #(
  parameter int WIDTH = 128,
  parameter int DEPTH = 512,
  parameter int SLICE = 32
) (
  input  logic                     clk,
  input  logic                     we,
  input  logic [$clog2(DEPTH)-1:0] waddr,
  input  logic [WIDTH-1:0]         wdata,
  input  logic                     re,
  input  logic [$clog2(DEPTH)-1:0] raddr,
  output logic [WIDTH-1:0]         rdata
);
  for (genvar s = 0; s < WIDTH / SLICE; s++) begin : g_slice
    sdp_ram #(.WIDTH(SLICE), .DEPTH(DEPTH)) u_ram (
      .clk, .we, .waddr, .wdata(wdata[SLICE*s +: SLICE]),
      .re, .raddr, .rdata(rdata[SLICE*s +: SLICE]));
  end
endmodule
