// Shows the name of a KWS class on the Nexys A7's eight 7-segment digits, left-aligned.
// Segments and anodes are active low. seg = {g, f, e, d, c, b, a}.
module seg7_word (
  input  logic       clk,
  input  logic       show,        // blank the display until the first result
  input  logic [3:0] class_idx,
  output logic [6:0] seg,
  output logic       dp,
  output logic [7:0] an
);
  // 8 ASCII characters, first character in the top byte (leftmost digit, AN7).
  function automatic logic [63:0] word(input logic [3:0] c);
    unique case (c)
      4'd0:    return "SILENCE ";
      4'd1:    return "UNKNOWN ";
      4'd2:    return "YES     ";
      4'd3:    return "NO      ";
      4'd4:    return "UP      ";
      4'd5:    return "DOWN    ";
      4'd6:    return "LEFT    ";
      4'd7:    return "RIGHT   ";
      4'd8:    return "ON      ";
      4'd9:    return "OFF     ";
      4'd10:   return "STOP    ";
      4'd11:   return "GO      ";
      default: return "--------";
    endcase
  endfunction

  // Best-effort letter shapes, active high {g, f, e, d, c, b, a}.
  function automatic logic [6:0] glyph(input logic [7:0] ch);
    unique case (ch)
      "C":     return 7'h39;
      "D":     return 7'h5E;  // d
      "E":     return 7'h79;
      "F":     return 7'h71;
      "G":     return 7'h3D;
      "H":     return 7'h76;
      "I":     return 7'h30;
      "K":     return 7'h75;
      "L":     return 7'h38;
      "N":     return 7'h54;  // n
      "O":     return 7'h3F;
      "P":     return 7'h73;
      "R":     return 7'h50;  // r
      "S":     return 7'h6D;
      "T":     return 7'h78;  // t
      "U":     return 7'h3E;
      "W":     return 7'h1C;  // u, the closest a 7-segment digit gets
      "Y":     return 7'h6E;
      "-":     return 7'h40;
      default: return 7'h00;
    endcase
  endfunction

  // Scan one digit every 2^14 cycles (~6 kHz at 100 MHz, ~760 Hz per digit).
  logic [16:0] scan;
  logic [2:0]  digit;
  logic [63:0] text;

  assign digit = scan[16:14];
  assign text  = word(class_idx);

  always_ff @(posedge clk) begin
    scan <= scan + 1'b1;
    an   <= ~(8'd1 << digit);
    seg  <= show ? ~glyph(text[8*digit +: 8]) : 7'h7F;
  end

  assign dp = 1'b1;
endmodule
