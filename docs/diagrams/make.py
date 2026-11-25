"""Draw the repo's diagrams as SVG (light + dark).

    python docs/diagrams/make.py

Writes docs/diagrams/NAME.svg and NAME-dark.svg. The Markdown files embed them
with a <picture> element so GitHub shows the one matching the reader's theme.
To change a diagram, edit its function below and re-run.
"""
from pathlib import Path

from svgkit import Diagram

HERE = Path(__file__).parent
DIAGRAMS = []


def diagram(fn):
    DIAGRAMS.append(fn)
    return fn


# --------------------------------------------------------------------------- README
@diagram
def overview():
    d = Diagram("overview", 878, 180,
                "Trained ternary model, then the model compiler on the PC, then firmware that is the "
                "model, running on a Pico 2, Pi 4 or Pi 5 with no OS. Later the same compiler "
                "targets an FPGA via hls4ml or FINN.")
    h = 58
    d.node("M", 20, 20, 180, "Trained ternary model", kind="neutral", h=h)
    d.node("C", 236, 20, 180, "Model compiler on PC", h=h)
    d.node("F", 452, 20, 180, "Firmware = the model", kind="hw", h=h)
    d.node("B", 668, 20, 190, "Pico 2 / Pi 4 / Pi 5,\nno OS", kind="hw", h=h)
    d.node("FPGA", 452, 120, 200, "FPGA via hls4ml / FINN", kind="future")
    for a, b in (("M", "C"), ("C", "F"), ("F", "B")):
        d.edge([d.a(a, "r"), d.a(b, "l")])
    d.edge([d.a("C", "b"), (326, 140), d.a("FPGA", "l")], dashed=True)
    d.label(389, 132, "same compiler, later")
    return d


@diagram
def how_it_works():
    d = Diagram("how-it-works", 960, 706,
                "How it works. On the PC: BitNetMCU training of a ternary MNIST CNN exports a model "
                "header; the compiler front end (compiler/mafc) builds an IR of conv, pool, relu_norm, "
                "linear and argmax; bounds.py proves value ranges; the RP2350 back end (2 cores, SMLAD, "
                "branch-free ReLU) generates C, which the Pico SDK builds and picotool flashes. An IR "
                "interpreter must match the exact-integer golden model, whose expected logits feed "
                "harness/run.py and measure.sh, which write results/*.csv and notes/FINDINGS.md. "
                "The firmware is the model: on the Pico 2, bare metal, weights are const arrays in "
                "flash or SRAM; core 0 takes half the channels then the fc layers and argmax, core 1 "
                "the other half, joining core 0. The harness sends an image over USB serial and gets "
                "back the prediction, logits and cycle count.")
    d.panel(16, 16, 580, 650, "On the PC")
    d.panel(660, 16, 284, 650, "On the Pico 2", "bare metal, no OS", kind="hw")
    left = [
        ("T", "train/: BitNetMCU training", "ternary MNIST CNN"),
        ("H", "Exported model header", "train/export/*.h"),
        ("FE", "Compiler front end", "compiler/mafc"),
        ("IR", "IR", "conv, pool, relu_norm, linear, argmax"),
        ("BD", "bounds.py", "proves value ranges (int16 where safe)"),
        ("BE", "RP2350 back end", "2 cores, SMLAD, branch-free ReLU"),
        ("GEN", "Generated C", "firmware/rp2350/mafc/gen/"),
        ("SDK", "Pico SDK build + picotool flash", None),
    ]
    ys = [64 + i * 74 for i in range(len(left))]
    for (nid, t, s), y in zip(left, ys):
        d.node(nid, 36, y, 262, t, s, h=59)
    for (a, *_), (b, *_) in zip(left, left[1:]):
        d.edge([d.a(a, "b"), d.a(b, "t")])

    d.node("I", 318, ys[3] + 9.5, 262, "IR interpreter (numpy)", kind="check")
    d.node("G", 318, ys[3] + 92, 262, "Golden model", "harness/golden.py\n(exact integer numpy)", kind="check")
    d.node("R", 318, ys[3] + 210, 262, "harness/run.py, measure.sh", "send images, check, time", kind="check")
    d.node("CSV", 318, ys[-1] + 9.5, 262, "results/*.csv, notes/FINDINGS.md", kind="check")
    d.edge([d.a("IR", "r"), d.a("I", "l")])
    d.edge([d.a("I", "b"), d.a("G", "t")], dashed=True)
    d.label(459, d.a("I", "b")[1] + 25, "must match", "start", bg="pc")
    d.edge([d.a("G", "b"), d.a("R", "t")], dashed=True)
    d.label(459, d.a("G", "b")[1] + 25, "expected logits", "start", bg="pc")
    d.edge([d.a("R", "b"), d.a("CSV", "t")])

    d.node("MEM", 680, 76, 244, "Weights as const arrays", "flash (XIP cache) or SRAM", kind="hw")
    d.node("C1", 680, 220, 200, "Core 1", "other half\n(job loop on SIO FIFO)", kind="hw")
    ry = d.a("R", "l")[1]
    d.node("C0", 680, ry - 37.5, 244, "Core 0", "half the channels,\nthen fc layers, argmax", kind="hw")
    d.edge([d.a("MEM", "b", 100 / 244), d.a("C1", "t")])
    d.edge([d.a("MEM", "b", 224 / 244), d.a("C0", "t", 224 / 244)])
    d.edge([d.a("C1", "b"), d.a("C0", "t", 100 / 244)])
    d.label(790, (d.a("C1", "b")[1] + d.a("C0", "t")[1]) / 2 + 4, "join", "start", bg="hw")

    d.edge([(580, ry - 11), (680, ry - 11)])
    d.label(630, ry - 36, "image over\nUSB serial")
    d.edge([(680, ry + 11), (580, ry + 11)])
    d.label(630, ry + 29, "prediction, logits,\ncycle count")

    sx = d.a("SDK", "b")[0]
    d.edge([d.a("SDK", "b"), (sx, 690), (802, 690), (802, 667)], accent="hw")
    d.label(628, 684, "firmware = the model", accent="hw")
    return d


# ------------------------------------------------------------------ FEASIBILITY-STUDY
@diagram
def three_ways():
    d = Diagram("three-ways", 912, 332,
                "Three ways to run a model. Normal setup: model file (data), runtime or interpreter, "
                "operating system, generic CPU. This project, model as firmware: model, compiler on "
                "PC, firmware that is the model, existing chip with cores, SIMD, DMA and PIO. Taalas "
                "HC1, model as silicon: model, chip design, custom wiring for one model only.")
    cols = [
        (16, "Normal setup", "grey", [("Model file - data", "neutral"), ("Runtime / interpreter", "neutral"),
                                      ("Operating system", "neutral"), ("Generic CPU", "neutral")]),
        (314, "This project: model as firmware", "pc", [("Model", "build"), ("Compiler on PC", "build"),
                                                        ("Firmware = the model", "hw"),
                                                        ("Existing chip:\ncores, SIMD, DMA, PIO", "hw")]),
        (612, "Taalas HC1: model as silicon", "amber", [("Model", "check"), ("Chip design", "check"),
                                                        ("Custom wiring -\none model only", "check")]),
    ]
    for px, title, pk, nodes in cols:
        d.panel(px, 16, 284, 300, title, kind=pk)
        prev = None
        for i, (t, k) in enumerate(nodes):
            nid = f"{px}-{i}"
            d.node(nid, px + 22, 70 + i * 58, 240, t, kind=k)
            if prev:
                d.edge([d.a(prev, "b"), d.a(nid, "t")])
            prev = nid
    return d


@diagram
def hybrid():
    d = Diagram("hybrid", 936, 270,
                "The compiler splits a trained ternary model three ways: layer order, loops, addresses "
                "and activations become code in SRAM; small hot weights such as tree router nodes "
                "become instructions; bulk weights stay packed 2-bit as data in flash, loaded and "
                "unpacked at run time. Both feed the code in SRAM.")
    d.node("M", 16, 132, 190, "Trained ternary model", kind="neutral")
    d.node("C", 246, 132, 120, "Compiler", kind="decide", shape="pill")
    d.node("HOT", 486, 62, 210, "Weights as instructions")
    d.node("DATA", 486, 202, 210, "Data in flash", kind="store")
    d.node("CODE", 756, 132, 160, "Code in SRAM", kind="hw")
    d.edge([d.a("M", "r"), d.a("C", "l")])
    d.edge([d.a("C", "r"), d.a("CODE", "l")])
    d.label(561, 145, "layer order, loops, addresses, activations")
    d.edge([d.a("C", "t"), (306, 82), d.a("HOT", "l")])
    d.label(396, 54, "small hot weights,\ne.g. tree router nodes")
    d.edge([d.a("C", "b"), (306, 222), d.a("DATA", "l")])
    d.label(396, 241, "bulk weights,\npacked 2-bit")
    d.edge([d.a("HOT", "r"), (836, 82), d.a("CODE", "t")])
    d.edge([d.a("DATA", "r"), (836, 222), d.a("CODE", "b")])
    d.label(846, 196, "load + unpack\nat run time", "start")
    return d


@diagram
def bcd_in_a_layer():
    d = Diagram("bcd-in-a-layer", 852, 232,
                "Inside one layer: B, a tree search, makes a few yes/no decisions to pick one branch; "
                "C, DMA streams that branch's weights from flash while the core computes; only that "
                "branch is computed; D asks whether the answer is confident: if yes, answer now and "
                "skip the remaining layers, if no, go to the next layer.")
    d.node("IN", 16, 24, 140, "Layer input", kind="neutral", h=59)
    d.node("B", 196, 24, 270, "B: tree search", "a few yes/no decisions pick one branch")
    d.node("C", 506, 24, 330, "C: DMA streams that branch's weights", "from flash while the core computes",
           kind="store")
    d.node("CMP", 16, 145, 180, "Compute branch only", kind="hw")
    d.node("D", 236, 145, 150, "D: confident?", kind="decide", shape="pill")
    d.node("OUT", 486, 115, 290, "Answer now, skip remaining layers", kind="hw")
    d.node("NEXT", 486, 175, 290, "Next layer", kind="neutral")
    d.edge([d.a("IN", "r"), d.a("B", "l")])
    d.edge([d.a("B", "r"), d.a("C", "l")])
    d.edge([d.a("C", "b"), (671, 114), (106, 114), d.a("CMP", "t")])
    d.edge([d.a("CMP", "r"), d.a("D", "l")])
    d.edge([d.a("D", "r"), (430, 165), (430, 135), d.a("OUT", "l")])
    d.label(458, 128, "yes")
    d.edge([d.a("D", "r"), (430, 165), (430, 195), d.a("NEXT", "l")])
    d.label(458, 213, "no")
    return d


@diagram
def pipeline_vs_split():
    d = Diagram("pipeline-vs-split", 912, 472,
                "Pipeline, one layer group per board: board 1 has layers 1-4, board 2 layers 5-8, "
                "board 3 layers 9-12, board 4 layers 13-16; one token visits the boards in turn while "
                "the others wait, which adds capacity, not speed. Split layer, every board holds a "
                "slice of every layer: a coordinator (input, output, next token) feeds four boards "
                "each holding a quarter of the neurons, and they exchange small result vectors back, "
                "one message per layer.")
    xs = [46, 262, 478, 694]
    d.panel(16, 16, 880, 176, "Pipeline: one layer group per board", kind="hw")
    for i, x in enumerate(xs):
        d.node(f"p{i}", x, 64, 180, f"Board {i + 1}: layers {4 * i + 1}-{4 * i + 4}", kind="hw")
        if i:
            d.edge([d.a(f"p{i - 1}", "r"), d.a(f"p{i}", "l")])
    d.node("pn", 236, 122, 440, "One token visits boards in turn; others wait.\nAdds capacity, not speed.",
           kind="note")

    d.panel(16, 208, 880, 248, "Split layer: every board holds a slice of every layer")
    d.node("co", 286, 256, 340, "Coordinator: input, output, next token")
    cb = d.a("co", "b")
    for i, x in enumerate(xs):
        d.node(f"s{i}", x, 356, 180, f"Board {i + 1}: 1/4 neurons", kind="hw")
        d.edge([cb, (cb[0], 326), (x + 90, 326), d.a(f"s{i}", "t")])
        d.edge([d.a(f"s{i}", "b"), (x + 90, 420)], head=False)
    d.edge([(xs[0] + 90, 420), (866, 420), (866, 276), d.a("co", "r")])
    d.label(456, 440, "exchange small result vectors, one message per layer", bg="pc")
    return d


@diagram
def sram_cache():
    d = Diagram("sram-cache", 916, 374,
                "Larger-than-SRAM loop: the router or tree picks the next block; if it is in the SRAM "
                "cache (hit), compute from the 520 KB SRAM; on a miss, do an aligned block read from "
                "QSPI flash (4-16 MB) or PSRAM (8 MB) into the SRAM working-set cache, then compute. "
                "A prefetch of the most likely next block via DMA fills the cache ahead of time. "
                "Compute then goes back to the router.")
    d.node("R", 16, 122, 190, "Router / tree picks\nnext block")
    d.node("Q", 246, 131, 160, "In SRAM cache?", kind="decide", shape="pill")
    d.node("CMP", 640, 46, 250, "Compute from SRAM 520 KB", kind="hw")
    d.node("RD", 456, 200, 220, "Aligned block read", "QSPI flash 4-16 MB or PSRAM 8 MB", kind="store")
    d.node("CACHE", 700, 210, 190, "SRAM working-set cache", kind="hw")
    d.node("PF", 690, 300, 210, "Prefetch most likely\nnext block via DMA")
    d.edge([d.a("R", "r"), d.a("Q", "l")])
    d.edge([d.a("Q", "r"), (430, 151), (430, 66), d.a("CMP", "l")])
    d.label(440, 112, "hit", "start")
    d.edge([d.a("Q", "r"), (430, 151), (430, 229.5), d.a("RD", "l")])
    d.label(440, 196, "miss", "start")
    d.edge([d.a("RD", "r"), (700, 229.5)])
    d.edge([(795, 210), (795, 86)])
    d.edge([d.a("PF", "t"), (795, 250)], dashed=True)
    d.edge([(700, 46), (700, 20), (111, 20), d.a("R", "t")])
    return d


@diagram
def architecture():
    d = Diagram("architecture", 936, 450,
                "Build time, on the PC: train a ternary model in PyTorch; the model compiler (Python) "
                "emits code per layer with fixed addresses and a core split; the result is a .uf2 "
                "firmware image, flashed over USB or the Debug Probe. Run time, on the board with no "
                "OS: input via UART or USB goes to core 0 (first half of layers), activations pass "
                "via SRAM to core 1 (second half), DMA sends the result out as output, which the PC "
                "test harness compares with PyTorch.")
    d.panel(16, 16, 888, 130, "Build time - on the PC")
    d.node("tr", 46, 66, 200, "Train ternary model\nin PyTorch", h=59)
    d.node("cp", 296, 66, 300, "Model compiler (Python)", "code per layer, fixed addresses, core split")
    d.node("fw", 646, 66, 190, "Firmware image .uf2", kind="hw", h=59)
    d.edge([d.a("tr", "r"), d.a("cp", "l")])
    d.edge([d.a("cp", "r"), d.a("fw", "l")])

    d.panel(16, 212, 888, 130, "Run time - on the board, no OS", kind="hw")
    y = 262
    d.node("in", 46, y, 130, "Input via\nUART / USB", kind="neutral")
    d.node("c0", 212, y, 150, "Core 0: first\nhalf of layers", kind="hw")
    d.node("c1", 502, y, 130, "Core 1:\nsecond half", kind="hw")
    d.node("dma", 666, y, 120, "DMA sends\nresult out", kind="store")
    d.node("out", 816, y, 70, "Output", kind="neutral", h=58)
    for a, b in (("in", "c0"), ("c0", "c1"), ("c1", "dma"), ("dma", "out")):
        d.edge([d.a(a, "r"), d.a(b, "l")])
    d.label(432, y + 22, "activations via SRAM", bg="hw")

    d.edge([d.a("fw", "b"), (741, 212)], accent="hw")
    d.label(731, 182, "flash over USB or Debug Probe", "end", accent="hw")
    d.node("test", 781, 392, 140, "PC test harness", kind="check")
    d.edge([d.a("out", "b"), (851, 392)])
    d.label(841, 372, "compare with PyTorch", "end")
    return d


@diagram
def phases():
    d = Diagram("phases", 914, 291,
                "Six phases, each ending in a gate. Phase 0: train ternary MNIST (reuse BitNetMCU). "
                "Phase 1: compile to Pico 2 firmware vs the CMSIS-NN baseline. Phase 2: dual core, "
                "DMA, PIO plus ideas A-D. Phase 3: tiny story LM on the Pico, beat Atome at about "
                "1 tok/s. Phase 4: bare-metal Pi 4 / Pi 5, larger models, storage tier. "
                "Phase 5: FPGA via QONNX + hls4ml / FINN.")
    ph = [("Train ternary MNIST", "(reuse BitNetMCU)"),
          ("Compile to Pico 2 firmware", "vs CMSIS-NN baseline"),
          ("Dual core, DMA, PIO", "+ ideas A-D"),
          ("Tiny story LM on Pico", "beat Atome ~1 tok/s"),
          ("Bare-metal Pi 4 / Pi 5", "larger models, storage tier"),
          ("FPGA via QONNX", "+ hls4ml / FINN")]
    xs = [16, 352, 688]
    gx = [254, 590]
    for i, (a, b) in enumerate(ph):
        row, col = divmod(i, 3)
        y = 20 + row * 176
        d.node(f"P{i}", xs[col], y, 210, f"Phase {i}", f"{a}\n{b}")
        if i < 5 and col < 2:
            d.node(f"G{i}", gx[col], y + 19.5, 70, "Gate", kind="decide", shape="pill", h=36)
            d.edge([d.a(f"P{i}", "r"), d.a(f"G{i}", "l")])
            d.edge([d.a(f"G{i}", "r"), (xs[col + 1], y + 37.5)])
    d.node("G2", 758, 115, 70, "Gate", kind="decide", shape="pill", h=36)
    d.edge([d.a("P2", "b"), d.a("G2", "t")])
    d.edge([d.a("G2", "b"), (793, 172), (121, 172), d.a("P3", "t")])
    return d


# ---------------------------------------------------------------------------- HANDOFF
@diagram
def handoff_overview():
    d = Diagram("handoff-overview", 860, 275,
                "Train a ternary model (PyTorch, on the PC), then the model compiler (Python, to "
                "build) produces a firmware image that is the model, for the Pico 2 now. Later the "
                "compiler also targets a bare-metal Pi 4 / Pi 5 and an FPGA via QONNX + hls4ml / FINN.")
    d.node("T", 16, 24, 200, "Train ternary model", "(PyTorch, on PC)", kind="neutral")
    d.node("C", 252, 24, 190, "Model compiler", "(Python, to build)")
    d.node("F", 478, 24, 180, "Firmware image", "= the model", kind="hw")
    d.node("P", 694, 24, 150, "Pico 2", "(now)", kind="hw")
    d.node("PI", 478, 118, 180, "Pi 4 / Pi 5", "bare metal", kind="future")
    d.node("FP", 478, 196, 220, "FPGA via", "QONNX + hls4ml / FINN", kind="future")
    for a, b in (("T", "C"), ("C", "F"), ("F", "P")):
        d.edge([d.a(a, "r"), d.a(b, "l")])
    for n in ("PI", "FP"):
        y = d.a(n, "l")[1]
        d.edge([d.a("C", "b"), (347, y), d.a(n, "l")], dashed=True)
        d.label(412, y - 8, "later")
    return d


@diagram
def hardware_setup():
    d = Diagram("hardware-setup", 866, 376,
                "Hardware setup. The PC or laptop (build and test harness) connects over USB to the "
                "Debug Probe (CMSIS-DAP), which reaches Pico 2 #1, the target, by an SWD cable "
                "(3-pin JST) and a UART cable. The PC also powers and flashes the Pico over USB, "
                "through a USB power meter (mWh). Later, for idea F, Pico 2 #2 and #3 join with "
                "jumper wires in a star layout.")
    s = 16
    d.node("PC", 16, 130 + s, 180, "PC / laptop", "build + test harness")
    d.node("DP", 336, 20 + s, 180, "Debug Probe", "(CMSIS-DAP)", kind="check")
    d.node("P2", 676, 130 + s, 170, "Pico 2 #1", "target", kind="hw")
    d.node("P3", 676, 300 + s, 170, "Pico 2 #2, #3", kind="future")
    d.edge([d.a("PC", "t"), (106, d.a("DP", "l")[1]), d.a("DP", "l")])
    d.label(221, d.a("DP", "l")[1] - 8, "USB")
    d.edge([d.a("DP", "r", 0.28), (801, 36.5 + s), d.a("P2", "t", 0.74)])
    d.label(610, 8 + s, "SWD cable\n(3-pin JST)")
    d.edge([d.a("DP", "r", 0.72), (721, 62.5 + s), d.a("P2", "t", 0.26)])
    d.label(610, 81 + s, "UART cable")
    d.edge([d.a("PC", "r", 0.3), d.a("P2", "l", 0.3)])
    d.label(436, 126 + s, "USB\npower + flashing")
    d.edge([d.a("PC", "r", 0.72), d.a("P2", "l", 0.72)], head=False)
    d.label(436, 190 + s, "USB power meter\n(mWh)")
    d.edge([d.a("P2", "b"), d.a("P3", "t")], dashed=True, head=False)
    d.label(749, 246 + s, "idea F later:\njumper wires, star layout", "end")
    return d


@diagram
def first_steps():
    d = Diagram("first-steps", 912, 590,
                "Phase 0 and Phase 1 step by step. 1: toolchain (Pico SDK, arm-none-eabi-gcc, "
                "picotool, OpenOCD). 2: blink and UART hello on the Pico 2 via the Debug Probe. "
                "3: reproduce BitNetMCU, training ternary MNIST on the PC. 4: plain-C reference "
                "inference on the Pico 2 (packed weights). 5: CMSIS-NN int8 baseline, same task. "
                "6: test harness, a PC script comparing board and PyTorch. Gate 0: board output "
                "equals PyTorch and baseline numbers logged? 7: compiler v0, model to C per layer. "
                "8: idea A, three builds (all-code, all-data, hybrid). Gate 1: beats CMSIS-NN at "
                "equal accuracy?")
    a = [("S1", "1. Toolchain", "Pico SDK + arm-none-eabi-gcc + picotool + OpenOCD"),
         ("S2", "2. Blink + UART hello", "on the Pico 2 via Debug Probe"),
         ("S3", "3. Reproduce BitNetMCU", "train ternary MNIST on PC"),
         ("S4", "4. Plain-C reference inference", "on Pico 2 (packed weights)"),
         ("S5", "5. CMSIS-NN int8 baseline", "same task"),
         ("S6", "6. Test harness", "PC script compares board vs PyTorch")]
    for i, (nid, t, s) in enumerate(a):
        d.node(nid, 16, 20 + i * 79, 420, t, s)
        if i:
            d.edge([d.a(a[i - 1][0], "b"), d.a(nid, "t")])
    d.node("G0", 16, 494, 420, "Gate 0:", "board output = PyTorch?\nbaseline numbers logged?", kind="decide")
    d.edge([d.a("S6", "b"), d.a("G0", "t")])
    d.node("S7", 476, 20, 420, "7. Compiler v0", "model → C per layer")
    d.node("S8", 476, 99, 420, "8. Idea A: three builds", "all-code / all-data / hybrid")
    d.node("G1", 476, 178, 420, "Gate 1:", "beats CMSIS-NN\nat equal accuracy?", kind="decide")
    d.edge([d.a("G0", "r"), (456, d.a("G0", "r")[1]), (456, d.a("S7", "l")[1]), d.a("S7", "l")])
    d.edge([d.a("S7", "b"), d.a("S8", "t")])
    d.edge([d.a("S8", "b"), d.a("G1", "t")])
    return d


@diagram
def idea_order():
    d = Diagram("idea-order", 944, 310,
                "Idea test order after Phase 1: A (code vs data split), then C (DMA streaming), then "
                "B (tree-routed neurons, the most novel). B leads to D (early exit), H (prompt "
                "batching) and G (SRAM cache over flash). D leads to F (split layers across 3 Picos). "
                "G and H lead to the larger-than-memory goal on a Pi 4 with a USB SSD. "
                "E (lookup-table maths) only if time.")
    d.node("A", 16, 122.5, 140, "A", "code vs data split")
    d.node("C", 186, 122.5, 140, "C", "DMA streaming")
    d.node("B", 356, 114.5, 160, "B", "tree-routed neurons\n(most novel)")
    d.node("D", 552, 22.5, 150, "D", "early exit")
    d.node("H", 552, 122.5, 150, "H", "prompt batching")
    d.node("G", 552, 222.5, 150, "G", "SRAM cache over flash")
    d.node("F", 738, 14.5, 190, "F", "split layers\nacross 3 Picos")
    d.node("BIG", 738, 172.5, 190, "Larger-than-memory goal", "Pi 4 + USB SSD", kind="hw")
    d.node("E", 16, 214.5, 140, "E", "lookup-table maths\n(only if time)", kind="future")
    d.edge([d.a("A", "r"), d.a("C", "l")])
    d.edge([d.a("C", "r"), d.a("B", "l")])
    d.edge([d.a("B", "r"), d.a("H", "l")])
    d.edge([(516, 132), (534, 132), (534, 52), d.a("D", "l")])
    d.edge([(516, 172), (534, 172), (534, 252), d.a("G", "l")])
    d.edge([d.a("D", "r"), d.a("F", "l", 37.5 / 75)])
    d.edge([d.a("H", "r"), (720, 152), (720, 190), (738, 190)])
    d.edge([d.a("G", "r"), (720, 252), (720, 214), (738, 214)])
    return d


if __name__ == "__main__":
    for fn in DIAGRAMS:
        d = fn()
        for theme, suffix in (("light", ""), ("dark", "-dark")):
            (HERE / f"{d.name}{suffix}.svg").write_text(d.render(theme), encoding="utf-8")
        print(d.name)
