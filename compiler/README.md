# compiler/: mafc, the model-as-firmware compiler

Turns a trained model into firmware source for one target. Phase 1, v0.

```
train/export/<model>.h ──front end──► IR (integer ops + integer weight codes) ──back end──► gen/mafc_<model>.c
                                        │
                                        └─ interp.py: reference semantics (numpy, exact)
```

| File | Role |
|---|---|
| `mafc/ir.py` | The IR: `Graph` of `Node`s. Ops: `input`, `conv3x3` (broadcast/depthwise, ReLU, `>> shift`), `maxpool2`, `relu_norm`, `linear` (ternary / int4sym / int8), `argmax`. The docstring defines each op's exact arithmetic |
| `mafc/frontend_bitnetmcu.py` | BitNetMCU-exported header -> IR (CNNMNIST layout) |
| `mafc/bounds.py` | Worst-case value range of every tensor (exact interval analysis); decides int16 vs int32 |
| `mafc/interp.py` | Numpy interpreter of the IR, the definition of "correct" for every back end |
| `mafc/backend_rp2350_c.py` | IR -> C for the RP2350. Implements `ref_infer()` so the Phase 0 harness and firmware apply unchanged |
| `compile.py` | Command line |
| `mafc/lm_backend.py`, `compile_lm.py` | MafLM (Phase 3 language model): integer `.npz` (from `train/lm/intref.py --npz`) -> packed C data for `firmware/rp2350/maflm` (see notes/lm.md) |

## Use

```bash
train/.venv/Scripts/python.exe compiler/compile.py --model tern80     # -> firmware/rp2350/mafc/gen/mafc_tern80.c (+ .ir.json)
harness/measure.sh final results/phase1.csv mafc:tern80:flash mafc:tern80:ram
train/.venv/Scripts/python.exe compiler/compile.py --model tern80 --cores 2 --simd --ternary code --int4 code   # current best
harness/measure.sh final results/phase1.csv mafc2_s:tern80:ram:code:code
```

The generated C is committed, so building the firmware doesn't need Python. Regenerate it after changing the compiler or a model.

## Correctness chain

Each link must be bit-exact; all are checked on the 10k MNIST test set:

1. IR interpreter == harness golden model (`compile.py` checks 1000 images every time; 10k checked once).
2. Generated C compiled for the PC == golden (`harness/run.py --target host`).
3. Generated C on the Pico == golden (`harness/run.py --target pico`).

The back end also round-trips every packed weight array through a decoder before writing it.

## Back end v0 ("all-data")

- **One C function per layer**, every size a literal, weights as `static const` packed arrays (ternary: 10 trits per uint16; int4sym: 8 per uint32; int8: bytes), activations in static buffers at fixed addresses.
- **Channel fusion:** the leading run of per-channel ops (conv from one input channel, depthwise convs, pools) runs one channel at a time. Only two small planes are needed instead of whole feature maps.
- **conv1 reads the int8 image directly.** The hand-written reference widens it to int32 once per channel.
- Each conv/pool/linear node is a profiling stage (`-DPROFILE_STAGES=ON`).

## Back end v1: two cores (`--cores 2`)

`compile.py --cores 2` writes `gen/mafc_<model>_dual.c`. Build it with `-DDUAL_CORE=ON`, or as `mafc2` in `harness/measure.sh`. The conv channels and the big linear layers (at least 4,096 weights) are split between the two M33 cores. Core 1 runs a job loop on the SIO FIFO, started once at boot by `ref_init()`. On the PC the core-1 share runs inline, so the same split is checked there. Design and numbers are in [notes/compiler.md](../notes/compiler.md).

## Back end v1: SMLAD (`--simd`)

`compile.py --simd` (combine with `--cores 2`) writes `gen/mafc_<model>[_dual]_simd.c`. Build it with `-DSIMD=ON`, or as `mafc_s` / `mafc2_s` in `harness/measure.sh`.

- `mafc/bounds.py` proves each tensor's value range over all int8 inputs. Tensors that fit are stored as int16.
- Convs that read an int16 plane use an SMLAD kernel: two outputs per step, weights pre-packed at compile time as (k0,k1)/(k1,k2) pairs, and ReLU + shift as one `USAT`.
- conv1 (broadcast) reads `img16`, the image widened to int16 once per inference, so it uses the same kernel.
- The proven ranges are written as a comment at the top of the generated file.
- On the PC the DSP intrinsics fall back to plain C with identical results.

The plain kernels write the conv ReLU clamp-then-shift (`relu_shift()`): `(s > 0 ? s : 0) >> sh`, never `s < 0 ? 0 : s >> sh`. The results are the same, but the select form made gcc emit branches or IT blocks and cost conv layers 10–20% (FINDINGS F14). After any back-end change, run `harness/asm_loops.py` on the disassembly. Check the hot loops for branches and for stack accesses (`[sp]`, register spills): a spilling SMLAD loop ran at half the speed of one that fits (F15).

## Ternary layer formats (`--ternary`)

`compile.py --ternary {base3,sparse,code,dense}` sets the format of ternary linear layers (fc1). It adds a `_<format>` suffix to the generated file. Build with `-DTERNARY=<format>`, or give it as the 4th field in `harness/measure.sh` (`mafc2_s:tern80:ram:code`).

- **base3** (default): 10 trits per uint16 and a decode loop. Smallest (6 KB); best when fc1 must run from flash.
- **code** (idea A): one straight-line sum per row, with the indices as immediates. Fastest by far (5x base3), and it gets faster with more zeros. It is big (86 KB for tern80), and the function is always placed in SRAM.
- **sparse** (index lists) and **dense** (int8 + `SXTB16`/`SMLAD`, needs `--simd`): kept for comparison; both lose to code.

Numbers: FINDINGS F16.

## int4 layer formats (`--int4`)

`compile.py --int4 {packed,code,dense}` sets the format of int4sym layers (fc2, fc3). The file suffix is `_i4<format>`; build with `-DINT4=<format>`, or give it as the 5th field in `measure.sh` (`mafc2_s:tern80:ram:code:code`).

- **packed** (default): 8 weights per uint32 and a decode loop.
- **code:** per row, sums grouped by weight magnitude, one multiply per group; placed in SRAM. About 3.4x faster than packed, and level with CMSIS-NN.
- **dense:** int8 weights with `SXTB16`/`SMLAD` (needs `--simd`).

Numbers: FINDINGS F18.

## Tree layers (idea B)

A model exported by `train/export_fff.py` (`train/export/<model>.npz`, e.g. `fff80`) contains a fast feedforward tree in place of fc1 + fc2 (IR op `fff`). A few int8 routing neurons pick one leaf, and only that leaf's two small layers run. The leaves use the formats of their weight types (`--ternary` for the leaf input layer, `--int4` for the output layer; packed int4 needs widths divisible by 8, so use `--int4 code` or `dense` for 12-wide leaves). Example: `compile.py --model fff80 --simd --ternary code --int4 dense`. FINDINGS F25.

## int8 layers (`--int8`)

`--int8 {plain,dense,code}` sets the format of int8 linear layers (BitNetMCU `8bit`, e.g. the `int8_80` model): `plain` is a multiply loop over an int8 array, `dense` is the `SXTB16`/`SMLAD` kernel (needs `--simd`), `code` compiles the weights into instructions, grouped by magnitude. File suffix `_i8<fmt>`, CMake `-DINT8=`, `measure.sh` 8th field. Use `dense`: with almost no zero weights, code is slower and about 170 KB larger (FINDINGS F24).

All dense layers (ternary, int4 or int8) compute 4 rows at a time, sharing each activation load (`dense_rows()`, `DENSE_ROW_BLOCK`).

## Streamed layers (`--stream`)

`--stream` makes linear layers in a dense format (`--ternary dense` / `--int4 dense` / `--int8 dense`) stream their weights from flash in flash builds. The XIP stream FIFO feeds a DMA channel into a double buffer of 8-row blocks while the CPU computes, and the function runs from SRAM on core 0. In copy_to_ram builds the layer reads its weights directly. File suffix `_stream`, CMake `-DSTREAM=ON`, `measure.sh` 7th field `stream`. This is the placement for weights too big for SRAM; for MNIST, code in SRAM is faster (FINDINGS F23).

## Conv weights as code (`--conv code`)

`--conv code` (with `--simd`) writes one SMLAD function per conv channel with the kernel words as immediates, called through a table. File suffix `_cvcode`, CMake `-DCONV=code`, `measure.sh` 6th field. It exists for the idea A comparison and is not recommended. It is no faster from SRAM and 2–3x slower from flash, because conv weights are reused across a whole plane anyway (FINDINGS F20).

In two-core builds the SMLAD conv kernel is `noinline`, so both cores run one copy. An inlined copy in `core1_main` was compiled worse and made core 0 wait about 7% (FINDINGS F19).

## Limits of v0

- Front end: the BitNetMCU CNNMNIST header layout only. A QONNX or `.pth` front end comes later.
- Back end: 1x16x16 int8 input, a per-channel conv prefix, and a `linear -> 10` + `argmax` end. These match the harness protocol.
- SIMD covers the convs only. Weights as code (idea A) covers the linear layers (ternary and int4); conv weights are still data. See `notes/compiler.md`.
