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

## Use

```bash
train/.venv/Scripts/python.exe compiler/compile.py --model tern80     # -> firmware/rp2350/mafc/gen/mafc_tern80.c (+ .ir.json)
harness/measure.sh final results/phase1.csv mafc:tern80:flash mafc:tern80:ram
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
- Depthwise convs that read an int16 plane use an SMLAD kernel (two outputs per step, weights pre-packed at compile time).
- The proven ranges are written as a comment at the top of the generated file.
- On the PC the DSP intrinsics fall back to plain C with identical results.

All builds write the conv ReLU clamp-then-shift (`relu_shift()`): `(s > 0 ? s : 0) >> sh`, never `s < 0 ? 0 : s >> sh`. The results are the same, but the select form made gcc emit branches or IT blocks and cost conv layers 10–20% (FINDINGS F14). After any back-end change, check the hot loops' disassembly for branches.

## Limits of v0

- Front end: the BitNetMCU CNNMNIST header layout only. A QONNX or `.pth` front end comes later.
- Back end: 1x16x16 int8 input, a per-channel conv prefix, and a `linear -> 10` + `argmax` end. These match the harness protocol.
- SIMD only for depthwise convs on int16 planes so far (conv1 next); no weights as code yet (idea A). See `notes/compiler.md`.
