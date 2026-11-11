# Compiler (mafc): design notes and findings

*Started 11 Nov 2026 (Phase 1, compiler v0). How to use it: [compiler/README.md](../compiler/README.md).*

## Design decisions (v0)

- **Front end reads the exported header, not the `.pth`.** The header holds the exact integer codes the Phase 0 builds were verified with, so the compiler starts from a proven single source of truth. A `.pth` or QONNX front end comes later; it only has to produce the same IR.
- **The IR stores weights as unpacked integer codes.** Packing is the back end's decision. That's what makes idea A (weights as code vs data) and sparse formats possible without touching the front end.
- **The IR defines exact integer semantics** (`ir.py` docstring + `interp.py`). A back end is correct only if it is bit-exact with the interpreter. There are no tolerances anywhere.
- **Our own ternary packer:** word = ceil(N · 2^16 / 3^10) for the 10-digit base-3 number N, with digit 0 -> +1, 1 -> −1, 2 -> 0. That's exact because 2^16/3^10 ≈ 1.11 > 1, so every N has a unique 16-bit word in its interval. Rows that aren't a multiple of 10 are padded with zero trits. The back end decodes every packed array before writing it.
- **Channel fusion** is a pass in the back end: the leading run of per-channel ops runs one channel at a time with two small ping-pong planes (14×14 int32). The hand-written reference does this by hand; the compiler finds it.

## Results

See [results/README.md](../results/README.md). Profiling (stage cycles, tern80, image 0, profiling builds):

| Stage | ref_c flash | mafc v0 flash | ref_c SRAM | mafc v0 SRAM |
|---|---|---|---|---|
| conv1 | 698k | **459k** | 703k | **466k** |
| dw1 | 402k | 449k | 402k | 450k |
| pool1 + dw2 + pool2 | 97k | 102k | 88k | 98k |
| fc1 | 307k | 302k | 255k | 253k |
| fc2 + fc3 | 94k | 96k | 66k | 67k |
| **total** | 1.60M | 1.41M | 1.51M | 1.33M |

- **conv1 is 34% faster** because the generated code reads the int8 image directly. ref_c widens it to an int32 plane once per channel, 80 times per image, and its conv kernel loads int32.
- **dw1 is 12% slower** although the arithmetic is identical. Everything is inlined into `ref_infer()` in both builds (checked with `nm`), so the difference is gcc's code generation for fully constant loops (unrolling and register pressure on the M33's 13 usable registers). That's the first sign that "specialize everything" isn't automatically faster. The v1 back end should control unrolling explicitly (e.g. `#pragma GCC unroll`) and check the disassembly.
- fc layers are unchanged, as expected, since it's the same decode loop.

## Where the time goes, and v1 candidates

At v0 the conv stack is about 70% of the cycles and fc1 about 20%. Candidates, by expected gain:

1. **Second core.** Every channel of the conv stack is independent, so splitting the 80 channels across both M33 cores should take the conv stack's cycles to about half (the original plan says "both cores"). It needs a core-1 launch, a FIFO handshake and a split `feat` buffer. Cost: about 1 µs per handshake. Expected: about 1.5x end to end.
2. **SIMD (SMLAD, two 16x16 MACs per cycle) in the convs**, using bound analysis done by the compiler:
   - conv1 outputs are provably ≤ 3,448 (tern80z70; 2,648 for tern80), so **dw1 can run on int16 inputs with SMLAD, provably exactly**.
   - dw1 outputs are provably ≤ 60,830: they fit *uint16* but not int16, and SMLAD is signed. dw2 outputs are only bounded by about 820k. The largest seen on 20k images is 12,094 for dw2 and 5,658 for dw1. So int16 storage there would need either a range proof the bounds can't give, or a different decomposition. Never use a saturating shortcut: it breaks bit-exactness.
   - conv1 itself takes int8 input with int8 weights, so it can pack pairs with `SXTB16` + `SMLAD` directly.
3. **fc1 ternary (about 255k cycles from SRAM, about 8 per weight):** this is idea A. Candidates: weights as code (the add/subtract lists are about 15k non-zero instructions, too big for flash bandwidth?), a 2-bitplane format (positive and negative masks per 32 inputs), lookup tables over groups of activations (idea E), or sparse index lists for the z70/z90 models.
4. **Fixing the dw1 code-generation regression** (see above), plus the int8 → int16 widening.

## v1: second core (`compile.py --cores 2`, build `mafc_v1_dual`)

**Design.**
- **Conv segment:** core 1 takes channels 0–39, core 0 takes 40–79. Each core has its own two planes and writes its own slice of `feat`, so there's no sharing to synchronize except the start and the end.
- **Big linear layers:** fc1 (30.7k weights) and fc2 (6.1k) are split by output rows. fc3 (640 weights, under `PAR_MIN_MACS` = 4096) stays on core 0. `relu_norm` stays on core 0, because it needs the whole vector.
- **Handshake:** core 1 runs a job loop on the SIO FIFO: pop a job number, `__dmb()`, run, `__dmb()`, push the number back. Core 0 does `start_job(k)`, its own share, `join_job()`. That's 3 handshakes per image.
- **Start-up:** `ref_init()` (a new optional hook, called once by `pico_main.c` at boot) launches core 1, so the launch cost never lands in a timed inference.
- **PC build:** `start_job()` just runs core 1's share inline, so the host check covers exactly the same partitioning. The PC run was bit-exact on 10k images for all three models.
- **Profiling:** stages are core 0's own work plus a `join` stage (time spent waiting for core 1 after the conv segment). Linear stages are wall time, waits included.

**Profile** (tern80, image 0, profiling builds; v0 numbers in brackets):

| Stage | flash | SRAM |
|---|---|---|
| conv segment (core 0's half) | 562k [1,011k] | 534k [1,014k] |
| join (waiting for core 1) | 21k | 12k |
| fc1 | 215k [302k] | 139k [253k] |
| fc2 + fc3 | 75k [96k] | 40k [67k] |
| **total** | **880k = 5.86 ms** [1.41M] | **724k = 4.83 ms** [1.33M] |

- **From SRAM, close to the ideal 2x:** 1.84x overall. The conv segment is 1.90x and fc1 1.83x. The join wait (12k cycles, 1.6%) shows the two halves are well balanced; core 1 is slightly slower, likely from SRAM bank contention.
- **From flash, the second core helps less:** 1.60x overall, and fc1 only 1.40x. Both cores now pull fc1's 6 KB of ternary words and both cores' code through the one 16 KB XIP cache. Flash bandwidth and cache sharing are the limit again, which is the project's premise once more. The best dual-core setup runs from SRAM.

## v1 step 2: SMLAD on depthwise convs (`compile.py --simd`, builds `*_simd`)

**Bound analysis** (`compiler/mafc/bounds.py`): an exact interval analysis over every possible int8 input. For each conv channel it takes the sum over taps of min/max(k·lo, k·hi) per input channel, then applies ReLU and the shift. It also bounds |partial sums|. It's checked against 20k real images (train + test) for all three models: every observed value lies inside its bound.

| tern80 | proven range | observed max (20k images) | storage |
|---|---|---|---|
| conv1 out | [0, 2,643] | 1,969 | int16 |
| dw1 out / pool1 out | [0, 19,743] | 5,658 | int16 |
| dw2 out / pool2 out | [0, 132,087] | 12,094 | int32 |

(tern80z70: dw1 ≤ 27,623; tern80z90: ≤ 22,846; all int16.) **This corrects F9's first estimate.** That estimate (dw1 ≤ 60,830, "not int16") charged negative weights as if they could add to the sum, but the inputs are ReLU outputs and never negative. The exact bound shows dw1/pool1 fit int16, so **dw2's input is int16 too, and both dw1 and dw2 get SMLAD**.

**Code generation:**
- A tensor in the fused segment is stored as int16 when its proven range fits. The last op (pool2) still writes `feat` as int32.
- A depthwise conv reading an int16 plane uses an SMLAD kernel that produces two outputs per step. Per kernel row it does two aligned word loads (i0,i1) and (i2,i3), then `SMLAD`+`SMLABB` for output x and `SMLAD` (on the repacked (i1,i2), one `ORR` with shift) + `SMLATB` for x+1.
- Weights are pre-packed per channel at compile time as (k1,k0) pairs plus k2. The plane width must be even for aligned word pairs (14 and 6 are).
- On the PC the DSP intrinsics are plain-C functions with the same result, so the PC check covers the arithmetic.
- Non-SIMD outputs are byte-identical to before (checked with `cmp` on all 9 previously generated files).

**Results** (10k images, bit-exact, `results/phase1.csv`):

| tern80 | before (v0 / v1) | with SMLAD | gain |
|---|---|---|---|
| one core, flash | 9.33 ms | 8.93 ms | 1.04x |
| one core, SRAM | 8.83 ms | 8.29 ms | 1.07x |
| two cores, flash | 5.80 ms | 5.54 ms | 1.05x |
| **two cores, SRAM** | 4.80 ms | **4.36 ms** | **1.10x** |

**Profile** (one core, SRAM, stage cycles): dw1 450k → 320k (**−29%**), dw2 53k → 35k (**−35%**), but **conv1 466k → 522k (+12%)**. conv1's arithmetic didn't change; it now stores int16 instead of int32, and gcc's code for that loop got worse. That's the same kind of effect as the v0 dw1 regression. Without it the gain would be about 1.11x on one core. With two cores from SRAM, core 1 now finishes before core 0 (join wait ≈ 0): core 0 carries the profiling counter reads, and the halves are close enough that profiling overhead decides which one waits.

**Why conv1 got slower (investigated 11 Nov 2026).** The C source is identical except for `int16_t *out` and the `(int16_t)` cast. In the disassembly (`ref_infer` from the `-ram-prof` ELFs), gcc unrolls the x loop by 2 in both builds, but:

| | v0 (int32 store) | SIMD build (int16 store) |
|---|---|---|
| ReLU, 1st output of pair | `cmp; ite lt; movlt; asrge` | same |
| ReLU, 2nd output of pair | same, branch-free | `cmp; asr; blt` → out-of-line block that stores 0 and branches back |
| loop shape | one `bne` back-edge | `beq` exit + unconditional `b.n` back-edge |
| taken branches per pair | 1 | 2 if 2nd sum ≥ 0, 3 if < 0 |
| stack reloads of weights per pair | 11 | 12 (output pointer now takes a register) |
| measured cycles per pair (SRAM) | 59.5 | 66.6 |

conv1 pre-ReLU sums are negative 49% of the time (1,000 test images, same at even and odd x), so the slow path is taken about every other pair, and there is no branch predictor to hide it. The standalone compile with the same flags (`-O3 -mcpu=cortex-m33 -march=armv8-m.main+fp+dsp`) reproduces it: the current form has 5 branches, against 2 for the int32 form. Two equivalent rewrites, `t = s > 0 ? s : 0; o[x] = t >> 4` and `(s & ~(s >> 31)) >> 4`, both compile to `bic r, r, r, asr #31; asrs; strh`: no branches, no IT, 2 instructions shorter per output than v0. Exact, because `0 >> 4 == 0`. The fix goes in the back end's ReLU emission for every layer.

### v1 step 3: branch-free ReLU (11 Nov 2026)

`backend_rp2350_c.relu_shift()` emits `(ctype)((s > 0 ? s : 0) >> sh)` for every conv (plain and SMLAD). gcc compiles it to `bic r, r, r, asr #31; asrs`. Every generated file changed, including the v0 and dual ones, so they're no longer byte-identical to the earlier outputs. All 12 variants are bit-exact on the PC (10k images), and the 8 tern80 builds are bit-exact on the board.

Loops in the disassembly (`ref_infer`, RAM profile ELFs; `harness/asm_loops.py` lists every loop with its instruction, IT and branch counts):

| Loop | before | after |
|---|---|---|
| v0 conv1 (2 outputs) | 49 instr, 2 IT | 45, branch-free |
| v0 dw1 (2 outputs) | 61 instr, 2 IT | 53, branch-free |
| SMLAD conv1 (2 outputs) | 50 instr, 1 IT + `blt` path | 46, branch-free |
| SMLAD dw1 (2 outputs) | 48 instr, 2 IT | 36, branch-free |
| SMLAD dw2 (4 outputs) | 95 instr, 7 IT | 69, branch-free |

Measured (stage cycles, one core, SRAM): SMLAD build conv1 522k → 451k, dw1 320k → 273k, dw2 35k → 34k; v0 conv1 466k → 460k, dw1 450k → 360k, dw2 53k → 45k. Whole model (10k images, SRAM): v0 8.83 → 8.18 ms, v1 two cores 4.80 → 4.53 ms, v0 + SMLAD 8.29 → 7.44 ms, **v1 + SMLAD two cores 4.36 → 4.23 ms (3.44x faster than CMSIS-NN)**. All numbers are in `results/README.md` and FINDINGS F14.

Lesson: the select form cost far more than its own instructions; it changed register allocation and scheduling of whole loops. That's also most of the old v0 dw1 regression (F6). Check branch and IT counts in the hot loops after every back-end change.

**Next:** conv1 is still the largest stage (38–40% of cycles). It needs its own SIMD path. Its input is the int8 image, so pairs come from `SXTB16` on a word of four pixels, then `SMLAD` against packed (k1,k0) pairs. Its outputs are provably ≤ 2,643 and its partial sums ≤ 42k, so int16 arithmetic is exact. (Done differently, in step 4.)

### v1 step 4: conv1 on SMLAD, kernel v2 (11 Nov 2026)

**conv1 without SXTB16.** conv1 is a broadcast conv: all 80 output channels read the same 16×16 image. So `widen_img()` copies the int8 image into an int16 plane `img16` once per inference (256 values, inside the conv1 stage). Every channel then runs the same SMLAD kernel as the depthwise convs. Unpacking with `SXTB16` per channel would redo the same work 80 times. In the dual build, core 0 widens the image before starting core 1, and the FIFO push is fenced with a DMB.

**First try: old kernel on `img16`.** conv1 only dropped from 451k to 423k cycles (−6%). The disassembly showed why: 45 instructions per output pair, 11 of them stack accesses. gcc needs 6 kernel words, 2 sums, 4 loaded words, the repack temp, 2 output pointers and a counter. That doesn't fit in the M33's 13 usable registers.

**Kernel v2** (`emit_conv_ch_smlad`, used by conv1, dw1 and dw2):

| Change | Why |
|---|---|
| Kernel words A = (k0,k1), B = (k1,k2) per row: s0 += SMLAD(p,A) + SMLABT(q,B), s1 += SMLATB(p,A) + SMLAD(q,B) | No `(p >> 16) \| (q << 16)` repack: 2 fewer instructions and one fewer temp per row. Still 6 words per kernel |
| Row 0 uses `SMUAD` (no accumulator) | gcc had kept a register holding 0 |
| ReLU + shift as `USAT(s >> sh, 15)` (31 for int32 outputs) | One `usat r, #15, r, asr #4` instead of `bic` + `asr`. Exact: negative → 0; the top clamp can't be reached (proven range) |
| int16 output pair stored as one word, `v0 \| v1 << 16` | One store and one output pointer instead of two. Exact because both values are in [0, 2^15) |
| `#pragma GCC unroll 1` on the x loop | Full unrolling (tried with `#pragma GCC unroll 7`) hoisted loads and spilled about 10 times per pair. Without the pragma, gcc fully unrolled dw1 by itself, with the same result |

Inner loops (two outputs; `harness/asm_loops.py --max 2000` now also counts `[sp]` accesses):

| Loop | F14 kernel | first conv1 try | v2 |
|---|---|---|---|
| conv1 | 46 instr (int8 input, plain) | 45 instr, 11 `[sp]` | **24 instr, 0 `[sp]`** |
| dw1 | 37 instr, 4 `[sp]` | same | **24 instr, 0 `[sp]`** |
| dw2 (int32 out) | 73 instr | same | 47 instr |

Measured, stage cycles on one core from SRAM: conv1 451k → 232k, dw1 273k → 170k, dw2 34k → 24k. That is about 29.5 cycles per output pair for both conv1 and dw1, with 24 instructions in the loop plus the row overhead.

Whole model, 10k images, bit-exact (tern80 on the board; all 6 SIMD variants on the PC):

| Build | flash | SRAM |
|---|---|---|
| v0 + SMLAD | 8.11 → 5.97 ms | 7.44 → 5.23 ms |
| v1 + SMLAD, two cores | 5.40 → 4.38 ms | **4.23 → 3.05 ms (4.76x CMSIS-NN)** |

Two-core SRAM profile now: conv1 128k (28%), dw1 94k (20%), pool1 23k, dw2 13k, join 25k, **fc1 139k (30%)**, fc2 32k, fc3 7k. **Next: fc1** (weight formats, then weights as code, idea A).

Possible later conv work (not done):
- About 5.5 cycles per pair over the instruction count is row overhead and load-use stalls.
- Two channels per pass would share the 6 loads in conv1, but would need 12 kernel words in registers.
- pool1 (23k on core 0) is still plain C.

### v1 step 5: fc1 formats and fc1 as code (`compile.py --ternary`, 11 Nov 2026)

`backend_rp2350_c.emit_ternary()` writes the ternary layer in one of three new formats. base3 stays the default and stays in `emit_linear`. The build and file names get a `_<format>` suffix: `mafc_tern80_dual_simd_code.c`, `-DTERNARY=code` in CMake, and `mafc2_s:tern80:ram:code` in `measure.sh`.

| Format | Weights | Inner work | Notes |
|---|---|---|---|
| base3 | 10 trits per uint16, 6 KB | `c *= 3; d = c >> 16`, two compares, per weight | about 8.3 cycles per weight |
| sparse | uint16 indices, +1 list then −1 list per row, 30 KB (+ 2 offset tables) | `s += in[*p++]` (unrolled ×4) | pays only for nonzeros, but about 6 cycles each |
| code | none: the indices are immediates in the code, 86 KB of Thumb | `ldrsb` once per input, then one add/sub per nonzero | function `noinline`, always placed in SRAM (`__not_in_flash_func`) |
| dense | int8, 4 per word in byte order w0 w2 w1 w3, 30 KB | `SXTB16`, `SXTB16 ror 8`, 2× `SMLAD` per 4 weights against an int16 copy of the activations | 10 instructions per 4 weights; gcc doesn't fold the `ror` into `sxtb16` |

Measured: see FINDINGS F16. From SRAM, code takes fc1 from 256k to 50k cycles (one core) and from 139k to 28k (two cores). From flash, every format larger than the 16 KB cache gets slower; code ran 12x slower than from SRAM (611k cycles) until it was placed in SRAM.

The code format scales with sparsity (two cores, SRAM): fc1 28k (tern80, 50% zeros), 19k (z70), 11k (z90). base3 stays at 131–139k for all three.

What gcc does with the code format: it loads each of the 320 inputs once (`ldrsb`) at the start and keeps most of them in a 1.5 KB stack frame. Each row then becomes `ldr` from the stack + `add`/`sub` per nonzero. That is about 2 instructions per nonzero, measured at about 3.3 cycles. Ideas to go further: SIMD sums of byte groups shared by several rows, or common-subexpression sharing between rows (idea A's "shared partial sums").

Two-core SRAM profile with fc1 as code: conv1 127k (36%), dw1 94k (27%), pool1 23k, dw2 13k, join 24k, fc1 28k, fc2 33k, fc3 7k; 2.32 ms. **Next candidates:**
- Rebalance the channel split: core 0 waits about 24k cycles for core 1, because core 1 does its conv half plus the FIFO handshake.
- fc2 as code or SMLAD (int4, 6,144 weights).
- pool1.

### v1 step 6: int4 layers as code (`compile.py --int4`, 11 Nov 2026)

`emit_linear_fmt()` (renamed from `emit_ternary`) now handles int4sym layers too:
- **code:** per row, the terms are grouped by |weight| and each group is multiplied once, `3 * (in[2] - in[9] …) + 5 * (…)`. Ternary rows are a single group with no multiply, so the generated ternary code is unchanged (apart from an `#ifndef MAFC_IN_SRAM` guard).
- **dense:** int8 weights with `SXTB16`/`SMLAD`, the same as the ternary dense format.

Files get an `_i4<format>` suffix (`mafc_tern80_dual_simd_code_i4code.c`); CMake takes `-DINT4=`, and `measure.sh` takes a 5th field (`mafc2_s:tern80:ram:code:code`).

fc2 (one core, SRAM): packed 59.0k → dense 22.6k → code **17.4k** cycles (CMSIS-NN int8: 16.6k). fc3: 7.1k → 3.3k → 2.9k (CMSIS-NN 2.1k). The code is about 13.4k instructions for fc2's 6,144 weights.

Whole model: **2.15 ms** (two cores, SRAM), 2.35 ms (two cores, flash), 3.56 ms (one core, SRAM), 3.59 ms (one core, flash). FINDINGS F18.

Two-core SRAM profile now: conv1 127k (39%), dw1 94k (29%), pool1 23k (7%), dw2 13k, join 24k (7%), fc1 28k (9%), fc2 10k, fc3 3k. What's left is almost all the conv stack and the core imbalance.

### v1 step 7: shared kernel for two cores; all-code convs (`--conv code`) (12 Nov 2026)

- **Shared kernel:** in two-core builds `emit_conv_ch_smlad(shared=True)` marks the kernel `noinline`.
  - Inlined, core 1's copy inside `core1_main` was compiled worse: dw1 loop 30 instructions and 5 `[sp]`, against 24 and 0 on core 0. Core 0 waited about 24k cycles at the join.
  - Now the join wait is 24 cycles: 2.15 → 1.99 ms (two cores, SRAM). One-core builds stay inline (noinline cost 1% there). FINDINGS F19.
  - Check both `ref_infer` and `core1_main` with `asm_loops.py` in two-core builds.
- **`--conv code`:**
  - One function per conv channel with the 6 kernel words as immediates, plus a function table `<layer>_tab[c]`. File suffix `_cvcode`, CMake `-DCONV=code`, `measure.sh` 6th field.
  - The result (F20): no gain from SRAM, 2–3x slower convs from flash (41 KB of code through the 16 KB cache).
  - It is kept as the "all-code" point of the idea A test; it is not a recommended build.

## Lessons

- The worst-case bound analysis belongs in the compiler: it decides data types (int16 vs int32) and proves no overflow. The current interpreter only *asserts* int32 safety at run time.
- The fc1 zero fraction of the base `tern80` model is **50.2%**, not 24.8% as `train/README.md` said on 10 Nov (now corrected). The compiler's IR summary caught it, and `sparsity_report.py` on the committed checkpoint confirms it.
