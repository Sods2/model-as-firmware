# Findings so far

*Consolidated 11 Nov 2026, covering Phase 0 and Phase 1 up to compiler v1 (dual core), the core-contention test, SMLAD on all three convs, and four fc1 weight formats (fc1 as code). One place to read and analyse the conclusions. Each finding gives its evidence, where the raw data lives, and what it means for the project. The narrative is in [LOG.md](LOG.md), and the details are in the topic notes linked below.*

Setup for all numbers:
- **Board and clock:** Raspberry Pi Pico 2 (RP2350, Cortex-M33) at 150 MHz.
- **Model:** BitNetMCU CNN `tern80` (unless stated): an 8-bit conv stack, ternary fc1 (320→96), and 4-bit fc2/fc3.
- **Test set:** the full 10,000-image MNIST test set.
- **Latency:** `ref_infer()` only, measured with the DWT cycle counter. "Flash" means code and weights run in place from QSPI flash (XIP, 16 KB cache). "SRAM" means a `copy_to_ram` build.

## Headline numbers (tern80)

| Build | Where | Latency | Cycles | vs CMSIS-NN, same memory | Flash | RAM | Data |
|---|---|---|---|---|---|---|---|
| CMSIS-NN int8 | flash | 18.70 ms | 2.81M | 1.00x | 93.6 KB | 33.5 KB | `results/phase0.csv` |
| CMSIS-NN int8 | SRAM | 14.52 ms | 2.18M | 1.00x | 93.5 KB | 127.0 KB | `results/phase0.csv` |
| ref_c (hand-written plain C) | flash | 10.21 ms | 1.53M | 1.83x | 47.3 KB | 7.7 KB | `results/phase0.csv` |
| ref_c | SRAM | 9.52 ms | 1.43M | 1.53x | 47.2 KB | 54.9 KB | `results/phase0.csv` |
| mafc v0 (compiled, one core) | flash | 8.66 ms | 1.30M | 2.16x | 46.4 KB | 9.2 KB | `results/phase1.csv` |
| mafc v0 | SRAM | 8.18 ms | 1.23M | 1.77x | 46.3 KB | 55.5 KB | `results/phase1.csv` |
| mafc v1 (two cores) | flash | 5.46 ms | 0.82M | 3.43x | 48.9 KB | 10.8 KB | `results/phase1.csv` |
| mafc v1 (two cores) | SRAM | 4.53 ms | 0.68M | 3.21x | 48.8 KB | 59.5 KB | `results/phase1.csv` |
| mafc v0 + SMLAD (one core) | flash | 5.97 ms | 0.90M | 3.13x | 49.6 KB | 9.7 KB | `results/phase1.csv` |
| mafc v0 + SMLAD (one core) | SRAM | 5.23 ms | 0.78M | 2.78x | 49.5 KB | 59.3 KB | `results/phase1.csv` |
| mafc v1 + SMLAD (two cores) | flash | 4.38 ms | 0.66M | 4.27x | 51.7 KB | 11.3 KB | `results/phase1.csv` |
| mafc v1 + SMLAD (two cores) | SRAM | 3.05 ms | 0.46M | 4.76x | 51.6 KB | 62.9 KB | `results/phase1.csv` |
| mafc v0 + SMLAD + fc1 as code (one core) | flash, fc1 in SRAM | 4.11 ms | 0.62M | 4.55x | 132.6 KB | 101.5 KB* | `results/phase1.csv` |
| mafc v0 + SMLAD + fc1 as code (one core) | SRAM | 3.87 ms | 0.58M | 3.75x | 132.5 KB | 135.7 KB* | `results/phase1.csv` |
| mafc v1 + SMLAD + fc1 as code (two cores) | flash, fc1 in SRAM | 2.89 ms | 0.43M | 6.47x | 128.8 KB | 97.5 KB* | `results/phase1.csv` |
| mafc v1 + SMLAD + fc1 as code (two cores) | SRAM | 2.32 ms | 0.35M | 6.25x | 128.6 KB | 133.4 KB* | `results/phase1.csv` |
| mafc v0 + SMLAD + all linear layers as code (one core) | flash, linear in SRAM | 3.59 ms | 0.54M | 5.22x | 160.9 KB | 133.6 KB* | `results/phase1.csv` |
| **mafc v0 + SMLAD + all linear layers as code (one core)** | **SRAM** | **3.56 ms** | **0.53M** | **4.08x** | 160.8 KB | 164.0 KB* | `results/phase1.csv` |
| mafc v1 + SMLAD + all linear layers as code (two cores) | flash, linear in SRAM | 2.35 ms | 0.35M | 7.95x | 156.5 KB | 129.2 KB* | `results/phase1.csv` |
| mafc v1 + SMLAD + all linear layers as code (two cores) | SRAM | 2.15 ms | 0.32M | 6.77x | 156.3 KB | 161.1 KB* | `results/phase1.csv` |
| mafc v1 + SMLAD + linear as code + shared kernel (two cores) | flash, linear in SRAM | 2.27 ms | 0.34M | 8.23x | 156.5 KB | 129.2 KB* | `results/phase1.csv` |
| **mafc v1 + SMLAD + linear as code + shared kernel (two cores)** | **SRAM** | **1.99 ms** | **0.30M** | **7.28x** | 156.4 KB | 161.1 KB* | `results/phase1.csv` |

\* RAM counted the new way (from 11 Nov, F16): every section linked at an SRAM address, excluding heap and stacks. The older rows used data + bss (+ text for SRAM builds). That overstated SRAM builds by about 5–6 KB, understated flash builds by about 2 KB, and would have missed code placed in SRAM entirely.

All mafc rows were re-measured on 11 Nov with the branch-free ReLU (F14), which sped up every mafc build. The SMLAD rows are the v2 kernel (F15: conv1 on SMLAD too, no repacking, ReLU as one `USAT`). The earlier numbers (v0 8.83 ms, v1 4.80 ms, v1 + SMLAD 4.36 → 4.23 ms from SRAM) are in F13/F14, `results/phase1.csv` (the `notes` column says which version) and `results/README.md`.

The Gate 1 bar is CMSIS-NN in its best setup (SRAM, 14.52 ms, **one core**). Our two-core rows compare two cores with one; one core against one core, the best is 3.56 ms = **4.08x** (F17, F18). Accuracy: 99.47% for every ternary-pipeline build and 99.44% for CMSIS-NN, against 99.45% for the PyTorch model. Energy is **not measured yet** (step 2b, waiting for the USB-C cable).

## Findings

### F1. Correctness holds end to end, bit for bit
- **Evidence:** every build matches its golden model's logits on 10,000/10,000 images: ref_c, CMSIS-NN, mafc v0 and v1, on the PC and on the board, for tern80, z70 and z90. The integer pipeline agrees with the float PyTorch model on 99.98% of predictions (2 images differ). CMSIS-NN agrees on 99.91%.
- **Meaning:** "matches PyTorch" was redefined as "bit-exact with an independent integer golden model, plus agreement logged", because 100% agreement with float is not a meaningful target for integer arithmetic. Any later speed-up must keep `golden_match_pct` at 100.
- **Details:** [bitnetmcu-integer-inference.md](bitnetmcu-integer-inference.md), [harness/README.md](../harness/README.md).

### F2. Flash bandwidth is a real limit, already at MNIST size
- **Evidence:** CMSIS-NN's fc1 runs **7.4x faster from SRAM** (524k → 71k cycles). Its 30 KB of int8 weights don't fit the 16 KB XIP cache. ref_c's packed ternary fc1 (6 KB) fits, so SRAM gains only 17% there.
- **Whole-build penalty for running from flash:** CMSIS-NN +29%, ref_c +7%, mafc v0 +6%, mafc v1 (dual) **+21%**.
- **Meaning:** this confirms the project's premise (packed weights win through memory traffic, not arithmetic) in Phase 0. Weight size relative to the 16 KB cache is a first-order design parameter.
- **Data:** `results/profile_stages.csv`. **Details:** [cmsis-nn-baseline.md](cmsis-nn-baseline.md).

### F3. Two cores give about 2x from SRAM but much less from flash
- **Evidence:** the dual-core build is 1.84x faster than one core from SRAM (fc1 1.83x, conv segment 1.90x), but only **1.61x from flash, with fc1 at just 1.40x**. Join wait: 12k cycles (SRAM) and 21k (flash), so the halves are balanced.
- **Meaning:** both cores share the one XIP cache, so a second core makes flash pressure worse (same root cause as F2). The fast configuration is two cores running from SRAM. That works while the model fits in 520 KB; for larger models, idea G (an SRAM cache over flash) becomes the key.
- **Data:** `results/profile_stages_dual.csv`. **Details:** [compiler.md](compiler.md).

### F4. Decoding ternary weights costs compute
- **Evidence:** from SRAM, our ternary fc1 takes about 255k cycles (about **8.3 cycles per weight**: a multiply and two branches per trit). CMSIS-NN's int8 fc1 with SMLAD takes 71k (about 2.3 per weight).
- **Meaning:** packing saves memory traffic (F2) but pays in decode work. The best format depends on where the weights live. This is the core trade-off for idea A (weights as code vs data) and idea E (lookup tables). fc1 is the next target after the conv stack.

### F5. The conv stack dominates, and CMSIS-NN handles this shape badly
- **Evidence:** the conv layers are about 70% of single-core cycles in every build. CMSIS-NN's conv1 (1 input channel → 80) takes 1.07M cycles against 0.47M for mafc. Its depthwise 3x3 takes 0.82M against 0.45M.
- **Meaning:** much of our 1.5–3x lead over CMSIS-NN comes from this model's unusual shape (a single-input-channel first layer, depthwise convs). That is a weakness of the baseline more than a strength of our method. **Don't overclaim at Gate 1.** Test on a wide-channel model too before drawing general conclusions. *(Done in F17: on a wide conv, ternary-as-code still wins once at least ~14% of weights are zero.)*

### F6. Specialising everything isn't automatically faster
- **Evidence:** mafc v0's constant-size loops made conv1 34% faster (it reads the int8 image directly instead of widening it per channel) but dw1 12% slower, with the same arithmetic. Everything was inlined in both builds, so the difference is gcc's unrolling and register pressure on the M33.
- **Meaning:** the compiler has to control code generation (unrolling, instruction choice) and check the disassembly, not just emit constants.
- **Update (F14):** most of the dw1 gap came from how the ReLU was written. With the branch-free ReLU, mafc v0's dw1 drops from 450k to 360k cycles, now 10% *faster* than ref_c's 402k.

### F7. Sparsity costs no accuracy and, so far, buys no speed
- **Evidence:** forcing 70% or 90% of fc1's weights to zero keeps accuracy (99.47% and 99.51%). The speed gain is 0.5–1% with the packed format, because every trit is still decoded. The base model is already **50.2% zeros** (an earlier note said 24.8%; that was wrong and has been corrected).
- **Meaning:** sparsity only pays with a sparse format (index lists, bitplanes, or code that skips zeros). That's a candidate for the compiler's fc1 work, alongside F4.

### F8. The network is positively homogeneous, which makes the int8 baseline fair
- **Evidence:** there are no biases, and only ReLU and max-pool non-linearities. So a static-scale int8 build (CMSIS-NN, calibrated on training images only) runs the *same* integer weights as the same network: 99.44% accuracy, bit-exact with its own golden model.
- **Meaning:** the speed comparison against CMSIS-NN is apples to apples. The same property would let the compiler swap BitNetMCU's per-image shifts for static scales if that ever helps.

### F9. Int16 bounds decide where SIMD is provably exact *(corrected 11 Nov)*
- **Evidence (exact interval analysis, `compiler/mafc/bounds.py`):** for tern80, conv1 outputs are ≤ 2,643, dw1 and pool1 outputs ≤ 19,743, and dw2 outputs ≤ 132,087. The largest observed over 20k images are 1,969, 5,658 and 12,094. So conv1, dw1 and pool1 fit int16 provably, and **both dw1 and dw2 can use SMLAD exactly**. dw2's output doesn't fit int16.
- **Correction:** the first estimate (dw1 ≤ 60,830, "not int16") was a loose |weight|-sum bound. It ignored that the inputs are ReLU outputs and never negative, so negative weights can only lower the sum.
- **Meaning:** let the compiler prove ranges and choose types; never rely on observed ranges or saturation, because that breaks bit-exactness. A loose bound costs real speed: it would have ruled out SIMD on dw2.

### F10. Latency is deterministic from SRAM, and less so from flash
- **Evidence:** the max over 10k images is within 0.1% of the mean from SRAM. From flash it is +6% (ref_c), +7% (mafc v0) and **+13% (mafc v1 dual)** above the mean.
- **Meaning:** cache misses make flash builds both slower and jittery, and the second core makes it worse. That matters for any real-time claim.

### F11. CMSIS-NN uses about 2x the flash of the ternary builds
- **Evidence:** about 94 KB vs about 46–49 KB. Weights: about 39 KB unpacked int8 vs about 11 KB packed. The CMSIS-NN library code is the rest.

### F12. The two cores slow each other down, mostly through flash, and mostly in fc1
- **Evidence:** in the dual-core build, each core's half of the conv layers takes longer than half the single-core time: **+6–8% from SRAM**, **+12–16% from flash** (core 0 / core 1). The join wait is 21k cycles from flash against 12k from SRAM. On fc1, the flash penalty is +18.5% with one core and **+54.6% with two**. Repeats agree within 0.1–1%.
- **Meaning:** about 6–8% is plain two-core sharing (SRAM and bus), which no placement removes. On top of that, flash costs about 41k critical-path cycles in the convs and about 76k in fc1. fc1's weights (only 6 KB) are evicted from the 16 KB XIP cache between inferences and then streamed by both cores at once. Moving **fc1's weights to SRAM** is the cheapest likely fix, ahead of moving the conv code. Test it as U1's mixed placement.
- **Data:** `results/contention*.csv`, `harness/contention.py`. **Details:** [IDEAS.md](IDEAS.md) U3.

### F13. SMLAD speeds up the depthwise convs by about a third, but a store-type change slowed conv1
- **Evidence:** with bound-proven int16 planes and SMLAD (two outputs per step), dw1 runs 29% faster and dw2 35% faster (one core, SRAM). Whole model: 1.07x on one core from SRAM, **1.10x on two cores from SRAM (4.80 → 4.36 ms, 3.33x faster than CMSIS-NN)**, about 1.05x from flash. Bit-exact on 10k images.
- **But:** conv1 got 12% *slower* only because it now stores int16, while its arithmetic is unchanged. That's the second time gcc's code generation has moved a stage by about 10% (see F6).
- **Root cause (checked in the disassembly):** with an int16 store, gcc -O3 compiles the ReLU `s < 0 ? 0 : (int16_t)(s >> 4)` of the *second* output in each unrolled pair as a conditional branch (`blt`), not IT-predicated. It also rotates the loop, so there's an unconditional `b.n` back-edge plus a `beq` exit. 49% of conv1 sums are negative, so the branchy path runs about every other output. That's 1 taken branch per pair in v0, against 2 (non-negative) or 3 (negative) now, plus one more stack reload. Measured: 59.5 → 66.6 cycles per output pair. Writing the ReLU as clamp-then-shift (`t = s > 0 ? s : 0; o = t >> 4`) compiles to a branch-free `bic r, r, r, asr #31`, which is shorter than v0's IT block too Fixed and timed in F14. Details: notes/compiler.md.
- **Meaning:** conv1 is now the biggest stage (about 40%). It needs its own SIMD path and a look at the generated assembly (done in F15). Gains from flash are smaller, because flash time is partly memory-bound (F2, F12).

### F14. How the ReLU is written moves the conv layers by 10–20%
- **Evidence:** the back end now writes every conv ReLU as clamp-then-shift, `(s > 0 ? s : 0) >> sh`, instead of `s < 0 ? 0 : s >> sh`. The results are identical, because `0 >> sh` is 0. All 12 generated variants are bit-exact on the PC (10k images, 3 models), and the 8 tern80 builds are bit-exact on the board (10k images). gcc now compiles the ReLU as `bic r, r, r, asr #31; asrs`, with no branch or IT block in any conv loop.
- **Stage cycles (one core, SRAM, image 0):**

  | Stage | v0 before | v0 after | SMLAD build before | SMLAD build after |
  |---|---|---|---|---|
  | conv1 | 466k | 460k (−1%) | 522k | **451k (−14%)** |
  | dw1 | 450k | **360k (−20%)** | 320k | **273k (−15%)** |
  | dw2 | 53k | 45k (−15%) | 35k | 34k (−2%) |

  Inner-loop instruction counts (per two outputs, or four for SMLAD dw2): v0 conv1 49 → 45, dw1 61 → 53; SMLAD build conv1 50 → 46, dw1 48 → 36, dw2 95 → 69. The old select form cost more than its own 4 instructions: it changed gcc's register allocation and scheduling of the whole loop.
- **Whole model (10k images):** v0 −7% (8.83 → 8.18 ms from SRAM), v1 two cores −6% (4.80 → 4.53 ms), v0 + SMLAD −10% (8.29 → 7.44 ms), **v1 + SMLAD −3% (4.36 → 4.23 ms from SRAM, 3.44x faster than CMSIS-NN)**. From flash: 8.66, 5.46, 8.11 and 5.40 ms. On two cores the gain is smaller, because each core's conv half shrank but fc1 and the join didn't.
- **Meaning:** F6 and F13 were the same thing: small source-level choices change gcc's code for the hot loops by 10–20%. The back end should keep checking its loops (branch and IT counts in the disassembly) after every change. conv1 still has no SIMD path and is the largest stage (38–40%).

### F15. A register-tight SMLAD kernel halves the conv layers; conv1 uses it on a widened image
- **Evidence:** a second SMLAD kernel, used by conv1, dw1 and dw2. All 6 SIMD variants are bit-exact on the PC (10k images, 3 models), and the 4 tern80 SIMD builds are bit-exact on the board (10k images).
  - **conv1:** it is a broadcast conv, so all 80 output channels read the same image. The int8 image is widened to an int16 plane once per inference (256 values). Every channel then runs the SMLAD kernel on it, so no per-channel `SXTB16` unpacking is needed.
  - **Kernel packing:** two halfword pairs per kernel row, A = (k0, k1) and B = (k1, k2). For the input words p = (i0, i1) and q = (i2, i3): s0 += SMLAD(p, A) + SMLABT(q, B), and s1 += SMLATB(p, A) + SMLAD(q, B). That is 4 MACs and 2 loads per row with no halfword repacking; the old kernel needed `lsl` + `orr` per row.
  - **ReLU + shift:** one `USAT #15, s, asr #sh` (int16 output) instead of `bic` + `asr`. This is exact: a negative sum shifts to a negative value and clamps to 0, and the upper clamp can't be reached because the output range is proven to fit.
  - **Rest of the kernel:** sums start with `SMUAD` (frees the zero register), the two int16 results go out as one word store, and the x loop is kept rolled (`#pragma GCC unroll 1`).
- **Inner loop, two outputs:** conv1 45 → **24 instructions**, dw1 37 → **24**, dw2 73 → 47. Stack accesses per iteration: conv1 11 → **0**, dw1 4 → 0.
  - The first attempt reused the old kernel on the widened image. conv1 dropped only 6% (451k → 423k cycles), because gcc ran out of registers and spilled weights and pointers (45 instructions, 11 stack accesses).
  - Fully unrolling the x loop made it worse: gcc hoisted the loads and spilled about 10 times per pair.
- **Stage cycles (one core, SRAM, image 0):**

  | Stage | before (F14) | after |
  |---|---|---|
  | conv1 | 451k | **232k (−49%)** |
  | dw1 | 273k | **170k (−38%)** |
  | dw2 | 34k | 24k (−29%) |

  On two cores (SRAM), core 0 timings: conv1 239k → 128k, dw1 143k → 94k, join 33k → 25k. That is about 29.5 cycles per output pair for both conv1 and dw1, against 24 instructions in the loop.
- **Whole model (10k images):**

  | Build | before | after | vs CMSIS-NN |
  |---|---|---|---|
  | v1 + SMLAD, two cores, SRAM | 4.23 ms | **3.05 ms** | **4.76x** |
  | v1 + SMLAD, two cores, flash | 5.40 ms | 4.38 ms | |
  | v0 + SMLAD, one core, SRAM | 7.44 ms | 5.23 ms | |
  | v0 + SMLAD, one core, flash | 8.11 ms | 5.97 ms | |

- **Meaning:** the convs went from about 75% of the time to about 50%.
  - fc1 is now the largest stage: 30% on two cores from SRAM (139k cycles) and 35% on one core from flash. It is the next target: weight formats and idea A.
  - The lesson from F14 holds again. On the M33 the win came from fitting the loop into 13 registers, not from the SIMD instruction alone. Count stack accesses (`harness/asm_loops.py` shows them) as well as instructions.

### F16. Compiling fc1 into code (idea A) makes it 5x faster from SRAM, and sparsity becomes speed
- **Evidence:** `compile.py --ternary {base3, sparse, code, dense}` sets the format of the ternary layer (fc1).
  - All 18 variants (3 models × 1/2 cores × 3 new formats) are bit-exact on the PC (10k images).
  - The code builds are bit-exact on the board (10k images; tern80 on one and two cores, SRAM and flash; z70/z90 on two cores, SRAM).
  - The formats:
    - **base3:** the existing format, 10 trits per uint16 (6 KB), decoded with a multiply and a branch per weight.
    - **sparse:** per row, the uint16 input indices of the +1 weights, then of the −1 weights (30 KB for tern80), summed in a loop.
    - **code:** idea A. One straight-line expression per row, `out[o] = in[3] + in[17] … - in[5] …`: 86 KB of Thumb code for tern80's 15,360 nonzeros.
    - **dense:** int8 trits, 4 per word (30 KB), unpacked with `SXTB16` and multiplied with `SMLAD` against an int16 copy of the activations.
- **fc1 stage cycles (tern80, image 0, profile builds):**

  | Format | 1 core SRAM | 2 cores SRAM | 1 core flash | 2 cores flash |
  |---|---|---|---|---|
  | base3 | 256k | 139k | 317k | 225k |
  | sparse | 94k | 57k | 311k | 513k |
  | **code** | **50k** | **28k** | 611k (from flash) → **51k** (placed in SRAM) | 1,306k → **29k** |
  | dense | 103k | 74k | 307k | 489k |

  - **From SRAM:** code is 5.1x faster than base3, at about 3.3 cycles per nonzero weight. gcc loads each of the 320 activations once and sums each row with one add or sub per nonzero (about 2 instructions per nonzero).
  - **From flash:** every format bigger than the 16 KB XIP cache is limited by flash bandwidth, at roughly 7–10 cycles per byte of weights or code. Two cores make that worse, not better (F12). base3 (6 KB) is the best format that runs from flash.
  - **Placement fixes it:** the code format's function now always goes to `.time_critical` (copied to SRAM at boot, `__not_in_flash_func`), so flash builds run it at SRAM speed. This answers the main part of U1.
- **Sparsity (two cores, SRAM, fc1 stage cycles):**

  | Model (fc1 zeros) | base3 | code |
  |---|---|---|
  | tern80 (50%) | 139k | 28k |
  | tern80z70 (70%) | 134k | 19k |
  | tern80z90 (90%) | 131k | 11k |

  The packed format pays for every weight, zero or not. Code pays only for nonzeros, so pruning turns directly into speed. This is the project's core claim, now measured.
- **Whole model (10k images):**

  | Build | Latency | vs CMSIS-NN |
  |---|---|---|
  | Two cores, SRAM | 3.05 → **2.32 ms** | **6.25x** |
  | Two cores, flash (fc1 in SRAM) | 4.38 → 2.89 ms | 6.47x vs CMSIS-NN from flash; 5.0x vs the 14.52 ms bar |
  | One core, SRAM | 5.23 → 3.87 ms | |
  | One core, flash (fc1 in SRAM) | 5.97 → 4.11 ms | |
  | z70, two cores, SRAM | 2.26 ms | 99.47% accuracy |
  | z90, two cores, SRAM | 2.21 ms | 99.51% accuracy |

- **Cost:** code is big. tern80's fc1 is 86 KB (z70 about 50 KB, z90 about 22 KB) against 6 KB packed. That is fine on a 520 KB-SRAM chip at MNIST size, but it is the trade-off to track for larger models. The sparse and dense formats lost to code everywhere and are kept only for comparison.
- **Measurement fix:** `run.py` counted RAM as data + bss from Berkeley `size`. That reports executable `.data` (code copied to SRAM) as text, so the first flash code build showed 9.7 KB of RAM instead of about 97 KB. RAM is now summed from the sections linked at SRAM addresses (`size -A`); see the note under the headline table.
- **Meaning:**
  - With fc1 down to 8% of the time, the convs are about 70% again: conv1 36%, dw1 27%, with the join wait at 7% and fc2 at 9% (two cores, SRAM).
  - For larger models, the choice between data and code is a placement question: what fits in SRAM, and what streams from flash.

### F17. Fairness check: the lead holds one core against one core, and on a wide conv once at least ~14% of weights are zero
- **The baseline is CMSIS-NN at its best for these shapes:**
  - Built with `-O3 -mcpu=cortex-m33 +dsp`, so `ARM_MATH_DSP` is on.
  - The wrappers pick the dedicated `arm_depthwise_conv_3x3_s8` for dw1/dw2.
  - conv1 uses im2col + SMLAD (`arm_convolve_s8`); CMSIS-NN has no faster path for one input channel.
  - Nothing to fix there.
- **Cores:** CMSIS-NN uses one core. The headline 6.25x compares two cores with one.
  - One core against one core (both from SRAM): **3.87 ms vs 14.52 ms = 3.75x**.
  - A hypothetical perfectly parallel two-core CMSIS-NN (≥ 7.26 ms) would still be ≥ 3.1x slower than our two-core 2.32 ms.
- **Per layer, one core, SRAM (stage cycles, image 0; CMSIS-NN against mafc v0 + SMLAD + fc1 as code):**

  | Stage | CMSIS-NN | mafc | mafc faster by |
  |---|---|---|---|
  | conv1 (1→80) | 1,070k | 231k | 4.6x |
  | dw1 | 822k | 171k | 4.8x |
  | pool1 | 96k | 43k | 2.2x |
  | dw2 | 92k | 24k | 3.7x |
  | fc1 (ternary in mafc, int8 in CMSIS-NN) | 71k | 50k | 1.4x |
  | fc2 | 17k | 59k → 17k as code (F18) | 0.28x → **0.96x** |
  | fc3 | 2k | 7k → 3k as code (F18) | 0.29x → 0.71x |
  | total | 2.18M | 0.59M → 0.54M (F18) | 3.7x → 4.0x |

  - The big wins are the layers CMSIS-NN handles badly (one input channel, depthwise; F5).
  - Our int4 fc2/fc3 still use a slow decode loop and lose clearly.
- **Reverse test: a wide conv, the shape CMSIS-NN is built for.**
  - Setup: one 3x3 conv, 10×10×32 → 8×8×32 (NHWC), ternary weights at 0–90% zeros, int8 activations, identical requantization.
  - Kernels: CMSIS-NN `arm_convolve_wrapper_s8` (int8 weights holding the ternary values) against mafc's method (weights compiled into code, one straight-line sum per output channel).
  - Checks: every kernel is bit-exact against the numpy reference, on the PC and on the board.
  - Firmware `firmware/bench/conv_wide`, driver `harness/bench_conv.py`, data `results/bench_conv.csv`.

  | Zero fraction | CMSIS-NN | mafc code | mafc faster by |
  |---|---|---|---|
  | 0% (dense ternary) | 1,257k | 1,457k | **0.86x (CMSIS-NN wins)** |
  | 33% | 1,257k | 1,053k | 1.19x |
  | 50% | 1,257k | 761k | 1.65x |
  | 70% | 1,257k | 458k | 2.74x |
  | 90% | 1,257k | 140k | 9.0x |

  (one core, everything in SRAM, minimum of 20 runs; the means are within 0.02%)

  - CMSIS-NN costs about 2.1 cycles per MAC, whatever the weights.
  - The code kernel costs about 2.5 cycles per nonzero weight per output pixel, so its time is proportional to the nonzeros. Break-even is at about **14% zeros**.
  - From flash, CMSIS-NN slows down by 3–50%. The code kernel was in SRAM in both builds, so the flash rows are not a like-for-like comparison and aren't used here.
- **Meaning and limits:**
  - Our method is not a trick of the MNIST shape. On CMSIS-NN's home ground it wins for ternary layers with typical sparsity (BitNet-style ternary layers are often 30–50% zeros), and it wins big with pruning.
  - It loses on dense ternary layers (about 1.16x). For int4 layers, mafc's fc2/fc3 were 3.6x slower than CMSIS-NN. Compiled into code they are about level (fc2 0.96x, fc3 0.71x; F18), but not ahead.
  - It only applies to ternary (multiply-free) weights. An int8 model would need multiplies, and CMSIS-NN's SMLAD path would be the bar.
  - The code costs memory: about 6 bytes per nonzero weight (55 KB for the dense 9,216-weight layer), which must sit in SRAM to be fast (F16).
  - On MNIST, the 4–5x conv wins come mostly from CMSIS-NN's weakness on this shape. The general claim is the 1.2–9x of the wide-conv test, depending on sparsity.

### F18. int4 layers as code: fc2 3.4x faster, level with CMSIS-NN
- **Evidence:** `compile.py --int4 {packed, code, dense}` sets the format of the int4sym layers (fc2 96→64, fc3 64→10).
  - **code:** per row, the inputs are summed per weight magnitude, and each group is multiplied once: `3 * (in[2] - in[9] …) + 5 * (…) + …`. That is one add/sub per weight and at most 8 multiplies per row; gcc emits about 2.2 instructions per weight. Placed in SRAM like fc1.
  - **dense:** the weights as int8, with `SXTB16` + `SMLAD`.
  - Checks: all 12 variants are bit-exact on the PC (10k images); the board finals are bit-exact on 10k images.
- **Stage cycles, one core, SRAM:**

  | Layer | packed (decode loop) | dense SMLAD | **code** | CMSIS-NN int8 |
  |---|---|---|---|---|
  | fc2 | 59.0k | 22.6k | **17.4k** | 16.6k |
  | fc3 | 7.1k | 3.3k | **2.9k** | 2.1k |

  From flash, dense loses badly (fc2 45.7k, fc3 12.4k), because its weights stream through the cache. Code stays at SRAM speed because it is placed in SRAM.
- **Whole model (10k images, fc1 + fc2/fc3 as code):**

  | Build | Before (fc1 only) | Now | vs CMSIS-NN |
  |---|---|---|---|
  | Two cores, SRAM | 2.32 ms | **2.15 ms** | 6.77x |
  | Two cores, flash (linear in SRAM) | 2.89 ms | 2.35 ms | 7.95x vs CMSIS-NN from flash; 6.2x vs the 14.52 ms bar |
  | One core, SRAM | 3.87 ms | **3.56 ms** | **4.08x** like-for-like |
  | One core, flash (linear in SRAM) | 4.11 ms | 3.59 ms | 5.22x vs CMSIS-NN from flash |
  | z70 / z90, two cores, SRAM | 2.26 / 2.21 ms | 2.09 / 2.03 ms | |

- **Meaning:**
  - The int4 gap that F17 found is closed to parity, not turned into a lead. Dense int4 weights have no zeros to skip, so code saves only the decode.
  - The one-core flash build (3.59 ms) is now within 1% of the all-SRAM build. With every linear layer in SRAM, what is left in flash (conv code and small conv weights) fits the 16 KB cache.
  - Cost: +28 KB (fc2 + fc3 code) on top of fc1's 86 KB.

### F19. The two-core join wait was core 1 running a worse copy of the same kernel
- **Evidence:** in two-core builds, gcc inlined the SMLAD conv kernel into both `ref_infer` (core 0) and `core1_main` (core 1), and compiled the two copies differently.
  - Core 1's dw1 loop was 30 instructions with 5 stack accesses, against 24 and 0 on core 0 (`harness/asm_loops.py` on `core1_main`).
  - Core 0 therefore waited about 24k cycles (7%) at the join.
  - The all-code build of F20 (one out-of-line function per channel) had no wait. That gave it away.
- **Fix:** in two-core builds the SMLAD kernel is `noinline`, so both cores run one shared copy. Join wait: 24k → 24 cycles.
- **Results (10k images, bit-exact):**

  | Build | Before | After |
  |---|---|---|
  | Two cores, SRAM, linear as code | 2.15 ms | **1.99 ms (7.28x CMSIS-NN)** |
  | Two cores, flash, linear as code | 2.35 ms | 2.27 ms |
  | Two cores, SRAM, all data | 3.05 ms | 2.91 ms |
  | Two cores, flash, all data | 4.38 ms | 4.32 ms |

  One-core builds were about 1% slower with `noinline` (call overhead: 3.56 → 3.59 ms), so they keep the inline kernel. Those rows are in `phase1.csv`, labelled `noinline SMLAD conv kernel`.
- **Meaning:** the "core imbalance" was a code-generation artefact, not a work split problem. It is the third time gcc's treatment of an identical loop has cost 10–25% (F6, F14).

### F20. Idea A verdict: compile the linear layers into code, keep the conv weights as data (keep the hybrid)
- **Test (roadmap):** three builds of the same model.
  - **All-data:** packed weights everywhere.
  - **Hybrid:** linear layers as code (fc1 ternary, fc2/fc3 int4; F16, F18), conv weights as data.
  - **All-code:** the conv weights are code too. One specialised SMLAD function per channel with the kernel as immediates (`--conv code`): 240 functions, called through a table.
  - All are bit-exact on 10k images. Two-core builds use the F19 shared kernel; the conv code is in flash in flash builds, and the linear code is always in SRAM.

  | Build | All-data | **Hybrid** | All-code |
  |---|---|---|---|
  | Two cores, SRAM | 2.91 ms | **1.99 ms** | 1.95 ms |
  | Two cores, flash | 4.32 ms | **2.27 ms** | 6.70 ms |
  | One core, SRAM | 5.23 ms | **3.56 ms** | 3.58 ms |
  | One core, flash | 5.97 ms | **3.59 ms** | 6.29 ms |
  | Flash used | 52 KB | 156 KB | 197 KB |

- **Against the keep criterion** ("hybrid ≥ the faster of the other two, with less flash"):
  - The hybrid is the fastest from flash by 1.9–2.8x.
  - It ties all-code from SRAM (within 2%), with 41 KB less flash than all-code.
  - It uses 3x the flash of all-data, but it is 1.5–1.9x faster.
  - **Keep.**
- **Why the split falls there:**
  - A linear-layer weight is used once per inference. As data it costs a load plus decode; as code, an add or sub with the index as an immediate. Code wins, and it gets to skip zeros (F16).
  - A conv kernel word is loaded once per channel and then reused for 144–196 outputs. Turning it into an immediate saves nothing (one core, SRAM: 3.58 vs 3.56 ms).
  - Each specialised channel function adds about 170 bytes of instructions, 41 KB in all. From flash that code streams through the 16 KB XIP cache and the conv stack becomes 2–3x slower.
  - The rule for the compiler: **weights with no reuse → code (in SRAM); weights with reuse → data.**

### F21. Idea C: streaming weights from flash with DMA while computing saves 31%, but only through the XIP stream FIFO
- **Test:** `firmware/bench/stream`, driven by `harness/bench_stream.py`.
  - One dense int8 linear layer, 256→256, whose 64 KB of weights stay in flash (4x the 16 KB XIP cache).
  - The same `SXTB16`/`SMLAD` kernel runs in every variant. Kernels, input and buffers are in SRAM, so only the weights touch flash.
  - One core; flash SCK 75 MHz (`PICO_FLASH_SPI_CLKDIV` 2); 2 KB blocks (8 rows).
  - Every variant is bit-exact (output checksum checked on the board); minimum of 10 runs.

  | Variant | Cycles | vs xip_cached |
  |---|---|---|
  | weights in SRAM (compute only) | 216k | |
  | **xip_cached**: read in place through the cache (what flash builds do today) | 646k | |
  | xip_nocache: read in place through the uncached alias `0x14000000` | 654k | +1% |
  | dma_seq: DMA a block from the uncached alias, then compute it | 723k | +12% |
  | dma_alias: the same, double-buffered | 822k | **+27%** |
  | stream_seq: DMA from the XIP stream FIFO (`XIP_AUX`, `DREQ_XIP_STREAM`), then compute | 697k | +8% |
  | **dma_stream**: stream FIFO, double-buffered | **443k** | **−31%** |

- **Reading:**
  - From flash, this layer costs about 430k cycles more than from SRAM, about 6.6–6.8 cycles per byte (≈ 22 MB/s, against 37.5 MB/s raw for a quad SPI at 75 MHz). That is the same for cache-line fills and for the stream FIFO.
  - Double-buffered streaming hides all of the compute: 443k is about the transfer time alone. In place, the CPU stalls on every cache miss, so compute and transfer add up.
  - DMA from the uncached alias is slower than the cache and gets *worse* with overlap. Each word is a separate flash transaction, and the overlapped DMA is exposed to XIP bus stalls; the datasheet gives this as the reason the `XIP_AUX` stream port exists.
- **Verdict:** **keep.** The roadmap target was ≥20% lower latency. The mechanism to use is the XIP stream FIFO plus double-buffered DMA into SRAM.
- **Where it applies:** layers whose weights stay as data and don't fit SRAM or the cache. In today's best MNIST build no layer is like that: the linear layers are code in SRAM, and the conv weights fit the cache. So it matters for the larger models of later phases (and for idea G, the SRAM cache over flash). The compiler doesn't emit it yet.
- **Correction:** the RP2350 *does* have XIP cache counters (`xip_ctrl_hw->ctr_hit`, `ctr_acc` in `hardware/structs/xip.h`); [pico-toolchain.md](pico-toolchain.md) wrongly said it had none. Bytes read from flash per inference can be measured with them (open question below).

### F22. Flash traffic measured: the hybrid reads 1.2 KB from flash per inference, CMSIS-NN 45 KB
- **Method:** the board firmware (`pico_main.c`, shared by every build) clears the RP2350 XIP cache counters `ctr_acc`/`ctr_hit` before each `ref_infer()` and adds them up afterwards. The PC reads them with the new `X` command.
  - Flash read = misses × 8-byte lines (`XIP_CACHE_LINE_SIZE` in the SDK).
  - `run.py` now fills the `flash_read_kb` column, empty since Phase 0, and prints accesses, misses and the hit rate.
  - The counters cover both cores and include instruction fetches. They do **not** see DMA/stream reads, which bypass the cache (F23).
- **Results (tern80, flash builds, 10k images, bit-exact):**

  | Build | Latency | XIP accesses | Misses | Flash read / inference | SRAM build | Cost per miss* |
  |---|---|---|---|---|---|---|
  | ref_c | 10.08 ms | 1.03M | 1,027 | 8.0 KB | 9.52 ms | ~82 cycles |
  | CMSIS-NN int8 | 18.70 ms | 1.67M | 5,821 | **45.5 KB** | 14.52 ms | ~108 |
  | mafc v0 (all data) | 8.67 ms | 0.90M | 893 | 7.0 KB | 8.18 ms | ~82 |
  | mafc v0 + SMLAD (all data) | 5.97 ms | 0.66M | 1,271 | 9.9 KB | 5.23 ms | ~87 |
  | **mafc one core, hybrid** (linear as code in SRAM) | 3.58 ms | 0.39M | 149 | **1.2 KB** | 3.56 ms | (noise) |
  | mafc one core, all-code | 6.30 ms | 0.39M | 6,274 | 49.0 KB | 3.58 ms | ~65 |
  | mafc two cores, all data | 4.32 ms | 0.64M | 1,678 | 13.1 KB | 2.91 ms | (two cores) |
  | **mafc two cores, hybrid** | 2.30 ms | 0.39M | 172 | **1.3 KB** | 1.99 ms | (two cores) |
  | mafc two cores, all-code | 6.67 ms | 0.39M | 6,496 | 50.8 KB | 1.95 ms | (two cores) |

  \* (flash latency − SRAM latency) / misses, one core. A rough figure: it also folds in the slower hits.
  The two-core hybrid SRAM build reads nothing from flash (0 accesses), as expected.
- **Reading:**
  - Hit rates are 98.4–99.96% everywhere, so hits are cheap. The misses set the flash penalty, at roughly 65–110 cycles per 8-byte line (≈ 8–14 cycles per byte). That is a bit worse than the 6.7 cycles/byte of sequential streaming (F21), because misses are scattered.
  - The hybrid reads **35x less** from flash than CMSIS-NN (1.2 vs 45.5 KB). That is why its flash build runs within 1% of its SRAM build, while CMSIS-NN loses 29%.
  - All-code reads 49 KB. The conv functions' code doesn't fit the cache, which confirms F20's explanation with a direct measurement.
  - With two cores, the all-data build misses 32% more than with one (13.1 vs 9.9 KB): the cores evict each other's lines, as F12 suspected.
- **Meaning:** "bytes read from flash per inference" (a metric in the logging table since Phase 0) is now measured for every flash build. For MNIST it confirms the premise: what the compiler places in SRAM, and what it keeps compact, decides flash traffic, and flash traffic decides the flash penalty.

### F23. Idea C in the compiler: streaming a 30 KB layer from flash makes it 29% faster on one core, and 58% faster than two cores fighting over the cache
- **Evidence:** `compile.py --stream` makes the dense-format linear layers (here fc1 with `--ternary dense`, 30 KB of int8 weights) stream their weights in flash builds.
  - The XIP stream FIFO feeds a DMA channel into a double buffer of 8-row blocks, while the CPU computes the previous block (F21's mechanism).
  - The function runs from SRAM. A streamed layer runs on core 0 only (there is one stream FIFO).
  - In a copy_to_ram build it detects that the weights are in SRAM and reads them directly.
  - Checks: bit-exact on the PC, where `memcpy` stands in for the DMA so the block logic is exercised, and on the board (10k images).

  | Build (fc1 dense) | fc1 stage | Whole model | Flash read via cache (F22) |
  |---|---|---|---|
  | one core, flash, in place through the cache | 308k | 6.29 ms | 39.9 KB |
  | one core, flash, **streamed** | **219k (−29%)** | **5.12 ms (−19%)** | 2.9 KB + 30 KB streamed |
  | two cores, flash, in place (both cores read fc1) | 526k | 6.65 ms | 42.5 KB |
  | two cores, flash, **streamed** (core 0) | **219k (−58%)** | **3.97 ms (−40%)** | 4.4 KB + 30 KB streamed |
  | one core, SRAM build (direct, for reference) | 107k | 4.29 ms (profile) | 0 |

- **Reading:**
  - The streamed fc1 is transfer-bound: 30 KB at about 7 cycles per byte is about 210k cycles, and the 107k of compute is hidden.
  - It also stops fc1's weights from evicting everything else from the cache. Misses through the cache fell from 5.1k to 0.4k per inference, and the other flash-resident layers sped up: dw2 52k → 30k, fc2 105k → 72k (one core).
  - Splitting a flash-bound layer across two cores is counter-productive (526k against 308k on one core). Streaming it on one core beats both.
- **Caveat for F22:** the XIP counters see only cached accesses. Streamed bytes bypass the cache, so for streamed builds `flash_read_kb` must be read as "via the cache"; the streamed bytes (the layer size) come on top.
- **Meaning:**
  - The compiler can now choose per layer between three placements: code in SRAM (best, F16/F18), data in the cache, and data streamed from flash.
  - For MNIST, code in SRAM still wins: fc1 as code takes 51k cycles. Streaming is the fallback for weights that are too big for SRAM, which is where later phases go.

### F24. int8 layers: level with CMSIS-NN once the dense kernel does 4 rows at a time; the whole int8 model is 3.9x faster on one core, and weights as code stop paying from SRAM
- **Setup:**
  - New model `int8_80`: the same CNN as tern80 with every linear layer int8 (BitNetMCU `8bit`, config `train/configs/int8_cnn80.yaml`). 99.43% PyTorch, 99.45% integer pipeline.
  - The convs were already int8, so only fc1 (30,720 weights, 5% zero), fc2 (6,144) and fc3 (640) change.
  - Golden model, ref_c and the front end decode BitNetMCU's 8-bit layers. `compile.py --int8 {plain,dense,code}` picks the format. CMSIS-NN runs the same weights (`export_cmsis.py --model int8_80`).
  - Checks: all variants bit-exact on the PC (10k images) and on the board (10k images).
- **First result:** CMSIS-NN won the int8 linear layers. Our dense format (`SXTB16`/`SMLAD`, one row at a time) took 111k cycles for fc1 against CMSIS-NN's 71k.
- **Fix: 4 rows at a time.** `dense_rows()` computes 4 rows together, so each pair of int16 activations is loaded once and feeds 4 rows' `SMLAD`s. CMSIS-NN's `vec_mat_mult` does the same with 2 rows, and has to unpack int8 activations.
  - The rotate is written as `sxtb16 r, r, ror #8` (inline asm); gcc never folded `__ror` into it.
  - The inner loop is 24 instructions per 16 MACs, with no spills (`asm_loops.py`). It still takes about 40 cycles per iteration; load latency and SRAM bus contention are the likely cause (not measured).

  | Layer (one core, SRAM, cycles) | CMSIS-NN | plain int8 loop | dense, 1 row | dense, 2 rows | **dense, 4 rows** | dense, 8 rows (spills) | code |
  |---|---|---|---|---|---|---|---|
  | fc1 (30,720 weights) | 70.9k | 199.6k | 110.7k | 85.1k | **75.9k** | 86.7k | 85.3k |
  | fc2 (6,144) | 16.4k | 46.9k | 22.6k | 17.3k | **16.5k** | 19.0k | 21.3k |
  | fc3 (640) | 2.1k | 5.3k | 3.3k | | **3.1k** | 3.3k | 3.3k |

- **Whole int8 model (10k images, bit-exact):**

  | Build | SRAM build | Flash build | SRAM used (flash build) | Flash read / inference |
  |---|---|---|---|---|
  | CMSIS-NN int8 | 14.54 ms | 18.74 ms | 37.8 KB | 45.5 KB |
  | ref_c | 8.74 ms | 10.63 ms | 12.0 KB | 40.9 KB |
  | mafc one core, plain int8 loop | 4.97 ms | | | |
  | **mafc one core, dense** | **3.73 ms (3.90x)** | 7.97 ms | 13.0 KB | 44.9 KB |
  | mafc one core, code (in SRAM) | 3.82 ms | **3.85 ms (4.87x)** | 225 KB | 1.2 KB |
  | mafc one core, dense streamed (`--stream`) | | 4.95 ms (3.79x) | 23.7 KB | 1.6 KB + 37 KB streamed |
  | **mafc two cores, dense** | **2.12 ms (6.86x)** | 7.11 ms | 14.5 KB | 47.9 KB |
  | mafc two cores, code (in SRAM) | 2.13 ms | 2.44 ms | 222 KB | 1.35 KB |

- **Reading:**
  - On int8 linear layers mafc is now level with CMSIS-NN: fc2 16.5k against 16.4k, fc1 7% behind (75.9k against 70.9k). The whole-model lead still comes from the conv stack (conv1 + dw1: 0.40M cycles against CMSIS-NN's 1.89M), which is the same as in tern80.
  - **Weights as code stop paying from SRAM.** Only 5% of int8 weights are zero and grouping by magnitude still leaves about one multiply per weight, so code is slower than dense (85k against 76k for fc1) and takes about 170 KB of SRAM.
  - In flash builds, dense layers thrash the 16 KB cache (45 KB read per inference, 7.97 ms). Code in SRAM is fastest (3.85 ms) but needs 225 KB of SRAM; streaming the dense layers (F23) gets 4.95 ms with 24 KB.
- **Side effect on tern80:** the 4-row kernel also speeds up the ternary and int4 dense formats.
  - fc1 dense: 103k → 76k cycles.
  - fc2 dense (int4): 22.6k → 16.5k. That is now faster than fc2 as code (17.4k, F18) and level with CMSIS-NN (16.6k).
  - Whole model, one core, SRAM: fc1 as code with fc2/fc3 dense is 3.56 ms, equal to the hybrid with everything as code (3.56 ms), and its binary is 24 KB smaller (136.8 against 160.8 KB).
- **Meaning:** the right format depends on the weight type. Ternary: code (zeros become speed, F16). int4: dense or code (a tie). int8: dense in SRAM; in flash builds, code in SRAM if there is room, otherwise streamed. mafc's general advantage on int8 layers is small; its large wins are the conv kernels (F5) and weights with many zeros (F17).

### F25. Idea B: a tree layer reads 6.6x fewer weights and runs 3.8-6.2x faster than its dense twin, for 0.30 points of accuracy (keep)
- **Model `fff80`:** tern80 with its fc1 + fc2 block (320 → 96 → 64) replaced by a fast feedforward tree (Belcak & Wattenhofer 2023) of the same training width.
  - Depth 3: 7 routing nodes (one int8 neuron each, no bias) choose one of 8 leaves.
  - Each leaf is 320 → 12 (ternary) → relu_norm → 64 (int4), like fc1 and fc2. 8 × 12 = 96, so tern80 is the dense twin.
  - Inference: at each node go right if `nodes[k] · x >= 0`; only the reached leaf is computed.
  - Weights read per inference: 3 × 320 (nodes) + 12 × 320 + 64 × 12 = **5,568, against 36,864** for fc1 + fc2: **6.6x fewer**. Stored: 39,104 against 36,864 (+6%, the nodes).
- **Training** (BitNetMCU patch: `FFFLayer`, `CNNMNIST_FFF`; config `train/configs/fff_cnn80.yaml`):
  - Soft routing: the output mixes all leaves, weighted by the path probabilities (products of node sigmoids).
  - A hardening loss (the mean entropy of the node decisions) ramps up over 10 epochs.
  - From epoch 40: hard routing with a straight-through estimator, so training matches inference.
  - **The first run collapsed:** hardening alone sent every image to one leaf (all node logits large and positive), the cheapest way to make every decision confident. Hard test accuracy hovered at 84–95%, from one 12-neuron leaf.
  - **Fix: a load-balancing loss**, KL(mean leaf probability over the batch ‖ uniform), as in mixture-of-experts models.
  - Result: 99.15% final (99.19% best), against tern80's 99.45%: **−0.30 points**. The integer pipeline agrees with PyTorch on 99.88% of images (tern80: 99.98%); a few routing decisions flip with the integer requantization.
  - Test images use the 8 leaves evenly (10.9–13.8% each). Each leaf specialises in 2–3 digits (e.g. one leaf takes 94% of the 0s, another the 7s and 9s); per-leaf accuracy is 98.5–99.6%.
- **Compiler:** a new IR op `fff` (reference semantics in `interp.py`, an independent numpy golden model `GoldenFFF`, an interval bound).
  - Routing: int8 dot products with `SMLAD` on the widened input.
  - Leaves: each leaf's two layers are emitted by the ordinary linear emitters in the format chosen for their weight type (`--ternary`, `--int4`), one function per leaf, called through a table.
  - Dense leaves share one int16 copy of their input. Packed int4 needs widths divisible by 8, so the 12-neuron leaves use `--int4 code` or `dense`.
  - Checks: 7 variants bit-exact on the PC (10k images), 14 board finals bit-exact (10k images).

  | Replaced layers, one core (cycles) | Dense twin: fc1 + fc2 | Tree (`fff` stage) | Speedup |
  |---|---|---|---|
  | SRAM, code (fc1 code, fc2 dense) | 67.2k | 17.5k | **3.8x** |
  | SRAM, dense | 94.2k | 22.3k | **4.2x** |
  | Flash, code (code in SRAM, int4 dense in flash) | 109.1k | 32.5k | **3.4x** |
  | Flash, dense | 645k | 103k | **6.2x** |

  | Whole model (10k images) | Dense twin (tern80) | Tree (fff80) |
  |---|---|---|
  | One core, SRAM, best (twin: code + code; tree: code + dense) | 3.56 ms | **3.23 ms** (4.50x CMSIS-NN) |
  | One core, SRAM, dense | 3.73 ms | 3.26 ms |
  | One core, flash, dense | 8.05 ms, 45.0 KB read from flash | **4.09 ms, 10.7 KB** |
  | One core, flash, fc1/leaf-in as code | 4.11 ms | 3.45 ms |
  | Two cores, SRAM (twin: hybrid code + code; tree: code + dense, tree on core 0) | 1.99 ms | **1.86 ms** (7.79x CMSIS-NN) |
  | Two cores, flash (same formats) | 2.28 ms | 2.57 ms* |

  \* The tree's int4 layers (leaf out, fc3) were dense in flash here, while the twin had every linear layer as code in SRAM (1.3 KB read from flash against 6.3 KB). It is a format difference, not a tree effect; tree + code + code from flash was not measured.

- **Keep criterion** (HANDOFF: ≥3x fewer bytes and lower latency for the tree layers, <1 point accuracy loss): **met.** 6.6x fewer weights read, 3.4–6.2x faster, −0.30 points.
- **Reading:**
  - For MNIST the whole-model gain is small from SRAM (9%), because fc1 + fc2 were only about 13% of the time; the conv stack dominates.
  - Where weights come from flash, the tree halves the whole model (8.05 → 4.09 ms) without moving anything to SRAM (12.9 KB of RAM). The bytes it skips are the bytes that cost.
  - That is the case idea B is for: models whose feedforward weights don't fit in SRAM, where flash traffic sets the speed (F22). The next test is the tiny story model (HANDOFF section 7), where feedforward layers dominate.
  - Training needs care: without load balancing the tree collapses silently, and the soft-phase accuracy hides it (hard routing must be what is evaluated).

### F26. Phase 3: our own ternary story model beats Atome by 0.5 bits/byte; its tree twin runs at 310 tok/s on the Pico 2, 2.1–2.5x faster than the dense twin for 4.7% in bits/byte
- **Models:** MafLM (`train/lm/`), d=128, 4 layers, GQA + ALiBi, byte vocabulary, all linear layers ternary with int8 activations. The dense twin has a 128 → 640 → 128 feedforward layer; the tree twin has a depth-3 fast feedforward tree with 8 leaves of 80, the same total width. About 0.92M weights each, like Atome's 944K.
- **Training:** 2B bytes each on TinyStories V2, Atome's budget (overnight, 2 h 46 min and 3 h 25 min).
- **Quality** (bits/byte on V2-valid, first 2 MB, 256-byte window):
  - Atome: 1.711.
  - **Dense: 1.152. Tree: 1.206.**
  - The integer reference is within 0.0006 bits/byte of PyTorch.
  - The tree's routing stays balanced (leaf entropy 2.98–2.99 of 3 bits).
- **Pico 2, two cores, bit-exact on all four builds:**

  | Build | Positions 0–23 | Whole 256 window | Flash per token |
  |---|---|---|---|
  | Tree, SRAM | **3.22 ms, 310 tok/s** | 4.99 ms, 201 tok/s | 0 |
  | Tree, flash | 8.47 ms, 118 tok/s | 10.41 ms, 96 tok/s | 82 KB |
  | Dense, SRAM | 6.77 ms, 148 tok/s | 8.56 ms, 117 tok/s | 0 |
  | Dense, flash | 20.81 ms, 48 tok/s | 22.82 ms, 44 tok/s | 223 KB |
  | Atome's engine, SRAM, window 24 | 1,414 ms, 0.71 tok/s | – | 0 |

- **Reading:**
  - Idea B holds on a language model: 2.8x fewer weights read per token, 2.1x faster from SRAM and 2.5x from flash, for 0.054 bits/byte.
  - Most of the gain over Atome's engine comes from the design for integer inference (a KV cache instead of recomputing the window, ternary × int8 lookup-table kernels, two cores), not from the tree.
  - Details: [lm.md](lm.md).

## Gate 1

**Met (11 Nov 2026), with a stated scope.** At equal accuracy (99.47% against 99.44%, bit-exact on 10k images), mafc is faster than CMSIS-NN in every like-for-like setup:
- **4.08x on one core** (3.56 ms vs 14.52 ms, both from SRAM; 3.75x before F18);
- 7.28x with both cores against CMSIS-NN's one (1.99 ms, F19);
- 5.22x for a one-core flash build with the linear layers in SRAM, against CMSIS-NN from flash (18.70 ms).

Scope (F5, F17):
- The MNIST margin is inflated by CMSIS-NN's weak kernels for one-input-channel and depthwise convs.
- On a wide conv, the general win from compiling ternary weights into code is 1.2x (33% zeros) to 9x (90% zeros), and it loses at 0% zeros.
- For int4 and int8 layers mafc is only level with CMSIS-NN (F18, F24). On an all-int8 version of the model it is still 3.90x faster on one core (6.86x on two), all of it from the conv stack (F24).
- Energy (step 2b) is not measured yet.

## Open questions and next experiments

| Question | How to test | Status |
|---|---|---|
| Energy per inference (mJ) for each build | USB meter, idle vs busy (step 2b). Tooling ready: firmware command `L`, `harness/energy.py`, `measure.sh energy` | Cable here (13 Nov); the USB meter is too coarse for the Pico 2 (reads 0–4 mA). Needs a finer instrument |
| Bytes actually read from flash per inference | F22: measured with the XIP counters for every flash build (`flash_read_kb`): hybrid 1.2 KB, CMSIS-NN 45.5 KB | Done |
| Idea B: tree over neurons (fast feedforward) | F25: depth-3 tree in place of fc1 + fc2: 6.6x fewer weights read, 3.8–6.2x faster for those layers, −0.30 points (99.15%); keep. Needs load balancing in training. Next: the tiny story model | Done for MNIST |
| Idea C: DMA streaming of weights | F21: the stream FIFO with double-buffered DMA is 31% faster than reading in place for a 64 KB layer; keep. F23: in the compiler (`--stream`), the 30 KB fc1 runs 29% faster on one core and 58% faster than two cores sharing the cache | Done |
| How much does SMLAD save? | All three convs (F13–F15): with the v2 kernel, 1.39x on two cores from SRAM (4.23 → 3.05 ms) | Done |
| Best fc1 format: decode loop, code, bitplanes, lookup tables, sparse lists (idea A/E) | base3, sparse lists, code and dense SXTB16/SMLAD compared (F16): code wins from SRAM, and only code turns zeros into speed. Not tried: bitplanes, lookup tables | Done for 4 formats |
| Where does the remaining time go? (two cores, all linear as code) | Join wait fixed (F19). Now conv1 42%, dw1 31%, pool1 8%, fc1 9%: the conv stack is about 85%. Candidates: fuse pool1 into dw1, conv1 channel pairs | Open |
| Idea A: code vs data vs hybrid | F20: keep the hybrid (linear layers as code, conv weights as data) | Done |
| Does the lead over CMSIS-NN survive a wide-channel model? (F5) | Wide-conv microbenchmark (F17): yes for ternary weights with ≥ ~14% zeros (1.19x at 33%, 1.65x at 50%, 9x at 90%); no for dense ternary (0.86x). F24: int8 layers are level with CMSIS-NN (no zeros to exploit). Still open: a whole wide model | Answered for one layer |
| Make int4/int8 layers competitive (fc2/fc3 were 3.6x slower than CMSIS-NN) | F18: fc2 as code is level with CMSIS-NN (17.4k vs 16.6k cycles). F24: the 4-row dense kernel is level too (int4 fc2 16.5k; int8 fc2 16.5k vs 16.4k, int8 fc1 75.9k vs 70.9k) | Done (level, not ahead) |
| Is the second core's flash penalty about cache thrashing or bandwidth? | Contention measured (F12): moderate on convs, strong on fc1. With fc1 as code in SRAM (F16), the flash build is 2.89 ms against 2.32 ms all-SRAM. The remaining 25% is the convs and fc2 from flash. F22: thrashing contributes; with two cores the all-data build misses 32% more (13.1 vs 9.9 KB per inference) | Answered |

## Caveats

- **Scope:** one board, one model family, a tiny task (MNIST, 16×16 input). Absolute numbers won't transfer; ratios and mechanisms are what matter.
- **Measurement scope:** cycles cover `ref_infer()` on core 0 only. USB transfer time and core 1's idle time are excluded.
- **Profiling overhead:** profiling builds add about 3% (one counter read per stage, per channel). Headline numbers come from builds without it, and stage tables from builds with it.
- **Flash state:** flash-build timings depend on the cache, and the first image after a reset is slowest. Means over 10k images absorb this; the max does not.

## Where the data and details are

| What | Where |
|---|---|
| Timed runs (one row per build) | `results/phase0.csv`, `results/phase1.csv` (same columns; see [results/README.md](../results/README.md)) |
| Stage cycles | `results/profile.csv`, `results/profile_stages.csv`, `results/profile_stages_dual.csv` |
| Core-contention test (F12) | `results/contention.csv`, `results/contention_stages.csv`, `results/contention_stages_dual.csv`; `harness/contention.py` |
| Dated narrative | [LOG.md](LOG.md) |
| Baseline design | [cmsis-nn-baseline.md](cmsis-nn-baseline.md) |
| Integer semantics and licence | [bitnetmcu-integer-inference.md](bitnetmcu-integer-inference.md) |
| Compiler design and analysis | [compiler.md](compiler.md) |
| Toolchain and gotchas | [pico-toolchain.md](pico-toolchain.md) |
| Re-measure everything | `harness/phase0_matrix.sh final`; `harness/measure.sh final results/phase1.csv mafc:tern80:flash mafc:tern80:ram mafc2:tern80:flash mafc2:tern80:ram` |
