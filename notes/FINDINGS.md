# Findings so far

*Consolidated 11 Nov 2026, covering Phase 0 and Phase 1 up to compiler v1 (dual core) the core-contention test, and SMLAD on the depthwise convs. One place to read and analyse the conclusions. Each finding gives its evidence, where the raw data lives, and what it means for the project. The narrative is in [LOG.md](LOG.md), and the details are in the topic notes linked below.*

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
| mafc v1 + SMLAD (two cores) | flash | 5.40 ms | 0.81M | 3.46x | 50.9 KB | 10.8 KB | `results/phase1.csv` |
| **mafc v1 + SMLAD (two cores)** | **SRAM** | **4.23 ms** | **0.63M** | **3.44x** | 50.8 KB | 61.6 KB | `results/phase1.csv` |

All mafc rows were re-measured on 11 Nov with the branch-free ReLU (F14), which sped up every mafc build. The earlier numbers (v0 8.83 ms, v1 4.80 ms, v1 + SMLAD 4.36 ms from SRAM) are in F13, `results/phase1.csv` (rows with an empty `notes` column) and `results/README.md`.

The Gate 1 bar is CMSIS-NN in its best setup (SRAM, 14.52 ms). Accuracy: 99.47% for every ternary-pipeline build and 99.44% for CMSIS-NN, against 99.45% for the PyTorch model. Energy is **not measured yet** (step 2b, waiting for the USB-C cable).

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
- **Meaning:** much of our 1.5–3x lead over CMSIS-NN comes from this model's unusual shape (a single-input-channel first layer, depthwise convs). That is a weakness of the baseline more than a strength of our method. **Don't overclaim at Gate 1.** Test on a wide-channel model too before drawing general conclusions.

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
- **Meaning:** conv1 is now the biggest stage (about 40%). It needs its own SIMD path (`SXTB16` on the int8 image + `SMLAD`) and a look at the generated assembly. Gains from flash are smaller, because flash time is partly memory-bound (F2, F12).

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

## Open questions and next experiments

| Question | How to test | Status |
|---|---|---|
| Energy per inference (mJ) for each build | USB meter, idle vs busy (step 2b) | Waiting for cable |
| Bytes actually read from flash per inference | No XIP counters on the RP2350 ([pico-toolchain.md](pico-toolchain.md)); estimate from weight layout, or the Debug Probe | Open |
| How much does SMLAD save? | Done for dw1 and dw2 (F13), 1.07–1.10x overall. The conv1 int16-store regression is fixed (F14). Next: conv1 SIMD (SXTB16) | dw1/dw2 done; conv1 SIMD next |
| Best fc1 format: decode loop, code, bitplanes, lookup tables, sparse lists (idea A/E) | Three-plus builds of fc1, same harness | Next |
| Does the lead over CMSIS-NN survive a wide-channel model? (F5) | Train or borrow a model with many input channels | Not started |
| Is the second core's flash penalty about cache thrashing or bandwidth? | Contention measured (F12): moderate on convs, strong on fc1. Next: fc1 weights in SRAM vs conv code in SRAM vs both (U1) | Measured; placement test next |

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
