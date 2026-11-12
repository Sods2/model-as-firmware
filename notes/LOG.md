# Lab log

Newest first. Numbers are on the full 10k MNIST test set unless stated otherwise. Raw results are in [results/](../results/).

## 12 Nov 2026 (evening): energy tooling, int8 layers, 4-row dense kernel

- **Energy tooling (step 2b, ready for the cable):** firmware command `L` + 2 bytes runs `ref_infer()` back to back for N seconds with no USB traffic; `harness/energy.py` walks through an idle and a busy phase and asks for the meter readings (mWh, mW or mA); `measure.sh energy` builds, flashes and runs it per build. Checked with `--no-meter`: 2,508 inferences in 5 s = 1.994 ms, the same as the final.
- **int8 model:** `train/configs/int8_cnn80.yaml`, the tern80 CNN with all linear layers int8. 99.43% final (99.50% best, epoch 45); integer pipeline 99.45%.
- Golden model, ref_c and the front end decode BitNetMCU 8-bit layers (`decode_8bit`, `fc_8bit`); `compile.py --int8 {plain,dense,code}`; CMake `-DINT8=`; `measure.sh` 8th field. tern80 output unchanged by the plumbing (regenerated files identical).
- First board profile: CMSIS-NN beat our dense format on int8 linear layers (fc1 71k vs 111k cycles).
- **Dense kernel, 4 rows at a time** (activations loaded once per 4 rows) and `sxtb16 ror #8` written directly: fc1 111k → 76k, fc2 22.6k → 16.5k. 2 rows: 85k; 8 rows spill: 87k.
- The helper change touched every `--simd` file: all 53 generated variants regenerated and bit-exact on the PC.
- Finals (10k, all bit-exact): int8 model one core SRAM 3.73 ms (CMSIS-NN 14.54 ms, 3.90x), two cores 2.12 ms (6.86x). Code no longer beats dense from SRAM (3.82 ms). tern80 fc2 dense is now faster than fc2 as code. FINDINGS F24.

## 12 Nov 2026 (later): idea C in the compiler (`--stream`)

- `compile.py --stream`: dense-format linear layers stream their weights from flash (XIP stream FIFO, DMA, 8-row double buffer, SRAM function, core 0 only). Bit-exact on the PC (memcpy blocks) and on the board.
- fc1 dense (30 KB), flash builds:
  - fc1 stage: 308k → **219k** cycles on one core; two cores in place 526k → **219k** streamed.
  - Whole model: 6.29 → 5.12 ms (one core), 6.65 → 3.97 ms (two cores).
- Side effect: fc1 no longer thrashes the cache, so dw2 (52k → 30k) and fc2 (105k → 72k) got faster too.
- The XIP counters don't see streamed bytes (they bypass the cache); noted under F22. FINDINGS F23.

## 12 Nov 2026: flash traffic measured (XIP counters)

- `pico_main.c`: clears `xip_ctrl_hw->ctr_acc/ctr_hit` before each `ref_infer()` and sums them; new `X` command. `run.py` reads it and fills `flash_read_kb` = misses × 8 B per inference.
- First attempt failed to build: a heredoc turned `\n` inside a C string into a real newline. measure.sh hides compiler output, so the batch stopped silently (gotcha in pico-toolchain.md).
- 10k-image finals of 10 builds:
  - flash read per inference: ref_c 8.0 KB; CMSIS-NN 45.5 KB; mafc all-data 7.0–13.1 KB; **hybrid 1.2–1.3 KB**; all-code 49–51 KB; SRAM build 0.
  - Hit rates are 98.4–99.96%; a miss costs ~65–110 cycles. FINDINGS F22.

## 12 Nov 2026 (night): idea C, DMA streaming from flash

- New benchmark `firmware/bench/stream` + `harness/bench_stream.py`: a dense int8 256→256 layer, 64 KB of weights in flash, the same SMLAD kernel fed seven ways. All bit-exact.
- Cycles:
  - SRAM 216k (compute only);
  - in place through the cache 646k; through the uncached alias 654k;
  - DMA from the alias 723k sequential, 822k double-buffered (worse);
  - DMA from the XIP stream FIFO 697k sequential, **443k double-buffered (−31%)**.
- Idea C: **keep** (target ≥20%). Use the stream FIFO, not the alias. The compiler doesn't need it for MNIST yet: nothing in the best build streams.
- First run had the input vector in flash (`const`), which inflated every flash variant; it was fixed (input copied to SRAM), and those rows are labelled superseded.
- Found: the RP2350 has XIP hit/access counters (`ctr_hit`, `ctr_acc`). pico-toolchain.md was corrected; flash traffic is now measurable.

## 12 Nov 2026 (later): idea A three-way test; join wait fixed

- `--conv code`: conv weights as immediates, one SMLAD function per channel (240 functions), for the all-code build. Bit-exact on the PC and the board.
- All-code (10k images): 1.95 ms two cores SRAM, 3.58 ms one core SRAM, but 6.70 / 6.29 ms from flash (the conv code thrashes the XIP cache).
- Surprise: in the all-code build the join wait was 29 cycles instead of 24k. The cause was core 1's inlined copy of the SMLAD kernel: worse code, with spills in dw1. Fix: `noinline` kernel in two-core builds (F19). Hybrid two cores: 2.15 → **1.99 ms (7.28x CMSIS-NN)**; flash 2.35 → 2.27 ms; all-data two cores 3.05 → 2.91 ms. One-core builds stay inline (noinline was 1% slower there).
- Idea A verdict (F20): **keep the hybrid**.
  - SRAM: all-data / hybrid / all-code = 2.91 / 1.99 / 1.95 ms (two cores) and 5.23 / 3.56 / 3.58 ms (one core).
  - Flash: 4.32 / 2.27 / 6.70 ms and 5.97 / 3.59 / 6.29 ms.
  - Flash used: 52 / 156 / 197 KB.
  - Rule: weights with no reuse → code; weights with reuse → data.

## 12 Nov 2026: fc2/fc3 as code

- `compile.py --int4 {packed,code,dense}`. Code groups each row by weight magnitude (one add/sub per weight, one multiply per magnitude). All 12 variants are bit-exact on the PC.
- fc2 (one core, SRAM): 59.0k → 17.4k cycles as code (22.6k dense), against CMSIS-NN's 16.6k: parity. fc3: 7.1k → 2.9k (CMSIS-NN 2.1k).
- Whole model, 10k images, bit-exact:
  - **2.15 ms** two cores SRAM (6.77x); 2.35 ms two cores flash;
  - **3.56 ms one core SRAM = 4.08x CMSIS-NN like-for-like** (was 3.75x); 3.59 ms one core flash;
  - z70 2.09 ms, z90 2.03 ms. FINDINGS F18.

## 11 Nov 2026 (late night, 2): fairness check, Gate 1

- Audited the CMSIS-NN baseline. DSP is on, dw1/dw2 use the dedicated 3×3 depthwise kernel, and conv1 uses im2col (no faster path exists for one input channel). Nothing to fix.
- Cores: the headline 6.25x is two cores against one. One core against one core: 3.87 vs 14.52 ms = **3.75x**.
- Per layer (one core, SRAM): conv1 4.6x, dw1 4.8x, dw2 3.7x, fc1 1.4x faster; fc2/fc3 are 3.6x *slower* than CMSIS-NN (int4 decode).
- New wide-conv microbenchmark (`firmware/bench/conv_wide`, `harness/bench_conv.py`): a 3×3 conv, 10×10×32 → 8×8×32, ternary weights at 0/33/50/70/90% zeros. CMSIS-NN against weights-as-code, bit-exact on the PC and the board.
  - CMSIS-NN: 1.26M cycles at any sparsity.
  - Code: 1.46M / 1.05M / 761k / 458k / 140k cycles.
  - Break-even at about 14% zeros.
- The first benchmark run had 3 cases and a different random draw; its rows in `bench_conv.csv` are labelled as superseded.
- Gate 1 is met, with the scope written down in FINDINGS (F17, "Gate 1"). Energy is still open.

## 11 Nov 2026 (late night): fc1 formats; fc1 as code (idea A)

- `compile.py --ternary {base3,sparse,code,dense}`, `-DTERNARY=` in CMake, and `build:model:mem:ternary` in `measure.sh`. All 18 new variants are bit-exact on the PC.
- fc1 stage cycles (tern80), base3 → code: 256k → 50k on one core and 139k → 28k on two cores, from SRAM. sparse (94k/57k) and dense (103k/74k) are in between.
- From flash, every format bigger than the 16 KB XIP cache is slower than base3. code from flash took 611k cycles (1.3M on two cores). Fix: the code function always goes in SRAM (`__not_in_flash_func`); now 51k/29k in flash builds too.
- Sparsity pays only with code: fc1 28k (tern80) → 19k (z70) → 11k (z90) on two cores; base3 stays at 131–139k.
- Whole model, 10k images, bit-exact:
  - two cores: **2.32 ms from SRAM (6.25x faster than CMSIS-NN)**, 2.89 ms from flash with fc1 in SRAM;
  - one core: 3.87 ms (SRAM), 4.11 ms (flash);
  - z70 2.26 ms, z90 2.21 ms (two cores, SRAM). FINDINGS F16.
- Measurement bug found and fixed: `run.py` RAM = data + bss missed the code copied to SRAM. Berkeley `size` counts executable `.data` as text, so a flash code build showed 9.7 KB of RAM instead of 97 KB. RAM now = sections at SRAM addresses (`size -A`), excluding heap and stacks. Rows before this change used the old formula (see results/README.md).
- Profile rows from the first conv1-SMLAD try were relabelled "first kernel ... superseded" in `profile.csv`.

## 11 Nov 2026 (night): conv1 on SMLAD, SMLAD kernel v2

- conv1 now runs on SMLAD. It is a broadcast conv, so the int8 image is widened to int16 once per inference (`widen_img`) and every channel runs the depthwise SMLAD kernel on it. No `SXTB16` needed.
- First try (old kernel): conv1 only 451k → 423k cycles. The loop spilled: 45 instructions, 11 stack accesses (6 kernel words + sums + pointers don't fit 13 registers). Profile rows labelled "first kernel".
- Kernel v2 for all three convs:
  - kernel words A = (k0,k1), B = (k1,k2), so no halfword repack;
  - `SMUAD` to start the sums;
  - `USAT` for ReLU + shift;
  - one word store per int16 output pair;
  - x loop kept rolled (full unroll spilled more).
- Result: conv1 and dw1 loops are 24 instructions with no spills (were 45 and 37).
- Stage cycles, one core, SRAM: conv1 451k → 232k, dw1 273k → 170k, dw2 34k → 24k.
- Whole model, 10k images, bit-exact: two cores SRAM **4.23 → 3.05 ms (4.76x faster than CMSIS-NN)**; flash 5.40 → 4.38 ms; one core 7.44 → 5.23 ms (SRAM), 8.11 → 5.97 ms (flash). FINDINGS F15.
- fc1 is now the largest stage (30% on two cores from SRAM). Next: fc1 formats / idea A.
- `harness/asm_loops.py` gained `--max N` (to see unrolled loops) and a `[sp]` column (spills).

## 11 Nov 2026 (evening): branch-free ReLU in the back end

- `relu_shift()` in the back end: every conv ReLU is now written clamp-then-shift. All 12 generated variants changed and are bit-exact on the PC (10k images); the 8 tern80 board builds are bit-exact (10k images).
- No branch or IT block left in any conv loop. Stage cycles (SRAM, one core): SMLAD conv1 −14%, dw1 −15%; v0 dw1 −20% (most of the old F6 regression).
- Whole model: v1 + SMLAD on two cores, SRAM **4.36 → 4.23 ms (3.44x faster than CMSIS-NN)**; v0 8.83 → 8.18 ms. FINDINGS F14.
- `measure.sh` takes `NOTES="..."` for the CSV notes column; the new rows are labelled `branch-free ReLU`.
- Gotcha: `run.py --target host` logs to `results/phase0.csv` by default. My 12 host checks appended rows there (mislabelled `ref_c`); the file was restored with `git checkout`.

## 11 Nov 2026 (later): why conv1 got slower with SMLAD

- Disassembled conv1 in mafc_v0 and mafc_v0_simd (RAM profile ELFs). The int16 store makes gcc compile the 2nd ReLU of each unrolled pair as a branch, and rotates the loop (2 to 3 taken branches per pair instead of 1). 49% of conv1 sums are negative. Measured 59.5 → 66.6 cycles per pair.
- Standalone compile check: clamp-then-shift or a `bic … asr #31` mask gives branch-free code, shorter than v0's. Not yet timed on the board. Details: compiler.md and FINDINGS F13.
- Also found that the watcher loop from the SMLAD run never ended: its own command line contained `harness/run.py`. Gotcha added to pico-toolchain.md.

## 11 Nov 2026: compiler v1 step 2, SMLAD on the depthwise convs

- New `compiler/mafc/bounds.py`: an exact interval analysis of every tensor's range over all int8 inputs, checked against 20k images. It corrects F9: dw1 outputs are ≤ 19,743 (tern80), not 60,830, so they fit int16. dw2's input is then int16 too.
- `compile.py --simd`: int16 planes where proven, and an SMLAD kernel (2 outputs per step, weights pre-packed) for depthwise convs with int16 input, so dw1 and dw2. Builds `mafc_s` / `mafc2_s` in `measure.sh`. Earlier generated files are byte-identical. Bit-exact on the PC and the board, 10k images, all three models on the PC and tern80 on the board.
- Timed (tern80): one core 8.93 ms (flash) / 8.29 ms (SRAM); two cores 5.54 ms (flash) / **4.36 ms (SRAM)**. That's 1.10x over v1 on two cores from SRAM, and **3.33x faster than CMSIS-NN**.
- Profile: dw1 −29%, dw2 −35%, conv1 +12% (int16 store; gcc code generation). conv1 is next (SXTB16 + SMLAD).
- Process note: a `measure.sh` started with `&` kept running unseen and held COM3 while a second run started. Recorded in pico-toolchain.md; the orphan's results were valid and are the ones logged.

## 11 Nov 2026: do the two cores slow each other down? (IDEAS U3, test 1)

- No new code. Profiled mafc v0 (one core) and v1 (two cores), from flash and from SRAM, twice each (`results/contention*.csv`). Compared each core's share of the conv layers with half the single-core time (`harness/contention.py`).
- Conv layers: **+6–8% per core from SRAM** (shared SRAM/bus) and **+12–16% from flash**. The join wait is 21k cycles from flash vs 12k from SRAM, so core 1 is starved more.
- fc1: the flash penalty is +18.5% with one core but **+54.6% with two**, so the second core gives only 1.40x there. Its 6 KB of weights get evicted between inferences and both cores stream them at once.
- Next (U1): test fc1's weights in SRAM first, then conv code in SRAM, then both. Written up as finding F12.
- One small thing: the single-core SRAM conv time differs by about 1% between two builds of the same source (503k vs 507k cycles), probably code layout. Comparisons of under 1% between separate builds aren't meaningful.

## 11 Nov 2026: Phase 1, compiler v1 step 1, second core

- `compile.py --cores 2` splits the conv channels (40/40) and fc1/fc2 rows across both M33 cores. Core 1 runs a job loop on the SIO FIFO with DMB fences, started once by a new `ref_init()` boot hook. The PC build runs core 1's share inline, so the same split is verified there.
- Bit-exact on 10k images (PC and board) for tern80, z70 and z90.
- Timed: **4.80 ms from SRAM** (v0: 8.83 ms, so 1.84x) and **5.80 ms from flash** (v0: 9.33 ms, so 1.61x). Against CMSIS-NN in the same memory that's **3.02x** (SRAM) and 3.23x (flash). Against the hand-written ref_c from SRAM it's 1.98x.
- Profile: core 0 waits only 12k cycles (SRAM) for core 1 at the join, so the halves are balanced. From flash, fc1 gains just 1.40x: both cores share the 16 KB XIP cache. That's the flash-bandwidth limit again.
- `run.py` now writes dual-core stage rows to `*_stages_dual.csv` and checks that columns match. Before this, mixed stage sets would have produced misaligned rows.

## 11 Nov 2026: Phase 1, compiler v0 (`compiler/mafc`, branch `phase1-compiler-v0`)

- Built the compiler pipeline: a front end that reads the exported header, an IR with exact integer semantics, a numpy interpreter, and an RP2350 C back end. The back end emits one function per layer with constant sizes, packed weights as data, static buffers, and channel fusion. See [compiler.md](compiler.md) and [compiler/README.md](../compiler/README.md).
- Correctness chain, all bit-exact on 10k images for tern80, z70 and z90: interpreter == golden, generated C on the PC == golden, generated C on the Pico == golden. Our own ternary packer round-trips.
- Timed (10k images, `results/phase1.csv`): **9.33 ms from flash, 8.83 ms from SRAM** for tern80. The hand-written ref_c took 10.21 and 9.52 ms, so this is 1.09x and 1.08x faster. Against the CMSIS-NN bar (14.52 ms from SRAM) it's **1.64x faster**. z70: 9.28 ms; z90: 9.22 ms.
- Profile: conv1 is 34% faster (reads the int8 image directly) and dw1 is 12% slower (gcc code generation for constant loops). fc layers are unchanged.
- Bound analysis: conv1 outputs are provably ≤ 3,448, so dw1 can use SMLAD on int16 exactly. dw1 and dw2 outputs aren't provably int16.
- Correction: the base tern80 fc1 is **50.2%** zeros, not 24.8% (fixed in `train/README.md`).
- Added `harness/measure.sh` (any list of build:model:mem specs); flashing now retries up to 5 times.

## 11 Nov 2026: live execution view (`harness/bridge.py`, `ui/live.html`)

- Added a browser view of a run. It shows the pipeline with per-stage cycle shares, feature maps for each layer, FC1's ternary weights, and logits against the golden model. It also has slow-mo and replay modes and pauses with a red banner on any golden mismatch.
- `golden.py` gained `trace(img)` (every stage's output for one image) for both golden models. `logits()` is unchanged: still 2000/2000 bit-exact against the host build.
- The bridge picks the model from the board's INFO line. A board flashed with `tern80z90` would otherwise be checked against `tern80` and fail every image.
- Checked on the board with a `mafc_v0_prof` build (500 images): 9.43 ms mean, 100% golden match. That figure is from the live view, not a logged run.
- Design mockups are on a private canvas: (private link removed). It covers live view, build comparison and states. The running page follows its "Live execution" board.

## 11 Nov 2026: Phase 0 steps 4–6 (Gate 0 correctness) while parts are in the post

**Hardware status:** a USB-C cable for the power meter and a Raspberry Pi Debug Probe are ordered, so step 2b (energy) waits for them. Everything below runs over the Pico's own USB.

**Step 4: plain-C reference** (`firmware/baselines/ref_c/`)
- Wrote our own C implementation of BitNetMCU's integer arithmetic rather than copying upstream's C, which is GPL-3 (see [bitnetmcu-integer-inference.md](bitnetmcu-integer-inference.md)).
- Wrote a numpy golden model (`harness/golden.py`) that decodes the exported header and repeats the C arithmetic. Upstream's own Python "quantized" inference differs from its C engine in the conv stack, so it can't serve as an exact reference.
- The integer pipeline gets 99.47% on tern80 and agrees with the PyTorch model on 99.98% of images (2 differ). z70: 99.47%; z90: 99.51%.
- PC build (MinGW gcc) and Pico build: logits identical to the golden model on 10,000/10,000 images, for all three models.
- The largest possible intermediate sum is about 1.4M, 0.07% of the int32 range.

**Step 6: harness** (`harness/run.py`, `harness/phase0_matrix.sh`)
- USB-serial protocol: `?` info, `I`+256 bytes runs one image and returns pred, cycles and logits, `T` returns stage cycles. A 10k run takes about 3 minutes.
- Logs one row per run to `results/phase0.csv` with the HANDOFF §6 columns. Energy and flash bytes read stay blank for now.

**Step 5: CMSIS-NN int8 baseline** (`firmware/baselines/cmsis_nn/`)
- Found that the network has no biases and only ReLU/max-pool, so it's positively homogeneous. That means a static-scale int8 build can use the very same integer weights, and it's the same network.
- Calibrated on 10k *training* images. Accuracy is 99.44%, and the board output is bit-exact with `GoldenS8` on 10k images.
- Details and profiling are in [cmsis-nn-baseline.md](cmsis-nn-baseline.md).

**Profiling findings** (stage cycles, tern80):
- CMSIS-NN's fc1 is **7.4x faster from SRAM than from flash** (524k -> 71k cycles). Its 30 KB of int8 weights don't fit the 16 KB XIP cache. The packed ternary weights (6 KB) fit, so ref_c barely cares where its weights live. This is the flash-bandwidth premise, confirmed early.
- From SRAM, int8 + SMLAD beats ternary decoding in fc1 (71k vs 255k cycles). Decoding 10 trits per uint16 costs about 8 cycles per weight. That's a clear target for the compiler.
- The conv stack is about 70% of the time. CMSIS-NN handles this shape badly (a single input channel into conv1, and its depthwise 3x3 kernel), so our plain C wins there. Don't overclaim it at Gate 1.

**Final timed runs** (10k images, profiling off; full table in [results/README.md](../results/README.md)):
- ref_c: 10.21 ms from flash, 9.52 ms from SRAM. CMSIS-NN: 18.70 ms from flash, 14.52 ms from SRAM. All bit-exact.
- The Gate 1 speed bar is CMSIS-NN from SRAM at 14.52 ms, and the plain-C reference is already 1.53x faster than it.
- Sparse z70/z90 gain only 0.5–1% in ref_c, because the packed decode still visits every trit.
- Gate 0: met on correctness and baselines. Energy (step 2b) is pending the USB-C cable.

**Gotchas recorded** in [pico-toolchain.md](pico-toolchain.md): `picotool load -f` sometimes reports failure while BOOTSEL is still appearing; retrying without `-f` works.

## 10 Nov 2026: Phase 0 steps 0–3

- Bring-up: the Pico 2 enumerates and flashes with picotool; the DWT cycle counter is calibrated (150,251 cycles per 1 ms at 150 MHz, ±108 cycles of jitter over 700k).
- Trained BitNetMCU's ternary CNN80: 99.45% final, 99.48% best (99.52% reported upstream). The fc1-sparse variants at 70% and 90% zeros lose no accuracy.
- Patched BitNetMCU export: it silently re-quantized a ternary fc1 as 2-bit. See `train/bitnetmcu.patch`.
- Two of the micro-USB cables on hand are charge-only. The power meter has a USB-C plug, which blocks step 2b.
