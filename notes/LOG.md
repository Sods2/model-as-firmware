# Lab log

Newest first. Numbers are on the full 10k MNIST test set unless stated otherwise. Raw results are in [results/](../results/).

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
