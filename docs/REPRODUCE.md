# Reproducing the results

Step-by-step recipes for every headline claim in the [README](../README.md), with what each one needs and what you should see.

## Before you start

**Hardware the numbers come from** ([HARDWARE.md](HARDWARE.md)): one Raspberry Pi Pico 2 (RP2350) at its default 150 MHz, flashed over SWD with a Raspberry Pi Debug Probe, talking to the PC over its own USB cable. Without a probe, flashing falls back to picotool over USB, which changes nothing in the results. The PC was Windows 11 with Git Bash, and training ran on an RTX 3060 Ti.

**Software:** follow [SETUP.md](SETUP.md) first. Each recipe lists the parts of the setup it needs.

**Conventions:**
- Run everything from the repo root, in bash (Git Bash on Windows).
- `python` is the venv's interpreter ([SETUP.md](SETUP.md), section 2).
- Set `PORT` to your board's serial port (`COM3` and `/dev/ttyACM0` are the defaults on Windows and Linux): `export PORT=COM5`.
- **The measurement scripts append to the CSV you name.** The repo's own `results/*.csv` hold the published numbers, so write your runs to your own file (the examples use `my_results.csv`) and compare.

**How one board measurement works:**
1. `harness/measure.sh` builds the firmware with CMake and Ninja.
2. It flashes the board (`harness/flash.sh`).
3. `harness/run.py` sends the 10,000 MNIST test images over USB serial.
4. For each image the board returns its prediction, the 10 output scores (logits) and the cycle count of the inference alone.
5. Every image's logits must equal the golden model's (an independent Python copy of the same integer arithmetic) exactly; `run.py` exits with an error if one doesn't.
6. One summary row goes to the CSV: accuracy, golden match, latency, flash and RAM size, flash bytes read.

A 10,000-image run takes about 3 minutes, mostly USB round trips; the build and flash take under a minute.

**Expect small differences.** Timings repeat to within 0.2% from SRAM and 1.2% from flash on the same board. Another toolchain version can move them by a few percent. Accuracy and the golden match are exact.

## Contents

| Claim (README) | Recipe | Needs a board | Time |
|---|---|---|---|
| Every build is bit-exact with its golden model | [A](#a-no-hardware-the-golden-model-and-the-pc-build), and every board recipe | no | 5 min |
| Plain C 10.21 / 9.52 ms; CMSIS-NN 18.70 / 14.52 ms (the bar) | [B1](#b1-the-baselines-plain-c-and-cmsis-nn-gate-0) | yes | 20 min |
| Compiler: 3.08 ms on one core (4.7x), 1.67 ms on two | [B2](#b2-the-compilers-best-builds-308-ms-and-167-ms) | yes | 10 min |
| Memory: 137 KB flash, 140 KB SRAM | [B2](#b2-the-compilers-best-builds-308-ms-and-167-ms) | yes | (same run) |
| All-int8 model: 3.9x | [B3](#b3-the-all-int8-model-39x) | yes | 10 min |
| Wide conv: break-even ~14% zeros, 1.19x at 33%, 9x at 90% | [B4](#b4-the-wide-conv-benchmark) | yes | 10 min |
| Idea B (tree): 6.6x fewer weights, 99.15% | [B5](#b5-idea-b-the-tree-model-fff80) | yes | 10 min |
| Training: 99.45% (99.52% reported by BitNetMCU) | [C](#c-training-the-mnist-models) | no (GPU) | 25 min per model |
| Story model: tree 310 tok/s, 2.1x / 2.5x faster than its dense twin | [D1](#d1-train-the-two-twins), [D2](#d2-the-twins-on-the-board) | yes (and a GPU) | 6.5 h training + 20 min |
| Atome LM: 0.71 tok/s; 0.50–0.56 bits/byte gap | [D3](#d3-the-atome-lm-reference) | yes | 20 min |
| Energy | not measured: see [E](#e-not-reproduced-energy) | | |

---

## A. No hardware: the golden model and the PC build

**Needs:** Python and MNIST ([SETUP.md](SETUP.md) 2–3). The PC build also needs a C compiler (SETUP 5).

1. **Golden model against the labels and PyTorch.**

   ```bash
   python harness/run.py --model tern80 --target golden --csv my_results.csv
   ```

   Expected: `accuracy 99.47 %`, `logits == golden 10000/10000`. With the BitNetMCU clone, also `agrees w/ PyTorch 99.98 %` (2 of 10,000 images differ from the float model, an expected rounding effect: [harness/README.md](../harness/README.md#what-matches-pytorch-means)).

2. **The plain-C reference, compiled for your PC.** This is the same C file that runs on the board.

   ```bash
   cd firmware/baselines/ref_c && mkdir -p build-host
   ${HOST_CC:-gcc} -O2 -std=c11 -I../../../train/export -DMODEL_HEADER='"tern80.h"' ref_infer.c host_main.c -o build-host/ref_host_tern80.exe
   cd ../../..
   python harness/run.py --model tern80 --target host --exe firmware/baselines/ref_c/build-host/ref_host_tern80.exe --csv my_results.csv
   ```

   Expected: `logits == golden 10000/10000`, accuracy 99.47%. (To build all four models, use the loop in [firmware/baselines/README.md](../firmware/baselines/README.md).)

3. **Watch it** in the browser: `python harness/launcher.py`, open http://127.0.0.1:8765 and choose MNIST, then "Golden model" or "PC build". The pages are explained in [UI-GUIDE.md](UI-GUIDE.md).

---

## B. On the board: the MNIST model

**Needs:** a Pico 2 connected as in [HARDWARE.md](HARDWARE.md); SETUP 2–4; MNIST; CMSIS-NN for the CMSIS-NN builds.

All builds use the generated C already committed in `firmware/rp2350/mafc/gen/`, so no training is needed. To regenerate it from the model yourself, see [compiler/README.md](../compiler/README.md); the output is byte-for-byte the committed file.

### B1. The baselines: plain C and CMSIS-NN (Gate 0)

```bash
CSV=my_results.csv harness/phase0_matrix.sh final
```

This builds, flashes and measures six builds: the plain-C reference and CMSIS-NN, each from flash and from SRAM, plus the reference for the two sparse model variants.

| Build (CSV `build` column) | Accuracy | Latency | Published |
|---|---|---|---|
| `ref_c` (flash) | 99.47% | 10.21 ms | `results/phase0.csv` |
| `ref_c_ram` (SRAM) | 99.47% | 9.52 ms | |
| `cmsis_nn` (flash) | 99.44% | 18.70 ms | |
| `cmsis_nn_ram` (SRAM) | 99.44% | **14.52 ms**: the bar the compiler is measured against | |

Every row must show `golden_match_pct` 100. CMSIS-NN is checked against its own golden model (`--golden s8`), because its int8 requantization rounds differently ([notes/cmsis-nn-baseline.md](../notes/cmsis-nn-baseline.md)); that's why its accuracy is 99.44%. A re-run of the CMSIS-NN SRAM build gave 14.54 ms, so expect ±0.02 ms.

### B2. The compiler's best builds: 3.08 ms and 1.67 ms

The build spec says which compiler options to use; the fields are explained at the top of [harness/measure.sh](../harness/measure.sh).

```bash
harness/measure.sh final my_results.csv \
    mafc_s:tern80:ram:code:dense:data:nostream:plain:fpoolrp \
    mafc2_s:tern80:ram:code:dense:data:nostream:plain:fpoolrp
```

- `mafc_s` is one core with the DSP (SMLAD) kernels, `mafc2_s` the same on both cores.
- `ram` runs from SRAM.
- `code:dense` puts FC1's ternary weights into instructions and FC2/FC3 into a dense 4-row format.
- `fpoolrp` fuses each conv with the max-pool after it, and computes conv1 two rows at a time.

| Build (CSV `build`) | Latency | Flash | RAM | vs the bar (14.52 ms) |
|---|---|---|---|---|
| `mafc_v0_simd_code_i4dense_fpool_rp_ram` (one core) | **3.08 ms** | 137.0 KB | 139.8 KB | **4.7x**, like for like |
| `mafc_v1_dual_simd_code_i4dense_fpool_rp_ram` (two cores) | **1.67 ms** | 133.3 KB | 136.8 KB | 8.7x, but against CMSIS-NN on one core |

Both must show 10000/10000 golden matches and 99.47% accuracy. The flash and RAM columns are the memory claim: about 1.5x CMSIS-NN's flash (94 KB) and 1.1x its SRAM (127 KB).

### B3. The all-int8 model: 3.9x

The same network with every linear layer in int8 (model `int8_80`), so the gain can't come from ternary weights:

```bash
harness/measure.sh final my_results.csv \
    mafc_s:int8_80:ram:base3:packed:data:nostream:dense \
    cmsis_nn:int8_80:ram
```

Expected: the mafc build (`mafc_v0_simd_i8dense_ram`) at **3.73 ms**, CMSIS-NN (`cmsis_nn_ram`) at **14.54 ms**: 3.9x. Accuracy 99.45% and 99.43% (the same int8 weights, rounded differently). All of the lead comes from the conv stack ([FINDINGS F24](../notes/FINDINGS.md)).

### B4. The wide-conv benchmark

One synthetic 3×3 conv layer with 32 input and 32 output channels, the shape CMSIS-NN is built for, with random ternary weights at several fractions of zeros. Every kernel's output is checksum-checked on the board.

```bash
python harness/bench_conv.py gen             # rewrites firmware/bench/conv_wide/gen/ (committed; the output is identical)
. harness/env.sh
cmake -S firmware/bench/conv_wide -B firmware/bench/conv_wide/build-ram -G Ninja -DCMAKE_BUILD_TYPE=Release -DRUN_FROM_RAM=ON
ninja -C firmware/bench/conv_wide/build-ram
. harness/flash.sh && flash firmware/bench/conv_wide/build-ram/bench_conv.uf2
python harness/bench_conv.py run --port "$PORT" --csv my_bench_conv.csv
```

Expected, weights-as-code against CMSIS-NN: 0.86x (slower) with no zero weights, break-even around 14% zeros, 1.19x at 33%, 1.65x at 50%, 2.74x at 70% and about 9x at 90%. Published in `results/bench_conv.csv`; analysis in [FINDINGS F17](../notes/FINDINGS.md). For the flash variant use `-DRUN_FROM_RAM=OFF` and a `build-flash` folder.

### B5. Idea B: the tree model (`fff80`)

`fff80` replaces FC1 + FC2 with a small tree of neurons that reads one leaf per input (fast feedforward network). Its trained weights are committed (`train/export/fff80.npz`).

```bash
harness/measure.sh final my_results.csv \
    mafc_s:fff80:ram:code:dense:data:nostream:plain:fpoolrp \
    mafc2_s:fff80:ram:code:dense:data:nostream:plain:fpoolrp
```

Expected: accuracy **99.15%** (against 99.45% for the PyTorch model of its dense twin), 10000/10000 golden matches, 2.77 ms on one core and 1.53 ms on two. The tree reads 6.6x fewer weights; the whole model gains little because the conv stack dominates ([FINDINGS F25](../notes/FINDINGS.md)).

---

## C. Training the MNIST models

**Needs:** SETUP 2–3 with the BitNetMCU clone and its patch; a GPU (about 24 s per epoch on an RTX 3060 Ti, so 25 minutes for 60 epochs).

The trained weights are committed (`train/models/*.pth`, exported to `train/export/`), so this is only needed to check the training claim. From `train/BitNetMCU`:

```bash
cd train/BitNetMCU
python training.py --params ../configs/ternary_cnn80.yaml
```

Expected: final test accuracy about **99.45%** (BitNetMCU reports 99.52% for this configuration). One run with one seed was made per model, so a few hundredths of a percent either way is normal. The other configs and the export step are in [train/README.md](../train/README.md).

A retrained model has different weights, so its exported header won't match the committed `train/export/*.h`. To run your own model through the pipeline, export it there and regenerate the C with `compiler/compile.py`.

---

## D. The tiny story model (Phase 3)

### D1. Train the two twins

**Needs:** SETUP 2–3 with TinyStories (train and valid); a GPU: about 6.5 hours for both twins on an RTX 3060 Ti. A C compiler for the PC check.

MafLM is our own ternary byte-level story model, trained twice with the same budget and data: once with ordinary feedforward layers (the **dense** twin) and once with tree feedforward layers (the **tree** twin).

```bash
bash train/lm/overnight.sh
```

For each twin, this:
1. trains for 61,000 steps (2.0B bytes);
2. checks the integer reference against PyTorch;
3. exports `train/runs/lm_<twin>.npz`;
4. regenerates `firmware/rp2350/maflm/gen/maflm_lm_<twin>.c`;
5. checks the C runtime on the PC bit for bit.

The summary is in `train/runs/overnight.txt`. Expected bits per byte on the validation text: about **1.15 (dense)** and **1.21 (tree)**.

`STEPS=40 EVAL=20 TAG=dry bash train/lm/overnight.sh` runs the whole pipeline in a couple of minutes, to check the setup first.

**Your training run gives different weights from the published ones.** It overwrites the committed `gen/maflm_lm_*.c` with C for your model, so the firmware and your `.npz` stay consistent; the timings don't depend on the weights.

### D2. The twins on the board

**Needs:** D1 done (the `.npz` files, and the gen files that match them); a Pico 2; TinyStories valid.

Build the four configurations (dense and tree, weights in SRAM or flash, both cores), then flash and measure each:

```bash
. harness/env.sh
for twin in tree dense; do for mem in ram flash; do
  ram=OFF; [ $mem = ram ] && ram=ON
  dir=firmware/rp2350/maflm/build-lm_$twin-$mem-dual
  cmake -S firmware/rp2350/maflm -B $dir -G Ninja -DCMAKE_BUILD_TYPE=Release -DMODEL=lm_$twin -DRUN_FROM_RAM=$ram -DDUAL_CORE=ON
  ninja -C $dir
  . harness/flash.sh && flash $dir/maflm.uf2
  python harness/lm_run.py --port "$PORT" --npz train/runs/lm_$twin.npz --csv my_phase3.csv
done; done
```

`lm_run.py` first checks 2 × 256 positions bit for bit against the integer reference (argmax and a 64-bit checksum of all logits), then times greedy generation.

| Build | Positions 0–23 | Whole 256-byte window |
|---|---|---|
| tree, SRAM | 3.22 ms/byte = **310 bytes/s** | 201 bytes/s |
| dense, SRAM | 6.77 ms/byte = 148 bytes/s | 117 bytes/s |
| tree, flash | 8.47 ms/byte = 118 bytes/s | 96 bytes/s |
| dense, flash | 20.81 ms/byte = 48 bytes/s | 44 bytes/s |

The tree twin is 2.1x faster from SRAM and 2.5x from flash over the first 24 positions. Published in `results/phase3.csv`; details in [notes/lm.md](../notes/lm.md).

### D3. The Atome LM reference

**Needs:** the Atome clone (SETUP 3); TinyStories valid; a Pico 2.

[Atome LM](https://github.com/TilelliLab/atome-lm) is a published ternary story model with its own C engine. It is an **outside reference only**, not a controlled comparison; read "Fairness of the Atome comparison" in [notes/lm.md](../notes/lm.md) before quoting it.

```bash
python train/atome/ref.py --checkpoint train/atome-lm/checkpoints/atome_1m_v1.pt --max-seq 24 --json my_atome_ref.json
. harness/env.sh
cmake -S firmware/baselines/atome -B firmware/baselines/atome/build-s24-ram -G Ninja -DCMAKE_BUILD_TYPE=Release -DMAX_SEQ=24 -DRUN_FROM_RAM=ON
ninja -C firmware/baselines/atome/build-s24-ram
. harness/flash.sh && flash firmware/baselines/atome/build-s24-ram/atome.uf2
python harness/atome_run.py --port "$PORT" --ref my_atome_ref.json --csv my_phase3.csv
python train/atome/fairness.py           # the quality comparison (needs D1's .pt files)
```

Expected:
- `ref.py`: 1.711 bits/byte at a 256-byte window.
- The board: bit-exact with the PC reference, at **0.71 tokens/s** from SRAM (0.64 from flash, with `-DRUN_FROM_RAM=OFF`).
- `fairness.py`: the table in notes/lm.md, where both MafLM twins score 0.50–0.56 bits/byte better on this text.

The speed gap compares engines, not models: Atome's engine recomputes the whole context for every byte.

---

## E. Not reproduced: energy

Energy per inference hasn't been measured: the USB power meter used reads the Pico 2's current only in 1 mA steps. The tooling is ready (`harness/measure.sh energy`, [harness/README.md](../harness/README.md)) for a finer meter such as an INA226 breakout or a Nordic PPK2. Until then, the results are about speed only.
