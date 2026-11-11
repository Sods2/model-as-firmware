# results/

| File | What |
|---|---|
| `phase0.csv` | One row per timed run (10k MNIST test images, profiling off), written by `harness/run.py` |
| `phase1.csv` | Phase 1 timed runs (mafc compiler builds) |
| `profile.csv`, `profile_stages.csv` | Profiling builds (200 images, stage counters on). Stage cycles are for test image 0 |
| `profile_stages_dual.csv` | The same for dual-core builds (extra `join` stage: core 0 waiting for core 1) |
| `contention.csv`, `contention_stages.csv`, `contention_stages_dual.csv` | Core-contention test: mafc v0 and v1 profiling builds, flash and SRAM, 2 repeats each. Analyse with `harness/contention.py` (notes/FINDINGS.md F12) |

## Columns (`phase0.csv`)

| Column | Meaning |
|---|---|
| `build` | Firmware build as the board reports it: `ref_c` / `cmsis_nn`, plus `_ram` if the whole program runs from SRAM (`copy_to_ram`) |
| `accuracy_pct` | Against the MNIST labels |
| `golden_match_pct` | % of images whose logits equal the golden model's bit for bit. **Must be 100** |
| `pytorch_agree_pct` | % of images where the prediction equals the PyTorch model's |
| `latency_ms_mean`, `cycles_*` | `ref_infer()` only, DWT cycle counter at `sys_mhz` |
| `flash_kb`, `ram_kb` | From `arm-none-eabi-size`: flash = text + data, RAM = data + bss (+ text for `_ram` builds). Stack and heap not included |
| `flash_read_kb`, `energy_mj` | Not measured yet. Energy needs the power meter cable; flash reads have no simple counter on the RP2350 (see notes/pico-toolchain.md) |

## Phase 0 summary (11 Nov 2026)

All builds are bit-exact with their golden models on all 10,000 test images.

| Build | Model | Accuracy | Agrees w/ PyTorch | Latency | Cycles | Flash | RAM |
|---|---|---|---|---|---|---|---|
| ref_c (flash) | tern80 | 99.47% | 99.98% | 10.21 ms | 1.53M | 47.3 KB | 7.7 KB |
| ref_c (SRAM) | tern80 | 99.47% | 99.98% | **9.52 ms** | 1.43M | 47.2 KB | 54.9 KB |
| cmsis_nn (flash) | tern80 | 99.44% | 99.91% | 18.70 ms | 2.81M | 93.6 KB | 33.5 KB |
| cmsis_nn (SRAM) | tern80 | 99.44% | 99.91% | 14.52 ms | 2.18M | 93.5 KB | 127.0 KB |
| ref_c (flash) | tern80z70 | 99.47% | 99.97% | 10.16 ms | 1.52M | 47.3 KB | 7.7 KB |
| ref_c (flash) | tern80z90 | 99.51% | 99.98% | 10.10 ms | 1.51M | 47.3 KB | 7.7 KB |

- **Speed bar for Gate 1:** CMSIS-NN in its best setup (SRAM) runs at **14.52 ms**. The plain-C reference already beats it, by 1.53x from SRAM and 1.83x from flash, mostly because CMSIS-NN handles this network's single-input-channel conv and its depthwise convs poorly. See [notes/cmsis-nn-baseline.md](../notes/cmsis-nn-baseline.md).
- **Flash vs SRAM:** CMSIS-NN loses 29% running from flash, because its int8 weights overflow the 16 KB XIP cache. ref_c loses 7%.
- **Sparsity:** 70% and 90% zeros in fc1 speed ref_c up by only 0.5–1%, because the packed format still decodes every trit. Gains will need a sparse format in the compiler.
- Latency jitter is under 0.1% from SRAM. From flash the max is about 6% above the mean (cache misses).

## Phase 1: compiler v0 (`phase1.csv`, 11 Nov 2026)

The mafc-generated C. Same integer arithmetic, bit-exact with the golden model on all 10,000 images.

| Build | Model | Accuracy | Latency | vs ref_c | vs CMSIS-NN (same memory) | Flash | RAM |
|---|---|---|---|---|---|---|---|
| mafc_v0 (flash) | tern80 | 99.47% | 9.33 ms | 1.09x | 2.00x | 46.4 KB | 7.2 KB |
| mafc_v0 (SRAM) | tern80 | 99.47% | **8.83 ms** | 1.08x | **1.64x** | 46.3 KB | 53.6 KB |
| mafc_v0 (flash) | tern80z70 | 99.47% | 9.28 ms | 1.09x | n/a | 46.4 KB | 7.2 KB |
| mafc_v0 (flash) | tern80z90 | 99.51% | 9.22 ms | 1.10x | n/a | 46.4 KB | 7.2 KB |

Stage profiles are in `profile_stages.csv` (`mafc_v0_prof`, `mafc_v0_ram_prof` rows); analysis in [notes/compiler.md](../notes/compiler.md).

### v1: two cores (`mafc_v1_dual`)

| Build | Model | Accuracy | Latency | vs mafc v0 | vs CMSIS-NN (same memory) | Flash | RAM |
|---|---|---|---|---|---|---|---|
| mafc_v1_dual (flash) | tern80 | 99.47% | 5.80 ms | 1.61x | 3.23x | 49.1 KB | 10.8 KB |
| mafc_v1_dual (SRAM) | tern80 | 99.47% | **4.80 ms** | 1.84x | **3.02x** | 49.0 KB | 59.8 KB |
| mafc_v1_dual (flash) | tern80z70 | 99.47% | 5.79 ms | 1.60x | n/a | 49.1 KB | 10.8 KB |
| mafc_v1_dual (flash) | tern80z90 | 99.51% | 5.78 ms | 1.60x | n/a | 49.1 KB | 10.8 KB |

Bit-exact on all 10,000 images. From SRAM the second core gives close to the ideal 2x (1.84x). From flash it gives less (1.61x), because both cores share the 16 KB XIP cache. The stage profile has an extra `join` column, so it's in `profile_stages_dual.csv`.

### v1 + SMLAD on depthwise convs (`*_simd`)

| Build | Model | Accuracy | Latency | vs same build without SMLAD | vs CMSIS-NN (same memory) | Flash | RAM |
|---|---|---|---|---|---|---|---|
| mafc_v0_simd (flash) | tern80 | 99.47% | 8.93 ms | 1.04x | 2.09x | 48.7 KB | 9.2 KB |
| mafc_v0_simd (SRAM) | tern80 | 99.47% | 8.29 ms | 1.07x | 1.75x | 48.5 KB | 57.8 KB |
| mafc_v1_dual_simd (flash) | tern80 | 99.47% | 5.54 ms | 1.05x | 3.38x | 51.1 KB | 10.8 KB |
| mafc_v1_dual_simd (SRAM) | tern80 | 99.47% | **4.36 ms** | 1.10x | **3.33x** | 51.0 KB | 61.7 KB |

Bit-exact on all 10,000 images. Analysis in [notes/compiler.md](../notes/compiler.md) and FINDINGS F9/F13.

### Branch-free ReLU (all mafc builds re-measured, `notes` = `branch-free ReLU`)

The back end now writes the conv ReLU as clamp-then-shift (FINDINGS F14). Same results, faster code. These rows replace the four tables above as the current numbers.

| Build | Before | After | Change | vs CMSIS-NN (same memory) | Flash | RAM |
|---|---|---|---|---|---|---|
| mafc_v0 (flash) | 9.33 ms | 8.66 ms | −7.1% | 2.16x | 46.4 KB | 9.2 KB |
| mafc_v0 (SRAM) | 8.83 ms | 8.18 ms | −7.4% | 1.77x | 46.3 KB | 55.5 KB |
| mafc_v1_dual (flash) | 5.80 ms | 5.46 ms | −5.8% | 3.43x | 48.9 KB | 10.8 KB |
| mafc_v1_dual (SRAM) | 4.80 ms | 4.53 ms | −5.7% | 3.21x | 48.8 KB | 59.5 KB |
| mafc_v0_simd (flash) | 8.93 ms | 8.11 ms | −9.2% | 2.31x | 48.6 KB | 9.2 KB |
| mafc_v0_simd (SRAM) | 8.29 ms | 7.44 ms | −10.3% | 1.95x | 48.5 KB | 57.7 KB |
| mafc_v1_dual_simd (flash) | 5.54 ms | 5.40 ms | −2.5% | 3.46x | 50.9 KB | 10.8 KB |
| **mafc_v1_dual_simd (SRAM)** | 4.36 ms | **4.23 ms** | −3.1% | **3.44x** | 50.8 KB | 61.6 KB |

All tern80, 99.47%, bit-exact on all 10,000 images. Stage profiles (`profile_stages*.csv`, 18:19 rows): conv1 −14%, dw1 −15% in the SMLAD build; dw1 −20% in v0.

### SMLAD kernel v2, conv1 on SMLAD (`notes` = `SMLAD v2 kernel ...`)

conv1 now uses SMLAD on a widened int16 copy of the image. The kernel was rewritten so it fits in registers (FINDINGS F15). Only the SIMD builds changed. These rows are the current SIMD numbers.

| Build | Before | After | Change | vs CMSIS-NN (same memory) | Flash | RAM |
|---|---|---|---|---|---|---|
| mafc_v0_simd (flash) | 8.11 ms | 5.97 ms | −26.4% | 3.13x | 49.6 KB | 9.7 KB |
| mafc_v0_simd (SRAM) | 7.44 ms | 5.23 ms | −29.7% | 2.78x | 49.5 KB | 59.3 KB |
| mafc_v1_dual_simd (flash) | 5.40 ms | 4.38 ms | −19.0% | 4.27x | 51.7 KB | 11.3 KB |
| **mafc_v1_dual_simd (SRAM)** | 4.23 ms | **3.05 ms** | −27.8% | **4.76x** | 51.6 KB | 62.9 KB |

All tern80, 99.47%, bit-exact on all 10,000 images. Stage profiles (20:09 rows), one core, SRAM: conv1 451k → 232k, dw1 273k → 170k, dw2 34k → 24k cycles.
- The 20:01 profile rows ("first kernel") are an intermediate version: the old kernel on the widened image, which spilled registers. conv1 was 423k there.

### fc1 formats; fc1 as code (`notes` = `fc1 format...` / `fc1 as code (idea A), in SRAM`)

Build names carry the format (`_sparse`, `_code`, `_dense`; no suffix = base3). Profile rows compare the formats (FINDINGS F16). The code builds were then measured on 10k images. In flash builds the code-format fc1 runs from SRAM (`.time_critical`).

| Build | Before (base3) | fc1 as code | Change | vs CMSIS-NN (same memory) | Flash | RAM* |
|---|---|---|---|---|---|---|
| mafc_v0_simd_code (flash, fc1 in SRAM) | 5.97 ms | 4.11 ms | −31.2% | 4.55x | 132.6 KB | 101.5 KB |
| mafc_v0_simd_code (SRAM) | 5.23 ms | 3.87 ms | −25.9% | 3.75x | 132.5 KB | 135.7 KB |
| mafc_v1_dual_simd_code (flash, fc1 in SRAM) | 4.38 ms | 2.89 ms | −34.0% | 6.47x | 128.8 KB | 97.5 KB |
| **mafc_v1_dual_simd_code (SRAM)** | 3.05 ms | **2.32 ms** | −23.9% | **6.25x** | 128.6 KB | 133.4 KB |
| mafc_v1_dual_simd_code, tern80z70 (SRAM) | | 2.26 ms | | 6.42x | 87.7 KB | 92.4 KB |
| mafc_v1_dual_simd_code, tern80z90 (SRAM) | | 2.21 ms | | 6.57x | 59.8 KB | 64.5 KB |

All bit-exact on all 10,000 images. Accuracy: 99.47% for tern80 and z70, 99.51% for z90.

### Wide-conv fairness benchmark (`bench_conv.csv`)

One 3×3 conv, 10×10×32 → 8×8×32, ternary weights. CMSIS-NN `arm_convolve_wrapper_s8` against mafc's weights-as-code kernel; same requantization, and every output is checked bit-exact on the board (`bit_exact` column). One row per kernel and zero fraction, with min and mean cycles over 20 runs. Produced by `harness/bench_conv.py run` (see FINDINGS F17).

| Zero fraction | CMSIS-NN (SRAM) | mafc code (SRAM) | mafc faster by |
|---|---|---|---|
| 0% | 1,256,567 | 1,456,839 | 0.86x |
| 33% | 1,256,567 | 1,052,553 | 1.19x |
| 50% | 1,256,567 | 760,851 | 1.65x |
| 70% | 1,256,567 | 458,377 | 2.74x |
| 90% | 1,256,567 | 139,842 | 8.99x |

- Build `bench_conv_ram` is all in SRAM. In build `bench_conv` (flash), CMSIS-NN runs from flash but the code kernel is still placed in SRAM, so those rows are not like-for-like.
- The rows labelled "first run" used 3 cases and different random weights.

\* **RAM method changed on 11 Nov 2026 (these rows on).** `ram_kb` is now the sum of every section linked at a main-SRAM address (`arm-none-eabi-size -A`), excluding heap and stacks. Before, it was data + bss (+ text for copy_to_ram builds) from Berkeley `size`. That missed code copied to SRAM in flash builds: the first fc1-as-code flash build showed 9.7 KB instead of 97.5 KB. Compared with the new method, the old one overstated copy_to_ram builds by about 5–6 KB (it counted flash-only boot sections) and understated flash builds by about 2 KB. For example, mafc_v1_dual_simd (SRAM) was 62.9 KB the old way and is 56.3 KB the new way. Older rows were not rewritten.
