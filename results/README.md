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
