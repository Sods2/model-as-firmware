# results/

| File | What |
|---|---|
| `phase0.csv` | One row per timed run (10k MNIST test images, profiling off), written by `harness/run.py` |
| `profile.csv`, `profile_stages.csv` | Profiling builds (200 images, stage counters on). Stage cycles are for test image 0 |

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
