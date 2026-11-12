# CMSIS-NN int8 baseline (Phase 0 step 5)

*Built 11 Nov 2026 against CMSIS-NN commit `ece8aa0df4bee4fc928c294e18059071649c88bb` (v8.0.0+18, 5 Nov 2026), Apache-2.0.*

## Getting CMSIS-NN

The library isn't vendored. Clone it next to the build:

```bash
git clone https://github.com/ARM-software/CMSIS-NN.git firmware/baselines/cmsis_nn/third_party/CMSIS-NN
git -C firmware/baselines/cmsis_nn/third_party/CMSIS-NN checkout ece8aa0df4bee4fc928c294e18059071649c88bb
```

CMSIS-NN is self-contained (its own `Internal/arm_nn_compiler.h`) and needs no CMSIS-Core. The build compiles `Source/{Convolution,FullyConnected,Pooling}Functions/*_s8*.c` plus all of `NNSupportFunctions/*.c`; the `_s8` glob alone misses `arm_q7_to_q15_with_offset`. DSP paths turn on automatically, because the Pico SDK compiles with `-march=armv8-m.main+fp+dsp`, which defines `__ARM_FEATURE_DSP`. The ELF has 334 `smlad`/`sxtb16` instructions.

## Why it's the same network

- **Weights:** the *same integer codes* as the BitNetMCU export: 8-bit convs, ternary fc1, 4-bit fc2/fc3. They're unpacked to int8 because CMSIS-NN only takes int8 weights. fc1 grows from 6.1 KB packed to 30 KB.
- **Activations:** int8, zero point 0. ReLU is fused as a clamp to 0..127, the same range BitNetMCU uses.
- **Requantization:** one static scale per layer instead of BitNetMCU's per-image power-of-two shift. Since the network has no biases and only ReLU/max-pool (it's positively homogeneous; see [bitnetmcu-integer-inference.md](bitnetmcu-integer-inference.md)), this changes the rounding, not the function.
- **Calibration:** TFLite-style min/max. Each layer's scale is 127 / max(accumulator) over the first 10,000 MNIST *training* images, computed layer by layer on the already-quantized outputs. The test set is never used.
- **Result:** 99.44% on the 10k test set, against 99.45% for the PyTorch model and 99.47% for the BitNetMCU integer pipeline. Accuracy is equal within noise, so the speed comparison is fair.

## Exactness

- The build defines `CMSIS_NN_USE_SINGLE_ROUNDING`, so requantization is `round_half_up(acc * mult / 2^(31 - shift))`, which is easy to emulate exactly. `harness/golden.py:GoldenS8` does that in numpy, and the board's int8 logits match it **bit for bit on all 10k images**.
- Multipliers come from TFLite's `QuantizeMultiplier` (frexp, Q31). All channels of a conv share one scale; CMSIS-NN still wants per-channel arrays.
- The layout is NHWC, so fc1's columns are permuted from PyTorch's NCHW flatten order (`nhwc (y*2+x)*C + c  <->  nchw c*4 + y*2 + x`).

## Files

| File | Role |
|---|---|
| `harness/export_cmsis.py` | calibrate, write `firmware/baselines/cmsis_nn/model_s8_<model>.h` |
| `firmware/baselines/cmsis_nn/cmsis_infer.c` | the network as CMSIS-NN calls: `arm_convolve_wrapper_s8`, `arm_depthwise_conv_wrapper_s8` (dispatches to `arm_depthwise_conv_3x3_s8`), `arm_max_pool_s8`, `arm_fully_connected_s8` |
| `firmware/baselines/cmsis_nn/CMakeLists.txt` | build; reuses `../ref_c/pico_main.c` so the harness protocol is identical |

## Audit: is this CMSIS-NN at its best? (11 Nov 2026, FINDINGS F17)

- **Compiler flags:** `-O3 -mcpu=cortex-m33 -march=armv8-m.main+fp+dsp`, so `arm_nn_math_types.h` turns on `ARM_MATH_DSP` and the SMLAD paths are used.
- **dw1/dw2:** `arm_depthwise_conv_wrapper_s8` picks the dedicated `arm_depthwise_conv_3x3_s8` (ch_mult 1, padding ≤ 1, no MVE). That is the fastest depthwise kernel CMSIS-NN has for a DSP-only core. It is still slow here, because it works on NHWC with per-tap 4-channel loads and per-channel requantization.
- **conv1:** `arm_convolve_wrapper_s8` falls through to `arm_convolve_s8` (im2col + `arm_nn_mat_mult_kernel_s8_s16`). The 1×1 and 1×N fast paths don't apply to a 3×3 kernel, and there is no special path for one input channel.
- **Conclusion:** nothing to fix; the baseline uses the library as intended. CMSIS-NN is single-core, so like-for-like comparisons are one core against one core (3.75x).
- **Wide-conv test:** on a 32→32 channel 3×3 conv, the shape this library is tuned for, CMSIS-NN runs at about 2.1 cycles per MAC (`firmware/bench/conv_wide`).

## What profiling showed (stage cycles, tern80, image 0)

| Stage | ref_c flash | ref_c SRAM | CMSIS-NN flash | CMSIS-NN SRAM |
|---|---|---|---|---|
| conv1 (1->80) | 698k | 703k | 1,091k | 1,070k |
| dw1 | 402k | 402k | 845k | 822k |
| pool1 | 41k | 36k | 102k | 96k |
| dw2 | 55k | 51k | 104k | 92k |
| fc1 (320->96) | 307k | 255k | **524k** | **71k** |
| fc2 + fc3 | 94k | 66k | 121k | 19k |
| **total** | 1.60M | 1.51M | 2.81M | 2.18M |

*(Profiling builds; the counters add about 3% in ref_c.)*

1. **Flash bandwidth is real, already at this size.** CMSIS-NN's fc1 runs 7.4x faster from SRAM than from flash: its 30 KB of int8 weights don't fit the RP2350's 16 KB XIP cache, so every inference streams them from QSPI flash. ref_c's packed ternary fc1 (6 KB) fits, so moving it to SRAM gains only 17%. This is the project's main premise (packed weights win because of memory traffic) showing up in Phase 0.
2. **Ternary decoding is compute-heavy.** From SRAM, CMSIS-NN's int8 fc1 (71k cycles, about 2.3 per weight with SMLAD) beats our ternary decode (255k cycles, about 8.3 per weight: a multiply and two branches per trit). The compiler has a lot to win here: unpacked trits as add/subtract lists, lookup tables (idea E), or SIMD-friendly packing.
3. **The conv stack is two-thirds of the time, and CMSIS-NN does it badly for this shape.** conv1 has a single input channel, so CMSIS-NN's im2col + matmul path works on 9-long dot products with heavy overhead, and its 3x3 depthwise kernel is slower than our plain per-channel loop. CMSIS-NN is tuned for wide-channel convolutions; this model is unusual for it. Gate 1 comparisons should keep this in mind and not overclaim a win in the conv layers.
