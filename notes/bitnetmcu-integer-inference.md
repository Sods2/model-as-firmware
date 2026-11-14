# BitNetMCU integer inference, and our reference

*Researched 11 Nov 2026 against BitNetMCU upstream commit `0715bfc` (the clone in `train/BitNetMCU/`, untracked).*

## Licence: why we don't copy BitNetMCU's C code

BitNetMCU is **GPL-3.0**. Copying `BitNetMCU_inference.c` into this repo would make our firmware GPL-3 too. So [firmware/baselines/ref_c/ref_infer.c](../firmware/baselines/ref_c/ref_infer.c) is our own implementation of the same integer arithmetic (the arithmetic is an algorithm, not copyrightable text). It is written to reproduce the upstream C arithmetic; it was not run against upstream's `BitNetMCU_inference.c` (it is checked bit-exact against our own golden model). Upstream reports 99.52% for this model; our training run reached 99.45% at the final epoch (one seed). The exported model headers in `train/export/` are data produced by their `exportquant.py`.

## The model (CNNMNIST, `cnn_width` 80)

| Layer | Header name | Shape | Weights |
|---|---|---|---|
| conv 3x3, 1 -> 80 | `L2` | 16x16 -> 14x14 | int8 (`int8_t[]`, 9 per channel) |
| depthwise conv 3x3 | `L4` | 14x14 -> 12x12 | int8 |
| maxpool 2 | `L6` | 12 -> 6 | |
| depthwise conv 3x3 | `L7` | 6x6 -> 4x4 | int8 |
| maxpool 2 | `L9` | 4 -> 2 | |
| fc1 320 -> 96 | `L11` | | **ternary**, `bitperweight 64` = 10 trits per `uint16_t` |
| fc2 96 -> 64 | `L13` | | 4bitsym, 8 per `uint32_t` |
| classifier 64 -> 10 | `L15` | | 4bitsym |

- **No layer has a bias** (`bias=False` in BitLinear and BitConv2d), and every non-linearity is ReLU or max-pool. The network is therefore *positively homogeneous*: scaling the input by a positive number scales every layer's output by the same amount, so per-sample normalizations (RMS norm, the dynamic shifts below) never change the argmax in exact arithmetic. Rounding and clipping in the integer pipelines do change a few predictions: CMSIS-NN agrees with PyTorch on 99.91% of test images, and our pipeline on 99.98%. This is what lets a static-scale int8 build (CMSIS-NN) run the very same weights as the "same network".
- Flatten order is PyTorch NCHW: feature index = `channel*4 + y*2 + x`.

## Weight packing formats

- **Ternary (`bitperweight 64`)**: each `uint16` holds 10 trits as a base-3 fraction. Decode: `c *= 3`; digit = `(c >> 16) & 3`, where 0 -> +1, 1 -> -1, 2 -> 0; then `c &= 0xFFFF`. That is 1.6 bits/weight, **whatever the zero fraction**, so sparsity saves no space in this format. Rows are contiguous: output `o` starts at word `o * n_in/10`.
- **4bitsym (`bitperweight 4`)**: 8 nibbles per `uint32`, most significant first. Bit 3 = sign (1 = negative), bits 2:0 = m, weight = ±(2m+1), so the levels are ±1, ±3, …, ±15.
- **Conv**: plain `int8`, 9 per channel, row-major 3x3.

All these integer codes fit in int8, which matters for the CMSIS-NN baseline.

## Integer arithmetic (what the MCU engine does)

1. **Input**: per image, `scale = 127 / max|x|` on the normalized 16x16 float image, `round` (numpy, half to even), clip to −128..127, `int8`.
2. **Conv stack**, one channel at a time, int32 activations: `s = sum(k*x)`; output `s < 0 ? 0 : s >> 4` (fixed shift of 4, no clipping). Pools take the max of each 2x2 block.
3. **ReLUNorm** between fc layers (and after the conv stack): find max m over the vector; shift = bit length of `m >> 7`; `out = x < 0 ? 0 : min((x + (1<<shift>>1)) >> shift, 127)`. This is a per-sample power-of-two rescale into 0..127, rounding half up.
4. **Logits** are the raw int32 sums of the classifier. Prediction = index of the first maximum.

### The Python "quantized" inference is not the same arithmetic

`QuantizedModel.inference_quantized` in `BitNetMCU.py` differs from the C engine in the conv stack: it normalizes each conv output to 0..127 by its exact max, where C uses a fixed `>> 4`. So upstream's own C-vs-Python check (`test_inference.py`) is expected to show a few mismatches. That's why we wrote our own golden model ([harness/golden.py](../harness/golden.py)), a numpy copy of the *C* arithmetic, which must match the C reference bit for bit.

## Numbers (11 Nov 2026, full 10k MNIST test set)

| Model | PyTorch (QAT, float) | Integer pipeline (golden = C) | Prediction agreement |
|---|---|---|---|
| tern80 | 99.45% | 99.47% | 99.98% (2 differ) |
| tern80z70 | 99.44% | 99.47% | 99.97% |
| tern80z90 | 99.49% | 99.51% | 99.98% |

- The largest possible |partial sum| anywhere is about 1.4M, only 0.07% of the int32 range, so int32 accumulation is very safe. int16 accumulators would *not* be safe in general.
- These integer accuracies agree with `exportquant.py`'s "quantized" accuracy (99.47% for tern80).

## Upstream files worth knowing

- `BitNetMCU_inference.c`: the reference C kernels (fc 1/2/4/8-bit, ternary, conv3x3, maxpool).
- `BitNetMCU_MNIST_test.c` / `BitNetMCU_MNIST_dll.c`: CNN inference driver (channel loop) and a DLL wrapper for Python.
- `test_inference.py`: compares the DLL against the Python quantized model. It needs a 64-bit DLL, and the MinGW gcc on this PC is 32-bit, which is why we use an exe + binary file instead.
- `verify_int_inference.py`: exact integer emulation for the newer table/ShiftNorm FC models (`act_pow2=True`). It doesn't cover our CNN config.
