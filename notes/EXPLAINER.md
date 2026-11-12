# How this project works, from zero

*A plain-language guide, written 11 Nov 2026. This is a living document: update the numbers and add sections as the project moves on. Exact figures live in [results/README.md](../results/README.md) and the analysis in [FINDINGS.md](FINDINGS.md).*

## 1. The goal in one sentence

Take a small AI model that recognises handwritten digits and turn it into the firmware of a $5 chip, with no operating system and no AI library in between, so that it runs as fast and efficiently as possible.

## 2. What a "model" is

A neural network is a long chain of very simple arithmetic.

- **The input** is a 16×16 grayscale picture of a digit: 256 numbers, one per pixel.
- **Each layer** multiplies those numbers by fixed numbers called **weights**, adds the products up, and passes the result on to the next layer.
- **The output** is 10 numbers, one score per digit 0–9. The highest score is the answer.

Training, done on the PC's graphics card, is the process of finding good weights. After training the weights never change, and running the model ("inference") just means doing those multiplications and additions. Our model has about 40,000 weights.

## 3. Why "ternary"

Normally each weight is a 32-bit decimal number. In a ternary model, weights are only **−1, 0 or +1**. That has two consequences:

- **No multiplication needed:** multiplying by +1 means "add the input", by −1 means "subtract it", and by 0 means "skip it".
- **Tiny storage:** each weight needs about 1.6 bits. Our format packs 10 weights into one 16-bit number.

The model must be *trained* ternary from the start; you can't just round an existing model. Phase 0 step 3 did this with BitNetMCU's training code and reached 99.45% accuracy on 10,000 test images it had never seen. In our model only fc1 is ternary; the convolutions use 8-bit weights and fc2/fc3 use 4-bit weights.

## 4. The model's layers

```
picture 16×16
 → conv1  : 80 small 3×3 pattern detectors scan the image (edges, curves…) → 80 maps of 14×14
 → dw1    : each map refined by its own 3×3 filter                         → 80 maps of 12×12
 → pool1  : keep the max of every 2×2 block (shrinks the maps)             → 80 maps of 6×6
 → dw2    : another 3×3 filter per map                                     → 80 maps of 4×4
 → pool2  :                                                               → 80 maps of 2×2 = 320 numbers
 → fc1    : 320 → 96, every output mixes all 320 inputs  (ternary, 30,720 weights, 50% zero)
 → fc2    : 96 → 64   (4-bit weights)
 → fc3    : 64 → 10   (the 10 digit scores)
```

- **conv** layers are convolutions: they slide a small window over the image to detect local patterns.
- **fc** layers are "fully connected": every output depends on every input.
- Between layers, a step called **ReLU** sets negatives to 0, and a **normalisation** step rescales the values so they fit in 8 bits.

## 5. The hardware

The board is a **Raspberry Pi Pico 2**, built around the RP2350 chip:

- **Processor:** 2 cores at 150 MHz, i.e. 150 million basic steps per second each.
- **SRAM, 520 KB:** fast working memory, gone when the power is off.
- **Flash, 4 MB:** slow storage where the program lives. Reading from it goes through a **16 KB cache**. If the data you need is in the cache it's fast; if not, the chip waits for the flash.
- **Cycle counter:** a built-in counter that ticks once per clock cycle. Every timing here comes from it: 1,500,000 cycles = 10 ms.

## 6. What was built, and why each piece exists

### The plain-C reference (Phase 0 step 4)
A straightforward C program that runs the model on the Pico, layer by layer. It isn't meant to be fast; it's meant to be **correct**, and everything faster is checked against it. It's our own code rather than a copy of BitNetMCU's, because theirs is GPL-licensed and would force that licence on this project.

### The golden model and "bit-exact"
A separate Python program that does exactly the same integer arithmetic. The board counts as correct only if its 10 output numbers equal the golden model's **exactly, to the last digit**, for every image. Close isn't good enough: any difference means a bug.

The board's answers agree with the original PyTorch model on 99.98% of images. The few that differ do so because PyTorch uses decimal numbers while the board rounds to integers. That's expected.

### The test harness (step 6)
A Python script that talks to the Pico over USB. For each image it:

1. sends the 256 pixels to the board;
2. the board runs the model and replies with its answer, its 10 scores and the cycle count;
3. the script compares the reply with the golden model and the true label, and logs the result to `results/*.csv`.

### The CMSIS-NN baseline (step 5): the bar to beat
CMSIS-NN is ARM's official, optimised AI library for this kind of chip. To claim our approach is faster, we must beat *it*, not just our own slow code. It runs **the same model with the same weights**, as standard 8-bit numbers.

## 7. Results so far

| Build | From flash | From SRAM |
|---|---|---|
| CMSIS-NN (ARM's library) | 18.70 ms | **14.52 ms**, the bar |
| Our plain-C reference | 10.21 ms | 9.52 ms |
| Our compiler v0 | 9.33 ms | 8.83 ms, 1.64× faster than the bar |

"From SRAM" means the whole program is copied into fast memory at start-up, so there are no flash waits. Dual-core (v1) results go here once they're measured.

## 8. What profiling (timing each layer separately) taught us

1. **Memory traffic matters, which is the project's core idea.** CMSIS-NN stores fc1's weights as 8-bit numbers: 30 KB. That doesn't fit the 16 KB flash cache, so the layer is **7.4× slower from flash than from SRAM**. Our packed ternary version of fc1 is only 6 KB, fits the cache, and barely cares where it lives.
2. **Unpacking ternary weights costs time:** about 8 cycles per weight. Once memory isn't the problem (in SRAM), CMSIS-NN's 8-bit fc1 is faster than ours. A clear target.
3. **The convolutions take about 70% of the time,** so they're the main target.
4. **Zeros alone don't buy speed.** The 70% and 90% zero models are only about 1% faster, because the current format still unpacks every weight. Skipping zeros needs a different format.

## 9. The compiler (Phase 1)

```
trained model ─► front end ─► IR (a clean list of layers + integer weights) ─► back end ─► C code for the Pico
```

- **Front end:** reads the exported model file.
- **IR (intermediate representation):** a neutral description of the model, with every layer's arithmetic defined exactly.
- **Back end:** writes code for one specific chip. Later back ends (Pi 4, FPGA) can start from the same IR.

The generated code beats the hand-written version for two reasons:

- **Sizes are hard-coded,** so the C compiler can optimise every loop for its exact length.
- **The image is read directly** instead of being copied 80 times, which made conv1 34% faster.

It was checked bit-exact at three levels: Python, the C on the PC, and the C on the Pico.

## 10. Where it's going next

- **v1, both cores:** the convolution channels are split between the two cores (core 1: channels 0–39, core 0: 40–79), and so are the big fc layers.
- **Dual 16-bit multiplies:** the chip can do two 16-bit multiplications in one step. A worst-case maths check proved this is exact for dw1.
- **Idea A:** turn the ternary weights directly into add/subtract instructions instead of unpacking them at run time.
- **Energy per inference:** needs the USB-C cable for the power meter.

Your own ideas and questions are tracked in [IDEAS.md](IDEAS.md).

## 11. Glossary

| Term | Meaning |
|---|---|
| Inference | Running the trained model on one input |
| Weight | A fixed number learned in training |
| Ternary | Weights limited to −1, 0, +1 |
| Quantisation | Storing numbers with fewer bits (8-bit, 4-bit, ternary) |
| Firmware | The program stored on the chip |
| Flash / SRAM | Slow, permanent storage / fast working memory |
| Cache | Small fast copy of recently used flash data |
| Cycle | One clock tick (1/150,000,000 s) |
| Bit-exact | Identical results, digit for digit |
| Golden model | Trusted reference implementation used for checking |
| Profiling | Measuring the time spent in each layer |
| IR | Intermediate representation: the compiler's internal description of the model |
