# How this project works, from zero

*A plain-language guide, written 11 Nov 2026, numbers updated 14 Nov 2026. This is a living document: update the numbers and add sections as the project moves on. Exact figures live in [results/README.md](../results/README.md) and the analysis in [FINDINGS.md](FINDINGS.md).*

## 1. The goal in one sentence

Take a small AI model that recognises handwritten digits and turn it into the firmware of a $5 chip, with no operating system and no AI library in between, so that it runs as fast as possible. (Efficiency, meaning energy per answer, is a goal too, but it hasn't been measured yet: the power meter on hand is too coarse.)

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

To be fair to it: CMSIS-NN has no ternary kernels. It does have 4-bit ones: using them for fc1 makes its whole model 7% faster from flash, though slightly slower from SRAM (FINDINGS F27). Its weights are still 2–2.5x larger than our packed ternary ones, and its kernels happen to be weak for this model's unusual shape (a first layer with one input channel, and depthwise convolutions). Part of our lead comes from that, not from our method; see FINDINGS F5 and F17.

## 7. Results so far

Digit recognition (MNIST), one image, time per answer. All builds give bit-exact answers on all 10,000 test images.

| Build | Cores | From flash | From SRAM |
|---|---|---|---|
| CMSIS-NN (ARM's library), 99.44% | 1 | 18.70 ms | **14.52 ms**, the bar |
| Our plain-C reference, 99.47% | 1 | 10.21 ms | 9.52 ms |
| Our compiler v0 | 1 | 8.66 ms | 8.18 ms |
| Our compiler, best build, 99.47% | **1** | 3.14 ms | **3.08 ms, 4.7× the bar (the fair comparison)** |
| Our compiler, best build | 2 | 1.99 ms | 1.67 ms (8.7× the bar, but the bar uses one core) |

"From SRAM" means the whole program is copied into fast memory at start-up, so there are no flash waits.

Things to keep in mind:
- **The fair number is one core against one core: 4.7×.** CMSIS-NN only uses one core.
- **Our fast builds use more memory:** 133–161 KB of flash and 128–160 KB of SRAM, against CMSIS-NN's 94 KB and 33–127 KB. Turning weights into instructions costs space.
- **Our "from flash" builds still keep the linear layers in SRAM**, while CMSIS-NN's flash build reads everything from flash.
- **On one wide layer, the shape CMSIS-NN is built for,** our approach only wins when enough weights are zero: it breaks even at about 14% zeros and loses when no weights are zero (FINDINGS F17).

For the tiny story model (Phase 3), see [lm.md](lm.md). There, the fair comparison is our model's two versions against each other (the tree version is 2.1–2.5× faster over the first 24 letters, 1.7–2.2× over a full 256-letter window). The comparison with the published Atome model has caveats that are spelled out in that file.

## 8. What profiling (timing each layer separately) taught us

1. **Memory traffic matters, which is the project's core idea.** CMSIS-NN stores fc1's weights as 8-bit numbers: 30 KB. That doesn't fit the 16 KB flash cache, so the layer is **7.4× slower from flash than from SRAM**. Our packed ternary version of fc1 is only 6 KB, fits the cache, and barely cares where it lives.
2. **Unpacking ternary weights costs time:** about 8 cycles per weight. Once memory isn't the problem (in SRAM), CMSIS-NN's 8-bit fc1 was faster than ours. *(Fixed later: compiled into instructions, our fc1 takes 50k cycles against CMSIS-NN's 71k; F16, F17.)*
3. **The convolutions take about 70% of the time,** so they're the main target.
4. **Zeros alone don't buy speed.** The 70% and 90% zero models are only about 1% faster, because the current format still unpacks every weight. Skipping zeros needs a different format. *(Done later: compiled into instructions, zero weights simply disappear, and the 90% zero model's fc1 runs 12× faster; F16.)*

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

Done since this guide was first written: both cores, dual 16-bit multiplies, idea A (weights as instructions), the tree layer (idea B), weight streaming (idea C), and a tiny story-writing model (Phase 3). Still open:

- **Energy per inference:** needs a finer current meter than the USB one on hand.
- **A model bigger than the chip's fast memory,** where reading from flash sets the speed.

Further ideas and questions, with their tests and outcomes, are in [IDEAS.md](IDEAS.md).

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
