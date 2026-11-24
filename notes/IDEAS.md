# Ideas and questions to test

*Started 11 Nov 2026. Each entry has the idea, an assessment, a test, and the outcome where the test has run. New entries go at the bottom. The plan's own ideas A–H are in [FEASIBILITY-STUDY.md](../FEASIBILITY-STUDY.md); results are in [FINDINGS.md](FINDINGS.md).*

| # | Idea | Status | Related plan idea |
|---|---|---|---|
| U1 | Cache placement: keep the most-used data in fast memory | Done for MNIST: fc1 as code placed in SRAM (F16), weights streamed by DMA (F23). A general planned cache over flash waits for a model larger than SRAM | G (SRAM cache), C (DMA) |
| U2 | Make weight unpacking cheaper | Done: sparse lists and code formats (F16, F18); lookup tables over activations in the story-model runtime ([lm.md](lm.md), "Runtime v1"). Not tried: two bit masks | A, E |
| U3 | Split cores by data residency (one computes, one loads) | Step 1 done (F12): contention is moderate on the convs, strong on fc1. Loading by DMA instead of a core done (F21, F23) | C, G |
| U4 | Weights as add/subtract instructions | Done: kept as a hybrid, linear layers as code and conv weights as data (F16, F20) | A |
| U5 | Microsoft bitnet.cpp for the Pi phase | Noted, not started (Phase 4) | E, big-model goal |

---

## U1. Cache optimisation: predict which data is used most

**Idea:** when not everything fits in fast memory, work out which data is needed most and keep it in the cache, letting the least-used data go.

**Assessment: yes, and here it's easier than usual,** because inference is completely deterministic. The same weights are read in the same order for every image, so nothing needs predicting: the compiler knows the exact access sequence at build time. That allows:

- **Static placement (simplest, strongest).** The compiler counts how many times each weight block is read per inference and how big it is, then puts the most-read-per-byte data in SRAM and leaves the rest in flash.
  - The conv weights (2.1 KB, re-read for every channel and pixel) and the code for the hot loops are the obvious SRAM candidates.
  - fc1's weights are read exactly once per inference, so by this measure they gain the least per byte. (U3 showed this measure is misleading; see below.)
- **Optimal replacement.** Because the full access order is known, the textbook-optimal policy (Belady's: evict whatever is needed furthest in the future) can actually be computed. The compiler can then plan which blocks get copied into a reserved SRAM area and when.
- **Cache pinning.** To check: the RP2350 datasheet may allow parts of its XIP cache to be locked or used as plain SRAM. If so, hot flash data could be pinned.

**Caveat:** the MNIST model (about 47 KB in total) fits entirely in the 520 KB SRAM, so the best placement for it is simply "everything in SRAM" (the `RUN_FROM_RAM` builds). This idea pays off for models larger than SRAM: idea G and the big-model goal, plus the Pimoroni Pico Plus 2 with 8 MB PSRAM.

**Test:**
1. Mixed placement: fc1's weights in SRAM with the rest in flash, conv code and weights in SRAM with fc1 in flash, and both. Compare them with all-flash and all-SRAM.
2. On a model larger than SRAM, compare compiler-planned placement with plain flash + cache.
3. A compiler pass that outputs the per-block "reads per inference ÷ size" table.

**Outcome:**
- Test 1 was answered by fc1's own change of format. In code format, fc1 always goes to SRAM (`.time_critical`), so flash builds run it at SRAM speed, and the two-core flash build is 2.89 ms against 2.32 ms all-SRAM (F16). Streaming fc1's weights from flash by DMA is the alternative when they stay as data (F23).
- Tests 2 and 3 are open; they need a model larger than SRAM.

## U2. Can weight unpacking be improved?

**Problem:** fc1 spent about 8 cycles per weight unpacking 10 trits per 16 bits (a multiply, a shift and two branches per weight).

**Options, roughly from simplest to boldest:**

1. **Two bit masks instead of base 3.** Store each group of 32 weights as two 32-bit masks: "which are +1" and "which are −1". The layer output is then (sum of inputs where the +1 mask is set) − (sum where the −1 mask is set). Cost: 2 bits per weight instead of 1.6, so fc1 grows from 6.1 KB to 7.7 KB, still within the cache. Decoding becomes bit tests with no multiplications.
2. **Skip zeros explicitly.** Store only the positions of the +1 and −1 weights (sparse lists). The work is then proportional to the number of non-zeros: 50% of the weights in tern80, 10% in tern80z90.
3. **Lookup tables (idea E).** Per group of 4 inputs, precompute all 81 possible ±/0 combinations of their sum once per inference. Each group of 4 weights then becomes a single table lookup. This pays off when the same inputs meet many output rows; fc1 has 96.
4. **Weights as code (idea A, U4).** No unpacking at all: the weights become instructions.

**Test:** implement the options as back-end weight formats (the IR already keeps unpacked weights, so only the back end changes), then time fc1 alone on tern80 and tern80z90, from flash and from SRAM.

**Outcome (F16, F18):**
- Sparse lists (option 2), code (option 4) and a dense `SXTB16`/`SMLAD` format were compared with base 3. From SRAM, code is 5.1x faster than base 3 and is the only format that turns zeros into speed. From flash, base 3 (6 KB) is the best format that stays in flash.
- Lookup tables over activations (a version of option 3) are used in the story-model runtime, where they help bring the matrix-vector products down to 1.2–1.7 cycles per weight ([lm.md](lm.md)).
- Two bit masks (option 1) were not tried.

## U3. One core computes while the other loads

**Idea:** with the conv channels split across cores, reserve a section of fast memory as a loading area. One core works on data that's already loaded while the other loads the data still needed.

**Assessment: worth including, with two adjustments.**

- **Use DMA for the loading, not a core.** The RP2350 has DMA hardware that copies memory on its own. A core spent copying is a core not computing. The usual pattern is *double buffering*: while the cores compute on buffer A, DMA fills buffer B, then they swap. That's the plan's idea C, and the reserved loading area is exactly that buffer pair.
- **It only helps when data doesn't fit in SRAM.** The MNIST model fits completely, so there is nothing to load and the test would show no gain. It matters for a larger model (idea G, big-model goal).

An asymmetric variant was still worth one test in flash mode: one core takes the layers whose weights are in SRAM, while the other takes the ones streamed from flash (such as fc1). The cores would then contend less for the shared flash cache.

**Test:**
1. In the flash build, measure the two cores' "join" wait times to see whether they slow each other down through the shared cache.
2. A double-buffered DMA weight stream for a model larger than SRAM: two cores computing against one core computing plus DMA loading.

**Result of test 1 (11 Nov 2026, F12): the cores do slow each other down, moderately on the convs and strongly on fc1.**

Method: no new code. The existing mafc v0 (one core) and v1 (two cores) builds were profiled, from flash and from SRAM, on 200 images, twice each, plus the earlier profile (`results/contention*.csv`; analysis with `harness/contention.py`). Each core's share of the conv layers in the dual build was compared with half the single-core conv time. Core 1 has no timer of its own, so its time is core 0's plus the join wait (both start within one FIFO push). Repeats agree within 0.1% on the convs and 1% on the join.

| tern80, cycles | single core ÷ 2 | dual, core 0 | join wait | dual, core 1 (est.) |
|---|---|---|---|---|
| conv layers, flash | 506k | 565k (+11.7%) | 21.0k | 586k (+15.8%) |
| conv layers, SRAM | 503k | 534k (+6.2%) | 11.7k | 545k (+8.5%) |

| Flash penalty (flash vs SRAM, same build) | one core | two cores |
|---|---|---|
| conv layers | +0.7% | +5.9% (core 0), +7.5% (core 1) |
| fc1 | +18.5% | **+54.6%** |

- **Two cores cost about 6–8% each even from SRAM.** That's shared SRAM and bus traffic, not flash. It's the floor for any placement.
- **Flash adds about 6 points more on the convs.** The join wait nearly doubles (12k → 21k cycles), so core 1 is the one starved more. conv1 is hit hardest (+19% from flash vs +9% from SRAM).
- **fc1 is where contention really bites.** On one core, flash already costs +18.5%: fc1's 6 KB of weights don't survive in the 16 KB XIP cache between inferences, because the conv code and data evict them. With two cores both streaming their halves at once it's +54.6%, and the second core gives only 1.40x on fc1 (1.83x from SRAM).
- **In cycles on the critical path:** fc1 loses about 76k cycles (0.5 ms) to flash with two cores; the convs lose about 41k (core 1's half).

This pointed at **fc1's weights**, not the conv code, as the first thing to move into SRAM: the opposite of U1's "reads per byte" measure, which misses that the weights are evicted between inferences and that both cores stream them at once.

**Test 2:** DMA streaming was built and measured on one core (F21, F23). Streaming the 30 KB fc1 from flash is 29% faster than reading it in place on one core, and 58% faster than two cores sharing the cache. The two-cores-plus-DMA comparison on a model larger than SRAM is still open.

## U4. Weights as add/subtract instructions (idea A)

**Idea:** instead of storing ternary weights and unpacking them, the compiler writes them straight into code. For example, `sum += x[3]; sum -= x[7]; …` for every non-zero weight, and zeros simply produce no instruction.

**Why it's attractive:**
- There's no unpacking at all.
- Zeros cost nothing, so it's the most direct way to turn sparsity into speed.

**The catch (from the feasibility study):** each non-zero weight becomes an instruction of 2–4 bytes, against 1.6 bits as packed data. fc1 has about 15,300 non-zeros, so roughly 30–60 KB of code, larger than the 16 KB flash cache.

- **From flash:** it may be *slower*, because the code has to be streamed.
- **From SRAM:** it should be fast.
- **The hybrid:** code for small or hot parts, data for the rest.

**Test:** build fc1 all-code, all-data and hybrid, on tern80 and tern80z90, from flash and from SRAM. Measure cycles, flash size and code size.

**Outcome (F16, F20):**
- The code is bigger than estimated: 86 KB for tern80's 15,360 non-zeros, about 6 bytes per weight.
- From SRAM, fc1 as code is 5.1x faster than the packed format. Read from flash it is about 2x slower than the packed format (611k against 317k cycles on one core), so the code is always placed in SRAM.
- Verdict: keep the hybrid, with the linear layers as code and the conv weights, which are reused, as data.

## U5. Microsoft BitNet / bitnet.cpp for the Pi phase (noted 13 Nov 2026)

[bitnet.cpp](https://github.com/microsoft/BitNet) (MIT) is Microsoft's inference engine for ternary models, built on llama.cpp. It has NEON (ARM) and AVX2 (x86) CPU kernels, and its flagship model is BitNet b1.58 2B4T (2.4B weights). Papers: [arXiv 2410.16144](https://arxiv.org/abs/2410.16144), [arXiv 2502.11880](https://arxiv.org/abs/2502.11880).

- **Use as the Pi 4/5 baseline** for the big-model goal. It runs under Linux as is, so it plays the role Atome plays on the Pico.
- **Weight formats worth borrowing for the compiler:**
  - I2_S: 2 bits per weight, as in our v0.
  - TL1: a pair of weights becomes a 4-bit index into a 9-entry table precomputed per pair of activations.
  - TL2: 5 bits per 3 weights (a sign bit plus a 4-bit index), so 1.67 bits per weight; meant for when bandwidth is the limit.
  - These lookup tables are our idea E, and on NEON they use `vqtbl1q_u8`. TL2 could also cut flash bytes per token on the Pico by about 17%.
- **Trained ternary models of 0.7B–10B** for the larger-than-memory experiments, without training our own.
- **Caveats:**
  - No Raspberry Pi numbers are published; the ARM results are from an M2 Ultra (1.4–5x faster than llama.cpp, 55–70% less energy), and the paper says bandwidth matters a lot.
  - Activations use one int8 scale per tensor, where ours use one per token.
  - Not yet checked: the 2B4T architecture (squared ReLU? subln?) and the licences of the individual models.
