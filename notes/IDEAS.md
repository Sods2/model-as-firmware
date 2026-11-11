# Ideas and questions to test

*Started 11 Nov 2026 from the author's notes while reading [EXPLAINER.md](EXPLAINER.md). Each entry has the idea as raised, an assessment, and a concrete test. Add new entries at the bottom and update the status as tests run.*

| # | Idea | Status | Related plan idea |
|---|---|---|---|
| U1 | Cache placement: keep the most-used data in fast memory | Proposed | G (SRAM cache), C (DMA) |
| U2 | Make weight unpacking cheaper | Proposed | A, E |
| U3 | Split cores by data residency (one computes, one loads) | Proposed, test only when data doesn't fit SRAM | C, G |
| U4 | Weights as add/subtract instructions | Liked; this is idea A, next after v1 | A |

---

## U1. Cache optimisation: predict which data is used most

**Idea:** when not everything fits in fast memory, work out which data is needed most and keep that in the cache, letting the least-used data go.

**Assessment: yes, and for us it's easier than usual,** because inference is completely deterministic. The same weights are read in the same order for every image, so nothing needs predicting: the compiler knows the exact access sequence at build time. That allows:

- **Static placement (simplest, strongest).** The compiler counts how many times each weight block is read per inference and how big it is, then puts the most-read-per-byte data in SRAM and leaves the rest in flash.
  - The conv weights (2.1 KB, re-read for every channel and pixel) and the code for the hot loops are the obvious SRAM candidates.
  - fc1's weights are read exactly once per inference, so they gain the least per byte.
- **Optimal replacement.** Because the full access order is known, the textbook-optimal policy (Belady's: evict whatever is needed furthest in the future) can actually be computed. The compiler can then plan which blocks get copied into a reserved SRAM area and when.
- **Cache pinning.** To check: the RP2350 datasheet may allow parts of its XIP cache to be locked or used as plain SRAM. If so, hot flash data could be pinned.

**Caveat:** today's model (about 47 KB in total) fits entirely in the 520 KB SRAM, so the best placement right now is simply "everything in SRAM" (that's the `RUN_FROM_RAM` builds). This idea pays off for models larger than SRAM. That's idea G and the big-model goal, plus the Pimoroni Pico Plus 2 with 8 MB PSRAM.

**Test:**
1. Build a mixed placement: hot data in SRAM, fc1 weights in flash. Compare it with all-flash and all-SRAM.
2. Later, on a model larger than SRAM, compare compiler-planned placement with plain flash + cache.
3. Add a compiler pass that outputs the per-block "reads per inference ÷ size" table.

## U2. Can weight unpacking be improved?

**Problem:** fc1 spends about 8 cycles per weight unpacking 10-trits-in-16-bits (a multiply, a shift and two branches per weight).

**Options, roughly from simplest to boldest:**

1. **Two bit masks instead of base-3.** Store each group of 32 weights as two 32-bit masks: "which are +1" and "which are −1". Then the layer output is (sum of inputs where the +1 mask is set) − (sum where the −1 mask is set). Cost: 2 bits per weight instead of 1.6, so fc1 grows from 6.1 KB to 7.7 KB, still within the cache. Decoding becomes bit tests with no multiplications.
2. **Skip zeros explicitly.** Store only the positions of the +1 and −1 weights (sparse lists). The work is then proportional to the number of non-zeros: 50% of the weights in tern80, 10% in tern80z90. This is the format that finally turns sparsity into speed (see finding 4 in EXPLAINER §8).
3. **Lookup tables (idea E).** Per group of 4 inputs, precompute all 81 possible ±/0 combinations of their sum once per inference. Each group of 4 weights then becomes a single table lookup. This pays off when the same inputs meet many output rows; fc1 has 96.
4. **Weights as code (idea A, U4).** No unpacking at all: the weights become instructions.

**Test:** implement options 1 and 2 as back-end weight formats (the IR already keeps unpacked weights, so only the back end changes), then time fc1 alone on tern80 and tern80z90, from flash and from SRAM.

## U3. One core computes while the other loads

**Idea:** with the conv channels split across cores, reserve a section of fast memory as a loading area. One core works on data that's already loaded while the other handles data that still needs loading in.

**Assessment: worth including, with two adjustments.**

- **Use DMA for the loading, not a core.** The RP2350 has DMA hardware that copies memory on its own, for free. A core spent copying is a core not computing. The usual pattern is *double buffering*: while both cores compute on buffer A, DMA fills buffer B, then they swap. That's the plan's idea C, and your "reserved section for loading in and out" is exactly that buffer pair.
- **It only helps when data doesn't fit in SRAM.** The current model fits completely, so today there's nothing to load and the test would show no gain. It becomes important with a larger model (idea G / big-model goal).

An asymmetric variant still worth one test now, in flash mode: one core takes the layers whose weights are SRAM-resident, while the other takes the flash-streamed ones (e.g. fc1). The cores then contend less for the shared flash cache.

**Test:**
1. Now: in the flash build, measure the two cores' "join" wait times to see if they slow each other down through the shared cache.
2. Later: a double-buffered DMA weight stream for a model larger than SRAM, two cores computing vs one core computing plus one loading.

## U4. Weights as add/subtract instructions (idea A)

**Idea:** instead of storing ternary weights and unpacking them, the compiler writes them straight into code. For example, `sum += x[3]; sum -= x[7]; …` for every non-zero weight, and zeros simply produce no instruction.

**Why it's attractive:**
- There's no unpacking at all.
- Zeros cost nothing, so it's the most direct way to turn sparsity into speed.

**The catch (from the feasibility study):** each non-zero weight becomes an instruction of 2–4 bytes, against 1.6 bits as packed data. fc1 has about 15,300 non-zeros, so roughly 30–60 KB of code, larger than the 16 KB flash cache.

- **From flash:** it may be *slower*, because the code has to be streamed.
- **From SRAM:** it should be fast.
- **The hybrid:** code for small or hot parts, data for the rest. That's exactly the three-way test in HANDOFF step 8 (all-code / all-data / hybrid).

**Test:** build fc1 all-code, all-data (v0) and hybrid, on tern80 and tern80z90, from flash and from SRAM. Measure cycles, flash size and code size.
