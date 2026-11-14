# Phase 3: the tiny story model

*Started 13 Nov 2026.*

## Atome LM baseline (measured 13 Nov 2026)

[Atome LM](https://github.com/TilelliLab/atome-lm) is the published reference: a 944,640-weight ternary byte-level model trained on TinyStories, with a C99 engine that ran at about 1.0 tok/s on an ESP32-WROOM (240 MHz).

- **Where it is:**
  - The repo is cloned to `train/atome-lm/` (gitignored, commit `be12186`). Install it with `pip install --no-deps -e train/atome-lm`.
  - The engine is vendored unmodified, with its Apache-2.0 LICENSE and NOTICE, in `firmware/baselines/atome/upstream/`.
- **PC reference:** `train/atome/ref.py`, written to `results/atome_ref_seq24.json`.
  - Measured on the first 2 MB of `TinyStoriesV2-GPT4-valid.txt`, in non-overlapping windows: **1.711 bits/byte (perplexity 3.27) at a 256-byte window**, 1.791 at 64 and 1.957 at 24.
  - Greedy text matches Atome's ESP32 log exactly ("Once" → " upon a time, there ").
  - Their training log reports perplexity 2.87 on their own random 10% split of their corpus (described in their `HONEST_RESULTS.md` as TinyStories train + valid, ~1.7 GB; the release isn't stated). That isn't comparable with ours, so we use V2-valid for every model.
- **Pico 2:** `firmware/baselines/atome` and `harness/atome_run.py`, written to `results/phase3.csv`. The window is 24, as on the ESP32.
  - **Bit-exact** with PyTorch on all 4 prompts.
  - **0.64 tok/s from flash** (1.56 s/token, 1.3 MB read from flash per token) and **0.71 tok/s from SRAM**.
  - The state takes 163 KB of RAM and the blob is 277 KB.
- **Why it is slow:** compute, not flash. SRAM is only 10% faster.
  - `atome_predict_next` re-runs the whole window for every token: no KV cache, and the SSM state is reset each time.
  - Each weight is unpacked one trit at a time with branches and a float add, about 17 cycles per weight.
- **Gotcha: unaligned floats.**
  - The `.atome` blob starts with a 7-byte magic, and `atome_load()` points the LayerNorm and SSM float arrays straight into it. On the Cortex-M33 an unaligned `VLDR` faults: the core locks up at `pc=0xeffffffe` on the first token, and the board stops answering.
  - The ESP32 tolerates this. Our fix is to place the blob at an address of the form 4k+1 (`model_blob.S`), which leaves the engine unmodified.
- **Gotcha: Windows console encoding.** Atome's scripts print `→`. Without `PYTHONIOENCODING=utf-8` they die with a `UnicodeEncodeError` on Windows (cp1252).

### Why we don't build on Atome

1. **No feedforward layer.** Atome's block is conv + SSM + top-k attention with a router, so idea B (a tree in place of the feedforward layer) has no natural place to go.
2. **Float activations, LayerNorm and an SSM** don't fit our integer-only compiler. A port would stop matching their model.
3. **"Beat 1 tok/s" is a soft target.** The engine's re-run-everything design alone costs about 10x.
4. **Their own A/B result:** at 944K weights, the ternary model loses about 11% to a plain FP32 model.
5. **The licence on the weights is unclear.** The code is Apache-2.0, but NOTICE says the trained weights are not part of the open-source release, while the README says they are included.

**Decision (13 Nov 2026):** train our own model, MafLM, as two twins of the same size: dense feedforward and tree feedforward. Atome stays as the external yardstick. (A KV-cache variant of its engine was planned as a second baseline and dropped on 14 Nov: it would be work on another project. Its effect is estimated instead, in "Fairness of the Atome comparison" at the end of this file.)

## MafLM (train/lm/)

`model.py` holds the model and `train.py` the trainer.

| | Dense twin | Tree twin |
|---|---|---|
| Shape | d=128, 4 layers, 4 query heads + 2 KV heads (d_head 32), FFN 128→640→128 | same, with the FFN replaced by a depth-3 tree: 7 node neurons, 8 leaves of 128→80→128 |
| Weights | 917,504 | 921,088 |
| Weights read per generated token | 884,864 | **315,008 (2.8x fewer)**; the runtime evaluates only the 3 nodes on the path, so it actually reads 312,960 |

Atome has 944,640 weights.

**Choices made for integer, bare-metal inference:**
- **Vocabulary:** bytes (256), so there are no tokenizer tables.
- **Weights:** every linear layer is ternary with one absmean scale per matrix. The embedding is int8, since only one row is read per token.
- **Activations:** each linear layer RMS-normalizes its input per token (no learned gain) and quantizes it to int8 by absmax. On the board that becomes int8 × ternary → int32.
- **Attention:**
  - Grouped key/value heads keep the KV cache small: 512 B per token across 4 layers, so a 256-token window is 128 KB.
  - ALiBi positions: an integer-friendly per-head distance penalty with no position table, so a rolling KV cache stays valid.
  - q, k and v are fake-quantized to int8, and the attention weights to uint8.
- **Other:** ReLU, no biases, and a learned scalar logit scale.

**Tree training** (as on MNIST, F25):
- Soft routing for the first 30% of steps, with a hardening loss (Bernoulli entropy of the decisions) that ramps up during that phase.
- Then hard routing, with a straight-through estimator to the nodes.
- A load-balancing loss (KL of the mean leaf usage from uniform) runs throughout.
- Evaluation always uses hard routing.

**Training speed** (RTX 3060 Ti, batch 128 × 256 = 32K tokens per step): 0.16 s/step dense, 0.21 s/step tree. Atome's budget (2B tokens) would take about 3–3.5 h per model; Atome itself trains at 0.78 s per step of 65K tokens, because of its Python-loop SSM.

### First check runs (13 Nov 2026, 2,000 steps = 65M tokens each, about 7 min)

| Model | Bits/byte, 256-byte window (V2-valid, first 2 MB) | Perplexity | Weights read per token |
|---|---|---|---|
| Atome 944K (shipped, 30k steps × 65K tokens = 2B tokens) | 1.711 | 3.27 | ~880K |
| MafLM dense | **1.427** | 2.69 | 885K |
| MafLM tree | **1.507** | 2.84 | 315K |

- **Both twins already score better than the fully trained Atome on our test text** after 3% of its token budget. The test text matches our training data, and maybe not Atome's; see "Fairness of the Atome comparison".
- The tree costs 0.08 bits/byte (5.6%) against its dense twin at this point.
- **Load balancing works.** Leaf usage stays near uniform in all 4 layers, with entropy 2.99 of 3 bits; each leaf gets 10–16% of tokens.
- **Greedy samples** are TinyStories-like but repetitive, as expected this early. For example, dense on "Once": "Once upon a time, there was a little boy named Tim. Tim loved to play with his friend, …".
- **Caveat:** these numbers come from the PyTorch model with fake quantization, in bf16. The integer reference that the board will run must reproduce them before they count (next step).

## Integer reference (13 Nov 2026)

`train/lm/intref.py` defines the exact integer arithmetic the board will run; the rules are in its docstring. In short:
- **Residual and embedding:** the residual stream is int32 with 12 fractional bits. The embedding is int8 codes times a per-row multiplier.
- **Each ternary layer:**
  - Shift the input down to 16 bits.
  - Quantize it to int8 by absmax, rounding half away from zero.
  - Take the ternary dot product into an int32 accumulator.
  - Compute one per-vector multiplier `G = wmul * max / rms`, using an integer square root.
  - Output `y = (acc * G + 2^15) >> 16`.
- **Attention:**
  - The KV cache holds int8 codes plus one max per token and head.
  - Scores are in 2^-12 units, with ALiBi as a shift.
  - exp is `2^x` from a 256-entry table, and the weights are uint8 (sum about 255).
  - The values path stays in int32.
- **Tree nodes** decide on the sign of the exact int32 accumulator. This equals the float model's decision, because every scale factor is positive.

**Checks** (on the 2,000-step check models):

| | PyTorch, fp32 with fake quantization | Integer reference | Difference |
|---|---|---|---|
| Dense, bits/byte (first 500 KB of V2-valid) | 1.4194 | 1.4262 | +0.0068 |
| Tree | 1.4998 | 1.5004 | +0.0006 |

- **Token by token:** `train/lm/intref_step.py` is a separate plain-numpy decoder that runs one token at a time with a KV cache, the shape of the firmware loop. Its logits are bit-identical to the batched GPU reference: for both twins on three 48-token sequences, and for the tree over a full 256-token window.
- **Exactness of the batched GPU reference:** matrix products of int8 × ternary run in float32, where every partial sum is an integer below 2^24, so they are exact. The attention values run in float64, below 2^53. TF32 is turned off.
- **Export:** `intref.py --npz` writes the integer model as a flat `.npz`, with the config as `cfg.*`. A round trip gives identical logits. The tree check model is 958 KB, because the npz stores one int8 per ternary weight; the board format will pack them.
- **Greedy integer text** is almost the same as the float model's. For example, tree on "Once": "Once upon a time, there was a little boy named Tim. Tim was very happy. He liked to play with his friend…".
- **Still open:** the 0.0068 bits/byte gap on the dense twin. Candidates are the uint8 attention weights and the `(p * mv) >> 12` truncation. It is small, but worth a look after the long runs.

## Compiler and firmware v0 (13 Nov 2026)

```
train/lm/intref.py --npz ──► compiler/compile_lm.py (mafc/lm_backend.py) ──► firmware/rp2350/maflm/gen/maflm_<model>.{c,h}
                                                                               + lm_rt.c (runtime) + pico_main.c (harness)
```

- **Back end v0** (`compiler/mafc/lm_backend.py`):
  - Packs every ternary matrix at 4 trits per byte (00 = 0, 01 = +1, 11 = −1), row-major, and checks each with a decode round trip.
  - Writes the int8 embedding, the multipliers and an `lm_model` descriptor.
  - The model is still data plus a generic runtime. Specialised code (weights as instructions, unrolling) comes in v1.
- **Runtime** (`firmware/rp2350/maflm/lm_rt.c`):
  - Processes one token per call, with a 256-token ring-buffer KV cache.
  - It mirrors `intref_step.py` line by line.
  - **Matvec kernel:** a 256-entry table turns each byte of 4 trits into two words of int16 pairs (w0,w2) and (w1,w3). These meet the int8 codes, pre-packed the same way, in 4 `SMLAD`s, processing two rows per pass.
  - The tree reads only the 3 node rows on its path and the chosen leaf's 80 rows of `leaf_in`.
- **Checks:**
  - **PC build:** `harness/lm_host_check.py`, with MinGW gcc and SMLAD emulated in C. Logits are identical to the batched reference at every position of 4 × 256-token sequences, for both twins, on the first try.
  - **Pico 2:** `harness/lm_run.py` compares the argmax and a 64-bit logits checksum at every position of 256-token sequences. **Bit-exact on all four builds.**
- **Memory:**
  - 216 KB of packed weights, a 33 KB embedding and a 144 KB KV cache (256 tokens).
  - SRAM builds use 484–490 KB of the 512 KB of main SRAM.
  - Flash builds use 202–207 KB of SRAM.

**Speed** (2,000-step check models, `results/phase3.csv`; greedy generation from 4 prompts until the window is full):

| Build | ms/token, positions 0–23 | tok/s | ms/token, whole 256 window | tok/s | Flash read per token |
|---|---|---|---|---|---|
| Atome 944K, upstream engine, flash (window 24) | 1,559 | 0.64 | – | – | 1,317 KB |
| Atome 944K, upstream engine, SRAM (window 24) | 1,414 | 0.71 | – | – | 0 |
| MafLM dense, flash | 44.5 | 22.5 | 52.2 | 19.2 | 226 KB |
| MafLM dense, SRAM | 22.0 | 45.4 | 29.3 | 34.2 | 0 |
| MafLM tree, flash | 18.4 | 54.3 | 26.6 | 37.6 | **83 KB** |
| MafLM tree, SRAM | **10.3** | **97.1** | **18.2** | **55.0** | 0 |

- **The tree is 2.1x faster than its dense twin from SRAM and 2.4x from flash** (positions 0–23). It reads 2.7x fewer flash bytes, in line with the 315K vs 885K weights read per token.
- **Against Atome's engine on the same chip, the gap is about 140x.** Most of that is Atome's engine recomputing the whole context for every token (~15x), plus a faster runtime; idea B contributes only ~2x. The breakdown is in "Fairness of the Atome comparison".

**Where the time goes** (SRAM profiling builds; cycles per token):

| Stage | Tree, positions 0–23 | Tree, 0–255 | Dense, 0–23 | Dense, 0–255 |
|---|---|---|---|---|
| q, k, v projections | 402k (29%) | 402k (16%) | 403k (13%) | 403k (9%) |
| Attention core | 256k (19%) | **1,385k (55%)** | 255k (8%) | 1,384k (32%) |
| Wo | 255k (18%) | 255k (10%) | 255k (8%) | 255k (6%) |
| FFN | 361k (26%) | 361k (14%) | **2,140k (68%)** | 2,140k (50%) |
| Unembed | 109k (8%) | 109k (4%) | 108k (3%) | 108k (3%) |

**v1 candidates:**

1. **Attention core.** It costs about 340 cycles per key. The causes:
   - a 64-bit division per key for the uint8 weight;
   - an int64 multiply-accumulate on the value path;
   - a scalar dot product.

   The weights fit a 32-bit division, and the dot product can use `SMLAD`. At the full window this is half the time.
2. **The matvec kernel.** It runs at 3.1–4.3 cycles per weight against the about 1.5 planned. One cause is that `quant()` does a 64-bit division per element (`(254|x| + m) / 2m`), where 32 bits suffice since the inputs are shifted to 16 bits. Others: two rows per pass instead of four, and byte loads.
3. **The second core:** split the matvec rows and the attention heads across both cores, as on MNIST.
4. **Weights as code** for the small hot matrices: idea A, which won on MNIST from SRAM.

## Runtime v1: activation tables (13 Nov 2026)

What changed (all still bit-exact: the PC check and the board check pass for both twins, 2 × 256 positions per build):

1. **32-bit divisions.**
   - `quant()` shifts its input to 16 bits, so the code division `(254|x| + m) / 2m` fits in 32 bits and uses the hardware UDIV, not a 64-bit library call per element.
   - The same applies to the attention weight `(255 e + S/2) / S` (below 2^25), and to the per-head codes when the max is below 2^24 (with an exact 64-bit fallback).
2. **Attention core:**
   - The q·k dot product uses `SMLAD` on `SXTB16`-unpacked int8 codes.
   - The value sum uses int32 whenever `127 * sum_j w_j < 2^31`, which proves no partial sum can overflow; otherwise it falls back to int64.
   - 1,385k → about 620k cycles per token at the full window.
3. **Matvec: lookup tables over activations (idea E).** The weight-table + SMLAD kernel was load-bound (11 loads plus 3 spills per 16 weights; about 2.5 cycles per weight even after interleaving rows).
   - **Weight format:** each byte holds 4 trits as a base-3 number 0..80. Groups of 4 rows are interleaved, so one 32-bit load fetches four rows' bytes. The tree nodes stay row-major, because they're read one row at a time.
   - **Tables:** for each group of 4 inputs, `tab[g][0..80]` holds every possible partial sum (int16). A row costs one table load and one add per 4 weights.
   - **Sharing:** tables are built 128 inputs at a time (5 KB) and cached by vector version, so q, k and v share one build, as do the tree's nodes and leaf.
   - **Building:** `T[a + 9b] = A[a] + B[b]`, with 4 `SADD16` and unaligned word stores per row of 9. That's about 160 cycles per group, down from about 310 for the first, scalar version.
   - The lookup loop handles two groups per iteration.
   - **Result:** about 1.15 cycles per weight in the lookups, and 1.2–1.7 per weight including table builds and the per-vector quantize/rescale, down from 3.1–4.3 in v0.

**Speed** (2,000-step check models, `results/phase3.csv`):

| Build | v0, positions 0–23 | **v1, positions 0–23** | v1 tok/s | v1, whole 256 window | v1 tok/s | Flash per token |
|---|---|---|---|---|---|---|
| Dense, flash | 44.5 ms | 23.2 ms | 43.1 | 27.0 ms | 37.1 | 246 KB |
| Dense, SRAM | 22.0 ms | 11.1 ms | 90.0 | 14.6 ms | 68.5 | 0 |
| Tree, flash | 18.4 ms | 9.4 ms | 106.1 | 13.2 ms | 76.0 | 85 KB |
| Tree, SRAM | 10.3 ms | **5.1 ms** | **195.1** | **8.5 ms** | **117.4** | 0 |

**Profile** (SRAM profiling builds; thousands of cycles per token, positions 0–23 / 0–255):

| Stage | Dense | Tree |
|---|---|---|
| q, k, v | 199 | 201 |
| Attention core | 108 / 620 | 109 / 622 |
| Wo | 132 | 133 |
| FFN first half (W1, or nodes + leaf_in) | 440 | 110 |
| FFN second half (W2, or leaf_out) | 650 | 93 |
| Unembed | 55 | 55 |
| Table builds (included above) | 194 | 91 |

- **The tree's FFN is 5.4x faster than the dense FFN** (203k vs 1,090k cycles), close to the 7.8x fewer FFN weights it reads.
- **Over the whole token, the tree is 2.2x faster from SRAM and 2.5x from flash.** Attention and the other layers are the same in both twins, so they cap the gain (Amdahl).
- **Next:**
  - The second core, splitting rows and heads.
  - The attention core, now the largest stage late in the window. Its value path still does 32 scalar multiply-adds per key.
  - Weights as code for the hot small matrices.

## Two cores (13 Nov 2026, `-DDUAL_CORE=ON`)

**Design:**
- `par(f, ctx, n, split)` in `lm_rt.c` is a parallel-for. Core 0 runs `[0, split)` while core 1 runs `[split, n)`.
- Core 1 sits in a job loop on the SIO FIFO: one word each way per job, with `__dmb()` around the shared job descriptor. It is launched once, by `lm_init()`.
- **What gets split:**
  - table builds, by groups of 4 inputs;
  - the interleaved matvec row groups, with each core also rescaling its own rows to `y`;
  - the attention heads, 2 per core, with per-core scratch.
- **What stays on core 0:** the 3 tree-node dot products, the per-vector quantization and the residual adds.
- **Checks:** the PC build compiles the same partitioning with `--dual`, running both halves on one core, and is bit-exact. On the board, all builds are bit-exact.

**Gotcha: flash-resident weights must not be split.** Splitting matvec rows whose weights are in flash made both cores miss the XIP cache at two addresses at once. Flash bytes per token stayed the same (81 vs 85 KB for the tree), but the builds got slower than one core: tree 9.43 → 11.33 ms, dense 23.2 → 29.1 ms (positions 0–23). The misses serialize on the one QSPI bus, and each becomes more expensive, because two interleaved streams break the sequential reads. MNIST showed the same effect in a milder form (F12). Now `matvec_y()` keeps rows on core 0 when the weights' address is in XIP (0x1xxxxxxx); heads and tables are still split.

**Speed** (2,000-step check models, `results/phase3.csv`):

| Build | One core, 0–23 | **Two cores, 0–23** | tok/s | Two cores, whole 256 window | tok/s |
|---|---|---|---|---|---|
| Dense, flash | 23.2 ms | 20.8 ms | 48.0 | 23.2 ms | 43.1 |
| Dense, SRAM | 11.1 ms | 6.8 ms | 146.9 | 8.9 ms | 111.9 |
| Tree, flash | 9.4 ms | 8.5 ms | 117.5 | 10.9 ms | 92.1 |
| Tree, SRAM | 5.1 ms | **3.3 ms** | **306.8** | **5.4 ms** | **184.6** |

- **From SRAM, two cores give 1.6x on both twins.** Every stage gains 1.6–1.9x. The rest is handshakes (about 6 jobs per layer) and serial per-vector work on core 0.
- **From flash, the gain is only 1.1x.** The weight stream from flash is the bottleneck, and it can't be split. To overlap it with compute, core 1 could prefetch the next layer into SRAM by DMA from the XIP stream FIFO (ideas C and G).
- **The tree stays faster than its dense twin in every configuration:** 2.1–2.5x over positions 0–23, and 1.65–2.2x over the whole window.

## Trained models (overnight run, 13–14 Nov 2026)

`train/lm/overnight.sh` with the defaults: 61,000 steps × 128 × 256 = **2.0B bytes per twin**, the same number of training bytes as Atome (not the same data). Dense took 2 h 46 min and the tree 3 h 25 min on the PC's GPU. Validation ran every 2,000 steps on the first 2 MB of V2-valid. The best checkpoints were dense at step 56,000 and tree at step 54,000. Late in training the evaluation set is fixed, but the number still varies between checkpoints, by up to about 0.01 for the tree.

| Model | Bits/byte (V2-valid, first 2 MB, 256 window) | Held out: last 2 MB | Perplexity | Integer reference | Weights read per token |
|---|---|---|---|---|---|
| Atome 944K (2B bytes, its own data) | 1.711 | 1.721 | 3.27 | – | ~880K |
| **MafLM dense** | **1.152** | 1.165 | **2.22** | 1.153 (+0.0002) | 885K |
| **MafLM tree** | **1.206** | 1.219 | **2.31** | 1.206 (+0.0006) | 315K |
| (2,000-step check models) | 1.427 / 1.507 | | 2.69 / 2.84 | | |

The first 2 MB is also where our best checkpoint was picked. The last 2 MB was never used for any choice.

- **On our test text both twins score well below Atome:** −0.56 bits/byte for the dense twin and −0.50 for the tree, on the held-out slice, with slightly fewer weights.
  - Our training data matches the test text and Atome's may not.
  - If Atome's own-split gap (0.19) is all domain shift, the lead becomes about 0.36 and 0.30.
  - All the caveats are in "Fairness of the Atome comparison".
- **The tree costs 0.054 bits/byte (4.7%) against its dense twin** for 2.8x fewer weights read. The gap shrank from 0.080 on the check models.
- **The tree's routing stayed balanced** throughout the hard phase: leaf entropy 2.98–2.99 of 3 bits in every layer.
- **The integer reference matches PyTorch to within 0.0006 bits/byte.**
- **Samples** (greedy, from the integer reference and identical on the board): fluent TinyStories openings, for example "Once upon a time, there was a little boy named Tim. Tim loved to play with his toy ball." Greedy decoding falls into repetition loops after one or two sentences ("The dog was sad and said, ..."), as small models do; sampling with a temperature would be the fix and is a firmware change only.

**On the Pico 2** (two cores, `build-lm_<twin>-<mem>-dual`, `results/phase3.csv`): bit-exact against the integer reference on 2 × 256 positions for all four builds, and the same generated text as the PC.

| Build | ms/token, positions 0–23 | tok/s | ms/token, whole 256 window | tok/s | Flash read per token |
|---|---|---|---|---|---|
| Dense, flash | 20.81 | 48.0 | 22.82 | 43.8 | 223 KB |
| Dense, SRAM | 6.77 | 147.7 | 8.56 | 116.8 | 0 |
| Tree, flash | 8.47 | 118.0 | 10.41 | 96.1 | **82 KB** |
| Tree, SRAM | **3.22** | **310.3** | **4.99** | **200.6** | 0 |
| Atome, upstream engine, SRAM (window 24; recomputes the context for every token) | 1,414 | 0.71 | – | – | 0 |

- Speed is the same as with the check models, as expected: same architecture, same kernels.
- **The tree is 2.1x faster than the dense twin from SRAM and 2.5x faster from flash** over positions 0–23 (1.7x and 2.2x over the whole window, where attention takes a larger share).
- **Against Atome's engine on the same board:** the tree from SRAM is about 440x faster per generated token at window 24 (about 300x if our prompt time is counted, as Atome's engine counts its own).
  - This compares engines, not models. About 15x of it is Atome's engine recomputing the context for every token, and 1.6x is the second core.
  - On the board, Atome can only run a 24-byte window, where it scores 1.957 bits/byte.
  - With a cache, Atome's engine would be at about 10 tok/s (estimate, not measured), so roughly 30x.
  - Breakdown in the next section.

## Fairness of the Atome comparison (14 Nov 2026)

Atome is an external yardstick, not a controlled experiment. **The controlled experiment in Phase 3 is dense twin against tree twin** (see "Caveats on the twin comparison" below). This section lists every known difference with Atome, and which way it biases the numbers.

**Decision (14 Nov 2026):** we don't build or retrain anything of Atome's (no cached engine, no retraining on our data). It is their project. The comparison stays as measured, with the caveats below.

### Quality (bits/byte; `train/atome/fairness.py` reproduces the table)

**Same measurement for both:**
- Byte vocabulary, cross-entropy per byte, non-overlapping 256-byte windows on the same text.
- Atome's own `AtomeLM.loss` and `ByteTokenizer` give exactly the numbers of `train/atome/ref.py`, so the method can't explain the gap.
- Atome also trained at a 256-byte context.

| Test text | Atome | Dense | Tree |
|---|---|---|---|
| V2-valid, first 2 MB (the reported numbers; also our checkpoint selection set) | 1.711 | 1.152 | 1.206 |
| **V2-valid, last 2 MB (never used for any choice)** | **1.721** | **1.165** | **1.219** |
| V2-valid cleaned with Atome's `build_corpus.py`, first 1 MB | 1.707 | 1.153 | 1.205 |
| Same, 1–3 MB | 1.712 | 1.156 | 1.208 |

**Differences, and which way they lean:**
1. **Training data differs** (likely favours us, size unknown).
   - We train on TinyStories V2-GPT4 train (2.2 GB) and test on V2-GPT4 valid.
   - Atome's `HONEST_RESULTS.md` (Table 2) gives its 944K model's corpus as "TinyStories full (train.txt + valid.txt concatenated, ~1.7 GB raw)". It doesn't say which TinyStories release. Its `REPRODUCE.md` command would instead build only 1 MB of V2-valid; we take `HONEST_RESULTS.md` as the description of the shipped model.
   - At 2B bytes, that is about 1.2 epochs over its corpus, similar to our 0.9. So "same bytes" here also means a similar amount of distinct text.
   - Atome's log reports 1.521 bits/byte (perplexity 2.87) on a random 10% of its own corpus, against 1.711 on ours. That 0.19 gap suggests its corpus is a different text distribution from V2-GPT4, possibly another TinyStories release.
   - **If all of that gap is domain shift against Atome**, our lead on the held-out slice shrinks from 0.56 / 0.50 bits/byte to **0.36 (dense) / 0.30 (tree)**: 1.521 against 1.165 and 1.219. This is an adjustment, not a true bound: we never measured our twins on Atome's distribution.
2. **Possible test-set overlap** (would favour Atome). If Atome's `valid.txt` was V2-valid, it trained on our test text. The shipped model shows no sign of that: it scores the same on V2-valid text in its own cleaned format (1.707–1.712) as on the held-out raw slice (1.721).
3. **Checkpoint selection** (favours us, slightly).
   - Our best checkpoints were picked out of 31 evaluations on the first 2 MB, which is also the reported test text: dense at step 56,000, tree at step 54,000.
   - Final-step numbers: 1.157 and 1.209. The last-2 MB row above is the selection-free number.
   - Atome's checkpoint was picked on its own split.
4. **Activations** (favour Atome). Atome keeps float activations with LayerNorm; ours are int8 everywhere, so they can run on the integer runtime.
5. **Hyperparameters.**
   - Atome's are its authors' choices.
   - Ours were fixed before the 2,000-step check runs and not changed afterwards. The check runs did evaluate on the same first 2 MB, so they were seen before the long run.
6. **Model size:** about the same. Atome has 944,640 weights; the dense twin has 917,504 and the tree 921,088.

### Speed (tokens/s on the Pico 2)

Both run on the same board at 150 MHz, from SRAM and from flash, and both are bit-exact with their own references. **But this compares engines, not models**, and Atome's engine was written for portability on an ESP32, not for speed on this chip.

1. **No cache in Atome's engine.** `atome_predict_next` reprocesses every token in the context to produce each new one: it resets the SSM state and keeps no key/value cache.
   - Over our measurement (4 prompts, generating up to its compiled window of 24), the context averages 14.9 tokens, so each new token costs about 15 full positions.
   - **This factor (~15x) comes from their engine design, not from our compiler or idea B.**
2. **Atome only runs at a 24-byte window on the board.** Its state holds about 7 float arrays of `[window][256]`: 163 KB at 24, so about 1.7 MB at 256, more than the SRAM.
   - At window 24, Atome scores 1.957 bits/byte, against its 1.711 at 256.
   - So the board-speed comparison is at window 24, while the quality comparison is at 256.
   - Our twins run the full 256 window on the board.
3. **The prompt is timed differently.**
   - Our tok/s counts generated tokens only. The prompt is fed through the KV cache first and timed separately.
   - Atome's engine reprocesses the prompt inside every timed call.
   - For the 4 prompts at window 24 (30 prompt steps beside 66 generated ones), counting our prompt time too would make ours about 1.45x slower. That turns 440x into about 300x.
   - Over the full 256 window, the effect is negligible.
4. **Portable float C against hand-tuned integer code.**
   - Atome unpacks one trit at a time with branches and float adds (~17 cycles per weight) and uses one core.
   - Ours uses activation lookup tables, SIMD and two cores, tuned for the Cortex-M33.
   - A faster Atome engine is possible, and nobody has written it.
5. **Split of the 440x** (MafLM tree, SRAM, two cores, positions 0–23, prompt excluded, against Atome's upstream engine, SRAM, one core, window 24; the four factors multiply to 440x):

   | Factor | Ratio | Basis |
   |---|---|---|
   | Recompute vs cache | ~15x | Measured 1,414 ms per token ÷ 14.9 positions = ~95 ms per position, an estimate |
   | Runtime and model at one position: Atome engine vs MafLM v1 dense, one core | ~8.6x | 95 vs 11.1 ms. Mixes kernel quality with the two architectures |
   | Idea B: tree vs dense twin, one core | 2.2x | 11.1 vs 5.1 ms, measured on the check models |
   | Two cores vs one | 1.6x | 5.1 vs 3.22 ms, measured |

   - **Estimate, not measured:** Atome's engine with a cache would run at about 10 tok/s on this board.
   - Against that, our dense twin on one core is about 9x faster and the tree on two cores about 30x, rather than 440x.
   - Atome's published 1.0 tok/s is on an ESP32 at 240 MHz.
6. **Energy was not measured** for either engine.

### Caveats on the twin comparison

The twins share everything that matters: the same runtime, data, budget (61,000 steps), batch order (seed 0), test text and board. Their training arguments differ only in `--ffn`. But:
- **The tree adds training machinery:** a 30% soft-routing phase, a hardening loss and a load-balancing loss.
  - That costs more training compute: 0.21 vs 0.16 s per step, 3 h 25 min vs 2 h 46 min.
  - Its settings were carried over from MNIST, untuned.
- **The tree has 3,584 more weights** (the routing nodes): 921,088 vs 917,504.
- **Each twin's checkpoint was picked on the test text,** at different steps (54k and 56k). The quality gap agrees across views:
  - 0.054 on the selection set;
  - 0.054 on the held-out slice;
  - 0.052 at the final step.
- **One seed and one board.**
- **The speed ratio depends on the window:**
  - 2.1x (SRAM) and 2.5x (flash) over positions 0–23;
  - 1.7x and 2.2x over the full 256 window, where attention, which the tree doesn't change, takes a larger share.

**What to quote:**
- **Idea B:** the tree is 2.1–2.5x faster than its dense twin over the first 24 positions, and 1.7–2.2x over the full 256 window, for +0.054 bits/byte (4.7%). One seed.
- **Against Atome:** "on our test text, 0.50–0.56 bits/byte better (about 0.30–0.36 if Atome's own-split gap is all domain shift)". And: "~440x faster per generated token than Atome's own engine at window 24 (~300x counting the prompt); ~15x of that is the engine recomputing the context, so against an engine with a cache we estimate ~30x".
