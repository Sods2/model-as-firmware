# Notes

Working notes kept for future reference: what was looked into, what was found, and how things are done. The original plan and decisions live in [docs/history/plan.md](../docs/history/plan.md); measured numbers live in [results/](../results/); how to reproduce them is in [docs/REPRODUCE.md](../docs/REPRODUCE.md).

| Note | What's in it |
|---|---|
| [FINDINGS.md](FINDINGS.md) | **Start here for analysis:** all conclusions so far (F1–F28) with evidence, headline numbers, open questions, caveats |
| [EXPLAINER.md](EXPLAINER.md) | **New to the project?** Plain-language guide from zero: models, ternary, the hardware, what each piece does, results so far, glossary. Keep it updated |
| [IDEAS.md](IDEAS.md) | Ideas and questions to test (U1–U5: cache placement, cheaper weight unpacking, compute/load core split, weights as code, bitnet.cpp for the Pi phase), each with an assessment, a test and its outcome |
| [LOG.md](LOG.md) | Dated lab log: what was done each session, with key numbers |
| [lm.md](lm.md) | Phase 3: the tiny story models (Atome baseline, our MafLM twins, integer reference, firmware, results) and the fairness of the Atome comparison |
| [bitnetmcu-integer-inference.md](bitnetmcu-integer-inference.md) | How BitNetMCU's integer inference works (packing formats, requantization), how our C reference and golden model relate to it, licence |
| [pico-toolchain.md](pico-toolchain.md) | Toolchain notes and gotchas: where the tools live, flashing over SWD and USB, talking to the board, quirks |
| [compiler.md](compiler.md) | mafc compiler: design decisions, v0 results vs the reference, where the time goes, v1 candidates (second core, SIMD with proven bounds, fc1 formats) |
| [cmsis-nn-baseline.md](cmsis-nn-baseline.md) | Design of the CMSIS-NN int8 baseline (Phase 0 step 5) and why it's a fair comparison |
