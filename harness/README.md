# harness/: PC-side test harness (Phase 0 step 6)

Sends MNIST test images to a target, checks the outputs bit for bit against a golden model, and logs one CSV row per run in `results/`.

| File | Role |
|---|---|
| `golden.py` | Numpy golden models. `Golden` repeats the plain-C reference arithmetic, reading weights straight from `train/export/<model>.h`. `GoldenS8` does the same for the CMSIS-NN int8 build, from `firmware/baselines/cmsis_nn/model_s8_<model>.h`. `GoldenFFF` is the tree model of idea B (`fff80`), read from `train/export/<model>.npz`. |
| `run.py` | Runs one target (`golden`, `host` = the C reference built for the PC, `pico` = the board over USB serial). Compares with the golden model, the PyTorch model and the labels, and appends a row to `results/phase0.csv`. On the board it also reads the XIP cache counters (command `X`) and fills `flash_read_kb` (FINDINGS F22) |
| `energy.py` | Energy per inference with a power meter. A typical USB meter is too coarse for the Pico 2 (0–4 mA in 1 mA steps), so this waits for a finer one (an INA226 or PPK2). The board idles for `--secs`, then runs inference back to back for `--secs` (firmware command `L`); you type in the meter reading after each phase (`--reading mwh|mw|ma`). Logs `results/energy.csv`: power idle and busy, energy per inference above idle and in total. `--no-meter` only checks the loop. Easiest through `measure.sh energy` |
| `export_cmsis.py` | Calibrates and writes the CMSIS-NN int8 model header |
| `env.sh` | Where the tools are: `PY`, `PORT`, picotool, OpenOCD and the Pico toolchain, each overridable from the environment ([docs/SETUP.md](../docs/SETUP.md), section 6). Sourced by every shell script |
| `flash.sh` | Flashing helper sourced by `measure.sh` and `phase0_matrix.sh`. `flash <build.uf2>` loads the matching `.elf` over SWD with the Debug Probe (OpenOCD: program, verify, `cold_reset`), falls back to `picotool` over USB if that fails, then waits until the board answers `?` on the port. `FLASH=swd` or `FLASH=usb` forces one way. About 3–4 s per build |
| `phase0_matrix.sh` | Builds, flashes and measures every Phase 0 configuration (`profile` or `final`) |
| `measure.sh` | Builds, flashes and measures any list of `build:model:mem[:ternary[:int4[:conv[:stream[:int8[:fuse]]]]]]` specs (builds `ref_c`, `cmsis_nn`, `cmsis_nn_s4`, `mafc`, `mafc2`, `mafc_s`, `mafc2_s`; models `tern80`, `tern80z70`, `tern80z90`, `int8_80`, `fff80`; ternary = the fc1 format, int4 = the fc2/fc3 format, conv = data|code, stream = stream|nostream, int8 = plain|code|dense, fuse = nofuse|fpool|fpoolrp for mafc builds; the full list is at the top of the script; mode `energy` runs `energy.py` instead of `run.py`), e.g. `harness/measure.sh final results/phase1.csv mafc2_s:tern80:ram:code:code`. `NOTES="..."` labels the CSV rows |
| `bench_stream.py` | Idea C benchmark (FINDINGS F21): `gen` writes `firmware/bench/stream/gen/stream_data.h`; `run` drives the flashed firmware and logs `results/bench_stream.csv` (seven ways to feed one 64 KB layer from flash, each checked bit-exact) |
| `bench_conv.py` | Wide-conv fairness benchmark (FINDINGS F17). `gen` writes the data and the generated code kernels to `firmware/bench/conv_wide/gen/`; `run` drives the flashed benchmark firmware and logs `results/bench_conv.csv` (every kernel's output is checksum-checked on the board) |
| `launcher.py` | The test launcher ([`ui/launcher.html`](../ui/launcher.html)): choose a test and a target, and it starts the matching live view |
| `bridge.py` | Live view: runs a target image by image and streams each result, the per-stage cycles and the golden model's layer outputs to [`ui/live.html`](../ui/live.html) in the browser |
| `story_live.py` | The story-model mode of the bridge, for [`ui/story.html`](../ui/story.html) |
| `lm_run.py`, `lm_host_check.py`, `atome_run.py` | Story models: the board run (exactness, then speed), the PC check of the C runtime, and the Atome LM reference on the board |

## Typical use

From the repo root, in bash (Git Bash on Windows), with the venv's `python` ([docs/SETUP.md](../docs/SETUP.md)). Step-by-step recipes for every result are in [docs/REPRODUCE.md](../docs/REPRODUCE.md).

```bash
# golden model alone (no hardware)
python harness/run.py --model tern80 --target golden --csv scratch.csv
# C reference on the PC (build it first, see firmware/baselines/README.md)
python harness/run.py --model tern80 --target host --exe firmware/baselines/ref_c/build-host/ref_host_tern80.exe --csv scratch.csv
# the board, after flashing a build (port: COM3 on Windows, /dev/ttyACM0 on Linux)
python harness/run.py --model tern80 --target pico --port COM3 --elf <build>/ref_c.elf --csv scratch.csv
python harness/run.py --model tern80 --target pico --port COM3 --golden s8 --elf <build>/cmsis_nn.elf --csv scratch.csv
# build, flash and measure (append to your own CSV, not the committed results)
CSV=scratch.csv harness/phase0_matrix.sh final
harness/measure.sh final scratch.csv mafc2_s:tern80:ram:code:dense:data:nostream:plain:fpoolrp
```

Without `--csv`, `run.py` appends to `results/phase0.csv`.

`run.py` exits with status 1 if any logits differ from the golden model.

## Choosing a test: the launcher

```bash
python harness/launcher.py            # then open http://127.0.0.1:8765
```

This is the easiest way in; [docs/UI-GUIDE.md](../docs/UI-GUIDE.md) explains the pages panel by panel. The home page (`ui/launcher.html`) asks three things:

1. **What to test:** MNIST digits (`ui/live.html`) or the tiny story model (`ui/story.html`).
2. **Where it runs:** the golden model (no hardware), the PC build of the C reference (MNIST only), or the Pico 2.
3. **Which model or build:**
   - for the golden model, the model (`train/export/*.h` or `train/runs/*.npz`);
   - for the PC build, the `ref_host_*.exe`;
   - for the board, either the build already on it or any existing `build-*/` folder that holds a `.uf2`. The list is searchable, for example "mafc2_s tern80 ram prof" or "lm_tree ram". The launcher flashes the chosen build over SWD (`harness/flash.sh`, picotool as the fallback) before starting. Nothing is compiled here; build first with `measure.sh` or CMake.

Start stops the running test (which frees the serial port), flashes if needed, starts `bridge.py` on port 8766, and opens the live page. Each live page has an "← All tests" link back. The header shows what is on the board, from its INFO line; "Check the board" reads it again. The launcher doesn't read the board while a test, `run.py`, `measure.sh` or picotool/OpenOCD holds it. Builds of the tree MNIST model (fff80) are listed but greyed out, because the live MNIST view doesn't support them yet. It greys out choices whose files are missing (MNIST data, a trained story model, PC builds, build folders) and says where the setup step is. Stop also cancels a test that is still flashing or starting. Options: `--port` (default `COM3` on Windows, `/dev/ttyACM0` elsewhere), `--http-port 8765`, `--view-port 8766`.

## Watching a run live

The launcher starts the bridge for you. By hand:

```bash
python harness/bridge.py --target pico --port COM3      # model and golden model picked from the board's INFO line
python harness/bridge.py --target host --exe firmware/baselines/ref_c/build-host/ref_host_tern80.exe
python harness/bridge.py --target golden                # no hardware
```

Then open http://127.0.0.1:8765. The page has three modes:

- **Live** follows the stream.
- **Slow-mo** steps the board one image at a time and moves through the stages: 1000× slower than real time on a profiling build, otherwise 8 s per image with the stages at equal width. The inspector follows along.
- **Replay** scrubs back through what has been received.

The inspector shows each layer's feature maps, FC1's −1/0/+1 weight matrix, and the FC outputs. These come from `golden.py`, so they are exactly what the board computed. Per-stage times need a `*_prof` build (`PROFILE_STAGES`). On other builds the bridge stops asking after the first all-zero `T` reply.

If the board disagrees with the golden model, the run pauses behind a red banner, and the Current image panel shows the differing logits. Pass `--no-stop-on-mismatch` to keep going. `--rate` sets images per second (the page has a speed control too). With `--target pico`, the bridge won't open the serial port while `harness/run.py`, `measure.sh`, `phase0_matrix.sh` or `picotool` is running, so it can't take the board from a measurement. `--force` overrides this. `golden` and `host` never touch the board. The bridge only listens on 127.0.0.1. It needs nothing beyond what `run.py` already uses, because it streams over Server-Sent Events from Python's standard-library HTTP server. It doesn't log to `results/`; use `run.py` for measured runs.

### The tiny story model (`ui/story.html`)

```bash
python harness/bridge.py --target pico --port COM3                                  # board running a maflm build
python harness/bridge.py --task story --target golden --npz train/runs/lm_tree.npz  # no hardware (needs a trained model)
```

The bridge switches to story mode by itself when the board's INFO line has `window=`, meaning a MafLM build (`harness/story_live.py`). The integer model comes from `train/runs/<board model>.npz` unless you pass `--npz`, and `--prompt "Once"` starts generating right away. In the page you type a prompt (up to 64 bytes) and press Generate:

- **Bytes stream in one at a time.** The board runs each position (`S` command), and the golden model, `train/lm/intref_step.py`, runs the same byte. A position counts as checked only if the board's argmax and 64-bit logits checksum both equal the golden model's. A mismatch pauses the run behind a red banner.
- **Greedy or sampled.** Greedy takes the board's argmax, as `lm_run.py` measures. With a temperature, the PC samples from the golden logits with a fixed seed; the checksum shows they are the board's logits. Sampling avoids the repetition loops of greedy decoding.
- **Per byte:** the top 5 candidates, and for the tree twin the leaf each layer picked. These come from the golden model, which the checksum ties to the board.
- **Board time per byte across the 256 window:** the summary tiles give the mean over positions 0–23 and over all generated bytes, the same definitions as `results/phase3.csv`.
- **Flash read per byte**, from the XIP counters.
- **Stage shares:** on `-DLM_PROFILE=ON` builds, for example `build-lm_tree-ram-dual-prof`.
- **Leaf usage** for the generation.
- **The saved board runs from `results/phase3.csv`,** with the fairness note on the Atome comparison.

Board times exclude USB transfer and the PC-side check, which limit how fast bytes appear on screen. The page's speed control slows generation down to watch it. The prompt goes into the KV cache one byte at a time, like the generated bytes, so its positions are timed too, but the summary tiles count generated bytes only.

## What "matches PyTorch" means

The board doesn't do float arithmetic, so its predictions can't equal PyTorch's on 100% of images. The two checks are:

1. **Correctness, which must be exact:** board logits == golden-model logits on every image. The golden model is an independent numpy implementation of the same integer arithmetic.
2. **Fidelity, which is logged:** prediction agreement with the PyTorch model, plus accuracy against the labels. For tern80 this is 99.98% agreement, and the 2 images that differ are within the noise of the power-of-two rescaling.
