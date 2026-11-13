# harness/: PC-side test harness (Phase 0 step 6)

Sends MNIST test images to a target, checks the outputs bit for bit against a golden model, and logs one CSV row per run in `results/`.

| File | Role |
|---|---|
| `golden.py` | Numpy golden models. `Golden` repeats the plain-C reference arithmetic, reading weights straight from `train/export/<model>.h`. `GoldenS8` does the same for the CMSIS-NN int8 build, from `firmware/baselines/cmsis_nn/model_s8_<model>.h`. `GoldenFFF` is the tree model of idea B (`fff80`), read from `train/export/<model>.npz`. |
| `run.py` | Runs one target (`golden`, `host` = the C reference built for the PC, `pico` = the board over USB serial). Compares with the golden model, the PyTorch model and the labels, and appends a row to `results/phase0.csv`. On the board it also reads the XIP cache counters (command `X`) and fills `flash_read_kb` (FINDINGS F22) |
| `energy.py` | Energy per inference with the USB power meter (step 2b). The board idles for `--secs`, then runs inference back to back for `--secs` (firmware command `L`); you type in the meter reading after each phase (`--reading mwh|mw|ma`). Logs `results/energy.csv`: power idle and busy, energy per inference above idle and in total. `--no-meter` only checks the loop. Easiest through `measure.sh energy` |
| `export_cmsis.py` | Calibrates and writes the CMSIS-NN int8 model header |
| `flash.sh` | Flashing helper sourced by `measure.sh` and `phase0_matrix.sh`. `flash <build.uf2>` loads the matching `.elf` over SWD with the Debug Probe (OpenOCD: program, verify, `cold_reset`), falls back to `picotool` over USB if that fails, then waits until the board answers `?` on the port. `FLASH=swd` or `FLASH=usb` forces one way. About 3–4 s per build |
| `phase0_matrix.sh` | Builds, flashes and measures every Phase 0 configuration (`profile` or `final`) |
| `measure.sh` | Builds, flashes and measures any list of `build:model:mem[:ternary[:int4[:conv[:stream[:int8]]]]]` specs (`ref_c`, `cmsis_nn`, `mafc`, `mafc2`, `mafc_s`, `mafc2_s`; ternary = the fc1 format, int4 = the fc2/fc3 format, conv = data|code, stream = stream|nostream, int8 = plain|code|dense for mafc builds; mode `energy` runs `energy.py` instead of `run.py`), e.g. `harness/measure.sh final results/phase1.csv mafc2_s:tern80:ram:code:code`. `NOTES="..."` labels the CSV rows |
| `bench_stream.py` | Idea C benchmark (FINDINGS F21): `gen` writes `firmware/bench/stream/gen/stream_data.h`; `run` drives the flashed firmware and logs `results/bench_stream.csv` (seven ways to feed one 64 KB layer from flash, each checked bit-exact) |
| `bench_conv.py` | Wide-conv fairness benchmark (FINDINGS F17). `gen` writes the data and the generated code kernels to `firmware/bench/conv_wide/gen/`; `run` drives the flashed benchmark firmware and logs `results/bench_conv.csv` (every kernel's output is checksum-checked on the board) |
| `bridge.py` | Live view: runs a target image by image and streams each result, the per-stage cycles and the golden model's layer outputs to [`ui/live.html`](../ui/live.html) in the browser |

## Typical use (Git Bash, from the repo root)

```bash
PY=train/.venv/Scripts/python.exe
# golden model alone (no hardware)
$PY harness/run.py --model tern80 --target golden --csv /tmp/scratch.csv
# C reference on the PC (build it first, see firmware/baselines/README.md)
$PY harness/run.py --model tern80 --target host --exe firmware/baselines/ref_c/build-host/ref_host_tern80.exe
# the board, after flashing a build
$PY harness/run.py --model tern80 --target pico --port COM3 --elf <build>/ref_c.elf
$PY harness/run.py --model tern80 --target pico --port COM3 --golden s8 --elf <build>/cmsis_nn.elf
# everything
harness/phase0_matrix.sh final
```

`run.py` exits with status 1 if any logits differ from the golden model.

## Watching a run live

```bash
$PY harness/bridge.py --target pico --port COM3      # model and golden model picked from the board's INFO line
$PY harness/bridge.py --target host --exe firmware/baselines/ref_c/build-host/ref_host_tern80.exe
$PY harness/bridge.py --target golden                # no hardware
```

Then open http://127.0.0.1:8765. The page has three modes:

- **Live** follows the stream.
- **Slow-mo** steps the board one image at a time and moves through the stages 1000× slower than real time. The inspector follows along.
- **Replay** scrubs back through what has been received.

The inspector shows each layer's feature maps, FC1's −1/0/+1 weight matrix, and the FC outputs. These come from `golden.py`, so they are exactly what the board computed. Per-stage times need a `*_prof` build (`PROFILE_STAGES`). On other builds the bridge stops asking after the first all-zero `T` reply.

If the board disagrees with the golden model, the run pauses and the page shows a red banner with the differing logits. Pass `--no-stop-on-mismatch` to keep going. `--rate` sets images per second (the page has a speed control too). With `--target pico`, the bridge won't open the serial port while `harness/run.py`, `measure.sh`, `phase0_matrix.sh` or `picotool` is running, so it can't take the board from a measurement. `--force` overrides this. `golden` and `host` never touch the board. The bridge only listens on 127.0.0.1. It needs nothing beyond what `run.py` already uses, because it streams over Server-Sent Events from Python's standard-library HTTP server. It doesn't log to `results/`; use `run.py` for measured runs.

## What "matches PyTorch" means

The board doesn't do float arithmetic, so its predictions can't equal PyTorch's on 100% of images. The two checks are:

1. **Correctness, which must be exact:** board logits == golden-model logits on every image. The golden model is an independent numpy implementation of the same integer arithmetic.
2. **Fidelity, which is logged:** prediction agreement with the PyTorch model, plus accuracy against the labels. For tern80 this is 99.98% agreement, and the 2 images that differ are within the noise of the power-of-two rescaling.
