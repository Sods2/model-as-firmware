# harness/: PC-side test harness (Phase 0 step 6)

Sends MNIST test images to a target, checks the outputs bit for bit against a golden model, and logs one CSV row per run in `results/`.

| File | Role |
|---|---|
| `golden.py` | Numpy golden models. `Golden` repeats the plain-C reference arithmetic, reading weights straight from `train/export/<model>.h`. `GoldenS8` does the same for the CMSIS-NN int8 build, from `firmware/baselines/cmsis_nn/model_s8_<model>.h`. |
| `run.py` | Runs one target (`golden`, `host` = the C reference built for the PC, `pico` = the board over USB serial). Compares with the golden model, the PyTorch model and the labels, and appends a row to `results/phase0.csv` |
| `export_cmsis.py` | Calibrates and writes the CMSIS-NN int8 model header |
| `phase0_matrix.sh` | Builds, flashes and measures every Phase 0 configuration (`profile` or `final`) |
| `measure.sh` | Builds, flashes and measures any list of `build:model:mem` specs (`ref_c`, `cmsis_nn`, `mafc`), e.g. `harness/measure.sh final results/phase1.csv mafc:tern80:flash` |
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

If the board disagrees with the golden model, the run pauses and the page shows a red banner with the differing logits. Pass `--no-stop-on-mismatch` to keep going. `--rate` sets images per second (the page has a speed control too). The bridge only listens on 127.0.0.1. It needs nothing beyond what `run.py` already uses, because it streams over Server-Sent Events from Python's standard-library HTTP server. It doesn't log to `results/`; use `run.py` for measured runs.

## What "matches PyTorch" means

The board doesn't do float arithmetic, so its predictions can't equal PyTorch's on 100% of images. The two checks are:

1. **Correctness, which must be exact:** board logits == golden-model logits on every image. The golden model is an independent numpy implementation of the same integer arithmetic.
2. **Fidelity, which is logged:** prediction agreement with the PyTorch model, plus accuracy against the labels. For tern80 this is 99.98% agreement, and the 2 images that differ are within the noise of the power-of-two rescaling.
