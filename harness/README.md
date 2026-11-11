# harness/: PC-side test harness (Phase 0 step 6)

Sends MNIST test images to a target, checks the outputs bit for bit against a golden model, and logs one CSV row per run in `results/`.

| File | Role |
|---|---|
| `golden.py` | Numpy golden models. `Golden` repeats the plain-C reference arithmetic, reading weights straight from `train/export/<model>.h`. `GoldenS8` does the same for the CMSIS-NN int8 build, from `firmware/baselines/cmsis_nn/model_s8_<model>.h`. |
| `run.py` | Runs one target (`golden`, `host` = the C reference built for the PC, `pico` = the board over USB serial). Compares with the golden model, the PyTorch model and the labels, and appends a row to `results/phase0.csv` |
| `export_cmsis.py` | Calibrates and writes the CMSIS-NN int8 model header |
| `phase0_matrix.sh` | Builds, flashes and measures every Phase 0 configuration (`profile` or `final`) |

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

## What "matches PyTorch" means

The board doesn't do float arithmetic, so its predictions can't equal PyTorch's on 100% of images. The two checks are:

1. **Correctness, which must be exact:** board logits == golden-model logits on every image. The golden model is an independent numpy implementation of the same integer arithmetic.
2. **Fidelity, which is logged:** prediction agreement with the PyTorch model, plus accuracy against the labels. For tern80 this is 99.98% agreement, and the 2 images that differ are within the noise of the power-of-two rescaling.
