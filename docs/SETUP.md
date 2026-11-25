# Setup

Everything you need installed before following [REPRODUCE.md](REPRODUCE.md). How much you need depends on what you want to do:

| Goal | Hardware | Software sections below |
|---|---|---|
| Watch the models run on your PC, no board ("golden model" and "PC build" in the launcher) | none | 1, 2, 3 (MNIST; a story model for the story view), 5 for the PC build |
| Reproduce the board timings | a Pico 2, ideally a Debug Probe ([HARDWARE.md](HARDWARE.md)) | all of 1–5 |
| Retrain the models | a CUDA GPU helps a lot | 1, 2, 3 (BitNetMCU, TinyStories) |

**Tested on:** Windows 11 with Git Bash, Python 3.12, an RTX 3060 Ti for training. The scripts are bash and Python and should also work on Linux and macOS, but that hasn't been tested; tool locations can all be overridden (section 6). On Windows, run every command in this guide from **Git Bash** (installed with Git for Windows).

## 1. Get the repo

```bash
git clone https://github.com/Sods2/model-as-firmware.git
cd model-as-firmware
```

All commands in these guides run from the repo root.

## 2. Python environment

Python **3.12**, in a virtual environment at `train/.venv`. The scripts look for that location first, so you don't need to activate it for `harness/*.sh`; for Python commands, either activate it or call its `python` directly.

```bash
# Windows (Git Bash)
py -3.12 -m venv train/.venv
source train/.venv/Scripts/activate

# Linux / macOS
python3.12 -m venv train/.venv
source train/.venv/bin/activate
```

Then install the packages. For training on an NVIDIA GPU, install the CUDA build of PyTorch first (the results used torch 2.11.0 with CUDA 12.8); without a GPU, skip that line and the CPU build is installed.

```bash
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128   # GPU only
pip install -r requirements.txt
```

`requirements.txt` pins the versions the results were measured with: numpy, torch, torchvision, pyserial and PyYAML. That covers the compiler, the harness, the launcher and the story-model scripts. Retraining the MNIST models also needs BitNetMCU's own extras (section 3).

In the guides, `python` means this environment's interpreter.

## 3. Data and external code

None of these are in the repo. Each one goes to a gitignored folder, at a pinned version.

### MNIST test images (needed for every MNIST test, including the no-hardware ones)

```bash
python -c "from torchvision import datasets; [datasets.MNIST('train/BitNetMCU/data', train=t, download=True) for t in (False, True)]"
```

About 60 MB, into `train/BitNetMCU/data/MNIST/`. That is where BitNetMCU's training puts it too, so if you train first you can skip this.

### BitNetMCU (to retrain the MNIST models; also enables the "agrees with PyTorch" check)

The MNIST models were trained with [BitNetMCU](https://github.com/cpldcpu/BitNetMCU) (GPL-3.0) plus a small patch kept in this repo:

```bash
mkdir -p train/BitNetMCU
git -C train/BitNetMCU init -q
git -C train/BitNetMCU fetch https://github.com/cpldcpu/BitNetMCU.git
git -C train/BitNetMCU checkout 0715bfc4ed9f2578496e17b4b0e13f2297e3cc0f
git -C train/BitNetMCU apply ../bitnetmcu.patch
pip install tensorboard matplotlib torchsummary seaborn      # BitNetMCU's training extras
```

(This is a `git clone` that also works when the MNIST download above has already created `train/BitNetMCU/data`. The patch is described in [train/README.md](../train/README.md).)

Without the clone, everything else still works: the harness and the live view skip the comparison with the float PyTorch model and say so.

### CMSIS-NN (for the CMSIS-NN baseline and the wide-conv benchmark)

```bash
git clone https://github.com/ARM-software/CMSIS-NN.git firmware/baselines/cmsis_nn/third_party/CMSIS-NN
git -C firmware/baselines/cmsis_nn/third_party/CMSIS-NN checkout ece8aa0df4bee4fc928c294e18059071649c88bb
```

Why this is a fair baseline: [notes/cmsis-nn-baseline.md](../notes/cmsis-nn-baseline.md).

### TinyStories (to train or evaluate the story models)

```bash
mkdir -p train/data/tinystories
curl -L -o train/data/tinystories/TinyStoriesV2-GPT4-valid.txt https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt
curl -L -o train/data/tinystories/TinyStoriesV2-GPT4-train.txt https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
```

The validation file (about 22 MB) is enough for the PC checks and the board's exactness check; the training file (about 2.2 GB) is only for training. Dataset: Eldan & Li, 2023; see its [dataset card](https://huggingface.co/datasets/roneneldan/TinyStories) for the licence.

### Atome LM (only for the Atome comparison)

```bash
git clone https://github.com/TilelliLab/atome-lm.git train/atome-lm
git -C train/atome-lm checkout be12186
pip install --no-deps -e train/atome-lm
```

Its checkpoints (`train/atome-lm/checkpoints/`) come with the clone. Their licence is unclear (Atome's NOTICE and README disagree), which is why nothing of theirs except the unmodified engine source in `firmware/baselines/atome/upstream/` is in this repo.

### Trained story models

The trained story-model weights (`train/runs/lm_tree.npz`, `lm_dense.npz`) are not in the repo. The story view and the story-model board runs need them, so train them first: [REPRODUCE.md](REPRODUCE.md), "MafLM". The firmware's generated C *is* committed, so you can build and flash the story firmware without training, but the harness checks it against the `.npz`.

## 4. Pico 2 toolchain (for building and flashing firmware)

The versions the results were built with:

| Tool | Version |
|---|---|
| Pico SDK | 2.3.1 |
| Arm GNU toolchain (arm-none-eabi-gcc) | 15.2.Rel1 |
| CMake | 4.3.4 (3.13 or later should work) |
| Ninja | 1.13.2 |
| picotool | 2.3.1 |
| OpenOCD (Raspberry Pi fork) | 0.12.0+dev, for the Debug Probe |

Compiler and SDK versions affect the timings: a different GCC can move the numbers by a few percent.

**Route A, what was used: the VS Code "Raspberry Pi Pico" extension.** Install VS Code and the extension, then let it install the SDK 2.3.1 toolchain (for example by creating or importing any Pico 2 project, board `pico2`). Everything lands in `~/.pico-sdk/`, and the repo's scripts and CMake files find it there by themselves.

**Route B, without VS Code.** Install the Arm GNU toolchain, CMake, Ninja, picotool and OpenOCD from your package manager or their release pages, put them on PATH, and clone the SDK:

```bash
git clone -b 2.3.1 --recurse-submodules https://github.com/raspberrypi/pico-sdk.git ~/pico-sdk
export PICO_SDK_PATH=~/pico-sdk
```

Every firmware project includes the SDK's standard `pico_sdk_import.cmake`, so `PICO_SDK_PATH` is all CMake needs.

**Check:** build and flash the bring-up firmware. It blinks the LED and prints a cycle-counter timing over USB serial.

```bash
. harness/env.sh
cmake -S firmware/rp2350/hello -B firmware/rp2350/hello/build -G Ninja -DCMAKE_BUILD_TYPE=Release
ninja -C firmware/rp2350/hello/build
. harness/flash.sh && flash firmware/rp2350/hello/build/hello.uf2
```

The first time, if the board has never been flashed with this repo's firmware and there is no Debug Probe, hold BOOTSEL while plugging it in and copy `hello.uf2` onto the `RP2350` drive instead.

## 5. A C compiler for the PC (for the "PC build" target)

Any C11 compiler (gcc, clang). The results used 32-bit MinGW gcc 9.2 on Windows; on Linux and macOS, the system gcc or clang works. Set `HOST_CC` if it isn't called `gcc`. Build steps: [firmware/baselines/README.md](../firmware/baselines/README.md).

## 6. Where the scripts look for things

[harness/env.sh](../harness/env.sh), sourced by every shell script, sets these. Set any of them yourself to override.

| Variable | Default |
|---|---|
| `PY` | `train/.venv/Scripts/python.exe`, then `train/.venv/bin/python`, then `python3` |
| `PORT` | `COM3` on Windows, `/dev/ttyACM0` elsewhere. Set it to your board's port. |
| `PICO_SDK_HOME` | `~/.pico-sdk` (the VS Code extension's folder) |
| `PICOTOOL`, `OPENOCD`, `OPENOCD_SCRIPTS` | the extension's copies, else PATH |
| `FLASH` | `auto`: SWD with the Debug Probe, then picotool over USB. `swd` or `usb` forces one. |
| `ARM_SIZE` | `arm-none-eabi-size`, used by `harness/run.py` for flash/RAM sizes |
| `HOST_CC` | the PC's C compiler, used by `harness/lm_host_check.py` |

The Python tools take the serial port as `--port`.

More detail on the toolchain, the Debug Probe and their quirks: [notes/pico-toolchain.md](../notes/pico-toolchain.md).
