# train/ — ternary MNIST models (BitNetMCU)

Phase 0, step 3: reproduce BitNetMCU's ternary MNIST model, plus sparse variants for the sparse-storage experiment.

## Setup (Windows, Python 3.12, CUDA GPU)
```
py -3.12 -m venv .venv
.venv/Scripts/python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/python.exe -m pip install numpy PyYAML tensorboard matplotlib torchsummary seaborn
git clone https://github.com/cpldcpu/BitNetMCU.git BitNetMCU
```
`torchsummary` and `seaborn` are imported by BitNetMCU but missing from its `requirements.txt`.

## Train / export / inspect (run from `train/BitNetMCU`)
```
../.venv/Scripts/python.exe training.py    --params ../configs/ternary_cnn80.yaml
../.venv/Scripts/python.exe exportquant.py --params ../configs/ternary_cnn80.yaml   # writes BitNetMCU_model.h -> move to ../export/
../.venv/Scripts/python.exe ../sparsity_report.py ../configs/ternary_cnn80.yaml modeldata/<runname>.pth
```
About 24 s per epoch on an RTX 3060 Ti, so a 60-epoch run takes about 25 min. Export reads `modeldata/<runname>.pth`, which is the final epoch, not `_best`.

## Configs
| Config | fc1 zeros | Notes |
|---|---|---|
| `ternary_cnn80.yaml` | 50.2% (natural, BitNet absmean rule) | BitNetMCU's "80-wide ternary" run; they report 99.52% |
| `ternary_cnn80_z70.yaml` | 70% (forced) | sparse-storage variant |
| `ternary_cnn80_z90.yaml` | 90% (forced) | sparse-storage variant |
| `int8_cnn80.yaml` | 5% (natural) | the same CNN with every linear layer int8 (`8bit`), for the int8 compiler test (FINDINGS F24) |
| `fff_cnn80.yaml` | 55% (leaf inputs) | idea B: fc1 + fc2 replaced by a depth-3 fast feedforward tree (`CNNMNIST_FFF`), 8 leaves of 12; export with `train/export_fff.py` (FINDINGS F25) |

Model: 8-bit depthwise CNN (80 ch) → **fc1 320→96 ternary** → fc2 96→64 4-bit → classifier 64→10 4-bit. Only fc1 is ternary. It holds 30,720 of the 39,664 weights. Total size 11.42 kB.

## Local patches to BitNetMCU (marked `model-as-firmware patch`)
- `models.py` `CNNMNIST`: new `fc1_quant` arg (upstream hardcodes `'2bitsym'`) and `ternary_zero_frac` arg.
- `BitNetMCU.py` `weight_quant`: when `ternary_zero_frac > 0`, a Ternary layer zeroes exactly that fraction of its smallest-|w| weights. The remaining weights become ±1, scaled by the mean of the kept |w|.
- `exportquant.py` `load_model`: forwards `fc1_quant` and `ternary_zero_frac`. Without this, upstream export silently re-quantized a ternary fc1 as 2-bit.
- `models.py` `FFFLayer`, `CNNMNIST_FFF` (idea B): a fast feedforward tree layer with soft routing, a hardening loss, a load-balancing loss and straight-through hard routing from `fff_hard_epoch`. Evaluation (no grad) always routes hard.
- `training.py`: adds `model.aux_loss()` to the loss and calls `model.set_epoch(epoch)` when the model has them; prints `model.fff_info()` (entropy, balance, leaf use).

The clone isn't tracked in this repo. The patches are saved in `bitnetmcu.patch`, made against upstream commit `0715bfc`. Because it changes GPL-3.0 code, that one file is GPL-3.0; the rest of the repo is Apache-2.0 (see [NOTICE](../NOTICE)). To apply them after a fresh clone:
```
git -C BitNetMCU checkout 0715bfc4ed9f2578496e17b4b0e13f2297e3cc0f
git -C BitNetMCU apply ../bitnetmcu.patch
```

## Results
| Model | fc1 zeros | Best test acc | Final test acc | Quantized (integer) acc |
|---|---|---|---|---|
| tern80 | 50.2% | 99.48% (ep 52) | 99.45% | 99.47% |
| tern80z70 | 70.0% | 99.55% (ep 58) | 99.44% | 99.45% |
| tern80z90 | 90.0% | 99.54% (ep 59) | 99.50% | 99.49% |
| int8_80 (all linear int8) | 5% | 99.50% (ep 45) | 99.43% | 99.45% |
| fff80 (tree, hard routing) | 55% | 99.19% (ep 50) | 99.15% | 99.15% |

Even with 90% of fc1's weights set to zero, accuracy doesn't drop. The differences are within the epoch-to-epoch spread of a single run; each configuration was trained once (one seed), so there is no seed study behind this. The "best" column is the best epoch picked on the test set, which is optimistic; the final-epoch numbers are the fair ones, and the exports use the final epoch. Final-epoch weights are in `models/*.pth`, and the exported C headers are in `export/*.h`. Note that the packed ternary format is a fixed 1.6 bits per weight however many zeros there are, so the size savings would have to come from a sparse format in the compiler. *(Measured later: every format tried, sparse lists and code alike, is larger than the 6 KB packed one; zeros buy speed in the code format, not size. FINDINGS F16.)*

## Story models (Phase 3)

- **`lm/`: MafLM, our own ternary byte-level story model** (dense and tree twins), with its trainer, integer reference and overnight script. See [notes/lm.md](../notes/lm.md).
- **Data:** TinyStories (Eldan & Li, 2023; Hugging Face `roneneldan/TinyStories`), files `TinyStoriesV2-GPT4-train.txt` and `-valid.txt`, in `data/tinystories/` (gitignored). See the dataset card for its licence.
- **`atome/`: scripts that evaluate Atome LM** (TilelliLab, Apache-2.0 code), cloned to `atome-lm/` (gitignored). The licence of its trained weights is unclear (their NOTICE and README disagree), so neither the weights nor the cloned repo are committed here. `atome/fairness.py` reproduces the fairness table in notes/lm.md.
